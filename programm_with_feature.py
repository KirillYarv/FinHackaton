"""
Финансовый прогнозирующий сервис с поддержкой произвольных горизонтов k
"""

from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from scipy.interpolate import interp1d
from scipy.fft import fft, fftfreq
from scipy.stats import variation, moment
from scipy.stats.mstats import gmean
import requests
import time
import os
import json
import joblib
from typing import Dict, List, Optional, Tuple, Union
import random
from scipy.stats import kurtosis, skew, entropy
import nolds  # для фрактального анализа
import antropy as ent  # для энтропийных мер
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

# Фиксируем все сиды для воспроизводимости
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)
load_dotenv()

# Проверяем доступность GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🎯 Используется устройство: {device}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name()}")
    print(f"   Память GPU: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

class EvaluationMetrics:
    """Класс для расчета метрик оценки модели с поддержкой произвольных горизонтов"""

    def __init__(self, horizons: List[int] = [1, 20]):
        self.horizons = sorted(horizons)
        self.max_horizon = max(horizons)

    def calculate_targets(self, df: pd.DataFrame, price_col: str = 'close') -> Dict[int, pd.Series]:
        """Расчет целевых переменных для всех горизонтов"""
        targets = {}
        for k in self.horizons:
            targets[k] = (df[price_col].shift(-k) / df[price_col] - 1)
        return targets

    def calculate_mae(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Средняя абсолютная ошибка"""
        return np.mean(np.abs(y_true - y_pred))

    def calculate_brier(self, y_true: np.ndarray, prob_up: np.ndarray) -> float:
        """Brier score для вероятности роста"""
        binary_targets = (y_true > 0).astype(float)
        return np.mean((binary_targets - prob_up) ** 2)

    def calculate_direction_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Доля верно предсказанного знака (Direction Accuracy)"""
        correct_signs = np.sign(y_true) == np.sign(y_pred)
        return correct_signs.mean()

    def normalize_metrics(self, mae: float, brier: float,
                         mae_base: float, brier_base: float) -> Tuple[float, float]:
        """Нормировка метрик относительно бейзлайна"""
        mae_norm = max(0, 1 - (mae / mae_base)) if mae_base > 0 else 0
        brier_norm = max(0, 1 - (brier / brier_base)) if brier_base > 0 else 0
        return mae_norm, brier_norm

    def calculate_final_score(self, y_true: np.ndarray, y_pred: np.ndarray,
                            prob_up: np.ndarray, mae_base: float, brier_base: float) -> Dict:
        """Расчет итогового скора по формуле"""

        mae = self.calculate_mae(y_true, y_pred)
        brier = self.calculate_brier(y_true, prob_up)
        da = self.calculate_direction_accuracy(y_true, y_pred)

        mae_norm, brier_norm = self.normalize_metrics(mae, brier, mae_base, brier_base)

        # Итоговый скор с защитой от деления на ноль
        da_component = 0.1 * (1 / da) if da > 0 else 0
        final_score = 0.7 * mae_norm + 0.3 * brier_norm + da_component

        return {
            'mae': mae,
            'brier': brier,
            'direction_accuracy': da,
            'mae_norm': mae_norm,
            'brier_norm': brier_norm,
            'final_score': final_score
        }

class BaselineModel:
    """
    Улучшенная Baseline модель с поддержкой произвольных горизонтов k
    """

    def __init__(self, horizons: List[int] = [1, 20], window_size: int = 5):
        self.horizons = sorted(horizons)
        self.window_size = window_size

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Вычисление признаков (моментум, волатильность)"""
        df_processed = df.copy()

        # Группируем по тикерам
        for ticker in df_processed['ticker'].unique():
            mask = df_processed['ticker'] == ticker
            ticker_data = df_processed[mask].copy()

            # 1. Моментум = процентное изменение цены за window_size дней
            ticker_data['momentum'] = (
                ticker_data['close'].pct_change(self.window_size)
            )

            # 2. Волатильность = std доходностей за window_size дней
            ticker_data['volatility'] = (
                ticker_data['close'].pct_change().rolling(self.window_size).std()
            )

            # 3. Средняя цена за window_size дней
            ticker_data['ma'] = ticker_data['close'].rolling(self.window_size).mean()

            # 4. Расстояние от MA (нормализованное)
            ticker_data['distance_from_ma'] = (
                (ticker_data['close'] - ticker_data['ma']) / ticker_data['ma']
            )

            # Обновляем данные
            df_processed.loc[mask, 'momentum'] = ticker_data['momentum'].values
            df_processed.loc[mask, 'volatility'] = ticker_data['volatility'].values
            df_processed.loc[mask, 'ma'] = ticker_data['ma'].values
            df_processed.loc[mask, 'distance_from_ma'] = ticker_data['distance_from_ma'].values

        return df_processed

    def predict(self, df: pd.DataFrame) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
        """
        Создание предсказаний для всех горизонтов

        Returns:
            predictions: массив [return_k1, return_k2, ..., logit_k1, logit_k2, ...]
            prob_up_dict: словарь {k: prob_up_k}
        """
        # Вычисляем признаки
        df_with_features = self.compute_features(df)

        # Заполняем NaN нулями (для первых строк где нет истории)
        df_with_features['momentum'] = df_with_features['momentum'].fillna(0)
        df_with_features['volatility'] = df_with_features['volatility'].fillna(0.01)
        df_with_features['distance_from_ma'] = df_with_features['distance_from_ma'].fillna(0)

        predictions_list = []
        prob_up_dict = {}

        for k in self.horizons:
            # Стратегия: предсказываем что моментум продолжится
            # Коэффициент масштабирования зависит от горизонта
            scaling_factor = min(1.0, k / 5.0)  # Для длинных горизонтов сильнее эффект

            returns_k = df_with_features['momentum'] * scaling_factor

            # Предсказание вероятности роста
            def sigmoid(x, sensitivity=10):
                return 1 / (1 + np.exp(-sensitivity * x))

            # Чувствительность зависит от горизонта
            sensitivity = max(5, 15 - k)  # Для длинных горизонтов меньше чувствительность
            prob_up_k = sigmoid(df_with_features['momentum'], sensitivity=sensitivity)

            # Clipping: вероятности в диапазоне [0.1, 0.9] для стабильности
            prob_up_k = prob_up_k.clip(0.1, 0.9)

            # Clipping: доходности в разумном диапазоне
            max_return = min(0.5, k * 0.05)  # Зависит от горизонта
            returns_k = returns_k.clip(-max_return, max_return)

            predictions_list.extend([returns_k])
            prob_up_dict[k] = prob_up_k

        # Собираем все предсказания
        all_returns = np.column_stack(predictions_list)

        # Добавляем logits для совместимости с основной моделью
        all_logits = []
        for k in self.horizons:
            prob_up_k = prob_up_dict[k]
            logit_k = np.log(prob_up_k / (1 - prob_up_k))
            all_logits.append(logit_k)

        all_logits = np.column_stack(all_logits)
        predictions = np.column_stack([all_returns, all_logits])

        return predictions, prob_up_dict

    @staticmethod
    def naive_forecast(df: pd.DataFrame, horizon: int = 20) -> Tuple[np.ndarray, np.ndarray]:
        """Наивный прогноз: средняя историческая доходность"""
        returns = df['close'].pct_change().dropna()
        avg_return = returns.mean() if len(returns) > 0 else 0

        y_pred = np.full(len(df), avg_return)
        prob_up = np.full(len(df), 0.5)

        return y_pred, prob_up


class FinBertNewsProcessor:
    """Обработчик новостей с использованием FinBERT модели"""

    def __init__(self, model_name: str = "yiyanghkust/finbert-tone"):
        self.model_name = model_name
        self.device = device
        self.max_length = 512

        # Загружаем модель и токенизатор
        print(f"📰 Загрузка FinBERT модели: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        # Размер эмбеддингов для FinBERT
        self.embedding_dim = 768  # BERT base model

        print(f"✅ FinBERT модель загружена на устройство: {self.device}")
        print(f"   Размер эмбеддингов: {self.embedding_dim}")

    def get_news_embeddings(self, texts: List[str], batch_size: int = 16) -> np.ndarray:
        """Получение эмбеддингов новостей с помощью FinBERT"""
        if len(texts) == 0:
            return np.zeros((0, self.embedding_dim))

        # Фильтруем пустые тексты
        valid_texts = []
        valid_indices = []

        for i, text in enumerate(texts):
            if text and len(text.strip()) > 0:
                valid_texts.append(text.strip())
                valid_indices.append(i)

        if len(valid_texts) == 0:
            # Возвращаем нулевые эмбеддинги для всех исходных текстов
            return np.zeros((len(texts), self.embedding_dim))

        embeddings = np.zeros((len(texts), self.embedding_dim))

        # Остальной код без изменений...
        total_batches = (len(valid_texts) + batch_size - 1) // batch_size

        print(f"📡 Получение эмбеддингов через FinBERT...")
        print(f"   Модель: {self.model_name}, Текстов: {len(valid_texts)}/{len(texts)}, Батчей: {total_batches}")

        for i in range(0, len(valid_texts), batch_size):
            batch_texts = valid_texts[i:i + batch_size]
            batch_indices = valid_indices[i:i + batch_size]
            batch_num = i // batch_size + 1

            try:
                # Токенизация текстов
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                ).to(self.device)

                # Получение эмбеддингов
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    # Используем [CLS] токен как представление всего текста
                    batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

                    # Записываем эмбеддинги в соответствующие позиции
                    for j, idx in enumerate(batch_indices):
                        embeddings[idx] = batch_embeddings[j]

                print(f"   ✅ Батч {batch_num}/{total_batches} обработан ({len(batch_texts)} текстов)")

            except Exception as e:
                print(f"   ❌ Ошибка в батче {batch_num}: {e}")
                # Для неудачных текстов оставляем нулевые эмбеддинги

        print(f"✅ Все эмбеддинги успешно получены!")
        return embeddings


    def get_sentiment_scores(self, texts: List[str], batch_size: int = 16) -> np.ndarray:
        """Получение sentiment scores с помощью FinBERT (опционально)"""
        if len(texts) == 0:
            return np.zeros((0, 3))  # positive, negative, neutral

        # Загружаем модель для классификации тональности
        sentiment_model = AutoModelForSequenceClassification.from_pretrained(self.model_name).to(self.device)
        sentiment_model.eval()

        sentiment_scores = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = sentiment_model(**inputs)
                probs = F.softmax(outputs.logits, dim=-1).cpu().numpy()
                sentiment_scores.extend(probs)

        return np.array(sentiment_scores)


class HybridTransformerModel(nn.Module):
    """Гибридная модель: Transformer + FinBERT + линейные слои с поддержкой произвольных горизонтов"""

    def __init__(self, ts_input_size: int, llm_embedding_dim: int, horizons: List[int] = [1, 20], hidden_size: int = 256):
        super().__init__()
        self.horizons = sorted(horizons)
        self.num_horizons = len(horizons)

        # Каждый горизонт требует 2 выхода: return и logit
        output_size = 2 * self.num_horizons

        self.ts_transformer = TimeSeriesTransformer(
            input_size=ts_input_size,
            d_model=128,
            nhead=8,
            num_layers=3
        )

        self.llm_projection = nn.Linear(llm_embedding_dim, 128)

        total_input_size = 128 + 128

        self.fusion_network = nn.Sequential(
            nn.Linear(total_input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, output_size)
        )

    def forward(self, timeseries, news_embeddings):
        ts_features = self.ts_transformer(timeseries)
        news_features = self.llm_projection(news_embeddings)
        combined = torch.cat([ts_features, news_features], dim=1)
        output = self.fusion_network(combined)
        return output


class TimeSeriesDataset(Dataset):
    """Датасет для временных рядов с поддержкой произвольных горизонтов"""

    def __init__(self, features, news_embeddings, targets_dict: Dict[int, np.ndarray], seq_length=30):
        self.features = features
        self.news_embeddings = news_embeddings
        self.targets_dict = targets_dict
        self.horizons = sorted(targets_dict.keys())
        self.seq_length = seq_length

    def __len__(self):
        return len(self.features) - self.seq_length

    def __getitem__(self, idx):
        features_seq = self.features[idx:idx+self.seq_length]
        news_embedding = self.news_embeddings[idx+self.seq_length]

        # Собираем все таргеты для разных горизонтов
        targets = []
        for horizon in self.horizons:
            target = self.targets_dict[horizon][idx+self.seq_length]
            targets.append(target)

        return (
            torch.FloatTensor(features_seq),
            torch.FloatTensor(news_embedding),
            torch.FloatTensor(targets)
        )


class FinancialForecastService:
    """Сервис финансового прогнозирования с поддержкой произвольных горизонтов k"""

    def __init__(self, model_dir: str = "model_artifacts", seq_length: int = 30,
                 k_days: List[int] = [1, 20]):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        self.seq_length = seq_length
        self.k_days = sorted(k_days)
        self.device = device

        self.scaler = StandardScaler()
        self.news_processor = FinBertNewsProcessor()  # ✅ Заменяем на FinBERT
        self.feature_generator = FeatureGenerator()
        self.evaluator = EvaluationMetrics(horizons=self.k_days)
        self.baseline_model = BaselineModel(horizons=self.k_days, window_size=5)
        self.model = None

        self.feature_columns = None
        self.selected_features = None
        # Бейзлайн метрики для каждого горизонта
        self.mae_base_dict = {k: 0.01 for k in self.k_days}
        self.brier_base_dict = {k: 0.25 for k in self.k_days}

        print(f"🎯 Инициализирован сервис с горизонтами: {self.k_days}")

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Создание расширенных признаков для каждого тикера"""
        print("🔧 Генерация расширенных признаков...")

        all_features = []

        for ticker in df['ticker'].unique():
            ticker_data = df[df['ticker'] == ticker].copy().sort_values('begin')

            if len(ticker_data) < 30:
                continue

            prices = ticker_data['close'].values
            dates = pd.DatetimeIndex(ticker_data['begin'])

            ticker_features = self.feature_generator.create_features(prices, dates)

            ticker_features['ticker'] = ticker
            ticker_features['begin'] = dates

            all_features.append(ticker_features)

        if not all_features:
            return df

        features_df = pd.concat(all_features, ignore_index=True)

        result_df = pd.merge(df, features_df, on=['ticker', 'begin'], how='left')

        result_df = result_df.fillna(method='ffill').fillna(method='bfill')

        print(f"   ✓ Сгенерировано {len(features_df.columns) - 2} признаков")
        return result_df

    def prepare_news_features(self, news_df: pd.DataFrame, candles_df: pd.DataFrame) -> pd.DataFrame:
        """Подготовка признаков из новостей с использованием FinBERT"""
        print("📰 Обработка новостей через FinBERT...")

        if news_df is None or len(news_df) == 0:
            # Создаем список нулевых эмбеддингов правильной длины
            zero_embeddings = [np.zeros(768) for _ in range(len(candles_df))]
            candles_df = candles_df.copy()
            candles_df['news_embedding'] = zero_embeddings
            return candles_df

        news_df = news_df.copy()
        news_df['publish_date'] = pd.to_datetime(news_df['publish_date'])
        news_df['date'] = news_df['publish_date'].dt.date

        news_df['full_text'] = news_df['title'] + ". " + news_df['publication']

        # Группируем новости по дате
        grouped_news = news_df.groupby(['date'])['full_text'].apply(
            lambda x: ' '.join(x) if len(x) > 0 else ""
        ).reset_index()

        news_texts = grouped_news['full_text'].tolist()
        print(f"   📊 Получение эмбеддингов FinBERT для {len(news_texts)} групп новостей...")
        news_embeddings = self.news_processor.get_news_embeddings(news_texts)

        # Сохраняем эмбеддинги в DataFrame
        grouped_news['news_embedding'] = list(news_embeddings)

        # Подготавливаем candles_df для объединения
        candles_df = candles_df.copy()
        candles_df['date'] = pd.to_datetime(candles_df['begin']).dt.date

        # Объединяем данные
        merged_df = pd.merge(
            candles_df,
            grouped_news[['date', 'news_embedding']],
            on='date',
            how='left'
        )

        # Заполняем пропущенные эмбеддинги нулями - ИСПРАВЛЕННАЯ ЧАСТЬ
        mask = merged_df['news_embedding'].isna()
        if mask.any():
            # Создаем список нулевых эмбеддингов для пропущенных строк
            zero_embeddings_list = [np.zeros(768) for _ in range(mask.sum())]

            # Преобразуем в Series с правильным индексом
            zero_embeddings_series = pd.Series(zero_embeddings_list, index=merged_df[mask].index)

            # Присваиваем значения
            merged_df.loc[mask, 'news_embedding'] = zero_embeddings_series

        print(f"   ✅ Обработано {len(merged_df)} строк с новостными эмбеддингами")
        return merged_df

    def prepare_targets(self, df: pd.DataFrame) -> Dict[int, pd.Series]:
        """Подготовка целевых переменных для всех горизонтов по формуле: R_{t+N} = close_{t+N} / close_t - 1"""
        df = df.sort_values(['ticker', 'begin']).copy()
        targets_dict = {}

        for k in self.k_days:
            # Группируем по тикеру и вычисляем доходность за k дней
            targets = df.groupby('ticker').apply(
                lambda x: (x['close'].shift(-k) / x['close'] - 1)
            ).reset_index(level=0, drop=True)

            targets_dict[k] = targets

        return targets_dict

    def _calculate_baseline_metrics(self, df: pd.DataFrame) -> Tuple[Dict, Dict]:
        """Расчет бейзлайн метрик на исторических данных для всех горизонтов"""
        print("📊 Расчет бейзлайн метрик для всех горизонтов...")

        eval_df = df.tail(max(100, len(df) // 5)).copy()

        if len(eval_df) < 10:
            print("   ⚠️ Недостаточно данных для расчета бейзлайна, используются дефолтные значения")
            return {k: 0.01 for k in self.k_days}, {k: 0.25 for k in self.k_days}

        mae_base_dict = {}
        brier_base_dict = {}

        for k in self.k_days:
            actual_return_col = f'actual_return_{k}d'
            eval_df[actual_return_col] = (eval_df['close'].shift(-k) / eval_df['close'] - 1)
            eval_df_clean = eval_df.dropna(subset=[actual_return_col])

            if len(eval_df_clean) == 0:
                print(f"   ⚠️ Нет данных для горизонта {k}, используются дефолтные значения")
                mae_base_dict[k] = 0.01
                brier_base_dict[k] = 0.25
                continue

            # Используем улучшенный baseline для расчета метрик
            baseline_predictions, baseline_probs_dict = self.baseline_model.predict(eval_df_clean)
            y_true = eval_df_clean[actual_return_col].values

            # Индекс предсказания для текущего горизонта
            horizon_idx = self.k_days.index(k)
            y_pred_baseline = baseline_predictions[:, horizon_idx]
            prob_up_baseline = baseline_probs_dict[k]

            mae_base = self.evaluator.calculate_mae(y_true, y_pred_baseline)
            brier_base = self.evaluator.calculate_brier(y_true, prob_up_baseline)

            mae_base_dict[k] = mae_base
            brier_base_dict[k] = brier_base

            print(f"   ✅ Горизонт {k} дней: MAE={mae_base:.6f}, Brier={brier_base:.6f}")

        return mae_base_dict, brier_base_dict

    def select_features(self, X_train, y_train, X_val, y_val):
        """Отбор наиболее важных признаков"""
        print("🎯 Отбор признаков...")

        self.selected_features, scores = FeatureSelector.forward_selection(
            X_train, y_train, X_val, y_val, max_features=50, min_improvement=0.001
        )

        return self.selected_features

    def train(self, candles_df: pd.DataFrame, news_df: pd.DataFrame, val_ratio: float = 0.2):
        """Обучение модели с поддержкой произвольных горизонтов"""
        print(f"🎯 Обучение модели для горизонтов: {self.k_days}...")

        # Создаем копии чтобы не модифицировать оригинальные данные
        candles_df = candles_df.copy()
        if news_df is not None:
            news_df = news_df.copy()

        candles_df = self.create_features(candles_df)
        targets_dict = self.prepare_targets(candles_df)

        full_df = self.prepare_news_features(news_df, candles_df)

        # Удаляем строки с NaN в таргетах
        target_columns = [f'target_{k}d' for k in self.k_days]
        for k in self.k_days:
            full_df[f'target_{k}d'] = targets_dict[k]

        # Удаляем строки где все таргеты NaN
        full_df = full_df.dropna(subset=target_columns)

        if len(full_df) == 0:
            print("❌ Нет данных для обучения после очистки NaN")
            return

        # Расчет бейзлайн метрик
        self.mae_base_dict, self.brier_base_dict = self._calculate_baseline_metrics(full_df)

        # Выбираем только числовые колонки
        numeric_columns = full_df.select_dtypes(include=[np.number]).columns.tolist()
        # Исключаем таргеты и эмбеддинги
        exclude_columns = target_columns + ['news_embedding']
        self.feature_columns = [col for col in numeric_columns if col not in exclude_columns]

        print(f"   Доступно {len(self.feature_columns)} числовых признаков")

        # Сортируем по дате и разбиваем на train/val
        full_df = full_df.sort_values('begin')
        split_idx = int(len(full_df) * (1 - val_ratio))
        train_df = full_df.iloc[:split_idx]
        val_df = full_df.iloc[split_idx:]

        print(f"   Train samples: {len(train_df)}, Val samples: {len(val_df)}")

        if len(train_df) == 0 or len(val_df) == 0:
            print("❌ Недостаточно данных для обучения/валидации")
            return

        # Продолжение оригинального кода...
        X_train = train_df[self.feature_columns].values
        X_val = val_df[self.feature_columns].values

        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        # Используем первый горизонт для отбора признаков
        y_train = train_df[f'target_{self.k_days[0]}d'].values
        y_val = val_df[f'target_{self.k_days[0]}d'].values

        self.selected_features = self.select_features(X_train_scaled, y_train, X_val_scaled, y_val)

        print(f"   Используется {len(self.selected_features)} отобранных признаков")

        # Преобразуем эмбеддинги в numpy array
        news_embeddings = np.stack(full_df['news_embedding'].values)

        # Подготавливаем таргеты для обучения
        train_targets_dict = {}
        val_targets_dict = {}
        for k in self.k_days:
            train_targets_dict[k] = train_df[f'target_{k}d'].values
            val_targets_dict[k] = val_df[f'target_{k}d'].values

        self.model = HybridTransformerModel(
            ts_input_size=len(self.selected_features),
            llm_embedding_dim=news_embeddings.shape[1],  # FinBERT embedding size
            horizons=self.k_days,
            hidden_size=256
        ).to(self.device)

        self._train_model(train_df, val_df, train_targets_dict, val_targets_dict, news_embeddings)

        self._save_artifacts()

        print("✅ Модель обучена и артефакты сохранены")


    def _train_model(self, train_df: pd.DataFrame, val_df: pd.DataFrame,
                    train_targets_dict: Dict, val_targets_dict: Dict, news_embeddings: np.ndarray):
        """Обучение нейросетевой модели с произвольными горизонтами"""
        print("🧠 Обучение Transformer модели...")

        X_train = self.scaler.transform(train_df[self.feature_columns].values)
        X_train_selected = X_train[:, self.selected_features]

        train_news_embeddings = news_embeddings[:len(train_df)]

        train_dataset = TimeSeriesDataset(
            X_train_selected, train_news_embeddings,
            train_targets_dict, self.seq_length
        )
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=0.01)
        criterion = nn.MSELoss()

        self.model.train()
        for epoch in range(20):
            total_loss = 0
            for batch_ts, batch_news, batch_targets in train_loader:
                batch_ts = batch_ts.to(self.device)
                batch_news = batch_news.to(self.device)
                batch_targets = batch_targets.to(self.device)

                optimizer.zero_grad()

                outputs = self.model(batch_ts, batch_news)

                # Разделяем выходы на returns и logits
                num_horizons = len(self.k_days)
                pred_returns = outputs[:, :num_horizons]
                # pred_logits = outputs[:, num_horizons:]  # Не используем пока

                loss = criterion(pred_returns, batch_targets)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()

            if epoch % 5 == 0:
                val_loss = self._validate(val_df, val_targets_dict, news_embeddings[len(train_df):])
                print(f"   Epoch {epoch}, Train Loss: {total_loss/len(train_loader):.6f}, Val Loss: {val_loss:.6f}")

    def _validate(self, val_df: pd.DataFrame, val_targets_dict: Dict, val_news_embeddings: np.ndarray) -> float:
        """Валидация модели"""
        self.model.eval()

        X_val = self.scaler.transform(val_df[self.feature_columns].values)
        X_val_selected = X_val[:, self.selected_features]

        total_loss = 0
        criterion = nn.MSELoss()
        num_horizons = len(self.k_days)

        with torch.no_grad():
            for i in range(len(X_val_selected) - self.seq_length):
                ts_seq = torch.FloatTensor(X_val_selected[i:i+self.seq_length]).unsqueeze(0).to(self.device)
                news_embed = torch.FloatTensor(val_news_embeddings[i+self.seq_length]).unsqueeze(0).to(self.device)

                # Подготавливаем таргеты
                targets = []
                for k in self.k_days:
                    target_val = val_targets_dict[k][i+self.seq_length]
                    targets.append(target_val)
                batch_targets = torch.FloatTensor([targets]).to(self.device)

                output = self.model(ts_seq, news_embed)
                pred_returns = output[:, :num_horizons]

                loss = criterion(pred_returns, batch_targets)
                total_loss += loss.item()

        self.model.train()
        return total_loss / (len(X_val_selected) - self.seq_length)

    def predict(self, candles_df: pd.DataFrame, news_df: pd.DataFrame) -> pd.DataFrame:
        """Предсказание на новых данных в формате sample_submission.csv"""
        if self.model is None:
            self._load_artifacts()

        print(f"🎯 Прогнозирование для горизонтов 1-20 дней...")

        # Берем только последнюю дату для каждого тикера (дата t)
        latest_data = candles_df.sort_values(['ticker', 'begin']).groupby('ticker').last().reset_index()
        print(f"   Прогнозирование на дату: {latest_data['begin'].iloc[0]}")

        # Создаем фичи для последней даты
        candles_with_features = self.create_features(candles_df)
        full_df = self.prepare_news_features(news_df, candles_with_features)

        # Берем последовательности для последней даты каждого тикера
        predictions_data = []

        for ticker in latest_data['ticker'].unique():
            ticker_data = full_df[full_df['ticker'] == ticker].sort_values('begin')

            if len(ticker_data) < self.seq_length:
                print(f"   ⚠️ Недостаточно данных для тикера {ticker}, пропускаем")
                continue

            # Берем последнюю последовательность длины seq_length
            X_ticker = self.scaler.transform(ticker_data[self.feature_columns].values)
            X_selected = X_ticker[-self.seq_length:, self.selected_features]

            news_embedding = ticker_data['news_embedding'].values[-1]

            # Предсказываем
            self.model.eval()
            with torch.no_grad():
                ts_seq = torch.FloatTensor(X_selected).unsqueeze(0).to(self.device)
                news_embed = torch.FloatTensor(news_embedding).unsqueeze(0).to(self.device)

                output = self.model(ts_seq, news_embed)
                prediction = output.cpu().squeeze().numpy()

            # Берем только предсказания доходностей (первые num_horizons значений)
            num_horizons = len(self.k_days)
            returns_prediction = prediction[:num_horizons]

            predictions_data.append({
            'ticker': ticker,
                'predictions': returns_prediction
            })

        if not predictions_data:
            print("⚠️ Не удалось сгенерировать предсказания")
            return pd.DataFrame()

        # Формируем выход в формате sample_submission.csv
        output_rows = []

        for item in predictions_data:
            ticker = item['ticker']
            predictions = item['predictions']

            row = {'ticker': ticker}

            # Создаем 20 предсказаний (p1-p20)
            # Если у нас меньше 20 горизонтов, интерполируем
            if len(predictions) >= 20:
                # Берем первые 20 предсказаний
                for i in range(20):
                    row[f'p{i+1}'] = float(predictions[i])
            else:
                # Интерполируем имеющиеся предсказания до 20
                available_horizons = self.k_days[:len(predictions)]
                available_predictions = predictions

                # Создаем интерполятор

                try:
                    # Линейная интерполяция по горизонтам
                    interp_func = interp1d(
                        available_horizons,
                        available_predictions,
                        kind='linear',
                        fill_value='extrapolate'
                    )

                    # Генерируем предсказания для горизонтов 1-20
                    for i in range(1, 21):
                        row[f'p{i}'] = float(interp_func(i))

                except:
                    # Если интерполяция не удалась, дублируем последнее значение
                    for i in range(1, 21):
                        if i <= len(predictions):
                            row[f'p{i}'] = float(predictions[i-1])
                        else:
                            row[f'p{i}'] = float(predictions[-1])

            output_rows.append(row)

        result_df = pd.DataFrame(output_rows)

        # Сортируем по тикерам как в sample_submission.csv
        if not result_df.empty:
            all_tickers = ['AFLT', 'ALRS', 'CHMF', 'GAZP', 'GMKN', 'LKOH', 'MAGN', 'MGNT',
                        'MOEX', 'MTSS', 'NVTK', 'PHOR', 'PLZL', 'ROSN', 'RUAL', 'SBER',
                        'SIBN', 'T', 'VTBR']

            # Создаем DataFrame со всеми тикерами
            full_result = pd.DataFrame({'ticker': all_tickers})
            full_result = full_result.merge(result_df, on='ticker', how='left')

            # Заполняем пропущенные тикеры нулевыми предсказаниями
            for i in range(1, 21):
                col = f'p{i}'
                if col not in full_result.columns:
                    full_result[col] = 0.0
                else:
                    full_result[col] = full_result[col].fillna(0.0)

            print(f"✅ Сгенерировано предсказаний для {len(result_df)} тикеров")
            return full_result
        else:
            return pd.DataFrame()

    def evaluate_predictions(self, candles_df: pd.DataFrame, predictions: pd.DataFrame) -> Dict:
        """Оценка качества предсказаний для формата p1-p20"""
        print("📊 Оценка качества предсказаний...")

        evaluation_results = {}

        # Сортируем котировки по дате
        candles_sorted = candles_df.sort_values(['ticker', 'begin']).copy()

        # Для каждого горизонта от 1 до 20 дней
        for k in range(1, 21):
            return_col = f'p{k}'

            if return_col not in predictions.columns:
                print(f"⚠️ Отсутствует колонка {return_col}")
                continue

            # Вычисляем фактическую доходность за k дней для каждого тикера
            actual_returns = []
            predicted_returns = []

            for ticker in predictions['ticker'].unique():
                ticker_candles = candles_sorted[candles_sorted['ticker'] == ticker]
                ticker_pred = predictions[predictions['ticker'] == ticker]

                if len(ticker_candles) < 2 or len(ticker_pred) == 0:
                    continue

                # Берем последнюю дату (t) и дату t+k
                last_date = ticker_candles['begin'].iloc[-1]
                close_t = ticker_candles['close'].iloc[-1]

                # Ищем цену на k дней вперед
                future_idx = -1 - k
                if abs(future_idx) <= len(ticker_candles):
                    close_t_k = ticker_candles['close'].iloc[future_idx]
                    actual_return = (close_t_k / close_t) - 1

                    actual_returns.append(actual_return)
                    predicted_returns.append(ticker_pred[return_col].iloc[0])

            if len(actual_returns) < 5:
                print(f"⚠️ Недостаточно данных для оценки горизонта {k} дней")
                continue

            y_true = np.array(actual_returns)
            y_pred = np.array(predicted_returns)

            # Для Brier score преобразуем доходности в вероятности роста
            prob_up = 1 / (1 + np.exp(-y_pred * 10))

            mae_base = self.mae_base_dict.get(k, 0.01)
            brier_base = self.brier_base_dict.get(k, 0.25)

            scores = self.evaluator.calculate_final_score(y_true, y_pred, prob_up, mae_base, brier_base)

            evaluation_results[k] = {
                'horizon': k,
                **scores,
                'num_samples': len(actual_returns)
            }

            print(f"   📈 Горизонт {k} дней:")
            print(f"      Final Score: {scores['final_score']:.4f}")
            print(f"      MAE: {scores['mae']:.6f}")
            print(f"      Samples: {len(actual_returns)}")

        return evaluation_results

    def _save_artifacts(self):
        """Сохранение артефактов модели включая бейзлайн метрики и горизонты"""
        artifacts = {
            'model_state_dict': self.model.state_dict(),
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'selected_features': self.selected_features,
            'seq_length': self.seq_length,
            'k_days': self.k_days,  # ✅ Сохраняем горизонты
            'mae_base_dict': self.mae_base_dict,
            'brier_base_dict': self.brier_base_dict
        }

        torch.save(artifacts, self.model_dir / 'model_artifacts.pth')
        print("💾 Артефакты модели сохранены (включая горизонты и бейзлайн метрики)")

    def _load_artifacts(self):
        """Загрузка артефактов модели включая бейзлайн метрики и горизонты"""
        artifacts_path = self.model_dir / 'model_artifacts.pth'
        if not artifacts_path.exists():
            raise FileNotFoundError(f"Артефакты модели не найдены: {artifacts_path}")

        artifacts = torch.load(artifacts_path, map_location=self.device, weights_only=False)

        self.scaler = artifacts['scaler']
        self.feature_columns = artifacts['feature_columns']
        self.selected_features = artifacts['selected_features']
        self.seq_length = artifacts['seq_length']
        self.k_days = artifacts.get('k_days', [1, 20])  # ✅ Загружаем горизонты
        self.mae_base_dict = artifacts.get('mae_base_dict', {k: 0.01 for k in self.k_days})
        self.brier_base_dict = artifacts.get('brier_base_dict', {k: 0.25 for k in self.k_days})

        # Обновляем зависимости
        self.evaluator = EvaluationMetrics(horizons=self.k_days)
        self.baseline_model = BaselineModel(horizons=self.k_days, window_size=5)

        news_embeddings_dummy = np.zeros((1, 768))  # FinBERT embedding size
        self.model = HybridTransformerModel(
            ts_input_size=len(self.selected_features),
            llm_embedding_dim=news_embeddings_dummy.shape[1],
            horizons=self.k_days,  # ✅ Используем загруженные горизонты
            hidden_size=256
        ).to(self.device)

        self.model.load_state_dict(artifacts['model_state_dict'])
        print(f"💾 Артефакты модели загружены (горизонты: {self.k_days})")

# Остальные классы (FeatureGenerator, FeatureSelector, PositionalEncoding, TimeSeriesTransformer) остаются без изменений

class FeatureGenerator:
    """Генератор расширенных признаков для временных рядов"""

    @staticmethod
    def create_features(prices: pd.Series, dates: pd.DatetimeIndex) -> pd.DataFrame:
        """Создание расширенных технических индикаторов и признаков"""
        prices_series = pd.Series(prices, index=dates)

        features = pd.DataFrame(index=dates)

        # Лаговые переменные и доходности
        for lag in range(1, 7):
            features[f'lag_{lag}'] = prices_series.shift(lag)
            features[f'returns_{lag}'] = prices_series.pct_change(lag)
            features[f'log_{lag}'] = np.log(prices_series.shift(lag))
            features[f'diff_{lag}'] = prices_series - prices_series.shift(lag)

        # Скользящие средние и волатильность
        for window in [3, 5, 7, 14, 21, 30]:
            features[f'ma_{window}'] = prices_series.rolling(window).mean()
            features[f'std_{window}'] = prices_series.rolling(window).std()
            features[f'median_{window}'] = prices_series.rolling(window).median()
            features[f'min_{window}'] = prices_series.rolling(window).min()
            features[f'max_{window}'] = prices_series.rolling(window).max()
            features[f'range_{window}'] = prices_series.rolling(window).max() - prices_series.rolling(window).min()
            features[f'var_{window}'] = prices_series.rolling(window).var()

        # Экспоненциальные скользящие средние
        for alpha in [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]:
            features[f'ema_alpha_{alpha}'] = prices_series.ewm(alpha=alpha, adjust=False).mean()
            ema1 = prices_series.ewm(alpha=alpha, adjust=False).mean()
            ema2 = ema1.ewm(alpha=alpha, adjust=False).mean()
            features[f'dema_alpha_{alpha}'] = 2 * ema1 - ema2

        # RSI (Relative Strength Index)
        for period in [14, 21, 30]:
            delta = prices_series.diff()
            gain = delta.where(delta > 0, 0).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
            rs = gain / loss
            features[f'rsi_{period}'] = 100 - (100 / (1 + rs))

        # Стохастический осциллятор
        for k_period in [14, 21, 30]:
            low_min = prices_series.rolling(k_period).min()
            high_max = prices_series.rolling(k_period).max()
            features[f'stoch_k_{k_period}'] = 100 * ((prices_series - low_min) / (high_max - low_min))
            features[f'stoch_d_{k_period}'] = features[f'stoch_k_{k_period}'].rolling(3).mean()

        # Williams %R
        for period in [14, 21, 30]:
            lowest_low = prices_series.rolling(period).min()
            highest_high = prices_series.rolling(period).max()
            features[f'williams_r_{period}'] = -100 * ((highest_high - prices_series) / (highest_high - lowest_low))

        # Rate of Change (ROC)
        for period in [5, 10, 14, 21]:
            features[f'roc_{period}'] = ((prices_series - prices_series.shift(period)) / prices_series.shift(
                period)) * 100

        # Momentum
        for period in [5, 10, 14, 21]:
            features[f'momentum_{period}'] = prices_series - prices_series.shift(period)

        # MACD
        ema12 = prices_series.ewm(span=12, adjust=False).mean()
        ema26 = prices_series.ewm(span=26, adjust=False).mean()
        features['macd'] = ema12 - ema26
        features['macd_signal'] = features['macd'].ewm(span=9, adjust=False).mean()
        features['macd_histogram'] = features['macd'] - features['macd_signal']

        # Полосы Боллинджера
        for period in [20]:
            ma = prices_series.rolling(period).mean()
            std = prices_series.rolling(period).std()
            features[f'bb_upper_{period}'] = ma + (std * 2)
            features[f'bb_middle_{period}'] = ma
            features[f'bb_lower_{period}'] = ma - (std * 2)
            features[f'bb_width_{period}'] = (features[f'bb_upper_{period}'] - features[f'bb_lower_{period}']) / ma
            features[f'bb_position_{period}'] = (prices_series - features[f'bb_lower_{period}']) / (
                        features[f'bb_upper_{period}'] - features[f'bb_lower_{period}'])

        # Временные признаки
        features['hour'] = dates.hour
        features['dayofweek'] = dates.dayofweek
        features['dayofmonth'] = dates.day
        features['month'] = dates.month
        features['quarter'] = dates.quarter

        # Циклические кодирования времени
        features['hour_sin'] = np.sin(2 * np.pi * features['hour'] / 24)
        features['hour_cos'] = np.cos(2 * np.pi * features['hour'] / 24)
        features['dayofweek_sin'] = np.sin(2 * np.pi * features['dayofweek'] / 7)
        features['dayofweek_cos'] = np.cos(2 * np.pi * features['dayofweek'] / 7)
        features['month_sin'] = np.sin(2 * np.pi * features['month'] / 12)
        features['month_cos'] = np.cos(2 * np.pi * features['month'] / 12)

        # Календарные признаки
        features['is_weekend'] = features['dayofweek'].isin([5, 6]).astype(int)
        features['is_month_start'] = (features['dayofmonth'] == 1).astype(int)

        # Статистические моменты
        for window in [10, 20, 50]:
            features[f'skew_{window}'] = prices_series.rolling(window).apply(
                lambda x: skew(x) if len(x) > 2 else np.nan)
            features[f'kurtosis_{window}'] = prices_series.rolling(window).apply(
                lambda x: kurtosis(x) if len(x) > 3 else np.nan)
            features[f'cv_{window}'] = prices_series.rolling(window).apply(
                lambda x: variation(x) if len(x) > 1 and np.std(x) > 0 else np.nan)

        # Упрощенные спектральные признаки (без сложных вычислений)
        for window in [64]:
            features[f'spectral_mean_{window}'] = prices_series.rolling(window).apply(
                lambda x: np.mean(np.abs(fft(x))) if len(x) == window else np.nan
            )

        return features


class FeatureSelector:
    """Отбор признаков с помощью Forward Selection"""

    @staticmethod
    def forward_selection(X_train, y_train, X_val, y_val, max_features=50, min_improvement=0.001):
        """Forward Selection для отбора признаков"""

        n_features = X_train.shape[1]
        selected_features = []
        remaining_features = list(range(n_features))
        scores = []
        prev_score = -np.inf

        print(f"🎯 Начало Forward Selection (максимум {max_features} признаков)")

        for step in range(min(max_features, n_features)):
            best_score = -np.inf
            best_feature = None

            # Перебираем все оставшиеся признаки
            for feature in remaining_features:
                # Создаем текущий набор признаков
                current_features = selected_features + [feature]

                # Обучаем линейную регрессию
                model = LinearRegression()
                model.fit(X_train[:, current_features], y_train)

                # Оцениваем на валидации
                score = model.score(X_val[:, current_features], y_val)

                if score > best_score:
                    best_score = score
                    best_feature = feature

            if prev_score != -np.inf:
                improvement = best_score - prev_score
            else:
                improvement = best_score

            # Добавляем лучший признак
            selected_features.append(best_feature)
            remaining_features.remove(best_feature)
            scores.append(best_score)

            print(f"   Шаг {step + 1}: добавлен признак {best_feature}, R² = {best_score:.4f}")

            # Условие остановки
            if improvement < min_improvement and step > 10:  # Минимум 10 признаков
                print(f"   ⏹️ Остановка: улучшение менее {min_improvement * 100:.1f}%")
                break

            prev_score = best_score

        print(f"✅ Отобрано {len(selected_features)} признаков")
        return selected_features, scores


class PositionalEncoding(nn.Module):
    """Позиционное кодирование для Transformer"""

    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class TimeSeriesTransformer(nn.Module):
    """Transformer для временных рядов"""

    def __init__(self, input_size, d_model=64, nhead=8, num_layers=3, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.input_projection(x) * np.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.transformer(x)
        x = self.dropout(x)
        return x[:, -1, :]


def load_data(candles_paths: List[str], news_paths: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Загрузка данных из нескольких файлов"""
    candles_dfs = []
    for path in candles_paths:
        if Path(path).exists():
            df = pd.read_csv(path)
            df['begin'] = pd.to_datetime(df['begin'])
            candles_dfs.append(df)
            print(f"   📊 Загружены котировки: {path} ({len(df)} строк)")
        else:
            print(f"   ⚠️ Файл не найден: {path}")

    candles_df = pd.concat(candles_dfs, ignore_index=True) if candles_dfs else pd.DataFrame()

    news_dfs = []
    for path in news_paths:
        if Path(path).exists():
            df = pd.read_csv(path)
            df['publish_date'] = pd.to_datetime(df['publish_date'])
            news_dfs.append(df)
            print(f"   📰 Загружены новости: {path} ({len(df)} строк)")
        else:
            print(f"   ⚠️ Файл не найден: {path}")

    news_df = pd.concat(news_dfs, ignore_index=True) if news_dfs else pd.DataFrame()

    return candles_df, news_df


def main():
    """Основная функция запуска с поддержкой произвольных горизонтов"""
    import argparse

    parser = argparse.ArgumentParser(description='Сервис финансового прогнозирования')
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'predict', 'evaluate'],
                       help='Режим работы: train, predict или evaluate')
    parser.add_argument('--candles', type=str, nargs='+', required=True,
                       help='Пути к файлам с котировками (candles.csv, candles_2.csv, etc)')
    parser.add_argument('--news', type=str, nargs='+', required=True,
                       help='Пути к файлам с новостями (news.csv, news_2.csv, etc)')
    parser.add_argument('--k_days', type=int, nargs='+', default=[1, 20],
                       help='Горизонты прогнозирования в днях (например: 1 5 20)')
    parser.add_argument('--output', type=str, default='predictions.csv',
                       help='Путь для сохранения файла с предсказаниями')
    parser.add_argument('--model_dir', type=str, default='model_artifacts',
                       help='Директория для сохранения/загрузки артефактов модели')
    parser.add_argument('--evaluate', action='store_true',
                       help='Выполнить оценку после предсказания')

    args = parser.parse_args()

    print("=" * 70)
    print("🚀 СЕРВИС ФИНАНСОВОГО ПРОГНОЗИРОВАНИЯ С ПРОИЗВОЛЬНЫМИ ГОРИЗОНТАМИ")
    print("=" * 70)
    print(f"🎯 Запрошенные горизонты: {args.k_days}")

    print("📊 Загрузка данных...")
    candles_df, news_df = load_data(args.candles, args.news)

    if candles_df.empty:
        print("❌ Не загружены данные котировок")
        return

    print(f"   ✓ Всего котировок: {len(candles_df)}")
    print(f"   ✓ Всего новостей: {len(news_df) if not news_df.empty else 0}")
    print(f"   ✓ Тикеры: {candles_df['ticker'].unique().tolist()}")

    # ✅ Инициализируем сервис с произвольными горизонтами
    service = FinancialForecastService(model_dir=args.model_dir, k_days=args.k_days)

    if args.mode == 'train':
        print(f"\n🎯 ЗАПУСК ОБУЧЕНИЯ ДЛЯ ГОРИЗОНТОВ {args.k_days}...")
        service.train(candles_df, news_df)

    elif args.mode == 'predict':
        print(f"\n🎯 ЗАПУСК ПРОГНОЗИРОВАНИЯ ДЛЯ ГОРИЗОНТОВ {args.k_days}...")
        predictions = service.predict(candles_df, news_df)

        if not predictions.empty:
            predictions.to_csv(args.output, index=False)
            print(f"💾 Предсказания сохранены: {args.output}")
            print(f"📊 Структура предсказаний: {list(predictions.columns)}")

            if args.evaluate:
                print("\n📊 ЗАПУСК ОЦЕНКИ...")
                evaluation_results = service.evaluate_predictions(candles_df, predictions)

                if evaluation_results:
                    eval_df = pd.DataFrame([evaluation_results])
                    eval_df.to_csv('evaluation_results.csv', index=False)
                    print(f"💾 Результаты оценки сохранены: evaluation_results.csv")
        else:
            print("❌ Не удалось сгенерировать предсказания")

    elif args.mode == 'evaluate':
        print("\n📊 ЗАПУСК ОЦЕНКИ...")
        if Path(args.output).exists():
            predictions = pd.read_csv(args.output)
            evaluation_results = service.evaluate_predictions(candles_df, predictions)

            if evaluation_results:
                eval_df = pd.DataFrame([evaluation_results])
                eval_df.to_csv('evaluation_results.csv', index=False)
                print(f"💾 Результаты оценки сохранены: evaluation_results.csv")
        else:
            print("❌ Файл с предсказаниями не найден")

    print("\n" + "=" * 70)
    print("✅ ВЫПОЛНЕНИЕ ЗАВЕРШЕНО!")
    print("=" * 70)

if __name__ == "__main__":
    main()