import os
import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import matplotlib
from directional_metrics import DirectionalAccuracy
# from directional_loss import LastStepDirectionalQuantileOutput
from custom.timesfm_model import TimesFMModel
from custom.tirex_model import TiRexModel
from custom.preprocessing import (
    add_time_features,
    add_indicators,
    add_log_target,
    generate_future_known_covariates,
)

torch.set_float32_matmul_precision('highest')
torch.backends.cudnn.benchmark = True
matplotlib.use('Agg')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataProcessor:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.df = None

    def load_data(self):
        self.df = pd.read_csv(self.filepath)
        logging.info(f"Data loaded from {self.filepath}. Shape: {self.df.shape}")
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp']).dt.tz_localize(None)
        logging.info("Converted 'timestamp' to naive datetime64.")

    def preprocess_data(self):
        tailing = 10_000_000
        df = self.df.copy()
        df = df.sort_values(['item_id', 'timestamp'])
        before = df.shape[0]
        df = df.drop_duplicates(subset=['item_id', 'timestamp'], keep='last')
        after = df.shape[0]
        logging.info(f"Removed {before - after} duplicate rows.")

        df = df.groupby('item_id', group_keys=False).tail(tailing)
        logging.info(f"Truncated to last {tailing} rows per item_id. New shape: {df.shape}")

        df = add_time_features(df)
        df = add_indicators(df)
        df = add_log_target(df)

        keep_columns = [
            "item_id",
            "timestamp",
            "target",
            # "weekend",
            # "log_target",
        ]
        df = df[keep_columns]
        logging.info("Kept only required columns incl. new features.")

        df.dropna(inplace=True)
        logging.info("Dropped NaNs created during indicator calculation.")

        self.df = df
        print(self.df)
        logging.info("Pre-processing completed.")

    def remove_irregular_items(self):
        df = self.df.copy()
        irregular_items = []
        for item, group in df.groupby("item_id"):
            group = group.sort_values("timestamp")
            freq = pd.infer_freq(group["timestamp"])
            if freq is None:
                irregular_items.append(item)
        if irregular_items:
            logging.info(f"Removing items with irregular frequency: {irregular_items}")
            df = df[~df["item_id"].isin(irregular_items)]
        else:
            logging.info("All item_ids have regular frequency.")
        self.df = df

    def save_processed_data(self, output_filepath):
        self.df.to_csv(output_filepath, index=False)
        logging.info(f"Processed data saved to {output_filepath}.")

    def load_processed_data(self, input_filepath):
        self.df = pd.read_csv(input_filepath, parse_dates=['timestamp'])
        logging.info(f"Processed data loaded from {input_filepath}.")

class ModelTrainer:
    def __init__(
        self,
        prediction_length: int,
        target: str,
        known_covariates: list,
        output_dir: str = "AutogluonModels",
    ):
        self.prediction_length = prediction_length
        self.target = target
        self.known_covariates = known_covariates
        self.output_dir = output_dir
        self.predictor: TimeSeriesPredictor | None = None

    def prepare_time_series_data(self, df: pd.DataFrame):
        ts_df = TimeSeriesDataFrame.from_data_frame(
            df,
            id_column="item_id",
            timestamp_column="timestamp"
        )
        logging.info("Converted DataFrame --> TimeSeriesDataFrame.")
        train_data, test_data = ts_df.train_test_split(prediction_length=self.prediction_length)
        logging.info(f"Train / test split: {train_data.num_items} items train, {test_data.num_items} items test.")
        return train_data, test_data

    def train_model(self, train_data: TimeSeriesDataFrame, time_limit=118000):
        hyperparameters = {
            "SeasonalNaive": {
            },
            "AutoETS": {
            },
            "NPTS": {
            },
            "DynamicOptimizedTheta": {
            },
            "DLinear": [
                {},
            ],
            "TemporalFusionTransformer": [
                {
                    "ag_args": {"name_suffix": "BaseStdCtx"}, # Базовая, стандартный контекст
                    "context_length": 72,
                    "hidden_dim": 32,
                    "variable_dim": 32,
                    "num_heads": 4,
                    "dropout_rate": 0.1,
                    "lr": 1e-3,
                    "batch_size": 64,
                    "max_epochs": 100,
                    "early_stopping_patience": 20,
                },
                {
                    "ag_args": {"name_suffix": "ShortCtxFast"}, # Короткий контекст, быстрая итерация
                    "context_length": 24,
                    "hidden_dim": 16,
                    "variable_dim": 16,
                    "num_heads": 2,
                    "dropout_rate": 0.1,
                    "lr": 1e-3,
                    "batch_size": 128,
                    "max_epochs": 50,
                    "early_stopping_patience": 10,
                },
                {
                    "ag_args": {"name_suffix": "LongCtxLarge"}, # Длинный контекст, большая модель
                    "context_length": 168,
                    "hidden_dim": 64,
                    "variable_dim": 64,
                    "num_heads": 8,
                    "dropout_rate": 0.2,
                    "lr": 5e-4,
                    "batch_size": 32,
                    "max_epochs": 150,
                    "early_stopping_patience": 25,
                },
                {
                    "ag_args": {"name_suffix": "ComplexMidCtx"}, # Увеличенная сложность, средний контекст
                    "context_length": 96,
                    "hidden_dim": 48,
                    "variable_dim": 48,
                    "num_heads": 6,
                    "dropout_rate": 0.15,
                    "lr": 8e-4,
                    "batch_size": 64,
                    "max_epochs": 120,
                    "early_stopping_patience": 20,
                },
                {
                    "ag_args": {"name_suffix": "LowLRLongEpoch"}, # Низкий LR, много эпох
                    "context_length": 72,
                    "hidden_dim": 32,
                    "variable_dim": 32,
                    "num_heads": 4,
                    "dropout_rate": 0.1,
                    "lr": 1e-4,
                    "batch_size": 64,
                    "max_epochs": 200,
                    "early_stopping_patience": 30,
                },
                {
                    "ag_args": {"name_suffix": "HighDropoutReg"}, # Высокий dropout для регуляризации
                    "context_length": 72,
                    "hidden_dim": 32,
                    "variable_dim": 32,
                    "num_heads": 4,
                    "dropout_rate": 0.3,
                    "lr": 1e-3,
                    "batch_size": 64,
                    "max_epochs": 100,
                    "early_stopping_patience": 15,
                },
                {
                    "ag_args": {"name_suffix": "SmallModelLongCtx"}, # Маленькая модель, длинный контекст
                    "context_length": 168,
                    "hidden_dim": 16,
                    "variable_dim": 16,
                    "num_heads": 2,
                    "dropout_rate": 0.05,
                    "lr": 2e-3,
                    "batch_size": 128,
                    "max_epochs": 80,
                    "early_stopping_patience": 15,
                },
                {
                    "ag_args": {"name_suffix": "VarDimFocus"}, # Фокус на variable_dim (если есть ковариаты)
                    "context_length": 72,
                    "hidden_dim": 32,
                    "variable_dim": 64, # Увеличено
                    "num_heads": 4,
                    "dropout_rate": 0.1,
                    "lr": 1e-3,
                    "batch_size": 64,
                    "max_epochs": 100,
                    "early_stopping_patience": 20,
                },
                {
                    "ag_args": {"name_suffix": "LargeBatchAggro"}, # Большой batch_size, агрессивное обучение
                    "context_length": 64,
                    "hidden_dim": 32,
                    "variable_dim": 32,
                    "num_heads": 4,
                    "dropout_rate": 0.15,
                    "lr": 1.5e-3,
                    "batch_size": 256,
                    "max_epochs": 75,
                    "early_stopping_patience": 10,
                },
                {
                    "ag_args": {"name_suffix": "BalancedTune"}, # Сбалансированная конфигурация
                    "context_length": 120,
                    "hidden_dim": 40,
                    "variable_dim": 40,
                    "num_heads": 5, # 40 делится на 5
                    "dropout_rate": 0.12,
                    "lr": 7e-4,
                    "batch_size": 64,
                    "max_epochs": 150,
                    "early_stopping_patience": 25,
                }
            ],
            "PatchTST": [
                {
                    "ag_args": {"name_suffix": "BaseDefault"}, # Близко к значениям по умолчанию
                    "context_length": 96,    # Default
                    "patch_len": 16,         # Default
                    "stride": 8,             # Default
                    "d_model": 32,           # Default
                    "nhead": 4,              # Default (32 % 4 == 0)
                    "num_encoder_layers": 2, # Default
                    "lr": 1e-3,
                    "batch_size": 64,
                    "max_epochs": 100,
                    "weight_decay": 1e-8,    # Default
                },
                {
                    "ag_args": {"name_suffix": "ShortCtxSmallPatch"}, # Короткий контекст, маленькие патчи
                    "context_length": 48,    # 2 дня
                    "patch_len": 8,          # Меньше патч
                    "stride": 4,             # Меньше шаг, больше перекрытия
                    "d_model": 16,
                    "nhead": 2,              # (16 % 2 == 0)
                    "num_encoder_layers": 1, # Проще модель
                    "lr": 1e-3,
                    "batch_size": 128,       # Можно увеличить batch_size для меньшей модели
                    "max_epochs": 75,
                    "weight_decay": 1e-8,
                },
                {
                    "ag_args": {"name_suffix": "LongCtxLargePatchStd"}, # Длинный контекст, большие патчи, std скейлинг
                    "context_length": 168,   # 1 неделя
                    "patch_len": 32,         # Больше патч
                    "stride": 16,            # Шаг = patch_len / 2
                    "d_model": 64,
                    "nhead": 8,              # (64 % 8 == 0)
                    "num_encoder_layers": 3, # Глубже трансформер
                    "lr": 5e-4,              # Меньше lr для потенциально большей модели
                    "batch_size": 32,
                    "max_epochs": 150,
                    "weight_decay": 1e-7,    # Чуть больше регуляризации
                },
                {
                    "ag_args": {"name_suffix": "MorePatchesOverlap"}, # Больше перекрывающихся патчей
                    "context_length": 96,
                    "patch_len": 16,
                    "stride": 4,             # Меньший шаг для большего количества патчей
                    "d_model": 32,
                    "nhead": 4,
                    "num_encoder_layers": 2,
                    "lr": 1e-3,
                    "batch_size": 64,
                    "max_epochs": 100,
                    "weight_decay": 1e-8,
                },
                {
                    "ag_args": {"name_suffix": "DeepTransformerStd"}, # Более глубокий трансформер, std скейлинг
                    "context_length": 120,    # 5 дней
                    "patch_len": 16,
                    "stride": 8,
                    "d_model": 48,            # d_model=48, nhead=6 (48%6==0)
                    "nhead": 6,
                    "num_encoder_layers": 4,  # Глубже
                    "lr": 8e-4,
                    "batch_size": 64,
                    "max_epochs": 120,
                    "weight_decay": 1e-8,
                },
                {
                    "ag_args": {"name_suffix": "NoScalingTest"}, # Проверка без скейлинга (возможно, неоптимально, но для теста)
                    "context_length": 72,     # 3 дня
                    "patch_len": 12,          # patch_len < context_length
                    "stride": 6,
                    "d_model": 32,
                    "nhead": 4,
                    "num_encoder_layers": 2,
                    "scaling": None,     # Без скейлинга
                    "lr": 1e-3,
                    "batch_size": 64,
                    "max_epochs": 100,
                    "weight_decay": 1e-8,
                },
                {
                    "ag_args": {"name_suffix": "VLongCtxMean"}, # Очень длинный контекст, mean скейлинг
                    "context_length": 336,    # 2 недели
                    "patch_len": 16,          # Стандартный патч
                    "stride": 8,
                    "d_model": 64,
                    "nhead": 8,
                    "num_encoder_layers": 3,
                    "lr": 5e-4,
                    "batch_size": 32,
                    "max_epochs": 150,
                    "weight_decay": 1e-7,
                },
                {
                    "ag_args": {"name_suffix": "HighWeightDecay"}, # Повышенная L2 регуляризация
                    "context_length": 96,
                    "patch_len": 16,
                    "stride": 8,
                    "d_model": 32,
                    "nhead": 4,
                    "num_encoder_layers": 2,
                    "lr": 1e-3,
                    "batch_size": 64,
                    "max_epochs": 100,
                    "weight_decay": 1e-4,    # Значительно выше
                },
                {
                    "ag_args": {"name_suffix": "SmallModelLongCtxStdV3"}, # Маленькая модель на длинном контексте, std скейлинг
                    "context_length": 168,   # 1 неделя
                    "patch_len": 24,         # Средний патч
                    "stride": 12,
                    "d_model": 16,
                    "nhead": 2,
                    "num_encoder_layers": 1,
                    "lr": 2e-3,              # Можно чуть выше lr для маленькой модели
                    "batch_size": 512,
                    "max_epochs": 100,
                    "weight_decay": 1e-8,
                },
                {
                    "ag_args": {"name_suffix": "SmallModelLongCtxStdV1"}, # Маленькая модель на длинном контексте, std скейлинг
                    "context_length": 168,   # 1 неделя
                    "patch_len": 24,         # Средний патч
                    "stride": 12,
                    "d_model": 16,
                    "nhead": 2,
                    "num_encoder_layers": 1,
                    "lr": 2e-3,              # Можно чуть выше lr для маленькой модели
                    "batch_size": 128,
                    "max_epochs": 100,
                    "weight_decay": 1e-8,
                },
                {
                    "ag_args": {"name_suffix": "SmallModelLongCtxStdV2"}, # Маленькая модель на длинном контексте, std скейлинг
                    "context_length": 168,   # 1 неделя
                    "patch_len": 24,         # Средний патч
                    "stride": 12,
                    "d_model": 16,
                    "nhead": 2,
                    "num_encoder_layers": 1,
                    "lr": 2e-3,              # Можно чуть выше lr для маленькой модели
                    "batch_size": 256,
                    "max_epochs": 100,
                    "weight_decay": 1e-8,
                },
                {
                    "ag_args": {"name_suffix": "SmallModelLongCtxStdV4"}, # Маленькая модель на длинном контексте, std скейлинг
                    "context_length": 168,   # 1 неделя
                    "patch_len": 24,         # Средний патч
                    "stride": 12,
                    "d_model": 16,
                    "nhead": 2,
                    "num_encoder_layers": 1,
                    "lr": 2e-3,              # Можно чуть выше lr для маленькой модели
                    "batch_size": 128,
                    "max_epochs": 100,
                    "weight_decay": 1e-8,
                },
                {
                    "ag_args": {"name_suffix": "LargePatchShortCtx"}, # Большие патчи на коротком контексте
                    "context_length": 72,     # 3 дня
                    "patch_len": 24,          # Относительно большой патч для такого контекста
                    "stride": 12,             # (или stride: 24 для непересекающихся)
                    "d_model": 48,
                    "nhead": 6,               # (48 % 6 == 0)
                    "num_encoder_layers": 2,
                    "lr": 1e-3,
                    "batch_size": 64,
                    "max_epochs": 100,
                    "weight_decay": 1e-8,
                }
            ],
            "SimpleFeedForward": [
                {
                    "ag_args": {"name_suffix": "BaseStdCtxBNoff"}, # Базовая, контекст побольше, BN выключен
                    "context_length": 72,         # 3 дня (значительно больше дефолта)
                    "hidden_dimensions": [40, 20],# Чуть больше дефолта [20, 20]
                    "batch_normalization": False, # Default
                    "lr": 1e-3,
                    "batch_size": 64,
                    "max_epochs": 100,
                },
                {
                    "ag_args": {"name_suffix": "ShortCtxSimpleBNon"}, # Короткий контекст, простая сеть, BN включен
                    "context_length": 24,         # 1 день
                    "hidden_dimensions": [16],    # Один небольшой слой
                    "batch_normalization": True,
                    "lr": 1e-3,
                    "batch_size": 128,
                    "max_epochs": 75,
                },
                {
                    "ag_args": {"name_suffix": "LongCtxDeepBNon"}, # Длинный контекст, глубокая сеть, BN включен
                    "context_length": 168,        # 1 неделя
                    "hidden_dimensions": [64, 64, 32], # Три слоя, поглубже и пошире
                    "batch_normalization": True,
                    "lr": 5e-4,
                    "batch_size": 32,
                    "max_epochs": 150,
                },
                {
                    "ag_args": {"name_suffix": "WideShallowBNoff"}, # Широкая, но неглубокая сеть, BN выключен
                    "context_length": 96,         # 4 дня
                    "hidden_dimensions": [128],   # Один широкий слой
                    "batch_normalization": False,
                    "lr": 1e-3,
                    "batch_size": 64,
                    "max_epochs": 100,
                },
                {
                    "ag_args": {"name_suffix": "NarrowDeepBNon"}, # Узкая, но глубокая сеть, BN включен
                    "context_length": 96,
                    "hidden_dimensions": [32, 32, 32, 32], # Четыре узких слоя
                    "batch_normalization": True,
                    "lr": 8e-4,
                    "batch_size": 64,
                    "max_epochs": 120,
                },
                {
                    "ag_args": {"name_suffix": "NoMeanScalingBNon"}, # Без mean_scaling, BN включен
                    "context_length": 72,
                    "hidden_dimensions": [40, 20],
                    "batch_normalization": True,
                    "mean_scaling": False,        # Выключено
                    "lr": 1e-3,
                    "batch_size": 64,
                    "max_epochs": 100,
                },
                {
                    "ag_args": {"name_suffix": "LongTrainLowLRBNon"}, # Длительное обучение, низкий LR, BN включен
                    "context_length": 120,        # 5 дней
                    "hidden_dimensions": [50, 25],
                    "batch_normalization": True,
                    "mean_scaling": True,
                    "lr": 5e-4,                   # Ниже
                    "batch_size": 64,
                    "max_epochs": 200,            # Больше эпох
                    "early_stopping_patience": 30 # Увеличено терпение
                },
                {
                    "ag_args": {"name_suffix": "BalancedBNoffNoScale"}, # Сбалансированная архитектура, BN выключен, mean_scaling выключен
                    "context_length": 96,
                    "hidden_dimensions": [60, 30],
                    "batch_normalization": False,
                    "mean_scaling": False,
                    "lr": 1e-3,
                    "batch_size": 64,
                    "max_epochs": 100,
                },
                {
                    "ag_args": {"name_suffix": "VShortCtxFastBNon"}, # Очень короткий контекст (но больше дефолта), быстрое обучение, BN вкл
                    "context_length": 12,          # Всего полдня
                    "hidden_dimensions": [32],
                    "batch_normalization": True,
                    "mean_scaling": True,
                    "lr": 2e-3,                    # Выше LR
                    "batch_size": 128,
                    "max_epochs": 50,              # Меньше эпох
                    "early_stopping_patience": 10
                },
                {
                    "ag_args": {"name_suffix": "MedCtxThreeLayersBNoff"}, # Средний контекст, три одинаковых слоя, BN выкл
                    "context_length": 72,
                    "hidden_dimensions": [50, 50, 50],
                    "batch_normalization": False,
                    "mean_scaling": True,
                    "lr": 1e-3,
                    "batch_size": 64,
                    "max_epochs": 120,
                }
            ],
            "Chronos": [
                {
                    "ag_args": {"name_suffix": "ZeroMini"},
                    "model_path": "bolt_mini", 
                    "context_length": 2048
                },
                {
                    "ag_args": {"name_suffix": "ZeroSmall"}, 
                    "model_path": "bolt_small", 
                    "context_length": 2048
                },
                {
                    "ag_args": {"name_suffix": "ZeroBase"}, 
                    "model_path": "bolt_base", 
                    "context_length": 2048
                },
                {
                    "ag_args": {"name_suffix": "FineTunedSmall"},
                    "context_length": 2048,
                    "model_path": "bolt_small",
                    "fine_tune": True
                },
            ],
            TiRexModel: [
                {
                    "ag_args": {"name_suffix": "ZeroShot"},
                    "context_length": 2048,
                },
            ],
            TimesFMModel: [
                {
                    "ag_args": {"name_suffix": "200m"}, 
                    "context_length": 512,
                    "checkpoint_repo_id": "google/timesfm-1.0-200m-pytorch"
                },
                {
                    "ag_args": {"name_suffix": "599m"}, 
                    "context_length": 2048, 
                    "checkpoint_repo_id": "google/timesfm-2.0-500m-pytorch"
                },
            ],
            "TiDE": [
                {
                    "ag_args": {"name_suffix": "upd"},

                    "encoder_hidden_dim": 256,
                    "decoder_hidden_dim": 256,
                    "temporal_hidden_dim": 64,
                    "batch_size": 512,
                    "num_batches_per_epoch": 50,
                    "lr": 1e-4,
                },
                {
                    "ag_args": {"name_suffix": "def"},
                    "encoder_hidden_dim": 256,
                    "decoder_hidden_dim": 256,
                    "temporal_hidden_dim": 64,
                    "batch_size": 256,
                    "num_batches_per_epoch": 100,
                    "lr": 1e-4,
                },
                {
                    "ag_args": {"name_suffix": "wide_lowLR_512_50"},
                    "context_length": 64,
                    "encoder_hidden_dim": 512,
                    "decoder_hidden_dim": 512,
                    "temporal_hidden_dim": 128,
                    "batch_size": 512,
                    "num_batches_per_epoch": 50,
                    "lr": 5e-5,
                },
            ]
        }

        self.predictor = TimeSeriesPredictor(
            prediction_length=self.prediction_length,
            target=self.target,
            known_covariates_names=self.known_covariates,
            path=self.output_dir,
            eval_metric=DirectionalAccuracy(),
            quantile_levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        )
 
        self.predictor.fit(
            train_data,
            presets='best_quality',
            hyperparameters=hyperparameters,
            time_limit=time_limit,
            num_val_windows=24,
            refit_every_n_windows=2,
            enable_ensemble=True,
            verbosity=4,
            random_seed=42
        )
        logging.info("Model training completed.")

    def load_model(self):
        if not os.path.exists(self.output_dir):
            raise FileNotFoundError(f"Model dir '{self.output_dir}' not found.")
        self.predictor = TimeSeriesPredictor(verbosity=4).load(path=self.output_dir)
        self.predictor.persist()
        if hasattr(self.predictor, 'cache_predictions'):
            self.predictor.cache_predictions = False
        logging.info(f"Model loaded from '{self.output_dir}'.")

    def evaluate_model(self, test_data):
        if self.predictor is None:
            raise ValueError("Predictor not initialised.")
        evaluation = self.predictor.evaluate(data=test_data, metrics=['RMSSE', 'SMAPE', "SQL", 'MASE', 'MAE'])
        logging.info("Evaluation completed.")
        return evaluation

    # -----------------  Forecast / Plot  ---------------------- #
    def generate_forecasts(self, test_data, known_covariates_future, output_dir="plots"):
        os.makedirs(output_dir, exist_ok=True)
        predictions = self.predictor.predict(test_data, known_covariates=known_covariates_future)
        fig = self.predictor.plot(
            test_data,
            predictions,
            quantile_levels=[0.5],
            max_history_length=100,
            max_num_item_ids=16
        )
        plot_fp = os.path.join(output_dir, "forecasts_plot.png")
        fig.savefig(plot_fp)
        plt.close(fig)
        logging.info(f"Forecast plot saved: {plot_fp}")
        return predictions

    def compute_feature_importance(self, data, output_dir="plots"):
        os.makedirs(output_dir, exist_ok=True)
        for model in self.predictor.model_names():
            logging.info(f"Computing FI for {model}")
            fi = self.predictor.feature_importance(data=data, model=model, metric='MASE')
            if fi.empty:
                logging.warning("FI empty, skipping plot.")
                continue
            if 'feature' not in fi.columns:
                fi = fi.reset_index().rename(columns={'index': 'feature'})
            if 'importance' not in fi.columns and 'score' in fi.columns:
                fi = fi.rename(columns={'score': 'importance'})
            csv_path = os.path.join(output_dir, f'feature_importance_{model}.csv')
            fi.to_csv(csv_path, index=False)
            plt.figure(figsize=(12, 8))
            sns.barplot(x='importance', y='feature', data=fi.sort_values('importance', ascending=False))
            plt.title(f'Feature Importance – {model}')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'feature_importance_{model}.png'))
            plt.close()
            logging.info(f"FI saved for {model}")

    def save_leaderboard(self, test_data, output_dir="reports"):
        os.makedirs(output_dir, exist_ok=True)
        lb = self.predictor.leaderboard(test_data, silent=True, extra_metrics=['WQL', 'SMAPE', 'MASE', 'MAE', "SQL"], use_cache=False)
        lb.to_csv(os.path.join(output_dir, 'leaderboard.csv'), index=False)
        plt.figure(figsize=(12, 8))
        sns.scatterplot(x='fit_time_marginal', y='score_val', data=lb, hue='model', s=100)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'leaderboard_plot.png'))
        plt.close()
        logging.info("Leaderboard & plot saved.")


# -------------------------------------------------------------------------------- #
#                                      MAIN                                        #
# -------------------------------------------------------------------------------- #
def main():
    prediction_length = 1
    target = "target"
    data_filepath = "full_1h.csv"
    processed_csv = "full_1h.csv"

    models_dir = "1h_train_12_dloss_target_da"
    plots_dir = os.path.join(models_dir, "plots")
    reports_dir = os.path.join(models_dir, "reports")

    dp = DataProcessor(filepath=data_filepath)
    dp.load_data()
    dp.preprocess_data()
    # dp.save_processed_data(processed_csv)

    # del dp.df
    # logging.info("Cleared df from memory to free RAM.")

    # dp.load_processed_data(processed_csv)
    dp.remove_irregular_items()
    df = dp.df.sort_values(["item_id", "timestamp"])
    dp.df = df

    known_covariates = []

    trainer = ModelTrainer(
        prediction_length=prediction_length,
        target=target,
        known_covariates=known_covariates,
        output_dir=models_dir,
    )
    train_data, test_data = trainer.prepare_time_series_data(dp.df)

    predictor_file = os.path.join(models_dir, 'predictor.pkl')
    if os.path.exists(predictor_file):
        logging.info("Existing model found – load.")
        trainer.load_model()
    else:
        logging.info("No model found – train new.")
        trainer.train_model(train_data=train_data, time_limit=None)

    # --------  future covariates + forecast with categorical features
    future_cov = generate_future_known_covariates(
        ts_df=test_data,
        prediction_length=prediction_length,
        freq=test_data.freq,
        known_covariates=known_covariates,
    )
    predictions = trainer.generate_forecasts(test_data=test_data, known_covariates_future=future_cov, output_dir=plots_dir)

    # --------  reports
    trainer.save_leaderboard(test_data=test_data, output_dir=reports_dir)
    trainer.compute_feature_importance(data=test_data, output_dir=plots_dir)


if __name__ == "__main__":
    main()
