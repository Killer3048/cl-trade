import logging
import os
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch.nn as nn
from ta.momentum import RSIIndicator
from ta.trend import CCIIndicator, ADXIndicator, MACD
from ta.volume import MFIIndicator
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
from directional_metrics import DirectionalAccuracy
from custom.tirex_model import TiRexModel
from custom.preprocessing import (
    generate_future_known_covariates,
    compute_vwap,
    add_indicators,
    add_time_features
)

import warnings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
warnings.simplefilter(action="ignore", category=FutureWarning)

GLOBAL_KERNEL_DIR: Dict[str, int] = {}
USE_KERNEL_FILTER: bool = True
KERNEL_LOOKBACK: int = 8
KERNEL_R: float = 8.0
KERNEL_START_AT: int = 25

try:
    def _patched_to(self, *args, **kwargs):
        try:
            return nn.Module._orig_to(self, *args, **kwargs)
        except NotImplementedError as e:
            if "Cannot copy out of meta tensor" in str(e):
                device = kwargs.get("device") or (args[0] if args else None)
                if device is None:
                    raise ValueError("device must be specified in .to()") from e
                if hasattr(self, "to_empty"):
                    return self.to_empty(device=device)
            raise e

    nn.Module._orig_to = nn.Module.to
    nn.Module.to = _patched_to
except Exception as patch_err:
    logging.getLogger(__name__).warning(f"Патч nn.Module.to не применён: {patch_err}")


def _kernel_regression(
    series: pd.Series,
    span: int,
    r_param: float,
    start_at: int
) -> pd.Series:
    arr = series.to_numpy(dtype=float)
    
    win = span + start_at 
    
    if win <= 0 :
        return pd.Series([np.nan] * len(series), index=series.index)
    if span <= 0:
        return pd.Series([np.nan] * len(series), index=series.index)

    idx = np.arange(win)
    
    weights_denominator = (float(span) ** 2) * 2 * float(r_param)
    
    if weights_denominator == 0:
        return pd.Series([np.nan] * len(series), index=series.index)
        
    weights = (1 + (idx ** 2) / weights_denominator) ** (-r_param)
    
    sum_weights = weights.sum()
    if sum_weights == 0 or np.isnan(sum_weights):
         return pd.Series([np.nan] * len(series), index=series.index)
    weights = weights / sum_weights

    padded = np.pad(arr, (win - 1, 0), mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, win)
    
    kernel_values = (windows * weights[::-1]).sum(axis=1)
    
    return pd.Series(kernel_values, index=series.index)

def compute_kernel_direction(
    close_series_slice: pd.Series,
    char_length: int,
    alpha_r_param: float,
    n_window: int
) -> int:
    if close_series_slice.isna().any():
        close_series_slice = close_series_slice.dropna() 

    if len(close_series_slice) < 2 or n_window < 2:
        return 0
    
    if char_length <= 0:
        return 0

    py_func_start_at_arg = n_window - char_length 
    
    if py_func_start_at_arg < 0:
        return 0

    kernel_line = _kernel_regression(
        series=close_series_slice,
        span=char_length,
        r_param=alpha_r_param,
        start_at=py_func_start_at_arg
    )

    if len(kernel_line) < 2:
        return 0

    current_val = kernel_line.iloc[-1]
    prev_val = kernel_line.iloc[-2]

    if pd.isna(current_val) or pd.isna(prev_val):
        return 0

    if current_val > prev_val:
        return 1
    elif current_val < prev_val:
        return -1
    else:
        return 0

def assess_prediction_confidence(
    predictions: pd.DataFrame,
    reference_price: pd.Series,
    confidence_threshold: float = 0.62,
) -> pd.DataFrame:
    logger = logging.getLogger(__name__)
    q_levels = np.round(np.arange(0.1, 1.0, 0.1), 1)
    q_cols = [str(q) for q in q_levels]
    if any(c not in predictions.columns for c in q_cols):
        logger.error(f"Отсутствуют колонки квантилей: {q_cols}")
        return pd.DataFrame()

    last_preds = (
        predictions
        .groupby(level="item_id", group_keys=False)
        .tail(1)
        .droplevel("timestamp")
    )

    reference_price = reference_price.rename_axis("item_id").astype(float)
    common = last_preds.index.intersection(reference_price.index)
    if common.empty:
        logger.warning("Нет общих item_id.")
        return pd.DataFrame()

    last = last_preds.loc[common, q_cols].astype(float).values
    ref = reference_price.loc[common].values.astype(float)
    N = len(common)
    prob_down = np.empty(N, float)
    q0, q9 = last[:, 0], last[:, -1]

    mask_low = ref < q0
    mask_high = ref >= q9
    mask_mid = ~(mask_low | mask_high)

    prob_down[mask_low] = 0.1 * np.where(q0[mask_low] > 1e-9, ref[mask_low] / q0[mask_low], 0.1)
    prob_down[mask_high] = 1.0

    for i in np.where(mask_mid)[0]:
        arr = last[i]
        r = ref[i]
        idx = np.searchsorted(arr, r, side="left")
        ql, qu = arr[idx - 1], arr[idx]
        pl, pu = q_levels[idx - 1], q_levels[idx]
        prob_down[i] = pl + (pu - pl) * (r - ql) / max(qu - ql, 1e-9)

    prob_down = np.clip(prob_down, 0.0, 1.0)
    prob_up = 1.0 - prob_down
    decision = np.where(
        prob_up > confidence_threshold, "UP",
        np.where(prob_down > confidence_threshold, "DOWN", "UNCERTAIN")
    )

    return (
        pd.DataFrame({
            "item_id": common,
            "reference_price": ref,
            "prob_down": prob_down,
            "prob_up": prob_up,
            "decision": decision
        })
        .set_index("item_id")
    )

class ChronosPredictor:
    def __init__(self, config: dict):
        self.logger = logging.getLogger(f"{__name__}.ChronosPredictor")
        self.config = config
        self.model_type = config.get("model_type", "TimeSeriesModel")
        self.freq = config.get("freq", "5min")
        self.prediction_length = config.get("prediction_length", 1)
        self.mode = config.get("mode", "indicators")
        if self.mode == "indicators" and self.prediction_length != 1:
            self.logger.warning(
                f"Для режима 'indicators' с обновленной логикой ожидается prediction_length=1. "
                f"Текущее значение: {self.prediction_length}. "
                "Сигналы могут генерироваться некорректно."
            )
        self.results_output_dir = config.get("results_output_dir", "results")
        self.use_ensemble = config.get("use_ensemble", True)
        self.confidence_threshold = config.get("confidence_threshold", 0.62)
        self.all_time_retrain = config.get("all_time_retrain", False)
        self.use_kernel_filter = config.get("use_kernel_filter", USE_KERNEL_FILTER)
        self.kernel_lookback = config.get("kernel_lookback", KERNEL_LOOKBACK)
        self.kernel_r = config.get("kernel_r", KERNEL_R)
        self.kernel_start_at = config.get("kernel_start_at", KERNEL_START_AT)

        kc = config.get("known_covariates", [])
        if not isinstance(kc, (list, tuple, set)):
            kc = []
        self.known_covariates = tuple(kc)
        self.predictor: Optional[TimeSeriesPredictor] = None
        print("Conf level:", self.confidence_threshold*100, "%")

    def _ensure_target_column(self, df: pd.DataFrame) -> pd.DataFrame:
        if "target" in df.columns:
            return df
        if "close" in df.columns:
            self.logger.warning("'target' column not found, using 'close' instead.")
            df = df.rename(columns={"close": "target"})
        else:
            self.logger.error("Neither 'target' nor 'close' columns are present.")
        return df

    def load_model(self):
        if not os.path.isdir(self.results_output_dir):
            self.logger.error(f"Директория модели '{self.results_output_dir}' не найдена.")
            return
        try:
            self.predictor = TimeSeriesPredictor.load(self.results_output_dir)
            if hasattr(self.predictor, "cache_predictions"):
                self.predictor.cache_predictions = False
                self.predictor.persist()
            self.logger.info(f"Предиктор '{self.model_type}' успешно загружен.")
        except Exception as e:
            self.logger.exception(f"Ошибка загрузки модели: {e}")

    def _save_predictions_and_plot(
        self,
        ts_df: TimeSeriesDataFrame,
        predictions_price: TimeSeriesDataFrame,
        predictions_vwap: Optional[TimeSeriesDataFrame] = None,
        timestamp: Optional[str] = None,
    ):
        try:
            output_dir = os.path.join(self.results_output_dir, "predictions")
            os.makedirs(output_dir, exist_ok=True)

            timestamp_str = (
                pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
                if timestamp is None else timestamp
            )
            price_csv = os.path.join(output_dir, f"price_predictions_{timestamp_str}.csv")
            predictions_price.to_data_frame().to_csv(price_csv)
            self.logger.info(f"Price predictions saved to {price_csv}")

            if predictions_vwap is not None:
                vwap_csv = os.path.join(output_dir, f"vwap_predictions_{timestamp_str}.csv")
                predictions_vwap.to_data_frame().to_csv(vwap_csv)
                self.logger.info(f"VWAP predictions saved to {vwap_csv}")

            self.logger.info("Generating prediction plots...")
            fig = self.predictor.plot(
                ts_df, predictions_price,
                quantile_levels=[0.1, 0.5, 0.9],
                max_history_length=100,
                max_num_item_ids=16
            )
            plot_fp = os.path.join(output_dir, f"price_plots_{timestamp_str}.png")
            fig.savefig(plot_fp)
            self.logger.info(f"Plots saved to {plot_fp}")
        except Exception as e:
            self.logger.exception(f"Error in background task for saving predictions: {e}")

    def run(self, df: pd.DataFrame) -> Dict[str, str]:
        if self.mode == "target":
            return self._run_target(df)
        return self._run_indicators(df)

    def _run_indicators(self, df: pd.DataFrame) -> Dict[str, str]:
        original_item_ids = df.get("item_id", pd.Series("unknown_item")).unique()
        default_signals = {it: "NEUTRAL" for it in original_item_ids}

        if self.predictor is None:
            self.logger.error(f"{self.model_type}: Модель не загружена!")
            return default_signals

        df_prepared = self._prepare_indicators(df.copy())
        if df_prepared.empty:
            self.logger.warning(f"{self.model_type}: Подготовленный DataFrame пуст.")
            return default_signals

        predictions_dict: Dict[str, TimeSeriesDataFrame] = {}

        ts_df = self._prepare_indicator_training_ts(df_prepared)
        if self.mode == 'indicators':
            print(ts_df)
        if ts_df.num_items == 0:
            self.logger.warning("Training TimeSeriesDataFrame is empty")
            return default_signals

        if self.all_time_retrain:
            logging.info("Start tuning...")
            try:
                hyperparameters = {
                    "SimpleFeedForward": [
                        {
                            "ag_args": {"name_suffix": "ShortCtxSimpleBNon"}, # Короткий контекст, простая сеть, BN включен
                            "context_length": 24,         # 1 день
                            "hidden_dimensions": [16],    # Один небольшой слой
                            "batch_normalization": True,
                            "target_scaler": "min_max",
                            "scaling": None,
                            "lr": 1e-3,
                            "batch_size": 128,
                            "max_epochs": 75,
                        },
                    ]
                }
                self.predictor = TimeSeriesPredictor(
                    prediction_length=self.prediction_length,
                    target="target",
                    known_covariates_names=self.known_covariates,
                    path="1h_4pred",
                    eval_metric=DirectionalAccuracy(),
                    quantile_levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                )
                self.predictor.fit(
                    ts_df,
                    presets="best_quality",
                    hyperparameters=hyperparameters,
                    time_limit=None,
                    num_val_windows=1,
                    refit_every_n_windows=1,
                    enable_ensemble=False,
                    verbosity=2,
                    random_seed=42,
                )
            except Exception as e:
                self.logger.exception(f"Ошибка обучения модели: {e}")
                return default_signals

        future_covariates = None
        if self.known_covariates:
            future_covariates = generate_future_known_covariates(
                ts_df,
                self.prediction_length,
                self.freq,
                list(self.known_covariates),
            )
        try:
            preds_all = self.predictor.predict(
                ts_df,
                known_covariates=future_covariates,
                use_cache=False,
            )
        except Exception as e:
            self.logger.exception(f"Ошибка предсказания: {e}")
            return default_signals

        try:
            self.logger.debug(
                "Predictions head:\n%s",
                preds_all.to_data_frame().head().to_string(),
            )
        except Exception:
            pass

        predictions_dict = self._split_indicator_predictions(preds_all)
        signals = self._generate_signals(df_prepared, predictions_dict)
        if self.use_kernel_filter:
            signals = self._apply_kernel_filter(df_prepared, signals)
        signals_arr = ", ".join(f"{sym}: {sig}" for sym, sig in signals.items())
        self.logger.info(f"Signals array: [{signals_arr}]")
        return signals

    def _prepare_inference_df_multi(self, df: pd.DataFrame) -> pd.DataFrame:
        logger = self.logger
        if df.empty or "timestamp" not in df.columns:
            logger.warning("Входной DataFrame пуст или нет timestamp.")
            return pd.DataFrame()

        if "item_id" not in df.columns:
            df["item_id"] = "default_item"

        df["timestamp"] = (
            pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
            .dt.tz_localize(None)
        )
        df.dropna(subset=["timestamp"], inplace=True)

        df = self._ensure_target_column(df)

        for col in ["open", "high", "low", "target", "volume"]:
            if col not in df.columns:
                df[col] = df["target"] if col != "volume" else 0.0
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df.dropna(subset=["target"], inplace=True)

        df = add_time_features(df)
        df = add_indicators(df)
        df = df.groupby("item_id", group_keys=False).apply(
            lambda g: compute_vwap(g), include_groups=True
        )

        df_ret = df[[
            "item_id", "timestamp", "target",
            "open", "high", "low", "volume"
        ] + list(self.known_covariates)]
        df_ret = df_ret.sort_values(["item_id", "timestamp"]).dropna(subset=["target"])
        if self.mode == 'target':
            print(df_ret)
        return df_ret

    def _convert_predictions_to_signals_with_confidence(
        self, predictions: TimeSeriesDataFrame, df_prepared: pd.DataFrame
    ) -> Dict[str, str]:
        all_ids = df_prepared["item_id"].unique()
        default = {iid: "NEUTRAL" for iid in all_ids}
        if predictions.empty:
            logging.warning("Предсказания пусты.")
            return default

        last_idx = df_prepared.groupby("item_id")["timestamp"].idxmax()
        last_price = df_prepared.loc[last_idx, ["item_id","target"]].set_index("item_id")["target"].dropna()

        preds_df = predictions.to_data_frame()
        conf = assess_prediction_confidence(preds_df, last_price, confidence_threshold=self.confidence_threshold)
        if conf.empty:
            logging.warning("confidence_results пуст.")
            return default

        decision_up = conf["decision"] == "UP"
        decision_down = conf["decision"] == "DOWN"
        mask_confident = decision_up | decision_down
        ids = conf.index

        final = {iid: "NEUTRAL" for iid in ids}

        n_unc = np.count_nonzero(~mask_confident)
        logging.info(f"Всего item_id: {len(ids)}, неуверенных: {n_unc}")

        if n_unc == 1:
            uid = ids[~mask_confident][0]
            logging.info(f"1 неуверенный ({uid}) — исключаем.")

        conf_ids = ids[mask_confident]
        if len(conf_ids) == 0:
            logging.info("Все неуверенные — все NEUTRAL.")
            print(final)
            return final

        if len(conf_ids) == 2:
            d1 = "UP" if decision_up[conf_ids[0]] else "DOWN"
            d2 = "UP" if decision_up[conf_ids[1]] else "DOWN"
            if d1 != d2:
                logging.info("2 уверенных, но разное направление — NEUTRAL.")
                print(final)
                return final
            for cid in conf_ids:
                final[cid] = "AGREE_LONG" if decision_up[cid] else "AGREE_SHORT"
            print(final)
            return final

        if len(conf_ids) >= 3:
            arr = np.where(decision_up[conf_ids],1,-1)
            up_count = (arr==1).sum()
            down_count = (arr==-1).sum()
            if (up_count==1 and down_count>1) or (down_count==1 and up_count>1):
                outlier = conf_ids[np.where(arr==(1 if up_count==1 else -1))[0][0]]
                logging.info(f"Выброс {outlier} исключён.")
                conf_ids = conf_ids.drop(outlier)
            for cid in conf_ids:
                final[cid] = "AGREE_LONG" if decision_up[cid] else "AGREE_SHORT"
            print(final)
            return final

        if len(conf_ids) == 1:
            cid = conf_ids[0]
            final[cid] = "AGREE_LONG" if decision_up[cid] else "AGREE_SHORT"
            print(final)
            return final

        print(final)
        return final

    def _run_target(self, df: pd.DataFrame) -> Dict[str, str]:
        original_item_ids = df.get("item_id", pd.Series("unknown_item")).unique()
        default_signals = {it: "NEUTRAL" for it in original_item_ids}

        if self.predictor is None:
            self.logger.error(f"{self.model_type}: Модель не загружена!")
            return default_signals

        df_prepared = self._prepare_inference_df_multi(df.copy())
        if df_prepared.empty:
            self.logger.warning(f"{self.model_type}: Подготовленный DataFrame пуст.")
            return default_signals

        cols = ["item_id", "timestamp", "target"] + list(self.known_covariates) 
        ts_df = TimeSeriesDataFrame.from_data_frame(
            df_prepared[cols], id_column="item_id", timestamp_column="timestamp"
        )

        if self.all_time_retrain:
            self.logger.info("Start tuning...")
            try:
                hyperparameters = {
                    "DLinear": [
                        {},
                    ]
                }
                self.predictor = TimeSeriesPredictor(
                    prediction_length=self.prediction_length,
                    target="target",
                    known_covariates_names=self.known_covariates,
                    path="1h_4pred",
                    eval_metric=DirectionalAccuracy(),
                    quantile_levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
                )
                self.predictor.fit(
                    ts_df,
                    presets="best_quality",
                    hyperparameters=hyperparameters,
                    time_limit=None,
                    num_val_windows=1,
                    refit_every_n_windows=1,
                    enable_ensemble=False,
                    verbosity=2,
                    random_seed=42,
                )
            except Exception as e:
                self.logger.exception(f"Ошибка обучения модели: {e}")
                return default_signals

        future_covariates = None
        if self.known_covariates:
            future_covariates = generate_future_known_covariates(
                ts_df,
                self.prediction_length,
                self.freq,
                list(self.known_covariates),
            )

        try:
            preds = self.predictor.predict(ts_df, known_covariates=future_covariates, use_cache=False)
        except Exception as e:
            self.logger.exception(f"Ошибка предсказания target: {e}")
            return default_signals

        signals = self._convert_predictions_to_signals_with_confidence(preds, df_prepared)
        if self.use_kernel_filter:
            signals = self._apply_kernel_filter(df_prepared, signals)
        signals_arr = ", ".join(f"{sym}: {sig}" for sym, sig in signals.items())
        self.logger.info(f"Signals array: [{signals_arr}]")
        return signals

    def _prepare_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._prepare_inference_df_multi(df)
        if df.empty:
            return df
        df = df.sort_values(["item_id", "timestamp"])
        df["rsi"] = (
            df.groupby("item_id", group_keys=False).apply(lambda g: RSIIndicator(g["target"], window=14).rsi())
            .reset_index(level=0, drop=True)
        )
        df["cci"] = (
            df.groupby("item_id", group_keys=False).apply(lambda g: CCIIndicator(high=g["high"], low=g["low"], close=g["target"], window=20).cci())
            .reset_index(level=0, drop=True)
        )
        df["mfi"] = (
            df.groupby("item_id", group_keys=False).apply(lambda g: MFIIndicator(high=g["high"], low=g["low"], close=g["target"], volume=g["volume"], window=14).money_flow_index())
            .reset_index(level=0, drop=True)
        )
        df["adx"] = (
            df.groupby("item_id", group_keys=False).apply(lambda g: ADXIndicator(high=g["high"], low=g["low"], close=g["target"], window=14).adx())
            .reset_index(level=0, drop=True)
        )
        def calculate_macd(group_df: pd.DataFrame) -> pd.DataFrame:
            macd_indicator = MACD(
                close=group_df["target"],
                window_slow=26,
                window_fast=12,
                window_sign=9,
                fillna=False,
            )
            group_df["macd_line"] = macd_indicator.macd()
            group_df["macd_signal"] = macd_indicator.macd_signal()
            group_df["macd_hist"] = macd_indicator.macd_diff()
            return group_df

        df = df.groupby("item_id", group_keys=False).apply(calculate_macd)

        indicator_cols_to_check = [
            "rsi",
            "cci",
            "mfi",
            "adx",
            "macd_line",
            "macd_signal",
            "macd_hist",
        ]
        return df.dropna(subset=indicator_cols_to_check)

    def _prepare_indicator_training_ts(self, df: pd.DataFrame) -> TimeSeriesDataFrame:
        """
        Build a TimeSeriesDataFrame of each indicator **plus all known-covariate columns**
        so AutoGluon sees the same schema at train- and inference-time.
        """
        indicators = [
            "rsi",
            "cci",
            "mfi",
            "macd_line",
            "macd_hist",
            "macd_signal",
            "adx",
        ]

        records = []
        # --------------------------------------------------
        # 1) Build one tiny DataFrame per (symbol, indicator)
        #    and copy every column listed in `self.known_covariates`
        # --------------------------------------------------
        for sym, group in df.groupby("item_id"):
            for ind in indicators:
                tmp = pd.DataFrame({
                    "item_id": f"{sym}_{ind}",
                    "timestamp": group["timestamp"].values,
                    "target":   group[ind].astype(float).values,
                })

                # add known-covariate columns
                for cov in self.known_covariates:
                    # if a covariate is missing in the group, fill with NaNs
                    tmp[cov] = group[cov].values if cov in group else np.nan

                records.append(tmp)

        if not records:                      #  early-exit: nothing to build
            return TimeSeriesDataFrame()

        # --------------------------------------------------
        # 2) Concatenate and **keep only the requested cols**
        # --------------------------------------------------
        big  = pd.concat(records, ignore_index=True)

        cols = ["item_id", "timestamp", "target"] + list(self.known_covariates)
        big  = big[cols]                     # enforce column order / presence

        # --------------------------------------------------
        # 3) Return a TimeSeriesDataFrame with the right schema
        # --------------------------------------------------
        return TimeSeriesDataFrame.from_data_frame(
            big,
            id_column        = "item_id",
            timestamp_column = "timestamp",
        )


    def _split_indicator_predictions(self, predictions: TimeSeriesDataFrame) -> Dict[str, TimeSeriesDataFrame]:
        indicators = ["rsi", "cci", "mfi", "macd_line", "macd_hist", "macd_signal", "adx"]
        if predictions.empty:
            empty_df_for_ts = pd.DataFrame(columns=["item_id", "timestamp", "target"])
            empty_df_for_ts["timestamp"] = pd.to_datetime(empty_df_for_ts["timestamp"]) # Ensure correct dtype for timestamp
            
            empty_ts_df = TimeSeriesDataFrame.from_data_frame(
                empty_df_for_ts,
                id_column="item_id",
                timestamp_column="timestamp",
            )
            return {ind: empty_ts_df.copy() for ind in indicators}

        df = predictions.to_data_frame().reset_index()
        df[["symbol", "indicator"]] = df["item_id"].str.split("_", n=1, expand=True)
        result = {}
        for ind in indicators:
            sub = df[df["indicator"] == ind].drop(columns=["indicator", "item_id"]).rename(columns={"symbol": "item_id"})
            if sub.empty:
                empty_df_for_ts_sub = pd.DataFrame(columns=["item_id", "timestamp", "target"] + [col for col in sub.columns if col not in ["item_id", "timestamp"]]) # Keep other pred columns
                empty_df_for_ts_sub["timestamp"] = pd.to_datetime(empty_df_for_ts_sub["timestamp"])
                result[ind] = TimeSeriesDataFrame.from_data_frame(
                    empty_df_for_ts_sub,
                    id_column="item_id",
                    timestamp_column="timestamp"
                )
            else:
                result[ind] = TimeSeriesDataFrame.from_data_frame(
                    sub,
                    id_column="item_id",
                    timestamp_column="timestamp",
                )
        return result

    def _generate_signals(self, df_prepared: pd.DataFrame, preds: Dict[str, TimeSeriesDataFrame]) -> Dict[str, str]:
        signals: Dict[str, str] = {}

        def _get_predicted_value(indicator_name: str, current_item_id: str) -> float:
            if indicator_name not in preds:
                self.logger.warning(f"Словарь предсказаний не содержит ключ '{indicator_name}' для {current_item_id}.")
                raise KeyError(f"Missing indicator {indicator_name} in predictions_dict")

            indicator_ts_df = preds[indicator_name]

            if current_item_id not in indicator_ts_df.item_ids:
                self.logger.warning(f"Предсказания для индикатора {indicator_name} не содержат item_id '{current_item_id}'. Доступные item_ids: {list(indicator_ts_df.item_ids)}")
                raise KeyError(f"Missing item_id {current_item_id} in {indicator_name} predictions")
            
            item_specific_ts_df = indicator_ts_df.loc[current_item_id]
            item_pred_df = item_specific_ts_df.to_data_frame()

            if item_pred_df.empty:
                self.logger.warning(f"Пустой DataFrame предсказаний для {indicator_name} по {current_item_id} после фильтрации.")
                raise ValueError(f"Empty prediction DataFrame for {indicator_name} on {current_item_id}")

            if len(item_pred_df) != 1:
                self.logger.error(
                    f"Ожидалась 1 точка предсказания для {indicator_name} по {current_item_id}, "
                    f"получено: {len(item_pred_df)}. Данные:\n{item_pred_df.to_string()}"
                )
                raise ValueError(f"Expected 1 prediction point, got {len(item_pred_df)}")

            if "mean" in item_pred_df.columns:
                return float(item_pred_df["mean"].iloc[0])
            elif "0.5" in item_pred_df.columns:
                return float(item_pred_df["0.5"].iloc[0])
            else:
                cols_available = ", ".join(item_pred_df.columns)
                self.logger.error(
                    f"Колонки 'mean' или '0.5' не найдены в предсказаниях для {indicator_name} "
                    f"по {current_item_id}. Доступные колонки: [{cols_available}]"
                )
                raise KeyError("Missing 'mean' or '0.5' prediction column")

        for item_id, group in df_prepared.groupby("item_id"):
            if group.empty:
                signals[item_id] = "NEUTRAL"
                continue
            last_vals = group.iloc[-1]
            
            try:
                pred_rsi_val = _get_predicted_value("rsi", item_id)
                pred_cci_val = _get_predicted_value("cci", item_id)
                pred_mfi_val = _get_predicted_value("mfi", item_id)
                pred_adx_val = _get_predicted_value("adx", item_id)
                pred_macd_line_val = _get_predicted_value("macd_line", item_id)
                pred_macd_signal_val = _get_predicted_value("macd_signal", item_id)
            except (KeyError, ValueError) as e:
                self.logger.error(f"Ошибка получения предсказанного значения для {item_id}: {e}")
                signals[item_id] = "NEUTRAL"
                continue
            
            bull = 0.0
            bear = 0.0

            if pd.notna(pred_rsi_val) and pd.notna(last_vals.rsi):
                if pred_rsi_val > last_vals.rsi:
                    if last_vals.rsi < 30: bull += 1.5
                    elif last_vals.rsi < 50: bull += 1.0
                    else: bull += 0.5
                elif pred_rsi_val < last_vals.rsi:
                    if last_vals.rsi > 70: bear += 1.5
                    elif last_vals.rsi > 50: bear += 1.0
                    else: bear += 0.5

            if pd.notna(pred_cci_val) and pd.notna(last_vals.cci):
                if pred_cci_val > last_vals.cci:
                    if last_vals.cci < -100: bull += 1.5
                    elif last_vals.cci < 0: bull += 1.0
                    else: bull += 0.5
                elif pred_cci_val < last_vals.cci:
                    if last_vals.cci > 100: bear += 1.5
                    elif last_vals.cci > 0: bear += 1.0
                    else: bear += 0.5

            if pd.notna(pred_mfi_val) and pd.notna(last_vals.mfi):
                if pred_mfi_val > last_vals.mfi:
                    if last_vals.mfi < 20: bull += 1.5
                    elif last_vals.mfi < 50: bull += 1.0
                    else: bull += 0.5
                elif pred_mfi_val < last_vals.mfi:
                    if last_vals.mfi > 80: bear += 1.5
                    elif last_vals.mfi > 50: bear += 1.0
                    else: bear += 0.5
            
            if pd.notna(pred_macd_line_val) and pd.notna(pred_macd_signal_val) and pd.notna(last_vals.macd_line) and pd.notna(last_vals.macd_signal):
                if pred_macd_line_val > pred_macd_signal_val and last_vals.macd_line <= last_vals.macd_signal:
                    bull += 1.5
                elif pred_macd_line_val < pred_macd_signal_val and last_vals.macd_line >= last_vals.macd_signal:
                    bear += 1.5

                if pred_macd_line_val > 0 and last_vals.macd_line <= 0:
                    bull += 1.0
                elif pred_macd_line_val < 0 and last_vals.macd_line >= 0:
                    bear += 1.0

                pred_hist_val = pred_macd_line_val - pred_macd_signal_val
                last_hist_val = last_vals.macd_hist
                if pd.notna(last_hist_val) and pd.notna(pred_hist_val):
                    if pred_hist_val > last_hist_val:
                        if pred_hist_val > 0: bull += 0.5
                        else: bull += 0.25
                    elif pred_hist_val < last_hist_val:
                        if pred_hist_val < 0: bear += 0.5
                        else: bear += 0.25
            
            adx_threshold = 10.0
            adx_strong_enough = pd.notna(pred_adx_val) and pred_adx_val >= adx_threshold
            
            final_signal = "NEUTRAL"
            bull_threshold = 2.0 
            bear_threshold = 2.0

            if bull > bear and bull >= bull_threshold:
                if adx_strong_enough:
                    final_signal = "AGREE_LONG"
            elif bear > bull and bear >= bear_threshold:
                if adx_strong_enough:
                    final_signal = "AGREE_SHORT"
            
            signals[item_id] = final_signal
        return signals

    def _apply_kernel_filter(self, df: pd.DataFrame, signals: Dict[str, str]) -> Dict[str, str]:
        global GLOBAL_KERNEL_DIR
        if "target" not in df.columns:
            return signals

        N_window_data = self.kernel_lookback
        L_char_for_weights = self.kernel_lookback
        alpha_r_val = self.kernel_r
        
        if N_window_data < 2:
            return signals 

        for item_id, group in df.groupby("item_id"):
            series = group["target"].astype(float)
            
            if len(series) < N_window_data:
                direction = GLOBAL_KERNEL_DIR.get(item_id, 0)
            else:
                series_slice = series.tail(N_window_data)
                
                direction = compute_kernel_direction(
                    close_series_slice=series_slice,
                    char_length=L_char_for_weights,
                    alpha_r_param=alpha_r_val,
                    n_window=N_window_data
                )

            if direction != 0:
                GLOBAL_KERNEL_DIR[item_id] = direction
            else:
                direction = GLOBAL_KERNEL_DIR.get(item_id, 0)

            current_signal_status = signals.get(item_id, "NEUTRAL")
            if current_signal_status == "AGREE_LONG" and direction == -1:
                signals[item_id] = "NEUTRAL"
            elif current_signal_status == "AGREE_SHORT" and direction == 1:
                signals[item_id] = "NEUTRAL"
        return signals