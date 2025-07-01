import logging
from typing import Dict
import pandas as pd
import torch

from kronos.model import KronosTokenizer, Kronos, KronosPredictor

logger = logging.getLogger(__name__)

class KronosForecaster:
    """Wrapper around KronosPredictor for one-step forecasts."""

    def __init__(self, config: Dict):
        self.seq_len = config.get("seq_len", 512)
        self.pred_len = config.get("prediction_length", 1)
        self.model_name = config.get("model_name", "NeoQuasar/Kronos-mini")
        self.tokenizer_name = config.get(
            "tokenizer_name", "NeoQuasar/Kronos-Tokenizer-base"
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.predictor: KronosPredictor | None = None

    def load_model(self) -> None:
        """Load Kronos model and tokenizer from Hugging Face."""
        logger.info("Loading Kronos model %s", self.model_name)
        tokenizer = KronosTokenizer.from_pretrained(self.tokenizer_name)
        model = Kronos.from_pretrained(self.model_name)
        self.predictor = KronosPredictor(
            model, tokenizer, device=self.device, max_context=self.seq_len
        )

    def predict(self, df: pd.DataFrame, return_values: bool = False) -> Dict:
        """Return trading signals or raw predictions.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with columns ``item_id`` and OHLCV features.
        return_values : bool, optional
            If ``True`` each entry will contain the predicted close and last
            close in addition to the signal.

        Returns
        -------
        dict
            Mapping of ``item_id`` to signal string or to a dictionary with
            ``predicted_close`` and ``last_close``.
        """
        if self.predictor is None:
            neutral = {sid: "NEUTRAL" for sid in df["item_id"].unique()}
            return neutral if not return_values else {
                sid: {"signal": "NEUTRAL", "predicted_close": float("nan"), "last_close": float("nan")}
                for sid in df["item_id"].unique()
            }

        results: Dict = {}
        for item_id, group in df.groupby("item_id"):
            group = group.sort_values("timestamp")
            if len(group) < self.seq_len:
                results[item_id] = (
                    {"signal": "NEUTRAL", "predicted_close": float("nan"), "last_close": float("nan")}
                    if return_values
                    else "NEUTRAL"
                )
                continue

            context = group.iloc[-self.seq_len:]
            x_df = context[["open", "high", "low", "close", "volume"]].copy()
            if "amount" in context.columns:
                x_df["amount"] = context["amount"]
            x_timestamp = context["timestamp"]

            if len(x_timestamp) >= 2:
                delta = x_timestamp.iloc[-1] - x_timestamp.iloc[-2]
            else:
                delta = pd.Timedelta(minutes=1)
            y_timestamp = pd.Series([x_timestamp.iloc[-1] + delta])

            pred_df = self.predictor.predict(
                df=x_df,
                x_timestamp=x_timestamp,
                y_timestamp=y_timestamp,
                pred_len=1,
                T=1.0,
                top_p=0.9,
                sample_count=1,
                verbose=False,
            )
            predicted_close = float(pred_df["close"].iloc[0])
            last_close = float(context["close"].iloc[-1])

            if predicted_close > last_close:
                signal = "AGREE_LONG"
            elif predicted_close < last_close:
                signal = "AGREE_SHORT"
            else:
                signal = "NEUTRAL"

            if return_values:
                results[item_id] = {
                    "signal": signal,
                    "predicted_close": predicted_close,
                    "last_close": last_close,
                }
            else:
                results[item_id] = signal

        return results
