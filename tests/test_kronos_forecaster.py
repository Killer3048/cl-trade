import pandas as pd
import math
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from kronos_forecaster import KronosForecaster

def create_df(steps=20, item="BTCUSDT"):
    data = []
    for i in range(steps):
        val = math.sin(i/4)
        data.append({
            "item_id": item,
            "timestamp": pd.Timestamp("2024-01-01") + pd.Timedelta(minutes=i),
            "open": val,
            "high": val + 0.5,
            "low": val - 0.5,
            "close": val,
            "volume": 1.0,
        })
    return pd.DataFrame(data)

def test_predict_signal(tmp_path):
    cfg = {"seq_len": 8, "prediction_length": 1, "model_name": "NeoQuasar/Kronos-mini"}
    f = KronosForecaster(cfg)
    f.load_model()
    df = create_df(steps=10)
    signals = f.predict(df)
    assert set(signals.values()) <= {"AGREE_LONG", "AGREE_SHORT", "NEUTRAL"}


def test_evaluate_accuracy(tmp_path):
    cfg = {"seq_len": 8, "prediction_length": 1, "model_name": "NeoQuasar/Kronos-mini"}
    f = KronosForecaster(cfg)
    f.load_model()
    df = create_df(steps=12)
    from train import evaluate_directional_accuracy
    acc = evaluate_directional_accuracy(df, f)
    assert 0.0 <= acc <= 1.0
