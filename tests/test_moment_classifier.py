import math
import os
import pandas as pd
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from moment_classifier import MomentClassifier


def create_dummy_df(seq_len=16, steps=40, item="BTCUSDT"):
    data = []
    for i in range(steps):
        val = math.sin(i / 4)
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


def test_train_and_predict(tmp_path):
    config = {"seq_len": 16, "results_output_dir": tmp_path, "all_time_retrain": False}
    model = MomentClassifier(config)
    model.load_model()
    df = create_dummy_df()
    model.fit(df)
    signals = model.predict(df)
    assert signals["BTCUSDT"] in {"AGREE_LONG", "AGREE_SHORT", "NEUTRAL"}


def test_predict_without_training(tmp_path):
    config = {"seq_len": 8, "results_output_dir": tmp_path, "all_time_retrain": False}
    model = MomentClassifier(config)
    model.load_model()
    df = create_dummy_df(seq_len=8, steps=20)
    signals = model.predict(df)
    assert all(sig == "NEUTRAL" for sig in signals.values())


def test_auto_retrain(tmp_path):
    config = {"seq_len": 8, "results_output_dir": tmp_path, "all_time_retrain": True}
    model = MomentClassifier(config)
    model.load_model()
    df = create_dummy_df(seq_len=8, steps=32)
    model.fit(df)
    assert model._steps_since_train == 0
    model.predict(df)
    # After prediction, retrain should trigger when all_time_retrain is True
    assert model._steps_since_train == 0


def test_predict_small_dataset(tmp_path):
    config = {"seq_len": 16, "results_output_dir": tmp_path, "all_time_retrain": False}
    model = MomentClassifier(config)
    model.load_model()
    df = create_dummy_df(seq_len=16, steps=10)
    model.fit(df)
    signals = model.predict(df)
    assert signals["BTCUSDT"] == "NEUTRAL"
