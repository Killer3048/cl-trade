import math
import os
import pandas as pd
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from moment_classifier import MomentClassifier
from train import split_by_item, evaluate


def create_dummy_df(seq_len=16, steps=40, item="BTCUSDT"):
    data = []
    for i in range(steps):
        val = math.sin(i / 4)
        data.append(
            {
                "item_id": item,
                "timestamp": pd.Timestamp("2024-01-01") + pd.Timedelta(minutes=i),
                "open": val,
                "high": val + 0.5,
                "low": val - 0.5,
                "close": val,
                "volume": 1.0,
            }
        )
    return pd.DataFrame(data)


def test_train_and_predict(tmp_path):
    config = {
        "seq_len": 16,
        "results_output_dir": tmp_path,
        "all_time_retrain": False,
        "model_name": "AutonLab/MOMENT-1-small",
        "prediction_length": 1,
    }
    model = MomentClassifier(config)
    model.load_model()
    df = create_dummy_df()
    model.fit(df, df)
    signals = model.predict(df)
    assert signals["BTCUSDT"] in {"AGREE_LONG", "AGREE_SHORT", "NEUTRAL"}


def test_predict_without_training(tmp_path):
    config = {
        "seq_len": 8,
        "results_output_dir": tmp_path,
        "all_time_retrain": False,
        "model_name": "AutonLab/MOMENT-1-small",
        "prediction_length": 1,
    }
    model = MomentClassifier(config)
    model.load_model()
    df = create_dummy_df(seq_len=8, steps=20)
    signals = model.predict(df)
    assert all(sig == "NEUTRAL" for sig in signals.values())


def test_auto_retrain(tmp_path):
    config = {
        "seq_len": 8,
        "results_output_dir": tmp_path,
        "all_time_retrain": True,
        "model_name": "AutonLab/MOMENT-1-small",
        "prediction_length": 1,
    }
    model = MomentClassifier(config)
    model.load_model()
    df = create_dummy_df(seq_len=8, steps=32)
    model.fit(df, df)
    assert model._steps_since_train == 0
    model.predict(df)
    # After prediction, retrain should trigger when all_time_retrain is True
    assert model._steps_since_train == 0


def test_predict_small_dataset(tmp_path):
    config = {
        "seq_len": 16,
        "results_output_dir": tmp_path,
        "all_time_retrain": False,
        "model_name": "AutonLab/MOMENT-1-small",
        "prediction_length": 1,
    }
    model = MomentClassifier(config)
    model.load_model()
    df = create_dummy_df(seq_len=16, steps=10)
    model.fit(df, df)
    signals = model.predict(df)
    assert signals["BTCUSDT"] == "NEUTRAL"


def test_split_and_evaluate(tmp_path):
    df = create_dummy_df(seq_len=16, steps=40)
    train_df, test_df = split_by_item(df, test_ratio=0.25)
    config = {
        "seq_len": 16,
        "results_output_dir": tmp_path,
        "all_time_retrain": False,
        "model_name": "AutonLab/MOMENT-1-small",
        "prediction_length": 1,
    }
    model = MomentClassifier(config)
    model.load_model()
    model.fit(train_df, test_df)
    acc = evaluate(model, test_df)
    assert 0.0 <= acc <= 1.0


def test_split_by_item_multiple(tmp_path):
    df_a = create_dummy_df(seq_len=16, steps=20, item="A")
    df_b = create_dummy_df(seq_len=16, steps=20, item="B")
    df = pd.concat([df_a, df_b], ignore_index=True)
    train_df, test_df = split_by_item(df, test_ratio=0.3)

    for item_id, group in df.groupby("item_id"):
        total_len = len(group)
        train_len = len(train_df[train_df["item_id"] == item_id])
        test_len = len(test_df[test_df["item_id"] == item_id])
        assert train_len + test_len == total_len
        # Allow one-off rounding difference
        expected_train_len = int(total_len * 0.7)
        assert abs(train_len - expected_train_len) <= 1


def test_label_shift(tmp_path):
    df = create_dummy_df(seq_len=4, steps=12)
    config = {
        "seq_len": 4,
        "results_output_dir": tmp_path,
        "all_time_retrain": False,
        "model_name": "AutonLab/MOMENT-1-small",
        "prediction_length": 2,
    }
    model = MomentClassifier(config)
    model.load_model()
    X, y = model._build_sequences(df)
    assert len(y) == len(df) - config["seq_len"] - config["prediction_length"] + 1
    closes = df["close"].to_numpy(float)
    for i, target in enumerate(y):
        future = closes[i + config["seq_len"] + config["prediction_length"] - 1]
        past = closes[i + config["seq_len"] - 1]
        expected = 1 if future > past else 0
        assert target == expected
