import pandas as pd
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from moment_classifier import MomentClassifier

def test_train_and_predict():
    config = {"seq_len": 16, "all_time_retrain": 1}
    model = MomentClassifier(config)
    model.load_model()
    # create simple increasing close
    import math
    data = []
    for i in range(40):
        val = math.sin(i / 4)
        data.append({
            "item_id": "BTCUSDT",
            "timestamp": pd.Timestamp("2024-01-01") + pd.Timedelta(minutes=i),
            "open": val,
            "high": val + 0.5,
            "low": val - 0.5,
            "close": val,
            "volume": 1.0,
        })
    df = pd.DataFrame(data)
    model.fit(df)
    signals = model.predict(df)
    assert signals["BTCUSDT"] in {"AGREE_LONG", "AGREE_SHORT", "NEUTRAL"}

