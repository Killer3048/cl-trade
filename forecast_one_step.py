import logging
import json
import os
import pandas as pd
from kronos_forecaster import KronosForecaster

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")

logging.basicConfig(level=logging.INFO)


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def main(csv_path: str = "full_1h.csv"):
    config = load_config()
    tf_cfg = config["timeframe_config"]["LONG_TF"]
    model_cfg = {
        "seq_len": tf_cfg.get("seq_len", 512),
        "prediction_length": tf_cfg.get("prediction_length", 1),
        "model_name": tf_cfg.get("model_name", "NeoQuasar/Kronos-mini"),
    }

    df = pd.read_csv(csv_path, parse_dates=["timestamp"])

    forecaster = KronosForecaster(model_cfg)
    forecaster.load_model()
    signals = forecaster.predict(df)

    for item_id, signal in signals.items():
        print(f"{item_id}: {signal}")


if __name__ == "__main__":
    main()
