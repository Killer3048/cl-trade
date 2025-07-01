import argparse
import pandas as pd

from kronos_forecaster import KronosForecaster


def prepare_context_and_targets(df: pd.DataFrame, seq_len: int):
    """Split each item into context (all but last row) and target info."""
    contexts = []
    meta = {}
    for item_id, group in df.groupby("item_id"):
        group = group.sort_values("timestamp")
        if len(group) < seq_len + 1:
            continue
        context = group.iloc[:-1]
        contexts.append(context)
        meta[item_id] = {
            "next_close": float(group["close"].iloc[-1]),
            "last_close": float(group["close"].iloc[-2]),
        }
    if not contexts:
        raise ValueError("Not enough data for evaluation")
    return pd.concat(contexts).reset_index(drop=True), meta


def evaluate_directional_accuracy(df: pd.DataFrame, forecaster: KronosForecaster) -> float:
    context_df, meta = prepare_context_and_targets(df, forecaster.seq_len)
    predictions = forecaster.predict(context_df, return_values=True)

    correct = 0
    total = 0
    for item_id, pred in predictions.items():
        info = meta.get(item_id)
        if info is None:
            continue
        signal = pred.get("signal")
        if signal == "NEUTRAL":
            continue

        predicted_close = pred["predicted_close"]
        last_close = info["last_close"]
        actual_change = info["next_close"] - last_close
        predicted_change = predicted_close - last_close

        if actual_change > 0 and predicted_change > 0:
            correct += 1
        elif actual_change < 0 and predicted_change < 0:
            correct += 1
        total += 1

    return correct / total if total else 0.0


def main():
    parser = argparse.ArgumentParser(description="Evaluate Kronos forecasting model")
    parser.add_argument("csv_path", nargs="?", default="full_1h.csv")
    parser.add_argument("--model_name", default="NeoQuasar/Kronos-mini")
    parser.add_argument("--seq_len", type=int, default=512)
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path, parse_dates=["timestamp"])

    cfg = {"seq_len": args.seq_len, "prediction_length": 1, "model_name": args.model_name}
    forecaster = KronosForecaster(cfg)
    forecaster.load_model()

    acc = evaluate_directional_accuracy(df, forecaster)
    print(f"Directional accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
