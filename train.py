import argparse
import pandas as pd

from kronos_forecaster import KronosForecaster


def prepare_context_and_targets(
    df: pd.DataFrame, seq_len: int, num_windows: int = 1
) -> tuple[pd.DataFrame, dict]:
    """Prepare sliding windows for evaluation.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with an ``item_id`` column.
    seq_len : int
        Length of the context sequence for the model.
    num_windows : int, optional
        How many trailing windows to generate per item. Each window
        results in a separate ``item_id`` with ``__idx`` suffix.
    Returns
    -------
    pd.DataFrame
        Concatenated contexts for all items/windows.
    dict
        Mapping from the synthetic ``item_id`` to target metadata.
    """

    contexts: list[pd.DataFrame] = []
    meta: dict = {}

    for item_id, group in df.groupby("item_id"):
        group = group.sort_values("timestamp").reset_index(drop=True)
        required_len = seq_len + num_windows
        if len(group) < required_len:
            continue

        for idx in range(num_windows):
            context_start = len(group) - required_len + idx
            context_end = context_start + seq_len
            target_idx = context_end

            context = group.iloc[context_start:context_end].copy()
            synthetic_id = f"{item_id}__{idx}"
            context["item_id"] = synthetic_id
            contexts.append(context)

            meta[synthetic_id] = {
                "next_close": float(group["close"].iloc[target_idx]),
                "last_close": float(group["close"].iloc[context_end - 1]),
                "orig_item_id": item_id,
            }

    if not contexts:
        raise ValueError("Not enough data for evaluation")

    return pd.concat(contexts).reset_index(drop=True), meta


def evaluate_directional_accuracy(
    df: pd.DataFrame, forecaster: KronosForecaster, num_val_windows: int = 1
) -> float:
    """Calculate directional accuracy across multiple windows."""

    context_df, meta = prepare_context_and_targets(
        df, forecaster.seq_len, num_windows=num_val_windows
    )
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
    parser.add_argument("--num_val_windows", type=int, default=1,
                        help="number of evaluation windows per item")
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path, parse_dates=["timestamp"])

    cfg = {"seq_len": args.seq_len, "prediction_length": 1, "model_name": args.model_name}
    forecaster = KronosForecaster(cfg)
    forecaster.load_model()

    acc = evaluate_directional_accuracy(df, forecaster, num_val_windows=args.num_val_windows)
    print(f"Directional accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
