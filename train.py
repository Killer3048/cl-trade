import argparse
import pandas as pd
from sklearn.metrics import accuracy_score
import torch
from torch.utils.data import DataLoader, TensorDataset

from moment_classifier import MomentClassifier


def split_by_item(df: pd.DataFrame, test_ratio: float = 0.2):
    train_parts = []
    test_parts = []
    for item_id, group in df.groupby("item_id"):
        group = group.sort_values("timestamp")
        split_idx = int(len(group) * (1 - test_ratio))
        train_parts.append(group.iloc[:split_idx])
        test_parts.append(group.iloc[split_idx:])
    train_df = pd.concat(train_parts).reset_index(drop=True)
    test_df = pd.concat(test_parts).reset_index(drop=True)
    return train_df, test_df


def evaluate(model: MomentClassifier, df: pd.DataFrame) -> float:
    X, y = model._build_sequences(df)
    if len(y) == 0:
        return 0.0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long),
    )
    loader = DataLoader(dataset, batch_size=model.batch_size, shuffle=False)
    model.moment.to(device)
    model.moment.eval()
    preds = []
    with torch.no_grad():
        for batch_x, _ in loader:
            batch_x = batch_x.to(device)
            mask = torch.ones(batch_x.shape[0], model.seq_len, dtype=torch.long).to(device)
            logits = model.moment(x_enc=batch_x, input_mask=mask).logits
            preds.append(logits.argmax(dim=1).cpu())
    preds = torch.cat(preds).numpy()
    return accuracy_score(y, preds)


def main():
    parser = argparse.ArgumentParser(description="Train MOMENT classifier")
    parser.add_argument("csv_path", help="CSV file with columns item_id, timestamp, open, high, low, close, volume")
    parser.add_argument("--output_dir", default="moment_model")
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--prediction_length", type=int, default=1)
    parser.add_argument("--model_name", default="AutonLab/MOMENT-1-large")
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path, parse_dates=["timestamp"])
    train_df, test_df = split_by_item(df)

    config = {
        "seq_len": args.seq_len,
        "results_output_dir": args.output_dir,
        "model_name": args.model_name,
        "prediction_length": args.prediction_length,
        "all_time_retrain": False,
    }
    model = MomentClassifier(config)
    model.load_model()
    model.fit(train_df, test_df)

    signals = model.predict(test_df)
    print("Sample signals:", signals)

    acc = evaluate(model, test_df)
    print(f"Accuracy on test set: {acc:.4f}")


if __name__ == "__main__":
    main()
