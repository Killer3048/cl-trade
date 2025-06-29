import os
import json
import logging
import time
import pandas as pd
import torch

from moment_classifier import MomentClassifier

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")
try:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        CONFIG = json.load(f)
except FileNotFoundError:
    logging.critical(
        f"Configuration file not found at path: {CONFIG_PATH}"
    )
    exit(1)

def setup_logging():
    """Configure global logging format for the training script."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def split_by_item(df: pd.DataFrame, seq_len: int, pred_len: int, num_val_windows: int):
    """
    Split DataFrame into train and test parts based on a fixed number of windows.
    The test set for each item will contain exactly enough data to generate `num_val_windows`.
    """
    logging.info(f"Splitting data using fixed window method. Reserving data for {num_val_windows} windows per item.")
    
    # Calculate the exact number of rows to reserve for the test set.
    # This must match the `required_len` calculation in `_build_sequences` for validation.
    test_set_size = seq_len + pred_len + num_val_windows - 1
    logging.info(f"Calculated test set size per item: {test_set_size} rows.")

    train_parts, test_parts = [], []
    
    for item_id, group in df.groupby("item_id"):
        group = group.sort_values('timestamp')
        group_len = len(group)
        
        # Check if there's enough data in the group to create a test set AND
        # leave at least one sample for training.
        min_train_sample_size = seq_len + pred_len
        if group_len < test_set_size + min_train_sample_size:
            logging.warning(
                f"Skipping item '{item_id}': Not enough data ({group_len}) to create a test set of size "
                f"{test_set_size} and a minimal training sample."
            )
            continue
        
        # Split point: everything except the last `test_set_size` rows.
        split_idx = group_len - test_set_size
        train_parts.append(group.iloc[:split_idx])
        test_parts.append(group.iloc[split_idx:])

    if not test_parts:
         raise ValueError("Could not split data. No item had enough data points for the specified configuration.")

    train_df = pd.concat(train_parts).reset_index(drop=True)
    test_df = pd.concat(test_parts).reset_index(drop=True)
    
    logging.info(
        f"Split complete. Train records: {len(train_df)} ({train_df['item_id'].nunique()} items), "
        f"Test records: {len(test_df)} ({test_df['item_id'].nunique()} items)"
    )
    return train_df, test_df

def evaluate(model: MomentClassifier, df: pd.DataFrame, num_workers: int = 0) -> float:
    """Evaluate a trained model on a dataset and return accuracy."""
    logging.info("Starting final evaluation...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    X, y, _ = model._build_sequences(df, purpose="evaluation")
    if y.size == 0:
        logging.warning("No sequences generated for evaluation. Returning 0.0 accuracy.")
        return 0.0
        
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    loader = torch.utils.data.DataLoader(dataset, batch_size=model.batch_size, num_workers=num_workers, pin_memory=True)
    
    model.moment.to(device)
    return model.evaluate_on_loader(loader, device)

def main():
    """Entry point for training the MomentClassifier model."""
    setup_logging()
    
    logging.info("--- Starting Training Script ---")
    logging.info(f"Loading configuration from {CONFIG_PATH}")
    
    csv_path = "full_1h.csv"
    
    tf_config = CONFIG["timeframe_config"]["LONG_TF"]
    
    training_config = {
        "seq_len": tf_config.get("seq_len", 512),
        "prediction_length": tf_config.get("prediction_length", 1),
        "num_val_windows": tf_config.get("num_val_windows", 3),
        "results_output_dir": tf_config.get("results_output_dir", "1h_4pred"),
        "model_name": tf_config.get("model_name", "AutonLab/MOMENT-1-large"),
        "epochs": tf_config.get("epochs", 100),
        "batch_size": tf_config.get("batch_size", 32),
        "num_batches_per_epoch": tf_config.get("num_batches_per_epoch", 50),
        "lr": tf_config.get("lr", 1e-4),
        "num_workers": tf_config.get("num_workers", 4),
        "all_time_retrain": False,
        "save_on_improve": False, 
        "compile_model": True,
    }
    logging.info(f"Training configuration: {training_config}")
    
    try:
        df = pd.read_csv(csv_path, parse_dates=["timestamp"])
        logging.info(f"Loaded data with {len(df)} rows and {df['item_id'].nunique()} unique items.")
        
        train_df, test_df = split_by_item(
            df,
            seq_len=training_config["seq_len"],
            pred_len=training_config["prediction_length"],
            num_val_windows=training_config["num_val_windows"]
        )
    except (FileNotFoundError, ValueError) as e:
        logging.error(f"FATAL: Error loading or splitting data from '{csv_path}': {e}", exc_info=True)
        return

    model = MomentClassifier(training_config)
    
    try:
        model.load_model()
    except Exception as e:
        logging.error(f"FATAL: Could not proceed because initial model loading failed.", exc_info=True)
        return
    
    fit_start_time = time.time()
    # Pass the test_df for validation during training. This is standard practice.
    model.fit(train_df, test_df, num_workers=training_config["num_workers"])
    logging.info(f"Total fitting process took: {time.time() - fit_start_time:.2f} seconds.")

    logging.info("Reloading the best model for final evaluation...")
    best_model = MomentClassifier(training_config)
    try:
        best_model.load_model()
    except Exception as e:
        logging.error(f"FATAL: Could not load the best model for evaluation.", exc_info=True)
        return

    if not best_model.is_trained:
        logging.warning("Final evaluation is being run on a model that was not locally fine-tuned.")

    acc = evaluate(best_model, test_df, num_workers=training_config["num_workers"])
    print(f"\n=========================================")
    print(f"Final Accuracy on test set: {acc:.4f}")
    print(f"=========================================\n")

if __name__ == "__main__":
    main()