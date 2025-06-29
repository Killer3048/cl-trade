import os
import logging
from typing import Dict, List
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from momentfm import MOMENTPipeline

logger = logging.getLogger(__name__)

class MomentClassifier:
    def __init__(self, config: dict):
        self.seq_len = config.get("seq_len", 64)
        self.pred_len = config.get("pred_len", config.get("prediction_length", 1))
        self.model_name = config.get("model_name", "AutonLab/MOMENT-1-large")
        self.results_output_dir = config.get("results_output_dir", "moment_model")
        self.retrain_interval = 1 if config.get("all_time_retrain", False) else 0
        self.epochs = config.get("epochs", 1)
        self.batch_size = config.get("batch_size", 32)
        self.lr = config.get("lr", 1e-4)
        self._steps_since_train = 0
        self.moment: MOMENTPipeline | None = None
        self.is_trained = False

    def load_model(self):
        os.makedirs(self.results_output_dir, exist_ok=True)
        model_path = (
            self.results_output_dir
            if os.path.isfile(os.path.join(self.results_output_dir, "config.json"))
            else self.model_name
        )
        self.moment = MOMENTPipeline.from_pretrained(
            model_path,
            model_kwargs={"task_name": "classification", "n_channels": 5, "num_class": 2},
        )
        self.moment.init()
        self.is_trained = os.path.isfile(os.path.join(self.results_output_dir, "config.json"))

    def save_model(self):
        if self.moment is not None:
            self.moment.save_pretrained(self.results_output_dir)


    def _build_sequences(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        df = df.sort_values(["item_id", "timestamp"])
        X_list: List[np.ndarray] = []
        y_list: List[int] = []
        for item_id, group in df.groupby("item_id"):
            group = group.reset_index(drop=True)
            closes = group["close"].to_numpy(float)
            vals = group[["open", "high", "low", "close", "volume"]].to_numpy(float)
            for i in range(len(group) - self.seq_len - self.pred_len + 1):
                seq = vals[i : i + self.seq_len].T
                future_idx = i + self.seq_len + self.pred_len - 1
                past_idx = i + self.seq_len - 1
                target = 1 if closes[future_idx] > closes[past_idx] else 0
                X_list.append(seq)
                y_list.append(target)
        if not X_list:
            return np.empty((0, 5, self.seq_len), float), np.empty((0,), int)
        return np.stack(X_list), np.array(y_list)

    def _embed(self, x: np.ndarray) -> np.ndarray:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.moment.to(device)
        self.moment.eval()
        with torch.no_grad():
            tensor_x = torch.tensor(x, dtype=torch.float32).to(device)
            mask = torch.ones(tensor_x.shape[0], self.seq_len, dtype=torch.long).to(device)
            out = self.moment.embed(x_enc=tensor_x, input_mask=mask)
            emb = out.embeddings.detach().cpu().numpy()
        return emb

    def fit(self, df: pd.DataFrame, val_df: pd.DataFrame | None = None):
        X, y = self._build_sequences(df)
        if X.size == 0:
            logger.warning("No training data available")
            return
        if val_df is not None:
            X_val, y_val = self._build_sequences(val_df)
        else:
            X_val, y_val = None, None
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dataset = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.long),
        )
        if X_val is not None:
            val_dataset = TensorDataset(
                torch.tensor(X_val, dtype=torch.float32),
                torch.tensor(y_val, dtype=torch.long),
            )
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        else:
            val_loader = None
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        self.moment.to(device)
        self.moment.train()
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.moment.parameters(), lr=self.lr)
        for _ in range(self.epochs):
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                mask = torch.ones(batch_x.shape[0], self.seq_len, dtype=torch.long).to(device)
                out = self.moment(x_enc=batch_x, input_mask=mask)
                loss = criterion(out.logits, batch_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if val_loader is not None:
                self.moment.eval()
                total_correct = 0
                total_samples = 0
                with torch.no_grad():
                    for val_x, val_y in val_loader:
                        val_x = val_x.to(device)
                        val_y = val_y.to(device)
                        mask = torch.ones(val_x.shape[0], self.seq_len, dtype=torch.long).to(device)
                        logits = self.moment(x_enc=val_x, input_mask=mask).logits
                        total_correct += (logits.argmax(dim=1) == val_y).sum().item()
                        total_samples += val_y.size(0)
                val_acc = total_correct / max(total_samples, 1)
                logger.info("Validation accuracy: %.4f", val_acc)
                self.moment.train()
        self.save_model()
        self._steps_since_train = 0
        self.is_trained = True
        logger.info("Model trained on %d samples", len(y))

    def predict(self, df: pd.DataFrame) -> Dict[str, str]:
        result = {sid: "NEUTRAL" for sid in df["item_id"].unique()}
        if not self.is_trained:
            logger.warning("Model not trained")
            return result
        seqs = {}
        for item_id, group in df.groupby("item_id"):
            if len(group) < self.seq_len:
                continue
            seq = group[["open", "high", "low", "close", "volume"]].tail(self.seq_len).to_numpy(float).T
            seqs[item_id] = seq
        if not seqs:
            return result
        X = np.stack(list(seqs.values()))
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.moment.to(device)
        self.moment.eval()
        tensor_x = torch.tensor(X, dtype=torch.float32).to(device)
        mask = torch.ones(tensor_x.shape[0], self.seq_len, dtype=torch.long).to(device)
        with torch.no_grad():
            logits = self.moment(x_enc=tensor_x, input_mask=mask).logits
            preds = logits.argmax(dim=1).cpu().numpy()
        for item_id, pred in zip(seqs.keys(), preds):
            result[item_id] = "AGREE_LONG" if pred == 1 else "AGREE_SHORT"
        self._steps_since_train += 1
        if self.retrain_interval and self._steps_since_train >= self.retrain_interval:
            self.fit(df)
        return result
