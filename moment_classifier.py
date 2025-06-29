import os
import logging
from typing import Dict, List
import numpy as np
import pandas as pd
import torch
from momentfm import MOMENTPipeline
from momentfm.models.statistical_classifiers import fit_svm
import joblib

logger = logging.getLogger(__name__)

class MomentClassifier:
    def __init__(self, config: dict):
        self.seq_len = config.get("seq_len", 64)
        self.model_name = config.get("model_name", "AutonLab/MOMENT-1-large")
        self.results_output_dir = config.get("results_output_dir", "moment_model")
        self.retrain_interval = int(config.get("all_time_retrain", 0))
        self._steps_since_train = 0
        self.moment: MOMENTPipeline | None = None
        self.classifier = None

    def load_model(self):
        os.makedirs(self.results_output_dir, exist_ok=True)
        self.moment = MOMENTPipeline.from_pretrained(
            self.model_name,
            model_kwargs={"task_name": "embedding", "n_channels": 5},
        )
        self.moment.init()
        clf_path = os.path.join(self.results_output_dir, "svm.joblib")
        if os.path.isfile(clf_path):
            self.classifier = joblib.load(clf_path)
            logger.info("Classifier loaded from %s", clf_path)

    def save_classifier(self):
        if self.classifier is not None:
            clf_path = os.path.join(self.results_output_dir, "svm.joblib")
            joblib.dump(self.classifier, clf_path)

    def _build_sequences(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        df = df.sort_values(["item_id", "timestamp"])  # ensure ordering
        X_list: List[np.ndarray] = []
        y_list: List[int] = []
        for item_id, group in df.groupby("item_id"):
            group = group.reset_index(drop=True)
            closes = group["close"].to_numpy(float)
            vals = group[["open", "high", "low", "close", "volume"]].to_numpy(float)
            for i in range(len(group) - self.seq_len - 1):
                seq = vals[i : i + self.seq_len].T  # shape (5, seq_len)
                target = 1 if closes[i + self.seq_len] > closes[i + self.seq_len - 1] else 0
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
            out = self.moment(x_enc=tensor_x, input_mask=mask)
            emb = out.embeddings.detach().cpu().numpy()
        return emb

    def fit(self, df: pd.DataFrame):
        X, y = self._build_sequences(df)
        if X.size == 0:
            logger.warning("No training data available")
            return
        features = self._embed(X)
        self.classifier = fit_svm(features=features, y=y)
        self.save_classifier()
        self._steps_since_train = 0
        logger.info("Classifier trained on %d samples", len(y))

    def predict(self, df: pd.DataFrame) -> Dict[str, str]:
        result = {sid: "NEUTRAL" for sid in df["item_id"].unique()}
        if self.classifier is None:
            logger.warning("Classifier not trained")
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
        features = self._embed(X)
        preds = self.classifier.predict(features)
        for item_id, pred in zip(seqs.keys(), preds):
            result[item_id] = "AGREE_LONG" if pred == 1 else "AGREE_SHORT"
        self._steps_since_train += 1
        if self.retrain_interval and self._steps_since_train >= self.retrain_interval:
            self.fit(df)
        return result
