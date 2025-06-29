import os
import logging
import time
from typing import Dict, Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from momentfm import MOMENTPipeline
from torch.amp import autocast, GradScaler
from safetensors.torch import load_file

logger = logging.getLogger(__name__)

class MomentClassifier:
    def __init__(self, config: dict):
        self.seq_len = config.get("seq_len", 512)
        self.pred_len = config.get("prediction_length", 1)
        # --- ИЗМЕНЕНИЕ 1: Добавлен новый параметр для количества валидационных окон ---
        self.num_val_windows = config.get("num_val_windows", 1) 
        self.model_name = config.get("model_name", "AutonLab/MOMENT-1-large")
        self.results_output_dir = config.get("results_output_dir", "moment_model")
        self.epochs = config.get("epochs", 100)
        self.batch_size = config.get("batch_size", 32)
        self.num_batches_per_epoch = config.get("num_batches_per_epoch", 50)
        self.early_stopping = config.get("early_stopping", 20)
        self.lr = config.get("lr", 1e-4)
        self.compile_model = config.get("compile_model", True)
        self.save_on_improve = config.get("save_on_improve", False)
        self.all_time_retrain = config.get("all_time_retrain", False)
        
        self.moment: torch.nn.Module | None = None
        self.is_trained = False
        self.trained_epochs: int = 0
        
        # --- ИЗМЕНЕНИЕ 2: Обновлено логгирование для отображения нового параметра ---
        logger.info(f"MomentClassifier initialized. Seq Len: {self.seq_len}, Pred Len: {self.pred_len}, Num Val Windows: {self.num_val_windows}, Full Retrain: {self.all_time_retrain}")

    def load_model(self):
        """Load MOMENT model weights and prepare the pipeline."""
        logger.info("Starting model loading process...")
        start_time = time.time()
        os.makedirs(self.results_output_dir, exist_ok=True)
        
        model_kwargs = {"task_name": "classification", "n_channels": 5, "num_class": 2}
        
        try:
            logger.info(f"Step 1: Creating model skeleton from '{self.model_name}' with custom classification head.")
            pipeline = MOMENTPipeline.from_pretrained(self.model_name, model_kwargs=model_kwargs)
            pipeline.init()
            self.moment = pipeline

            local_weights_path = os.path.join(self.results_output_dir, "model.safetensors")
            
            if self.all_time_retrain:
                logger.info("Step 2: Full retrain mode is ON. Skipping loading of local weights.")
                self.is_trained = False
            elif os.path.exists(local_weights_path):
                logger.info(f"Step 2: Found local weights at '{local_weights_path}'. Loading them into the model.")
                state_dict = load_file(local_weights_path, device="cpu")
                self.moment.load_state_dict(state_dict)
                self.is_trained = True
                logger.info("Successfully loaded local weights.")
            else:
                logger.warning(f"Step 2: No local weights found at '{local_weights_path}'. Using base model weights.")
                self.is_trained = False

            if self.compile_model and torch.cuda.is_available():
                logger.info("Step 3: Compiling the model for faster performance...")
                self.moment = torch.compile(self.moment, mode="reduce-overhead")

        except Exception as e:
            logger.error(f"FATAL: Failed during model loading or initialization.", exc_info=True)
            raise RuntimeError(f"Model could not be initialized or loaded.") from e

        logger.info(f"Model ready in {time.time() - start_time:.2f}s. Is considered trained: {self.is_trained}")

    def save_model(self):
        """Persist the current model to ``results_output_dir``."""
        if self.moment is not None:
            logger.info(f"Saving model to {self.results_output_dir}...")
            start_time = time.time()
            
            model_to_save = self.moment._orig_mod if hasattr(self.moment, '_orig_mod') else self.moment
            
            if isinstance(model_to_save, MOMENTPipeline):
                 model_to_save.save_pretrained(self.results_output_dir)
            else:
                 logger.error(f"Cannot save model. Expected MOMENTPipeline, but got {type(model_to_save)}.")
            
            logger.info(f"Model saved in {time.time() - start_time:.2f} seconds.")

    def _build_sequences(self, df: pd.DataFrame, purpose: str = "training") -> tuple[np.ndarray, np.ndarray, list]:
        """Convert a DataFrame into model input sequences."""
        if df.empty:
            logger.warning(f"Input DataFrame for _build_sequences (purpose: {purpose}) is empty! Cannot generate sequences.")
            return np.empty((0, 5, self.seq_len), np.float32), np.empty((0,), np.int64), []

        logger.info(f"Building sequences for '{purpose}'. Input df shape: {df.shape}, Unique items: {df['item_id'].nunique()}.")
        start_time = time.time()
        
        df = df.sort_values(["item_id", "timestamp"])
        X_list, y_list, item_ids = [], [], []
        
        for item_id, group in df.groupby("item_id"):
            group_len = len(group)
            feats = group[["open", "high", "low", "close", "volume"]].to_numpy(dtype=np.float32).T
            closes = group["close"].to_numpy(dtype=np.float32)

            if purpose == "training":
                required_len = self.seq_len + self.pred_len
                if group_len < required_len:
                    logger.warning(f"Skipping '{item_id}' for training: not enough data. Have {group_len}, need {required_len}.")
                    continue
                for i in range(len(closes) - required_len + 1):
                    X_list.append(feats[:, i : i + self.seq_len])
                    y_list.append(1 if closes[i + required_len - 1] > closes[i + self.seq_len - 1] else 0)
            
            elif purpose == "prediction":
                required_len = self.seq_len
                if group_len < required_len:
                    logger.warning(f"Skipping '{item_id}' for prediction: not enough data. Have {group_len}, need {required_len}.")
                    continue

                seq = feats[:, -required_len:]
                X_list.append(seq)
                item_ids.append(item_id)
            
            # --- ИЗМЕНЕНИЕ 3: Полностью переработан блок для валидации и оценки ---
            # Теперь он обрабатывает и 'validation', и 'evaluation' одинаково
            elif purpose in ["validation", "evaluation"]:
                # Требуемая длина теперь зависит от количества окон, которые мы хотим извлечь.
                # seq_len + pred_len для одного окна, и +1 за каждое дополнительное окно.
                required_len = self.seq_len + self.pred_len + self.num_val_windows - 1
                if group_len < required_len:
                    logger.warning(f"Skipping '{item_id}' for {purpose}: not enough data. Have {group_len}, need {required_len} for {self.num_val_windows} windows.")
                    continue
                
                # Создаем N валидационных окон, идя с конца временного ряда
                for i in range(self.num_val_windows):
                    # i=0 - самое последнее окно, i=1 - предпоследнее и т.д.
                    
                    # Определяем индексы для среза признаков (X)
                    end_idx_x = -self.pred_len - i
                    start_idx_x = end_idx_x - self.seq_len
                    
                    # Python срезы: [start:end]. Если end_idx_x=0, это значит до самого конца.
                    # Но срез [X:0] пустой, поэтому нужно [X:].
                    seq = feats[:, start_idx_x : (end_idx_x if end_idx_x != 0 else None)]
                    
                    # Определяем индексы для расчета цели (y)
                    target_close_idx = -1 - i
                    prev_close_idx = -self.pred_len - 1 - i
                    target = 1 if closes[target_close_idx] > closes[prev_close_idx] else 0
                    
                    X_list.append(seq)
                    y_list.append(target)
                    item_ids.append(item_id)

        if not X_list:
            logger.warning(f"No sequences were generated for purpose '{purpose}' from the provided data.")
            return np.empty((0, 5, self.seq_len), np.float32), np.empty((0,), np.int64), []
        
        X = np.stack(X_list)
        y = np.array(y_list, dtype=np.int64) if y_list else np.empty((0,), dtype=np.int64)
        
        logger.info(f"Successfully generated {len(X)} sequences for '{purpose}' in {time.time() - start_time:.2f}s. Final shape of X: {X.shape}")
        return X, y, item_ids

    def fit(self, df: pd.DataFrame, val_df: pd.DataFrame | None = None, num_workers: int = 0):
        """Train the classifier on provided data."""
        logger.info("Starting model fitting process...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        X, y, _ = self._build_sequences(df, "training")
        if X.size == 0:
            logger.error("No training data could be generated. Aborting fit.")
            return

        logger.info(f"Training data prepared: {X.shape[0]} samples.")

        X_val, y_val = None, None
        if val_df is not None:
            # Используем purpose="validation"
            X_val, y_val, _ = self._build_sequences(val_df, "validation")
            if X_val.size > 0:
                logger.info(f"Validation data prepared: {X_val.shape[0]} samples.")
            else:
                logger.warning("Validation data was provided, but no validation sequences could be generated.")
        
        dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        
        val_loader = None
        if X_val is not None and y_val is not None and X_val.size > 0:
            val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        
        self.moment.to(device)
        optimizer = torch.optim.Adam(self.moment.parameters(), lr=self.lr)
        criterion = torch.nn.CrossEntropyLoss()
        scaler = GradScaler(enabled=(device == 'cuda'))
        
        best_val_acc = 0.0
        patience = 0
        best_model_state_dict: Dict[str, Any] | None = None
        
        data_iterator = iter(dataloader)

        for epoch in range(self.epochs):
            self.moment.train()
            epoch_start_time = time.time()
            total_correct, total_samples = 0, 0
            logger.info(f"--- Starting Epoch {epoch + 1}/{self.epochs} (Patience: {patience}/{self.early_stopping}) ---")
            
            for _ in range(self.num_batches_per_epoch):
                try: batch_x, batch_y = next(data_iterator)
                except StopIteration:
                    data_iterator = iter(dataloader)
                    batch_x, batch_y = next(data_iterator)

                batch_x, batch_y = batch_x.to(device, non_blocking=True), batch_y.to(device, non_blocking=True)
                mask = torch.ones(batch_x.shape[0], self.seq_len, dtype=torch.long, device=device)
                optimizer.zero_grad(set_to_none=True)
                
                with autocast(device_type=device, dtype=torch.float16, enabled=(device == 'cuda')):
                    out = self.moment(x_enc=batch_x, input_mask=mask)
                    loss = criterion(out.logits, batch_y)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                total_correct += (out.logits.argmax(dim=1) == batch_y).sum().item()
                total_samples += batch_y.size(0)
            
            logger.info(f"Epoch {epoch+1} took {time.time()-epoch_start_time:.2f}s. Train Acc: {total_correct/max(1,total_samples):.4f}")

            if val_loader:
                val_acc = self.evaluate_on_loader(val_loader, device, "Validation")
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience = 0
                    logger.info(f"New best validation accuracy: {best_val_acc:.4f}.")
                    
                    current_model_state = (self.moment._orig_mod if hasattr(self.moment, '_orig_mod') else self.moment).state_dict()
                    best_model_state_dict = {k: v.cpu() for k, v in current_model_state.items()}
                    
                    if self.save_on_improve:
                        self.save_model()
                else:
                    patience += 1
                    if patience >= self.early_stopping:
                        logger.info(f"Early stopping triggered after {epoch + 1} epochs.")
                        break
            self.trained_epochs = epoch + 1
        
        logger.info("Training finished.")
        if not self.save_on_improve and best_model_state_dict is not None:
            logger.info(f"Loading best model state (acc: {best_val_acc:.4f}) and saving to disk...")
            final_model = self.moment._orig_mod if hasattr(self.moment, '_orig_mod') else self.moment
            final_model.load_state_dict(best_model_state_dict)
            self.save_model()
            self.is_trained = True

    def evaluate_on_loader(self, loader: DataLoader, device: str, purpose: str = "Evaluation") -> float:
        """Evaluate the model on a dataloader and return accuracy."""
        self.moment.eval()
        total_correct, total_samples = 0, 0
        start_time = time.time()
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                mask = torch.ones(x.shape[0], self.seq_len, dtype=torch.long, device=device)
                with autocast(device_type=device, dtype=torch.float16, enabled=(device == 'cuda')):
                    logits = self.moment(x_enc=x, input_mask=mask).logits
                total_correct += (logits.argmax(dim=1) == y).sum().item()
                total_samples += y.size(0)
        
        val_acc = total_correct / max(1, total_samples)
        logger.info(f"{purpose} on {total_samples} samples took {time.time()-start_time:.2f}s. Accuracy: {val_acc:.4f}")
        return val_acc

    def predict(self, df: pd.DataFrame) -> Dict[str, str]:
        """Predict signals for a dataframe of market data."""
        logger.info(f"Starting prediction for {df['item_id'].nunique()} items...")
        result = {sid: "NEUTRAL" for sid in df["item_id"].unique()}

        if self.moment is None or not self.is_trained:
            return result
        
        X, _, item_ids = self._build_sequences(df, purpose="prediction")
        if X.size == 0: return result
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.moment.to(device)
        self.moment.eval()
        
        tensor_x = torch.from_numpy(X)
        predict_dataset = TensorDataset(tensor_x)
        predict_loader = DataLoader(predict_dataset, batch_size=self.batch_size * 2)

        all_preds = []
        with torch.no_grad():
            for (batch_x,) in predict_loader:
                batch_x = batch_x.to(device)
                mask = torch.ones(batch_x.shape[0], self.seq_len, dtype=torch.long, device=device)
                with autocast(device_type=device, dtype=torch.float16, enabled=(device=='cuda')):
                    logits = self.moment(x_enc=batch_x, input_mask=mask).logits
                    preds = logits.argmax(dim=1).cpu().numpy()
                    all_preds.extend(preds)
            
        for item_id, pred in zip(item_ids, all_preds):
            result[item_id] = "AGREE_LONG" if pred == 1 else "AGREE_SHORT"
        
        logger.info("Prediction finished.")
        return result