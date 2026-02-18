import torch
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
import torch

from toxicclf.training.metrics import compute_f1_score as compute_f1
from toxicclf.utils.logger import get_logger
logger = get_logger(__name__)


class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        config,
    ):
        self.best_val_macro = 0.0
        self.save_checkpoint_dir = Path(config.save_checkpoint_dir)
        self.save_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        self.device = torch.device(config.device)
        self.model.to(self.device)

        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
        )
    
    def save_checkpoint(self, epoch, val_macro_f1):
        checkpoint_path = self.save_checkpoint_dir / f"{self.config.saved_model_name}"
        logger.info(f"Saving checkpoint: {checkpoint_path}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_macro_f1': val_macro_f1
        }, checkpoint_path)
        
        logger.info(f"Saved checkpoint: {checkpoint_path}")

    def train_one_epoch(self):
        self.model.train()

        total_loss = 0.0
        all_preds = []
        all_labels = []

        for inputs, labels in tqdm(self.train_loader):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            logits = self.model(inputs)
            loss = self.criterion(logits, labels)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            all_preds.append(logits.detach())
            all_labels.append(labels.detach())

        avg_loss = total_loss / len(self.train_loader)

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        micro_f1, macro_f1 = compute_f1(all_labels, all_preds)

        return avg_loss, micro_f1, macro_f1

    def validate(self):
        self.model.eval()

        total_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in tqdm(self.val_loader):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                logits = self.model(inputs)
                loss = self.criterion(logits, labels)

                total_loss += loss.item()

                all_preds.append(logits)
                all_labels.append(labels)

        avg_loss = total_loss / len(self.val_loader)

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        micro_f1, macro_f1 = compute_f1(all_labels, all_preds)

        return avg_loss, micro_f1, macro_f1

    def fit(self):
        for epoch in range(self.config.epochs):
            logger.info(f"\nEpoch {epoch+1}/{self.config.epochs}")

            train_loss, train_micro, train_macro = self.train_one_epoch()
            val_loss, val_micro, val_macro = self.validate()

            logger.info(
                f"Train Loss: {train_loss:.4f} | "
                f"Micro F1: {train_micro:.4f} | "
                f"Macro F1: {train_macro:.4f}"
            )

            logger.info(
                f"Val Loss: {val_loss:.4f} | "
                f"Micro F1: {val_micro:.4f} | "
                f"Macro F1: {val_macro:.4f}"
            )
        
        if val_macro > self.best_val_macro:
            self.best_val_macro = val_macro
            self.save_checkpoint(epoch, val_macro)
