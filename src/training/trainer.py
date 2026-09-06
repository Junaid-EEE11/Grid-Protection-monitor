"""Multi-task PyTorch training engine for Graph Neural Networks and neural baselines."""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader as PyGDataLoader

from src.models.heads import MultiTaskOutput
from src.utils.logging_utils import get_logger

logger = get_logger("training_engine")


@dataclass
class TrainConfig:
    """Hyperparameter and optimization settings."""
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 50
    patience: int = 10
    lambda_det: float = 1.0
    lambda_type: float = 1.0
    lambda_loc: float = 1.5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class TrainHistory:
    """Epoch-by-epoch training and validation loss history."""
    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    val_det_accs: List[float] = field(default_factory=list)
    val_type_f1s: List[float] = field(default_factory=list)
    val_loc_accs: List[float] = field(default_factory=list)
    best_epoch: int = 0
    best_val_loss: float = float("inf")


class MultiTaskTrainer:
    """Trains multi-task neural architectures with composite loss balancing and early stopping."""

    def __init__(
        self,
        model: nn.Module,
        config: Optional[TrainConfig] = None,
        save_dir: Optional[Union[str, Path]] = None,
    ):
        self.config = config or TrainConfig()
        self.device = torch.device(self.config.device)
        self.model = model.to(self.device)
        self.save_dir = Path(save_dir) if save_dir else None
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5
        )
        self.det_criterion = nn.CrossEntropyLoss()
        self.type_criterion = nn.CrossEntropyLoss()
        self.loc_criterion = nn.CrossEntropyLoss()

    def compute_loss(
        self, outputs: MultiTaskOutput, y_det: torch.Tensor, y_type: torch.Tensor, y_loc: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute composite weighted multi-task loss."""
        loss_det = self.det_criterion(outputs.det_logits, y_det)
        loss_type = self.type_criterion(outputs.type_logits, y_type)

        # Localization loss only evaluated on faulted scenarios (y_det == 1)
        fault_mask = y_det == 1
        if fault_mask.any():
            loc_logits_faulted = outputs.loc_logits[fault_mask]
            y_loc_faulted = y_loc[fault_mask]
            loss_loc = self.loc_criterion(loc_logits_faulted, y_loc_faulted)
        else:
            loss_loc = torch.tensor(0.0, device=self.device)

        total_loss = (
            self.config.lambda_det * loss_det
            + self.config.lambda_type * loss_type
            + self.config.lambda_loc * loss_loc
        )
        return total_loss, loss_det, loss_type, loss_loc

    def train_epoch(self, dataloader: PyGDataLoader) -> float:
        """Run one training epoch across all batches."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()

            outputs = self.model(batch)
            loss, _, _, _ = self.compute_loss(outputs, batch.y_det, batch.y_type, batch.y_loc)

            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(1, num_batches)

    @torch.no_grad()
    def evaluate(self, dataloader: PyGDataLoader) -> Dict[str, float]:
        """Evaluate model on validation or test dataset."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        det_preds, det_trues = [], []
        type_preds, type_trues = [], []
        loc_preds, loc_trues = [], []

        for batch in dataloader:
            batch = batch.to(self.device)
            outputs = self.model(batch)
            loss, _, _, _ = self.compute_loss(outputs, batch.y_det, batch.y_type, batch.y_loc)

            total_loss += loss.item()
            num_batches += 1

            p_det = outputs.det_logits.argmax(dim=-1).cpu().numpy()
            p_type = outputs.type_logits.argmax(dim=-1).cpu().numpy()
            p_loc = outputs.loc_logits.argmax(dim=-1).cpu().numpy()

            det_preds.extend(p_det)
            det_trues.extend(batch.y_det.cpu().numpy())
            type_preds.extend(p_type)
            type_trues.extend(batch.y_type.cpu().numpy())
            loc_preds.extend(p_loc)
            loc_trues.extend(batch.y_loc.cpu().numpy())

        det_acc = float(np.mean(np.array(det_preds) == np.array(det_trues)))
        type_acc = float(np.mean(np.array(type_preds) == np.array(type_trues)))

        # Loc accuracy on faulted cases
        f_mask = np.array(det_trues) == 1
        if np.any(f_mask):
            loc_acc = float(np.mean(np.array(loc_preds)[f_mask] == np.array(loc_trues)[f_mask]))
        else:
            loc_acc = 1.0

        return {
            "loss": total_loss / max(1, num_batches),
            "det_acc": det_acc,
            "type_acc": type_acc,
            "loc_acc": loc_acc,
        }

    def train(
        self, train_loader: PyGDataLoader, val_loader: PyGDataLoader
    ) -> TrainHistory:
        """Execute full training loop with early stopping and checkpointing."""
        history = TrainHistory()
        patience_counter = 0

        for epoch in range(1, self.config.epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)
            val_loss = val_metrics["loss"]

            self.scheduler.step(val_loss)

            history.train_losses.append(train_loss)
            history.val_losses.append(val_loss)
            history.val_det_accs.append(val_metrics["det_acc"])
            history.val_type_f1s.append(val_metrics["type_acc"])
            history.val_loc_accs.append(val_metrics["loc_acc"])

            if val_loss < history.best_val_loss:
                history.best_val_loss = val_loss
                history.best_epoch = epoch
                patience_counter = 0
                if self.save_dir:
                    torch.save(self.model.state_dict(), self.save_dir / "best_model.pt")
            else:
                patience_counter += 1

            if epoch % 5 == 0 or epoch == 1:
                logger.info(
                    f"Epoch {epoch:03d}/{self.config.epochs:03d} | "
                    f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                    f"Det Acc: {val_metrics['det_acc']:.3f} | Loc Acc: {val_metrics['loc_acc']:.3f}"
                )

            if patience_counter >= self.config.patience:
                logger.info(f"Early stopping triggered at epoch {epoch}.")
                break

        # Load best weights
        if self.save_dir and (self.save_dir / "best_model.pt").exists():
            self.model.load_state_dict(torch.load(self.save_dir / "best_model.pt", map_location=self.device))

        return history
