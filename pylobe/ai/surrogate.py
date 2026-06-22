"""Neural network surrogate model for antenna performance prediction.

Replaces expensive EM simulation during optimisation.
Architecture: MLP [input_dim → 256 → 256 → 128 → output_dim]
with ReLU + BatchNorm.
"""
import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


def _build_mlp(input_dim: int, output_dim: int) -> "nn.Sequential":
    """Build MLP with BatchNorm and Dropout for uncertainty estimation."""
    if not _TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for NeuralSurrogate: pip install torch")
    import torch.nn as nn
    return nn.Sequential(
        nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.1),
        nn.Linear(256, 256),       nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.1),
        nn.Linear(256, 128),       nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.1),
        nn.Linear(128, output_dim),
    )


class NeuralSurrogate:
    """MLP surrogate model for fast antenna performance prediction.

    Input features: geometry parameters + frequency (normalised).
    Output targets: [S11_dB, gain_dBi, HPBW_deg, SLL_dB] (or custom).

    Parameters
    ----------
    input_dim : int
        Number of input features (geometry params + freq).
    output_dim : int
        Number of predicted quantities.
    device : str
        PyTorch device ('cpu' or 'cuda').
    """

    def __init__(self, input_dim: int, output_dim: int,
                 device: str = 'cpu'):
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch is required: pip install torch")
        import torch
        self.input_dim  = input_dim
        self.output_dim = output_dim
        self.device     = torch.device(device)
        self.model      = _build_mlp(input_dim, output_dim).to(self.device)

        # Normalisation statistics (set during train())
        self._X_mean = np.zeros(input_dim)
        self._X_std  = np.ones(input_dim)
        self._y_mean = np.zeros(output_dim)
        self._y_std  = np.ones(output_dim)
        self._trained = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(self, X: np.ndarray, y: np.ndarray,
              n_epochs: int = 500, val_split: float = 0.1,
              lr: float = 1e-3, weight_decay: float = 1e-4,
              batch_size: int = 32, verbose: bool = True):
        """Train surrogate on dataset.

        Parameters
        ----------
        X : ndarray, shape (N, input_dim)
        y : ndarray, shape (N, output_dim)
        n_epochs : int
        val_split : float
        lr : float
        weight_decay : float
        batch_size : int
        verbose : bool

        Returns
        -------
        dict {'train_loss': list, 'val_loss': list}
        """
        import torch
        import torch.nn as nn

        # Normalise
        self._X_mean = X.mean(axis=0)
        self._X_std  = X.std(axis=0) + 1e-8
        self._y_mean = y.mean(axis=0)
        self._y_std  = y.std(axis=0) + 1e-8

        Xn = (X - self._X_mean) / self._X_std
        yn = (y - self._y_mean) / self._y_std

        # Train/val split
        n_val = max(1, int(len(X) * val_split))
        X_val, y_val = Xn[:n_val], yn[:n_val]
        X_tr,  y_tr  = Xn[n_val:], yn[n_val:]

        X_tr_t  = torch.tensor(X_tr,  dtype=torch.float32).to(self.device)
        y_tr_t  = torch.tensor(y_tr,  dtype=torch.float32).to(self.device)
        X_val_t = torch.tensor(X_val, dtype=torch.float32).to(self.device)
        y_val_t = torch.tensor(y_val, dtype=torch.float32).to(self.device)

        ds    = TensorDataset(X_tr_t, y_tr_t)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

        optim    = torch.optim.Adam(self.model.parameters(), lr=lr,
                                    weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=n_epochs)
        criterion = nn.MSELoss()

        train_losses, val_losses = [], []
        for epoch in range(1, n_epochs + 1):
            self.model.train()
            epoch_loss = 0.0
            for Xb, yb in loader:
                optim.zero_grad()
                pred = self.model(Xb)
                loss = criterion(pred, yb)
                loss.backward()
                optim.step()
                epoch_loss += loss.item() * len(Xb)
            train_losses.append(epoch_loss / len(X_tr_t))

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_pred = self.model(X_val_t)
                val_loss = criterion(val_pred, y_val_t).item()
            val_losses.append(val_loss)
            scheduler.step()

            if verbose and epoch % 50 == 0:
                print(f"Epoch {epoch:4d}/{n_epochs} — "
                      f"train: {train_losses[-1]:.5f}, val: {val_loss:.5f}")

        self._trained = True
        return {'train_loss': train_losses, 'val_loss': val_losses}

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict antenna performance (denormalised).

        Parameters
        ----------
        X : ndarray, shape (N, input_dim)

        Returns
        -------
        ndarray, shape (N, output_dim)
        """
        import torch
        if not self._trained:
            raise RuntimeError("Call train() before predict().")
        Xn = (X - self._X_mean) / self._X_std
        Xt = torch.tensor(Xn, dtype=torch.float32).to(self.device)
        self.model.eval()
        with torch.no_grad():
            yn = self.model(Xt).cpu().numpy()
        return yn * self._y_std + self._y_mean

    def uncertainty(self, X: np.ndarray,
                    n_samples: int = 50) -> tuple:
        """MC Dropout uncertainty estimation.

        Parameters
        ----------
        X : ndarray, shape (N, input_dim)
        n_samples : int
            Number of stochastic forward passes.

        Returns
        -------
        tuple (mean, std) each ndarray shape (N, output_dim)
        """
        import torch
        if not self._trained:
            raise RuntimeError("Call train() before uncertainty().")
        Xn = (X - self._X_mean) / self._X_std
        Xt = torch.tensor(Xn, dtype=torch.float32).to(self.device)

        # Enable dropout during inference
        self.model.train()
        preds = []
        with torch.no_grad():
            for _ in range(n_samples):
                yn = self.model(Xt).cpu().numpy()
                preds.append(yn * self._y_std + self._y_mean)
        self.model.eval()

        preds = np.stack(preds, axis=0)   # (n_samples, N, output_dim)
        return preds.mean(axis=0), preds.std(axis=0)

    def save(self, path: str):
        """Save model weights and normalisation stats.

        Parameters
        ----------
        path : str
            .pt file path.
        """
        import torch
        torch.save({
            'model_state': self.model.state_dict(),
            'X_mean': self._X_mean, 'X_std': self._X_std,
            'y_mean': self._y_mean, 'y_std': self._y_std,
            'input_dim': self.input_dim, 'output_dim': self.output_dim,
        }, path)

    def load(self, path: str):
        """Load model weights from file.

        Parameters
        ----------
        path : str
        """
        import torch
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state'])
        self._X_mean = ckpt['X_mean']
        self._X_std  = ckpt['X_std']
        self._y_mean = ckpt['y_mean']
        self._y_std  = ckpt['y_std']
        self._trained = True
