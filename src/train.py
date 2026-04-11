"""
train.py — Training loop for PyTorch models; sklearn fit wrappers.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from config import EPOCHS, LR, PATIENCE, RANDOM_SEED, HORIZON
from models import TransformerLSTM, StandaloneLSTM, flatten_sequences

torch.manual_seed(RANDOM_SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
# Generic PyTorch trainer
# ─────────────────────────────────────────────────────────────────────────────
def train_torch_model(model, tr_loader, val_loader, epochs=EPOCHS,
                      lr=LR, patience=PATIENCE):
    model = model.to(DEVICE)
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val  = float("inf")
    wait      = 0
    tr_losses, val_losses = [], []
    best_state = None

    for epoch in range(1, epochs + 1):
        # ── train ──
        model.train()
        tr_loss = 0.0
        for Xb, yb in tr_loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            pred = model(Xb).squeeze(-1)
            loss = criterion(pred, yb.squeeze(-1))
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * len(Xb)
        tr_loss /= len(tr_loader.dataset)

        # ── validate ──
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                pred = model(Xb).squeeze(-1)
                val_loss += criterion(pred, yb.squeeze(-1)).item() * len(Xb)
        val_loss /= len(val_loader.dataset)

        tr_losses.append(tr_loss)
        val_losses.append(val_loss)

        if (epoch % 10 == 0) or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs} | tr_loss={tr_loss:.4f} | val_loss={val_loss:.4f}")

        # ── early stopping ──
        if val_loss < best_val:
            best_val   = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    return model, tr_losses, val_losses


# ─────────────────────────────────────────────────────────────────────────────
# Predict helper (returns numpy in original kWh scale)
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def predict_torch(model, loader):
    model.eval().to(DEVICE)
    preds = []
    for Xb, _ in loader:
        Xb = Xb.to(DEVICE)
        out = model(Xb).squeeze(-1).cpu().numpy()
        preds.append(out)
    return np.concatenate(preds, axis=0).ravel()


# ─────────────────────────────────────────────────────────────────────────────
# sklearn wrappers
# ─────────────────────────────────────────────────────────────────────────────
def train_sklearn(model, X_tr, y_tr):
    X_flat = flatten_sequences(X_tr)
    y_flat = y_tr.ravel()
    model.fit(X_flat, y_flat)
    return model


def predict_sklearn(model, X):
    return model.predict(flatten_sequences(X))


# ─────────────────────────────────────────────────────────────────────────────
# Full training pipeline: returns all models + history
# ─────────────────────────────────────────────────────────────────────────────
def train_all(data: dict) -> dict:
    tr_dl, val_dl, te_dl = data["loaders"]

    results = {}

    # ── Transformer-LSTM ──────────────────────────────────────────────────────
    print("\n[train] Transformer-LSTM")
    tl_model = TransformerLSTM()
    tl_model, tl_tr_hist, tl_val_hist = train_torch_model(tl_model, tr_dl, val_dl)
    tl_pred = predict_torch(tl_model, te_dl)
    tl_pred_val = predict_torch(tl_model, val_dl)
    results["TransformerLSTM"] = dict(
        model=tl_model,
        tr_hist=tl_tr_hist, val_hist=tl_val_hist,
        y_pred=tl_pred,
        y_pred_val=tl_pred_val,
    )

    # ── Standalone LSTM ───────────────────────────────────────────────────────
    print("\n[train] Standalone LSTM")
    lstm_model = StandaloneLSTM()
    lstm_model, lstm_tr_hist, lstm_val_hist = train_torch_model(lstm_model, tr_dl, val_dl)
    lstm_pred = predict_torch(lstm_model, te_dl)
    lstm_pred_val = predict_torch(lstm_model, val_dl)
    results["LSTM"] = dict(
        model=lstm_model,
        tr_hist=lstm_tr_hist, val_hist=lstm_val_hist,
        y_pred=lstm_pred,
        y_pred_val=lstm_pred_val,
    )

    # ── Random Forest ─────────────────────────────────────────────────────────
    print("\n[train] Random Forest")
    from models import get_random_forest
    rf = train_sklearn(get_random_forest(), data["X_tr"], data["y_tr"])
    rf_pred = predict_sklearn(rf, data["X_te"])
    results["RandomForest"] = dict(model=rf, y_pred=rf_pred)

    # ── Linear Regression ─────────────────────────────────────────────────────
    print("\n[train] Linear Regression")
    from models import get_linear_regression
    lr_mdl = train_sklearn(get_linear_regression(), data["X_tr"], data["y_tr"])
    lr_pred = predict_sklearn(lr_mdl, data["X_te"])
    results["LinearRegression"] = dict(model=lr_mdl, y_pred=lr_pred)

    return results
