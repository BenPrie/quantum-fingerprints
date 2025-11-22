# Imports, as always...
import time
import copy

import torch
import torch.nn as nn

import matplotlib
import matplotlib.pyplot as plt
from IPython.display import clear_output

seed = 42
torch.random.manual_seed(seed)
test_split, val_split = .2, .2

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def binary_accuracy(logits, targets):
    # Case 1: BCE-style output
    if logits.ndim == 1 or logits.shape[1] == 1:
        logits = logits.view(-1)
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).long()
    # Case 2: CE-style output
    else:
        preds = torch.argmax(logits, dim=1)

    return (preds == targets).float().mean().item()

def evaluate(model, dataloader):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_samples = 0

    loss_fn_bce = nn.BCEWithLogitsLoss()
    loss_fn_ce = nn.CrossEntropyLoss()

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)

            outputs = model(x)

            # Auto select loss
            if outputs.ndim == 1 or outputs.shape[1] == 1:
                outputs = outputs.view(-1)
                loss = loss_fn_bce(outputs, y.float())
            else:
                loss = loss_fn_ce(outputs, y)

            batch_size = y.size(0)
            acc = binary_accuracy(outputs, y)

            total_loss += loss.item() * batch_size
            total_acc += acc * batch_size
            total_samples += batch_size

    return total_loss / total_samples, total_acc / total_samples


def train(
        model,
        train_loader,
        val_loader,
        optimizer,
        epochs,
        verbose=False,
        seed=42,
        early_stopping=True,
        patience=10,
        min_delta=1e-4,
        restore_best_weights=True
):
    torch.random.manual_seed(seed)

    model.to(device)

    bce_loss = nn.BCEWithLogitsLoss()
    ce_loss = nn.CrossEntropyLoss()

    # Metrics history
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    start_time = time.time()

    # ----- Early stopping state -----
    best_val_loss = float("inf")
    best_model_state = None
    epochs_without_improve = 0

    if verbose:
        # For live plotting.
        matplotlib.use("TkAgg")
        plt.ion()

        fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 5))
        fig.canvas.manager.set_window_title("Training Progress")
        plt.show(block=False)

    for epoch in range(epochs):
        # ----- TRAIN -----
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        n = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            outputs = model(x)

            # Auto-detect output type
            if outputs.ndim == 1 or outputs.shape[1] == 1:
                outputs = outputs.view(-1)
                loss = bce_loss(outputs, y.float())
            else:
                loss = ce_loss(outputs, y)

            loss.backward()
            optimizer.step()

            bs = y.size(0)

            running_loss += loss.item() * bs
            running_acc += binary_accuracy(outputs, y) * bs
            n += bs

        train_loss = running_loss / n
        train_acc = running_acc / n

        # ----- VALIDATION -----
        val_loss, val_acc = evaluate(model, val_loader)

        # Store metrics
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # ----- Early Stopping Logic -----
        if early_stopping:
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                best_model_state = copy.deepcopy(model.state_dict())
                epochs_without_improve = 0
            else:
                epochs_without_improve += 1

            if epochs_without_improve >= patience:
                if restore_best_weights and best_model_state is not None:
                    model.load_state_dict(best_model_state)
                break

        if not verbose:
            continue

        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            # ----- ETA -----
            elapsed = time.time() - start_time
            done = epoch + 1
            time_per_epoch = elapsed / done
            remaining = time_per_epoch * (epochs - done)
            mins, secs = divmod(int(remaining), 60)

            # ----- Clear notebook output (text + figure) -----
            clear_output(wait=True)

            # ---- Update figure safely ----
            ax_loss.cla()
            ax_acc.cla()

            ax_loss.plot(train_losses, label="Train Loss")
            ax_loss.plot(val_losses, label="Val Loss")
            ax_loss.set_title("Loss")
            ax_loss.set_xlabel("Epoch")
            ax_loss.legend()
            ax_loss.set_xlim(0, epochs)

            ax_acc.plot(train_accs, label="Train Acc")
            ax_acc.plot(val_accs, label="Val Acc")
            ax_acc.set_title("Accuracy")
            ax_acc.set_xlabel("Epoch")
            ax_acc.legend(loc='lower right')
            ax_acc.set_ylim(.45, 1.01)
            ax_acc.set_xlim(0, epochs)

            fig.suptitle(f"Epoch {done}/{epochs} | ETA {mins:02d}:{secs:02d}")

            fig.canvas.draw()
            fig.canvas.flush_events()

            print(
                f"Epoch {done}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
            )

    if verbose:
        plt.close(fig)
        plt.ioff()

        # Back to inline.
        matplotlib.use("module://matplotlib_inline.backend_inline")

        fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 5))

        ax_loss.plot(train_losses, label="Train Loss")
        ax_loss.plot(val_losses, label="Val Loss")
        ax_loss.set_title("Loss")
        ax_loss.set_xlabel("Epoch")
        ax_loss.legend()
        ax_loss.set_xlim(0, epochs)

        ax_acc.plot(train_accs, label="Train Acc")
        ax_acc.plot(val_accs, label="Val Acc")
        ax_acc.set_title("Accuracy")
        ax_acc.set_xlabel("Epoch")
        ax_acc.legend(loc='lower right')
        ax_acc.set_ylim(.45, 1.01)
        ax_acc.set_xlim(0, epochs)

        plt.show()

    # Restore best model at end if requested
    if early_stopping and restore_best_weights and best_model_state is not None:
        model.load_state_dict(best_model_state)

    return train_losses, train_accs, val_losses, val_accs
