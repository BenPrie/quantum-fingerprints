# Imports, as always...
from os import makedirs, path
from tqdm.notebook import tqdm
import numpy as np
import torch

from torch.optim import Adam
from torch.nn.functional import one_hot


# Function to perform a single training step (i.e. one epoch).
def training_step(model, device, loader, optimiser, loss_fn):
    # Use train mode.
    model.train()

    # Track the running loss and accuracy.
    running_loss = 0
    running_acc = 0

    # Iterate over the loader (if verbose is true, give a progress bar).
    for xs, ys_true in loader:
        # Handling a singleton batch (PyTorch is not a fan).
        if xs.shape[0] == 1:
            continue

        # Move to device.
        xs = xs.to(device)
        ys_true = ys_true.to(device)

        # Zero gradients.
        optimiser.zero_grad()

        # Predict labels.
        ys_pred = model(xs)

        # Compute loss (adding to the running loss) and calculate gradients.
        loss = loss_fn(ys_pred, one_hot(ys_true.to(int), num_classes=ys_pred.shape[-1]).to(float))
        running_loss += loss.item()
        loss.backward()

        # Hardmax to get predicted class and compute accuracy.
        ys_pred_hardmax = torch.argmax(ys_pred, dim=1)
        running_acc += torch.sum(ys_pred_hardmax == ys_true).item()

        # Adjust weights.
        optimiser.step()

    # Return the average loss and accuracy over the epoch.
    return running_loss / len(loader), running_acc / len(loader.dataset)


# Function to evaluate with the given loader (i.e. validation/testing).
def evaluate(model, device, loader, loss_fn):
    # Use evaluation mode.
    model.eval()

    # Track the running loss and accuracy.
    running_loss = 0
    running_acc = 0

    # For evaluation, we do not track gradients.
    with torch.no_grad():
        # Iterate over the loader (if verbose is true, give a progress bar).
        for xs, ys_true in loader:
            if xs.shape[0] == 1:
                continue

            # Move to device.
            xs = xs.to(device)
            ys_true = ys_true.to(device)

            # Predict labels.
            ys_pred = model(xs)

            # Compute loss (adding to the running loss).
            running_loss += loss_fn(ys_pred, one_hot(ys_true.to(int), num_classes=ys_pred.shape[-1]).to(float)).item()

            # Hardmax to get predicted class and compute accuracy.
            ys_pred_hardmax = torch.argmax(ys_pred, dim=1)
            running_acc += torch.sum(ys_pred_hardmax == ys_true).item()

    # Return the average loss and accuracy over the loader.
    return running_loss / len(loader), running_acc / len(loader.dataset)


# Function to perform a full training routine.
def train(model, device, loss_fn, train_loader, val_loader, n_epochs, lr=1e-3, weight_decay=.0, verbose=False, print_interval=10, save_dir=None,
          save_file_name='model-dict', seed=0):
    # RNG.
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create the save directory.
    if save_dir: makedirs(save_dir, exist_ok=True)

    # Set up an optimiser and loss function.
    optimiser = Adam(params=model.parameters(), lr=lr, weight_decay=weight_decay)

    # Track the best validation loss so that we may return to the best weights on termination.
    best_val_loss = np.inf
    best_state = model.state_dict()

    # Remember the training statistics, just in case someone feels like plotting it.
    train_losses, train_accs, val_losses, val_accs = [], [], [], []

    for epoch_idx in (
    tqdm(np.arange(1, n_epochs + 1), desc='Top-level training') if verbose else np.arange(1, n_epochs + 1)):
        # Train the model.
        train_loss, train_acc = training_step(model, device, train_loader, optimiser, loss_fn)
        val_loss, val_acc = evaluate(model, device, val_loader, loss_fn)

        # Store those stats.
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # Update the best validation loss and remember the model's state.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()

            # Save the model (if a save path has been given).
            if save_dir: torch.save(best_state, path.join(save_dir, f'{save_file_name}.pt'))

        # Print the stats (if verbose).
        if verbose and epoch_idx % print_interval == 0:
            print(
                f'Epoch {epoch_idx:03d}: train loss - {train_loss:.3f}, val loss - {val_loss:.3f}, train acc - {train_acc:.3f}, val acc - {val_acc:.3f}')

    # Restore the best model in validation.
    model.load_state_dict(best_state)

    # Return the stats.
    return train_losses, train_accs, val_losses, val_accs
