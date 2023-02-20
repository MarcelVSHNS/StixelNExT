import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models.stixel_convnext import StixelNExT
from losses.stixel_loss import StixelLoss
from dataloader.waymo_multicut import MultiCutStixelData, transforming, target_transforming
from utilities.visualization import show_data


# 0.1 Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
# 0.2 Initialize the data logger for tensorboard
writer = SummaryWriter('runs')


# Training Function
def train(dataloader, model, loss_fn, optimizer, epoch):
    size = len(dataloader.dataset)
    model.train()
    # for every batch_sized chunk of data ...
    for batch, (X, y) in enumerate(dataloader):
        # copy data to the computing device (normally the GPU)
        X, y = X.to(device), y.to(device)

        # Compute prediction a prediction
        pred = model(X)
        # Compute the error (loss) of that prediction [loss_fn(prediction, target)]
        loss = loss_fn(pred, y)

        # Backpropagation strategy/ optimization "zero_grad()"
        optimizer.zero_grad()
        # Apply the prediction loss (backpropagation)
        loss.backward()
        # write the weights to the NN
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            # Log the loss by adding scalars
            writer.add_scalars("Training Loss", {'Training': loss},
                               epoch * len(dataloader) + batch)


# Validation Function
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def main():
    # Settings
    batch_size = 2
    num_epochs = 5
    save_model = False
    load_weights = False
    explore_data = False
    training = False
    # Paths
    training_data_path = "training_data.csv"
    validation_data_path = "validation_data.csv"

    # Load data
    test_data = MultiCutStixelData(validation_data_path, transforming, target_transforming)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)
    # training_data = MultiCutStixelData(training_data_path, transforming, target_transforming)
    # training_dataloader = DataLoader(training_data, batch_size=batch_size)

    # Explore data
    if explore_data:
        test_features, test_labels = next(iter(test_dataloader))
        show_data(test_features, test_labels, idx=1)

    # Define Model
    model = StixelNExT().to(device)
    print(model)
    # Load Weights
    if load_weights:
        model.load_state_dict(torch.load("saved_models/StixelNExT.pth"))
    # Loss function
    loss_fn = StixelLoss()
    # Optimizer definition
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    # Training
    if training:
        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}\n-------------------------------")
            train(test_dataloader, model, loss_fn, optimizer, epoch)
            test(test_dataloader, model, loss_fn)

    # Save model
    if save_model:
        torch.save(model.state_dict(), "saved_models/StixelNExT.pth")
        print("Saved PyTorch Model State to saved_models/StixelNExT.pth")


if __name__ == '__main__':
    main()
