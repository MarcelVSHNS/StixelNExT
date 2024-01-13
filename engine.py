import torch


# Training Function
def train_one_epoch(dataloader, model, loss_fn, optimizer, device, writer=None) -> float:
    num_batches = len(dataloader.dataset)
    train_loss = 0.0
    model.train()
    # for every batch_sized chunk of data ...
    for batch_idx, (samples, targets) in enumerate(dataloader):
        # copy data to the computing device (normally the GPU)
        samples = samples.to(device)
        targets = targets.to(device)
        # Compute prediction a prediction
        outputs = model(samples)
        # Compute the error (loss) of that prediction [loss_fn(prediction, target)]
        loss, loss_bce, loss_sum = loss_fn(outputs, targets)
        train_loss += loss.item()
        # Backpropagation strategy/ optimization "zero_grad()"
        optimizer.zero_grad()
        # Apply the prediction loss (backpropagation)
        loss.backward()
        # write the weights to the NN
        optimizer.step()
        if batch_idx % 10 == 0:
            loss, current = loss.item(), batch_idx * len(samples)
            print(f"loss: {loss:>7f}  [{current:>5d}/{num_batches:>5d}], loss_bce: {loss_bce}, loss_sum: {loss_sum}")
    train_loss /= num_batches
    if writer:
        # Log the average loss for the epoch
        writer.log({"Train loss": train_loss})
    return train_loss


# Validation Function
def evaluate(dataloader, model, loss_fn, device, writer=None):
    num_batches = len(dataloader)
    model.eval()
    eval_loss = 0
    with torch.no_grad():
        for (samples, targets) in dataloader:
            samples = samples.to(device)
            targets = targets.to(device)
            outputs = model(samples)
            loss, loss_bce, loss_sum = loss_fn(outputs, targets.squeeze(0))
            eval_loss += loss
    eval_loss /= num_batches
    print(f"Test Error: \n Avg loss: {eval_loss:>8f} \n")
    if writer:
        # Log the loss by adding scalars
        writer.log({"Eval loss": eval_loss})
    return eval_loss
