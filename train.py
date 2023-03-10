import torch
import wandb
import yaml
from torch.utils.data import DataLoader
from torchsummary import summary
import os

from models.ConvNeXt_implementation import ConvNeXt
from losses.stixel_loss import StixelLoss
from engine import train_one_epoch, evaluate
from dataloader.waymo_multicut import MultiCutStixelData, transforming, target_transforming
from utilities.visualization import show_data


# 0.1 Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
# 0.2 Load configfile
with open('config.yaml') as yamlfile:
    config = yaml.load(yamlfile, Loader=yaml.FullLoader)


def main():
    # Paths
    training_data_path = "training_data.csv"
    validation_data_path = "validation_data.csv"

    # Load data
    training_data = MultiCutStixelData(training_data_path, data_dir="data/training",
                                       transform=transforming,
                                       target_transform=target_transforming)
    train_dataloader = DataLoader(training_data, batch_size=config['batch_size'],
                                  num_workers=config['resources']['train_worker'], pin_memory=True)

    validation_data = MultiCutStixelData(validation_data_path, data_dir="data/validation",
                                         transform=transforming,
                                         target_transform=target_transforming)
    val_dataloader = DataLoader(validation_data, batch_size=config['batch_size'],
                                num_workers=config['resources']['val_worker'], pin_memory=True, shuffle=True)

    # Explore data
    test_features, test_labels = next(iter(val_dataloader))
    if config['explore_data']:
        show_data(test_features, test_labels, idx=-1)

    # Define Model
    model = ConvNeXt(depths=[3]).to(device)
    # Load Weights
    if config['weights']['load']:
        weights_file = config['weight']['file']
        model.load_state_dict(torch.load("saved_models/" + weights_file))
        print(f'Weights loaded from: {weights_file}')
    # Loss function
    loss_fn = StixelLoss()
    # Optimizer definition
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    # Inspect model
    if config['inspect_model']:
        summary(model, (3, 1280, 1920))
        data = test_features.to(device)
        print("Input shape: " + str(data.shape))
        print("Output shape: " + str(model(data).shape))
        print("Running on " + device)
        print("----------------------------------------------------------------")

    # testing loss
    if config['test_loss']:
        data = test_features.to(device)
        output = model(data)
        target = test_labels.to(device)
        print(loss_fn(output, target))
        # df_output = pd.DataFrame(output[0].cpu().detach().numpy())
        # df_target = pd.DataFrame(test_labels[0].numpy())
        # df_output.to_csv("Prediction.csv", index=False)
        # df_target.to_csv("Target.csv", index=False)

    # Training
    if config['training']:
        if config['logging']:
            wandb_logger = wandb.init(project=config['logging']['project'],
                                      config = {
                                          "learning_rate": config['learning_rate'],
                                          "architecture": config['logging']['architecture'],
                                          "dataset": config['logging']['dataset'],
                                          "epochs": config['num_epochs'],
                                      }
                                      )
            wandb_logger.watch(model)
        else:
            wandb_logger = None
        for epoch in range(config['num_epochs']):
            print(f"\n   Epoch {epoch + 1}\n----------------------------------------------------------------")
            train_one_epoch(train_dataloader, model, loss_fn, optimizer,
                            device=device, writer=wandb_logger)
            eval_loss = evaluate(val_dataloader, model, loss_fn,
                                 device=device)
            # Save model
            if config['weights']['save']:
                if os.path.isdir('saved_models'):
                    weights_path = f"saved_models/StixelNExT_{wandb_logger.name}_epoch-{epoch}_loss-{eval_loss}.pth"
                    torch.save(model.state_dict(), weights_path)
                    print("Saved PyTorch Model State to " + weights_path)
                else:
                    print("Directory doesn't exist.")


if __name__ == '__main__':
    main()
