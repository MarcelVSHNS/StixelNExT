import torch
import wandb
import yaml
from torch.utils.data import DataLoader
from torchsummary import summary
import os
import torch.nn as nn

from models.ConvNeXt_implementation import ConvNeXt
from losses.stixel_loss import StixelLoss
from engine import train_one_epoch, evaluate
from dataloader.waymo_multicut import MultiCutStixelData, transforming, target_transforming


# 0.1 Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
# 0.2 Load configfile
with open('config.yaml') as yamlfile:
    config = yaml.load(yamlfile, Loader=yaml.FullLoader)


def main():
    annotation = "targets_from_stereo"
    images = "STEREO_LEFT"
    # Load data
    training_data = MultiCutStixelData(data_dir=config['data_path'] + 'training',
                                       annotation_dir=annotation,
                                       img_dir=images,
                                       transform=transforming,
                                       target_transform=target_transforming)
    train_dataloader = DataLoader(training_data, batch_size=config['batch_size'],
                                  num_workers=config['resources']['train_worker'], pin_memory=True)

    validation_data = MultiCutStixelData(data_dir=config['data_path']+ 'validation',
                                         annotation_dir=annotation,
                                         img_dir=images,
                                         transform=transforming,
                                         target_transform=target_transforming)
    val_dataloader = DataLoader(validation_data, batch_size=config['batch_size'],
                                num_workers=config['resources']['val_worker'], pin_memory=True, shuffle=True)

    # Define Model
    model = ConvNeXt(depths=[3]).to(device)
    # Load Weights
    if config['weights']['load']:
        weights_file = config['weights']['file']
        model.load_state_dict(torch.load("saved_models/" + weights_file))
        print(f'Weights loaded from: {weights_file}')
    # Loss function
    # loss_fn = StixelLoss()
    loss_fn = nn.BCELoss()
    # Optimizer definition
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    # Initialize Logger
    if config['logging']['activate']:
        wandb_logger = wandb.init(project=config['logging']['project'],
                                  config={
                                      "learning_rate": config['learning_rate'],
                                      "architecture": config['logging']['architecture'],
                                      "dataset": config['logging']['dataset'],
                                      "epochs": config['num_epochs'],
                                  },
                                  tags=["training", "evaluation"]
                                  )
        wandb_logger.watch(model)
    else:
        wandb_logger = None

    # Explore data
    test_features, test_labels = next(iter(val_dataloader))
    if config['explore_data']:
        pass

    # Inspect model
    if config['inspect_model']:
        summary(model, (3, 1200, 1920))
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

    # Training
    if config['training']:
        for epoch in range(config['num_epochs']):
            print(f"\n   Epoch {epoch + 1}\n----------------------------------------------------------------")
            train_one_epoch(train_dataloader, model, loss_fn, optimizer,
                            device=device, writer=wandb_logger)
            eval_loss = evaluate(val_dataloader, model, loss_fn,
                                 device=device, writer=wandb_logger)
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
