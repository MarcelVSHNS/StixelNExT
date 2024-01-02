import torch
import wandb
import yaml
from torch.utils.data import DataLoader
from torchsummary import summary
from datetime import datetime
import os
from losses.stixel_loss import StixelLoss
from models.ConvNeXt import ConvNeXt
from engine import train_one_epoch, evaluate
from dataloader.stixel_multicut import MultiCutStixelData, target_transform_gaussian_blur
from dataloader.stixel_multicut_interpreter import StixelNExTInterpreter
from utilities.visualization import draw_stixels_on_image


# 0.1 Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
# 0.2 Load configfile
with open('config.yaml') as yamlfile:
    config = yaml.load(yamlfile, Loader=yaml.FullLoader)
overall_start_time = datetime.now()


def main():
    # Load data
    training_data = MultiCutStixelData(data_dir=config['data_path'],
                                       phase='training',
                                       transform=None,
                                       target_transform=target_transform_gaussian_blur)               # target_transform_gaussian_blur
    train_dataloader = DataLoader(training_data, batch_size=config['batch_size'],
                                  num_workers=config['resources']['train_worker'], pin_memory=True, drop_last=True)

    validation_data = MultiCutStixelData(data_dir=config['data_path'],
                                         phase='validation',
                                         transform=None,
                                         target_transform=target_transform_gaussian_blur)
    val_dataloader = DataLoader(validation_data, batch_size=config['batch_size'],
                                num_workers=config['resources']['val_worker'], pin_memory=True, shuffle=False, drop_last=True)

    testing_data = MultiCutStixelData(data_dir=config['data_path'],
                                      phase='testing',
                                      transform=None,
                                      target_transform=None,
                                      return_original_image=True)
    test_dataloader = DataLoader(testing_data, batch_size=config['batch_size'],
                                num_workers=config['resources']['test_worker'], pin_memory=True, shuffle=True,
                                drop_last=True)

    # Define Model
    model = ConvNeXt(stem_features=config['nn']['stem_features'],
                     depths=config['nn']['depths'],
                     widths=config['nn']['widths'],
                     drop_p=config['nn']['drop_p'],
                     out_channels=2).to(device)

    # Load Weights
    if config['load_weights']:
        weights_file = config['weights_file']
        model.load_state_dict(torch.load("saved_models/" + weights_file))
        print(f'Weights loaded from: {weights_file}')
    # Loss function
    loss_fn = StixelLoss(alpha=config['loss']['alpha'],
                         beta=config['loss']['beta'],
                         gamma=config['loss']['gamma'],
                         threshold=config['pred_threshold'])

    # Optimizer definition
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    # Initialize Logger
    if config['logging']['activate']:
        wandb_logger = wandb.init(project=config['logging']['project'],
                                  config={
                                      "learning_rate": config['learning_rate'],
                                      "loss_function": type(loss_fn).__name__,
                                      "loss_alpha": config['loss']['alpha'],
                                      "loss_beta": config['loss']['beta'],
                                      "loss_gamma": config['loss']['gamma'],
                                      "architecture": type(model).__name__,
                                      "dataset": training_data.name,
                                      "epochs": config['num_epochs'],
                                  },
                                  tags=["training"]
                                  )
        wandb_logger.watch(model)
    else:
        wandb_logger = None

    # Explore data
    test_features, test_labels, image = next(iter(test_dataloader))
    if config['explore_data']:
        pick = -1
        result_interpreter = StixelNExTInterpreter(detection_threshold=config['pred_threshold'],
                                                   hysteresis_threshold=config['pred_threshold'] - 0.05)

        # print ground truth
        result_interpreter.extract_stixel_from_prediction(test_labels[pick], detection_threshold=1.0,
                                                          hysteresis_threshold=0.9)
        result_interpreter.show_stixel(image[pick])
        result_interpreter.show_bottoms(image[pick])

        # print inference
        if config['load_weights']:
            sample = test_features.to(device)
            output = model(sample)
            output = output.cpu().detach()
            result_interpreter.extract_stixel_from_prediction(output[pick])
            result_interpreter.show_stixel(image[pick])
            result_interpreter.show_bottoms(image[pick])

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
            test_error = evaluate(val_dataloader, model, loss_fn,
                                 device=device, writer=wandb_logger)
            # Save model
            if config['logging']['activate']:
                saved_models_path = os.path.join('saved_models',wandb_logger.name)
                os.makedirs(saved_models_path, exist_ok=True)
                weights_name = f"StixelNExT_{wandb_logger.name}_epoch-{epoch}_test-error-{test_error}.pth"
                torch.save(model.state_dict(), os.path.join(saved_models_path, weights_name))
                print("Saved PyTorch Model State to " + os.path.join(saved_models_path, weights_name))
            step_time = datetime.now() - overall_start_time
            print("Time elapsed: {}".format(step_time))
        overall_time = datetime.now() - overall_start_time
        print(f"Finished training in {str(overall_time).split('.')[0]}")


if __name__ == '__main__':
    main()
