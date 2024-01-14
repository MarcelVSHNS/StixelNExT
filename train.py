import yaml
# 0.1 Load configfile
with open('config.yaml') as yamlfile:
    config = yaml.load(yamlfile, Loader=yaml.FullLoader)

import torch
import wandb
from torch.utils.data import DataLoader
from torchsummary import summary
from datetime import datetime
import os
import shutil
from losses.stixel_loss import StixelLoss
from models.ConvNeXt import ConvNeXt
from engine import train_one_epoch, evaluate
from dataloader.stixel_multicut_interpreter import StixelNExTInterpreter
from utilities.evaluation import create_result_file
from dataloader.stixel_multicut import MultiCutStixelData, target_transform_gaussian_blur as target_transform
if config['dataset'] == "kitti":
    from dataloader.stixel_multicut import feature_transform_resize as feature_transform
    config['grid_step'] = 4
    config['img_height'] = 376
    config['img_width'] = 1248
else:
    feature_transform = None
    config['grid_step'] = 8
    config['img_height'] = None
    config['img_width'] = None
#with open('config.yaml', 'w') as file:
#    yaml.dump(config, file, default_flow_style=False)

# 0.2 Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
overall_start_time = datetime.now()


def main():
    test_dataloader = None
    # Load data
    dataset_dir = os.path.join(config['data_path'], config['dataset'])
    # Training data
    training_data = MultiCutStixelData(data_dir=dataset_dir,
                                       phase='training',
                                       transform=feature_transform,
                                       target_transform=target_transform)               # target_transform_gaussian_blur
    train_dataloader = DataLoader(training_data, batch_size=config['batch_size'],
                                  num_workers=config['resources']['train_worker'], pin_memory=True, drop_last=True)
    # Validation data
    validation_data = MultiCutStixelData(data_dir=dataset_dir,
                                         phase='validation',
                                         transform=feature_transform,
                                         target_transform=target_transform)
    val_dataloader = DataLoader(validation_data, batch_size=config['batch_size'],
                                num_workers=config['resources']['val_worker'], pin_memory=True, shuffle=False, drop_last=True)

    # Testing data
    if config['explore_data'] or config['test_loss'] or config['inspect_model']:
        testing_data = MultiCutStixelData(data_dir=dataset_dir,
                                          phase='testing',
                                          transform=feature_transform,
                                          target_transform=None,
                                          return_original_image=True)
        test_dataloader = DataLoader(testing_data, batch_size=config['batch_size'],
                                     num_workers=config['resources']['test_worker'], pin_memory=True, shuffle=False,
                                     drop_last=True)

    # Define Model
    model = ConvNeXt(stem_features=config['nn']['stem_features'],
                     depths=config['nn']['depths'],
                     widths=config['nn']['widths'],
                     drop_p=config['nn']['drop_p'],
                     target_height=int(training_data.img_size['height'] / config['grid_step']),
                     target_width=int(training_data.img_size['width'] / config['grid_step']),
                     out_channels=2).to(device)

    # Load Weights
    if config['load_weights']:
        weights_file = config['weights_file']
        checkpoint = os.path.splitext(weights_file)[0]  # checkpoint without ending
        run = checkpoint.split('_')[1]
        model.load_state_dict(torch.load(os.path.join("saved_models", run, weights_file)))
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
    if config['explore_data']:
        import cv2
        import numpy as np
        test_features, test_labels, image = next(iter(test_dataloader))
        pick = -1
        result_interpreter = StixelNExTInterpreter(detection_threshold=config['pred_threshold'],
                                                   hysteresis_threshold=config['pred_threshold'] - 0.05)

        # print ground truth
        result_interpreter.extract_stixel_from_prediction(test_labels[pick], detection_threshold=1.0,
                                                          hysteresis_threshold=0.9)
        #result_interpreter.show_stixel(image[pick])
        #result_interpreter.show_bottoms(image[pick])

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
        test_features, test_labels, image = next(iter(test_dataloader))
        height = image[0].shape[0]
        width = image[0].shape[1]
        summary(model, (3, height, width))
        data = test_features.to(device)
        print("Input shape: " + str(data.shape))
        print("Output shape: " + str(model(data).shape))
        print("Running on " + device)
        print("----------------------------------------------------------------")

    # testing loss
    if config['test_loss']:
        test_features, test_labels, image = next(iter(test_dataloader))
        data = test_features.to(device)
        output = model(data)
        target = test_labels.to(device)
        print(loss_fn(output, target))

    # Training
    if config['training']:
        checkpoints = []
        early_stopping = EarlyStopping(tolerance=4, min_delta=0.001)
        for epoch in range(config['num_epochs']):
            print(f"\n   Epoch {epoch + 1}\n----------------------------------------------------------------")
            train_error = train_one_epoch(train_dataloader, model, loss_fn, optimizer,
                                          device=device, writer=wandb_logger)
            test_error = evaluate(val_dataloader, model, loss_fn,
                                  device=device, writer=wandb_logger)
            # Save model
            if config['logging']['activate']:
                saved_models_path = os.path.join('saved_models', wandb_logger.name)
                os.makedirs(saved_models_path, exist_ok=True)
                weights_name = f"StixelNExT_{wandb_logger.name}_epoch-{epoch}_test-error-{test_error}.pth"
                torch.save(model.state_dict(), os.path.join(saved_models_path, weights_name))
                checkpoints.append({'checkpoint': weights_name, 'test-error': test_error})
                print("Saved PyTorch Model State to " + os.path.join(saved_models_path, weights_name))
            step_time = datetime.now() - overall_start_time
            print("Time elapsed: {}".format(step_time))

            # early stopping
            early_stopping.check_stop(test_error)
            if early_stopping.early_stop:
                print("Early stopping at epoch:", epoch)
                break

        overall_time = datetime.now() - overall_start_time
        print(f"Finished training in {str(overall_time).split('.')[0]}")

        if config['logging']['activate']:
            best_checkpoint = min(checkpoints, key=lambda x: x['test-error'])
            source_path = os.path.join(saved_models_path, best_checkpoint['checkpoint'])
            destination_path = os.path.join("best_model_weights", best_checkpoint['checkpoint'])
            shutil.copy(source_path, destination_path)


class EarlyStopping:
    def __init__(self, tolerance=5, min_delta=0.0):
        self.tolerance = tolerance
        self.min_delta = min_delta
        self.validation_loss_minus_one = 10000
        self.counter = 0
        self.early_stop = False

    def check_stop(self, validation_loss):
        if (self.validation_loss_minus_one - validation_loss) > self.min_delta:
            self.counter = 0
        else:
            self.counter += 1
        if self.counter >= self.tolerance:
            self.early_stop = True
        self.validation_loss_minus_one = validation_loss


if __name__ == '__main__':
    main()
