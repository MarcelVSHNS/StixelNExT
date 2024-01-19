import torch
import wandb
import yaml
from torch.utils.data import DataLoader
from torchsummary import summary
from datetime import datetime
import os
from losses.stixel_loss_endtoend import StixelLoss
from models.ConvNeXt import ConvNeXt
from engine import train_one_epoch, evaluate, EarlyStopping
from dataloader.stixel_multicut_endtoend import MultiCutStixelData, target_transform_gaussian_blur
import optuna
from optuna.trial import TrialState
from optuna.visualization import plot_optimization_history


# 0.1 Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
# 0.2 Load configfile
with open('config.yaml') as yamlfile:
    config = yaml.load(yamlfile, Loader=yaml.FullLoader)
overall_start_time = datetime.now()


def objective(trial):
    num_epochs = config['num_epochs']
    learning_rate = config['learning_rate']
    depth0 = trial.suggest_int("depth0", 1, 9)
    depth1 = trial.suggest_int("depth1", 1, 9)
    depths = [depth0, depth1]
    stem_features = trial.suggest_categorical("stem_features", [32, 64, 96, 128, 160, 192, 224, 256])
    base_widths = [48, 96, 192, 386]
    width_factor = trial.suggest_categorical("width_factor", [0.5, 1.0, 2.0, 3.0, 4.0])
    widths = [int(base_width * width_factor) for base_width in base_widths]
    drop_p = trial.suggest_float("drop_p", 0.0, 0.3)

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

    # Define Model
    model = ConvNeXt(stem_features=config['nn']['stem_features'],
                     depths=config['nn']['depths'],
                     widths=config['nn']['widths'],
                     drop_p=config['nn']['drop_p'],
                     target_height=config['num_obj_per_col'],
                     target_width=int(config['img_width'] // config['grid_step']),
                     out_channels=3).to(device)
    # Inspect model
    summary(model, (3, 1200, 1920))
    print("----------------------------------------------------------------")

    # Loss function
    loss_fn = StixelLoss(alpha=1.0,
                         beta=1.0,
                         gamma=1.0,
                         delta=False)

    # Optimizer definition
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Initialize Logger
    if config['logging']['activate']:
        wandb_logger = wandb.init(project=config['logging']['project'],
                                  config={
                                      "learning_rate": config['learning_rate'],
                                      "loss_function": type(loss_fn).__name__,
                                      "architecture": type(model).__name__,
                                      "dataset": training_data.name,
                                      "epochs": config['num_epochs'],
                                  },
                                  tags=["training"]
                                  )
        wandb_logger.watch(model)
    else:
        wandb_logger = None

    # Training
    test_error = 0.0
    early_stopping = EarlyStopping(tolerance=config['early_stopping']['tolerance'],
                                   min_delta=config['early_stopping']['min_delta'])
    for epoch in range(num_epochs):
        print(f"\n   Epoch {epoch + 1}\n----------------------------------------------------------------")
        train_one_epoch(train_dataloader, model, loss_fn, optimizer,
                        device=device, writer=wandb_logger)
        test_error = evaluate(val_dataloader, model, loss_fn,
                             device=device, writer=wandb_logger)
        trial.report(test_error, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        # Save model
        if config['logging']['activate']:
            saved_models_path = os.path.join('saved_models',wandb_logger.name)
            os.makedirs(saved_models_path, exist_ok=True)
            weights_name = f"StixelNExT_{wandb_logger.name}_epoch-{epoch}_test-error-{test_error}.pth"
            torch.save(model.state_dict(), os.path.join(saved_models_path, weights_name))
            print("Saved PyTorch Model State to " + os.path.join(saved_models_path, weights_name))
        step_time = datetime.now() - overall_start_time
        print("Time elapsed: {}".format(step_time))
        # early stopping
        early_stopping.check_stop(test_error)
        if early_stopping.early_stop:
            print("Early stopping at epoch:", epoch)
            break

    overall_time = datetime.now() - overall_start_time
    print(f"Finished training x in {str(overall_time).split('.')[0]}")
    return test_error


def main():
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=60)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    # Export CSV
    df = study.trials_dataframe()
    df.to_csv("study_results.csv")
    # visual
    plot = plot_optimization_history(study)
    plot.write_image("optimization_history.png")


if __name__ == '__main__':
    main()
