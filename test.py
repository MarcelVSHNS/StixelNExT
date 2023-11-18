import torch
import wandb
import yaml
import time
import numpy as np
from torchmetrics.classification import BinaryROC
from torch.utils.data import DataLoader
from sklearn import metrics

from models.ConvNeXt import ConvNeXt
from dataloader.stixel_multicut import MultiCutStixelData, feature_transforming, target_transforming
from utilities.visualization import create_sample_comparison, show_data_pair, plot_roc_curve


# 0.1 Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
# 0.2 Load configfile
with open('config.yaml') as yamlfile:
    config = yaml.load(yamlfile, Loader=yaml.FullLoader)


def main():
    fpr_limit = 0.02
    # Data loading
    testing_data = MultiCutStixelData(data_dir=config['data_path'],
                                      phase='testing',
                                      transform=feature_transforming,
                                      target_transform=target_transforming)
    testing_dataloader = DataLoader(testing_data, batch_size=config['batch_size'],
                                    num_workers=config['resources']['test_worker'],
                                    pin_memory=True,
                                    shuffle=False,
                                    drop_last=True)

    # Set up the Model
    model = ConvNeXt(depths=[3]).to(device)
    weights_file = config['weights_file']
    model.load_state_dict(torch.load("saved_models/" + weights_file))
    print(f'Weights loaded from: {weights_file}')

    # Set up the Metric, 21 means 0.05 steps per threshold
    roc_curve = BinaryROC(thresholds=21)

    # Investigate some selected data
    if config['explore_data']:
        test_features, test_labels = next(iter(testing_dataloader))
        # Send to GPU
        data = test_features.to(device)
        start = time.process_time_ns()
        output = model(data)
        t_infer = time.process_time_ns() - start
        # Fetch from GPU
        output = output.cpu().detach()
        test_features = test_features.cpu().detach()
        # ROC
        fpr, tpr, thresholds = roc_curve(output, test_labels.squeeze().to(torch.int))
        idx = find_fpr_index(fpr, fpr_limit)
        plot_roc_curve(fpr, tpr, thres_idx=idx, display=True)
        # Scatter & Comparison
        sample_img = create_sample_comparison(test_features, output, test_labels, t_infer=t_infer,
                                              threshold=thresholds.numpy()[idx])
        show_data_pair(sample_img)

    # Create an export of analysed data incl. samples and ROC curve
    if config['logging']['activate']:
        # Init the logger
        # e.g. StixelNExT_ancient-silence-25_epoch-94_loss-0.09816327691078186.pth
        epochs = config['weights']['file'].split('_')[2].split('-')[1]
        checkpoint = config['weights']['file'].split('_')[1]
        wandb_logger = wandb.init(project=config['logging']['project'],
                                  config={
                                      "architecture": type(model).__name__,
                                      "dataset": testing_data.name,
                                      "checkpoint": checkpoint,
                                      "epochs": epochs,
                                      "fpr_limit": fpr_limit
                                  },
                                  tags=["metrics", "testing"]
                                  )

        for batch_idx, (samples, targets) in enumerate(testing_dataloader):
            # send data to GPU
            samples = samples.to(device)
            start = time.process_time_ns()
            output = model(samples)
            t_infer = time.process_time_ns() - start
            # fetch data from GPU
            output = output.cpu().detach()
            # Attach to ROC curve
            fpr, tpr, thresholds = roc_curve(output, targets.squeeze().to(torch.int))
            # https://github.com/wandb/wandb/issues/1076
            # wandb_logger.log({"roc": wandb.plot.roc_curve(targets, output)})

            if batch_idx % 100 == 0:
                # Create Image Sample
                samples = samples.cpu().detach()
                idx = find_fpr_index(fpr, fpr_limit)
                threshold = thresholds.numpy()[idx]
                sample_img = create_sample_comparison(samples, output, targets, t_infer=t_infer,
                                                      threshold=threshold)
                wandb_image = wandb.Image(sample_img,
                                          caption=f"Batch-ID= {batch_idx}\nTop: Output\nBottom: Target")
                wandb_logger.log({"Examples": wandb_image})
                # Create ROC snippet
                sample_roc = plot_roc_curve(fpr, tpr, thres_idx=idx)
                sample_auc = np.round(metrics.auc(fpr, tpr), decimals=3)
                wandb_roc = wandb.Image(sample_roc,
                                        caption=f"Batch-ID= {batch_idx}\nROC with {threshold}\nAUC: {sample_auc}")
                wandb_logger.log({"Examples ROC": wandb_roc})

        fpr, tpr, thresholds = roc_curve.compute()
        # plot_roc_curve(fpr, tpr, thres_idx=find_threshold_index(thresholds, threshold), display=True)

        data = [[x, y] for (x, y) in zip(fpr, tpr)]
        table = wandb.Table(data=data, columns=["False Positive Rate", "True Positive Rate"])
        wandb_logger.log({"ROC curve": wandb.plot.line(table, "True Positive Rate", "False Positive Rate",
                                                       title=f"ROC Curve over {testing_data.__len__()} samples")})
        wandb_logger.log({"AUC": metrics.auc(fpr, tpr)})


def find_fpr_index(fpr_array, fpr):
    fpr_array = fpr_array.numpy()
    fpr_idx = np.where(fpr_array >= fpr)
    return fpr_idx[0][0]

def find_threshold_index(thres_array, threshold):
    thres_array = thres_array.numpy().round(decimals=2)
    threshold_idx = np.where(thres_array == threshold)
    return threshold_idx


if __name__ == '__main__':
    main()
