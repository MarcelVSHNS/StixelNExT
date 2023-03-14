import torch
import wandb
import yaml
import time
from torch.utils.data import DataLoader
from torchsummary import summary
from PIL import Image

from models.ConvNeXt_implementation import ConvNeXt
from dataloader.waymo_multicut import MultiCutStixelData, transforming, target_transforming
from utilities.visualization import create_sample_comparison, show_data_pair


# 0.1 Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
# 0.2 Load configfile
with open('config.yaml') as yamlfile:
    config = yaml.load(yamlfile, Loader=yaml.FullLoader)


def main():
    # TODO: Adapt the threshold depending on the brightness conditions
    threshold = 0.4
    # Paths
    testing_data_path = "testing_data.csv"
    testing_data = MultiCutStixelData(testing_data_path, data_dir="data/testing",
                                      transform=transforming,
                                      target_transform=target_transforming)
    testing_dataloader = DataLoader(testing_data, batch_size=config['batch_size'],
                                    num_workers=config['resources']['test_worker'],
                                    pin_memory=True,
                                    shuffle=True)

    # Setup the Model
    model = ConvNeXt(depths=[3]).to(device)
    weights_file = config['weights']['file']
    model.load_state_dict(torch.load("saved_models/" + weights_file))
    print(f'Weights loaded from: {weights_file}')

    # Investigate some selected data
    if config['explore_data']:
        test_features, test_labels = next(iter(testing_dataloader))
        data = test_features.to(device)
        start = time.process_time_ns()
        output = model(data)
        t_infer = time.process_time_ns() - start
        sample_img = create_sample_comparison(test_features, output, test_labels, t_infer=t_infer, threshold=threshold)
        show_data_pair(sample_img)

    # Create an export of analysed data inkcl. samples and ROC curve
    if config['logging']['activate']:
        # Init the logger
        wandb_logger = wandb.init(project=config['logging']['project'],
                                  config={
                                      "architecture": config['logging']['architecture'],
                                      "dataset": config['logging']['dataset'],
                                      "checkpoint": config['weights']['file'],
                                  },
                                  tags=["metrics", "testing"]
                                  )

        for batch_idx, (samples, targets) in enumerate(testing_dataloader):
            samples = samples.to(device)
            targets = targets.to(device)
            start = time.process_time_ns()
            output = model(samples)
            t_infer = time.process_time_ns() - start
            # TODO: Enter ROC Curve function here
            if batch_idx % 50 == 0:
                sample_img = create_sample_comparison(samples, output, targets, t_infer=t_infer,
                                                      threshold=threshold)
                images = wandb.Image(sample_img, caption="Top: Output, Bottom: Target")
                wandb_logger.log({"Examples": images})


if __name__ == '__main__':
    main()
