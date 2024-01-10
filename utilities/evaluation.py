import os
import pickle
import time
import yaml
import torch
from torch.utils.data import DataLoader
from models.ConvNeXt import ConvNeXt
from dataloader.stixel_multicut import MultiCutStixelData
from dataloader.stixel_multicut_interpreter import StixelNExTInterpreter

# 0.1 Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
# 0.2 Load configfile
with open('config.yaml') as yamlfile:
    config = yaml.load(yamlfile, Loader=yaml.FullLoader)


def create_result_file(model, weights_file: str, output_path="predictions"):
    testing_data = MultiCutStixelData(data_dir=config['data_path'],
                                      phase='testing',
                                      transform=None,
                                      target_transform=None,
                                      return_name=True)
    testing_dataloader = DataLoader(testing_data, batch_size=config['batch_size'],
                                    num_workers=config['resources']['test_worker'], pin_memory=True, shuffle=True,
                                    drop_last=True)
    checkpoint = os.path.splitext(weights_file)[0]     # checkpoint without ending
    run = checkpoint.split('_')[1]
    model.load_state_dict(torch.load(os.path.join("saved_models", run, weights_file)))
    print(f'Weights loaded from: {weights_file}')

    stixel_reader = StixelNExTInterpreter(detection_threshold=config['pred_threshold'],
                                          hysteresis_threshold=config['pred_threshold'] - 0.05)

    stixel_lists = []
    for batch_idx, (samples, targets, names) in enumerate(testing_dataloader):
        samples = samples.to(device)
        start = time.process_time_ns()
        output = model(samples)
        t_infer = time.process_time_ns() - start
        # fetch data from GPU
        output = output.cpu().detach()
        for i in range(output.shape[0]):
            stixel_lists.append({"sample": names[i], "prediction": stixel_reader.extract_stixel_from_prediction(output[i]),
                                 "t_infer": t_infer})

    with open(os.path.join(output_path, checkpoint + ".pkl"), 'wb') as file:
        pickle.dump(stixel_lists, file)
    print(f"{checkpoint} exported to {output_path}!")


def main():
    weights_file = config['weights_file']
    model = ConvNeXt(stem_features=config['nn']['stem_features'],
                     depths=config['nn']['depths'],
                     widths=config['nn']['widths'],
                     drop_p=config['nn']['drop_p'],
                     out_channels=2).to(device)
    create_result_file(model=model, weights_file=weights_file)


if __name__ == '__main__':
    main()
