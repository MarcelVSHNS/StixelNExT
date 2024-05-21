import yaml
# 0.1 Load configfile
with open('config.yaml') as yamlfile:
    config = yaml.load(yamlfile, Loader=yaml.FullLoader)

import os
import torch
from models.ConvNeXt import ConvNeXt
from dataloader.mpeg4_video import VideoDataLoader, VideoExporter, resize_image_to_tensor
from dataloader.stixel_multicut_interpreter import StixelNExTInterpreter, draw_stixels_on_image


device = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    frame_size = (1920, 1200)
    video_path = 'input_video.mp4'
    output_path = 'output_video.mp4'
    color = [48, 213, 200]
    fps = 24.0

    mpeg_data = VideoDataLoader(video_path, img_size=frame_size, transform=resize_image_to_tensor)
    video_exporter = VideoExporter(output_path, frame_size, fps=fps)
    result_interpreter = StixelNExTInterpreter()

    model = ConvNeXt(stem_features=config['nn']['stem_features'],
                         depths=config['nn']['depths'],
                         widths=config['nn']['widths'],
                         drop_p=config['nn']['drop_p'],
                         target_height=int(frame_size[1] / config['grid_step']),
                         target_width=int(frame_size[0] / config['grid_step']),
                         out_channels=2).to(device)

    weights_file = config['weights_file']
    checkpoint = os.path.splitext(weights_file)[0]  # checkpoint without ending
    run = checkpoint.split('_')[1]
    model.load_state_dict(torch.load(os.path.join("saved_models", run, weights_file)))
    print(f'Weights loaded from: {weights_file}')

    for frame, image in mpeg_data:
        frame = frame.unsqueeze(0).to(device)
        output = model(frame)
        output = output.cpu().detach()
        output = output.squeeze()
        pred_stixel = result_interpreter.extract_stixel_from_prediction(output, detection_threshold=config['pred_threshold'])
        pred_stixel_img = draw_stixels_on_image(image, pred_stixel, color=color)
        video_exporter.add_frame(pred_stixel_img)

    video_exporter.save()


if __name__ == '__main__':
    main()
