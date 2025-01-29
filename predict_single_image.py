import argparse
from xmlrpc.client import DateTime

import torch
from torchsummary import summary
from torchvision.io import read_image, ImageReadMode
from dataloader.stixel_multicut import feature_transform_resize
from dataloader.stixel_multicut_interpreter import StixelNExTInterpreter
from models import ConvNeXt
from datetime import datetime
from PIL import Image
import os


def load_model(weights_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvNeXt(target_height=94,
                     target_width=312
                     ).to(device)
    model.load_state_dict(
        torch.load(weights_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))
    model.eval()
    return model


def preprocess_image(image_path):
    feature_image: torch.Tensor = read_image(image_path, ImageReadMode.RGB).to(torch.float32)
    image = Image.open(image_path).convert("RGB")
    return feature_transform_resize(feature_image).unsqueeze(0), image


def predict(model, image_tensor):
    with torch.no_grad():
        image_tensor = image_tensor.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        start_time = datetime.now()
        output = model(image_tensor)
        inference_time = (datetime.now() - start_time) * 1000
    print(f"t_inf: {inference_time.total_seconds():.2f} ms")
    return output


def main():
    parser = argparse.ArgumentParser(description="Inference for StixelNExT")
    parser.add_argument("--image_path", type=str, default="docs/set_0_2011_10_03_0027_16.png", help="path to image")
    parser.add_argument("--weights", type=str, default="saved_models/StixelNExT_prime-sunset-157_epoch-8_test-error-0.23861433565616608.pth", help="path to model weights")

    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        raise FileNotFoundError(f"Image not found: {args.image_path}")

    if not os.path.exists(args.weights):
        raise FileNotFoundError(f"Weights not found: {args.weights}")

    model = load_model(args.weights)
    summary(model, input_size=(3, 376, 1248))

    image_tensor, image = preprocess_image(args.image_path)

    # inference
    prediction = predict(model, image_tensor)
    prediction = prediction.cpu().detach()

    stixel_reader = StixelNExTInterpreter(detection_threshold=0.58)
    target_stixel = stixel_reader.extract_stixel_from_prediction(prediction)
    stixel_reader.show_stixel(image, stixel_list=target_stixel, color=[189, 195, 83]) #Color in BGR


if __name__ == "__main__":
    main()
