import cv2
import numpy as np
import torch
from typing import Tuple


class VideoDataLoader:
    def __init__(self, video_path, img_size, transform=None):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.transform = transform
        self.img_size = img_size
        if not self.cap.isOpened():
            raise IOError("Error opening video file")
        self.length = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx >= self.length:
            raise IndexError("Index out of bounds")
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.cap.read()
        if not ret:
            raise IOError("Error reading frame at index {}".format(idx))
        if self.transform is not None:
            return self.transform(frame, self.img_size)
        else:
            return frame

    def __del__(self):
        if self.cap.isOpened():
            self.cap.release()


def resize_image_to_tensor(image, img_size: Tuple[int, int]) -> np:
    resized_image = cv2.resize(image, img_size, interpolation=cv2.INTER_AREA)
    return torch.from_numpy(resized_image).to(torch.float32), image


class VideoExporter:
    def __init__(self, output_path, frame_size, fps=20.0, codec='MP4V'):
        self.output_path = output_path
        self.frame_size = frame_size
        self.fps = fps
        self.codec = cv2.VideoWriter_fourcc(*codec)
        self.writer = cv2.VideoWriter(output_path, self.codec, fps, frame_size)

    def add_frame(self, frame):
        if frame.shape[1::-1] != self.frame_size:
            raise ValueError("Check frame size")
        self.writer.write(frame)

    def save(self):
        self.writer.release()
        print(f'Video saved to: {self.output_path}')
