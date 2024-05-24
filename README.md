# [StixelNExT, 2D](https://ieeexplore.ieee.org/Xplore/home.jsp)

Official PyTorch implementation of **StixelNExT**, from the following paper:

[Toward Monocular Low-Weight Perception for Object Segmentation and Free Space Detection](https://ieeexplore.ieee.org/Xplore/home.jsp). IV 2024.\
[Marcel Vosshans](https://scholar.google.de/citations?user=_dbcdr4AAAAJ&hl=en), [Omar Ait-Aider](https://scholar.google.fr/citations?user=NIdLQnUAAAAJ&hl=en), [Youcef Mezouar](https://youcef-mezouar.wixsite.com/ymezouar) and [Markus Enzweiler](https://markus-enzweiler.de/)\
University of Esslingen, UCA Sigma Clermont\
[[`Xplore`](https://ieeexplore.ieee.org/Xplore/home.jsp)][[`video`](https://www.youtube.com/)] - t.b.d.

If you find our work useful in your research please consider citing our paper:
```
@INPROCEEDINGS{StixelNExT,
  title     = {StixelNExT: Toward Monocular Low-Weight Perception for Object Segmentation and Free Space Detection},
  author    = {Vosshans, Marcel and
               Ait-Aider, Omar  and
               Mezouar, Youcef and
               Enzweiler, Markus},
  booktitle = {2023 IEEE Intelligent Vehicles Symposium (IV)},
  month = {June},
year = {2024}
}
```

## ⚙️ Setup

Recommended is a fresh Python Venv (Version >= 3.7), you can install the dependencies with:
```shell
sudo apt-get install python3-venv 

python3 -m venv venv                  # on project folder level
source venv/bin/activate
pip install -r requirements.txt
```
We ran our experiments with PyTorch 2.1.2, CUDA 11.8, Python 3.8.10 and Ubuntu 20.04.05 LTS.

## 🖼️ Prediction for a single image
You can predict a single image with:

```shell
python test_simple.py --image_path assets/test_image.jpg --weights mono+stereo_640x192
```

### Model Weights
Pretrained model weights (used in our paper) can be downloaded [here](https://bwsyncandshare.kit.edu/s/y9Af5DfFrdcPSL6)


## 💾 Ground Truth Dataset generation
We also published our pipeline to generate ground truth from any dataset (dataloader necessary). Mandatory is a camera,
a LiDAR and the corresponding projection:
[StixelGENerator](https://github.com/MarcelVSHNS/StixelGENerator)

### KITTI Training Data
We also provide an already generated dataset, based on the public available KITTI dataset. It can be downloaded
[here](https://bwsyncandshare.kit.edu/s/FL4BDGe7FM2NjJK)

## 🏃 Training
1. Use the [config.yaml](./docs/config.yaml) file to configure your paths and settings for the training and copy it to project level
2. Run train.py (in your IDE or with `source venv/bin/activate && python train.py` in that case, don't forget to change the file permissions `chmod +x your_script.py`) 
3. dfsdf

## 📊 Evaluation
For evaluation, we provide another repository with both metrics: "The Naive" and the "The Fairer".
[StixelNExT-Eval](https://github.com/MarcelVSHNS/StixelNExT-Eval)