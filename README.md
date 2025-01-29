# [StixelNExT, 2D](https://ieeexplore.ieee.org/document/10588680)

Official PyTorch implementation of **StixelNExT**, from the following paper:

[Toward Monocular Low-Weight Perception for Object Segmentation and Free Space Detection](https://ieeexplore.ieee.org/Xplore/home.jsp). IV 2024.\
[Marcel Vosshans](https://scholar.google.de/citations?user=_dbcdr4AAAAJ&hl=en), [Omar Ait-Aider](https://scholar.google.fr/citations?user=NIdLQnUAAAAJ&hl=en), [Youcef Mezouar](https://youcef-mezouar.wixsite.com/ymezouar) and [Markus Enzweiler](https://markus-enzweiler.de/)\
University of Esslingen, UCA Sigma Clermont\
[[`Xplore`](https://ieeexplore.ieee.org/document/10588680)][[`arXiv`](https://arxiv.org/abs/2407.08277)]

If you find our work useful in your research please consider citing our paper:
```
@INPROCEEDINGS{StixelNExT,
  booktitle={2024 IEEE Intelligent Vehicles Symposium (IV)}, 
  title     = {StixelNExT: Toward Monocular Low-Weight Perception for Object Segmentation and Free Space Detection},
  author    = {Vosshans, Marcel and
               Ait-Aider, Omar  and
               Mezouar, Youcef and
               Enzweiler, Markus},
  booktitle = {2024 IEEE Intelligent Vehicles Symposium (IV)},
  year = {2024},
  pages={2154-2161},
  keywords={Training;Space vehicles;Adaptation models;Laser radar;Image recognition;Intelligent vehicles;Training data},
  doi={10.1109/IV55156.2024.10588680}
}
```
![Sample StixelNExT result](/docs/multistixel.png)
StixelNExT is a low-weight CNN with roughly 1.5 mio. parameters to segment obstacles in the 2D plane and divide them into 
multiple objects. It is trainable within ~10 epochs without pre-trained weights.


## ⚙️ Setup

Recommended is a fresh Python Venv (Version >= 3.7), you can install the dependencies with:
```shell
sudo apt-get install python3-venv 

python3 -m venv venv                  # on project folder level
source venv/bin/activate
pip install -r requirements.txt
```
We ran our experiments with PyTorch 2.1.2, CUDA 11.8, Python 3.8.10 and Ubuntu 20.04.05 LTS.

## 🖼️  Single Image Prediction
You can predict a single image with the following script, just needed a `target_image.png` and `weights`.

```shell
python predict_single_img.py --image_path test_image.png --weights StixelNExT_prime-sunset-157_epoch-8_test-error-0.23861433565616608
python predict_single_img.py  # or ... for default values
```

### Model Weights
Pretrained model weights (used in our paper) can be downloaded [here](https://drive.google.com/drive/folders/10uJ1LoY5YPeOOy6SiBAFpy-Ph8mw2gJH?usp=sharing) (KITTI).


## 💾 Ground Truth Dataset Generation
We also published our pipeline to generate ground truth from any dataset (dataloader necessary). Mandatory is a camera,
a LiDAR and the corresponding projection:
[StixelGENerator](https://github.com/MarcelVSHNS/StixelGENerator).

### KITTI Training Data
We also provide an already generated dataset, based on the public available KITTI dataset. It can be downloaded
[here](https://drive.google.com/drive/folders/1ft99z9F4053zDzyIDn2DZ_8qh5if-QvW?usp=sharing) (35,48 GB).

## 🏃 Training
We used [Weights & Biases](https://wandb.ai/site) for organizing our trainings, so check your W&B python
API key [login](https://docs.wandb.ai/quickstart#2-log-in-to-wb) or write a workaround.
1. Use the [config.yaml](./docs/config.yaml) file to configure your paths and settings for the training and copy it to project level
2. Run train.py (in your IDE or with `source venv/bin/activate && python train.py` in that case, don't forget to change the file permissions `chmod +x your_script.py`) 
3. Get your weights (checkpoints) from the [saved_models](./saved_models) folder and test it with `predict_single_img.py`

## 📊 Evaluation
For evaluation, we provide another repository with both metrics: "The Naive" and the "The Fairer".
[StixelNExT-Eval](https://github.com/MarcelVSHNS/StixelNExT-Eval).

## 🔮 Future Works
The next step involves the addition of depth estimation. Future research will focus on incorporating end-to-end 
monocular depth estimation into StixelNExT:
[StixelNExT Pro](https://github.com/MarcelVSHNS/StixelNExT_Pro).
