  <h1> 🍔 Bundle Adjusted Gaussian Avatars Deblurring </h1>
<div>
    <a href='https://myniuuu.github.io/' target='_blank'>Muyao Niu</a> <sup>1,2</sup> &nbsp;
    <a href='https://yifever20002.github.io/' target='_blank'>Yifan Zhan</a><sup>1,2</sup> &nbsp;
    <a href='https://qtzhu.me/' target='_blank'>Qingtian Zhu</a><sup>2</sup> &nbsp;
    Zhuoxiao Li<sup>2</sup> &nbsp;
    Wei Wang<sup>1</sup> &nbsp;<br>
    <a href='https://zzh-tech.github.io/' target='_blank'>Zhihang Zhong</a><sup>1,†</sup> &nbsp;
    <a href='https://jimmysuen.github.io/' target='_blank'>Xiao Sun</a><sup>1,†</sup> &nbsp;
    <a href='https://scholar.google.com/citations?user=JD-5DKcAAAAJ&hl=en' target='_blank'>Yinqiang Zheng</a><sup>2</sup> &nbsp;
</div>
<div>
    <sup>1</sup>Shanghai Artificial Intelligence Laboratory &nbsp; <sup>2</sup>The University of Tokyo
</div>
<div>
    <sup>†</sup>Corresponding Authors &nbsp; 
</div>

<br>

<a href='https://arxiv.org/abs/2411.16758'><img src='https://img.shields.io/badge/ArXiv-PDF-red'></a>


---

Stay tuned. Feel free to contact me for bugs or missing files.


## Setup Procedures

### Python Environment

```
conda create -n baga python==3.8 -y
conda activate baga
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install submodules/diff-gaussian-rasterization/
pip install submodules/simple-knn/
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
pip install -r requirement.txt
```

### SMPL files

Register and download SMPL models [here](https://smplify.is.tue.mpg.de/login.php) and put the downloaded models into the folder `assets`. Only the neutral one is needed. The folder structure should look like

```
./
└── assets/
    ├── SMPL_NEUTRAL.pkl
```

### Dataset

We contribute synthetic and real datasets for evaluating blur-aware 3DGS human avatar synthesis techniques. 

#### Synthetic Dataset
For the synthetic dataset, due to the [aggreement](https://pengsida.net/project_page_assets/files/ZJU-MoCap_Agreement.pdf) of ZJU-MoCap, we cannot re-distribute the sharp data of ZJU-MoCap. So you have to download the original dataset, and follow the following steps to construct the final synthetic dataset using our scripts:

1. Download the blurry frames and the calibrations from [here](https://drive.google.com/file/d/1ZLVpE-9zIobaY41-6bEsVXUEyILCycxw/view?usp=sharing) and unzip it to `./data/BlurZJU`.
2. Follow the procedure [here](https://github.com/zju3dv/neuralbody/blob/master/INSTALL.md) to download ZJU-MoCap (refined version). Unzip and put the six scenes (`my_377`, `my_386`, `my_387`, `my_392`, `my_393`, `my_394`) to `./data/ZJU-MoCap-Refine` (If you get scenes starting with `CoreView` instead of `my`, then you have downloaded the original ZJU-MoCap, not the Refined version).
3. Run `python rearrange_zju.py` to re-arrange the dataset.

#### Real Dataset (BS-Human)

Download the real dataset from this [link](https://drive.google.com/file/d/1FXFILsI3WjxVL5ercZUHnSatL9dAbEib/view?usp=sharing) and unzip them to the `./data` directory.

## Training

### Synthetic dataset

```
chmod 777 train_BlurZJU.sh
bash train_BlurZJU.sh
```

### Real dataset

```
chmod 777 train_BSHuman.sh
bash train_BSHuman.sh
```

## Acknowledgments

We appreciate [gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting), [GauHuman](https://github.com/skhu101/GauHuman), and [GSM](https://github.com/computational-imaging/GSM) for their wonderful work and code implementation. We would also like to deeply express our gratitude to the release of [NeuralBody](https://github.com/zju3dv/neuralbody) (as well as the ZJU-MoCap dataset) and [EasyMocap](https://github.com/zju3dv/EasyMocap) which we use to calibrate our dataset. 
