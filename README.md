# 🦁 Simba: Towards High-Fidelity and Geometrically-Consistent Point Cloud Completion via Transformation Diffusion

## Abstract
Simba is an advanced deep learning framework specifically designed for high-quality 3D point cloud completion tasks. By leveraging transformation diffusion techniques, Simba achieves high fidelity and geometric consistency in completing 3D point clouds. This repository provides the implementation, datasets, and tools necessary to reproduce the results presented in our work.

![Simba Overview](./fig/Simba_Overview.png) <!-- Replace with actual figure path -->

## 📰 News
- **2025-11-22** Initial release of Simba repository.
- **2025-11-08** Simba accepted at AAAI 2026.

## 🚀 Features
- High-fidelity 3D point cloud completion.
- Geometrically consistent results using transformation diffusion.
- Support for multiple datasets, including PCN, ShapeNet and KITTI.
- Modular and extensible codebase for research and development.

## 📂 Pretrained Models
Pretrained models will be provided soon. Stay tuned!

## 🛠️ Installation Guide

### Prerequisites
- CUDA 12.1 compatible GPU
- Anaconda or Miniconda installed
- Python 3.10

### Installation Steps

#### 1. Environment Setup
Create and activate a new conda environment:
```bash
conda create --name Simba python=3.10
conda activate Simba
```

#### 2. PyTorch Installation
Install PyTorch with CUDA 12.1 support:
```bash
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia
```

#### 3. Dependencies Installation
Install required Python packages and custom wheels:
```bash
pip install -r requirements.txt
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
```

#### 4. PyTorch3D Installation
Install PyTorch3D compatible with your PyTorch version:
```bash
conda install https://anaconda.org/pytorch3d/pytorch3d/0.7.8/download/linux-64/pytorch3d-0.7.8-py310_cu121_pyt231.tar.bz2
```

#### 5. Mamba and Causal Conv1D Installation
Install specialized neural network components by downloading the required `.whl` files from the following links:

- [Mamba SSM v1.2.1](https://github.com/state-spaces/mamba/releases?page=2) (File: `mamba_ssm-1.2.1+cu122torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl`)
- [Causal Conv1D v1.2.1](https://github.com/Dao-AILab/causal-conv1d/releases?page=2) (File: `causal_conv1d-1.2.1+cu122torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl`)

After downloading the specified files, install them using:
```bash
pip install mamba_ssm-1.2.1+cu122torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install causal_conv1d-1.2.1+cu122torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

#### 6. Extension Compilation
Compile various 3D processing extensions by running the provided `install.sh` script:
```bash
bash install.sh
```

The script will:
1. Navigate to each extension directory.
2. Run the installation command (`python setup.py install`).
3. Log the progress and errors to `install_extensions.log`.

If any step fails, the script will stop and display an error message.

## 📊 Datasets
Details about the datasets used in this project can be found in [DATASET.md](./DATASET.md).

## 🧪 Usage

### Two-Stage Training
Simba employs a two-stage training process to achieve high-fidelity and geometrically consistent point cloud completion. Below are the details for each stage:

#### Stage 1: Training the SymmGT Model
In the first stage, the SymmGT model is trained to generate high-quality intermediate representations. Use the following command to train the SymmGT model:
```bash
CUDA_VISIBLE_DEVICES=0,1,2 bash ./scripts/dist_train.sh 3 13232 \
    --config ./cfgs/ShapeNet55_models/SymmGT.yaml \
    --exp_name SymmGT_stage_1
```
The trained model weights will be saved in the `experiment/SymmGT_stage_1/` directory.

#### Stage 2: Training the Simba Model
In the second stage, the Simba model is trained using the pretrained weights from the first stage. Update the `pretrain` field in `cfgs/PCN_models/Simba.yaml` to point to the path of the trained SymmGT model (e.g., `experiment/SymmGT_stage_1/best_model.pth`). Then, run the following command:
```bash
CUDA_VISIBLE_DEVICES=0,1,2 bash ./scripts/dist_train.sh 3 13232 \
    --config ./cfgs/ShapeNet55_models/Simba.yaml \
    --exp_name Simba_stage_2
```

Alternatively, you can use the automated training script `train.sh`, which handles both stages and automatically sets the pretrained model path for the second stage:
```bash
bash train.sh
```
This script will:
1. Train the SymmGT model in the first stage.
2. Automatically retrieve the best model weights from the first stage and use them for training the Simba model in the second stage.

### Inference
To run inference with a pretrained model:
```bash
python tools/inference.py \
    --config <config_file> \
    --checkpoint <checkpoint_file> \
    --input <input_path> \
    --output <output_path>
```

### Training
To train a model from scratch:
```bash
bash scripts/train.sh \
    --config <config_file> \
    --output <output_dir>
```

## 📈 Results
Results on ShapeNet and KITTI datasets will be added soon.

## 📜 License
This project is licensed under the MIT License.

## 🙏 Acknowledgements
This project is inspired by [GRNet](https://github.com/hzxie/GRNet) and [mmdetection3d](https://github.com/open-mmlab/mmdetection3d).

## 📖 Citation
If you find this work useful, please consider citing:
```bibtex

```
