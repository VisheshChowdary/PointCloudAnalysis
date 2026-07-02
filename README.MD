# PointCloudAnalysis

Point Cloud Classification using PyTorch.

## Phase 1

- OFF file parser
- ModelNet40 dataset
- Data augmentation
- DataLoader
- Dummy baseline model
- Training loop
- Checkpoint saving
- Automatic dataset detection
- Windows / Colab / Kaggle support

## Project Structure

```
PointCloudAnalysis
│
├── configs
├── data
├── datasets
├── models
├── scripts
├── training
├── utils
│
├── checkpoints
├── outputs
│
├── main.py
├── requirements.txt
└── README.md
```

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

## Dataset

ModelNet40 (OFF format)

The project automatically detects the dataset on:

- Windows
- Google Colab
- Kaggle

## Next Phase

Implement PointNet.