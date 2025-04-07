Here's the README.md file for your project:

```markdown
# Handover Motion Prediction with siMLPe

This repository contains code for predicting human motion during handover interactions using the siMLPe (Simple MLP-based) model. The system predicts future human poses based on past motion and intention information.

## Project Structure

```
.
├── data/
│   ├── handover_test.txt       # Test subject IDs
│   └── handover_train.txt      # Training subject IDs
└── exps/
    └── baseline_handover/
        ├── config.py           # Main configuration file
        ├── test.py             # Testing script
        └── train.py            # Training script
```

## Dataset

The dataset consists of motion capture data from handover interactions with the following subject splits:

- **Training Subjects**: S3, S4, S5, S6, S8, S9, S10
- **Test Subject**: S7

## Configuration

The main configuration file (`config.py`) includes settings for:

- Model architecture (MLP layers, normalization, etc.)
- Training parameters (batch size, learning rate, etc.)
- Loss functions (MPJPE, velocity loss, intention classification, etc.)
- Data preprocessing (DCT transformations, etc.)

Key model configurations can be modified through command line arguments during training.

## Training

To train the model:

```bash
python train.py --exp-name [experiment_name] --seed [random_seed] [other_options]
```

Available training options:
- `--temporal-only`: Use temporal-only layers
- `--layer-norm-axis`: Set layer normalization axis
- `--with-normalization`: Enable layer normalization
- `--spatial-fc`: Use only spatial fully-connected layers
- `--num`: Number of MLP blocks
- `--weight`: Loss weight

The training script supports:
- Cosine learning rate scheduling
- Multiple loss functions (position, velocity, intention classification)
- Periodic evaluation on test set
- Model checkpointing

## Evaluation

To evaluate a trained model:

```bash
python test.py --model-pth [path_to_model_weights]
```

The evaluation script computes:
- MPJPE (Mean Per Joint Position Error) for full body
- Right hand specific error
- Percentage of predictions under various error thresholds (10cm, 15cm, etc.)
- Intention classification accuracy and F1 scores (if enabled)

## Model Architecture

The siMLPe model features:
- MLP-based architecture with configurable depth
- Optional DCT transformations for input/output
- Configurable normalization layers
- Multi-task learning (motion prediction + intention classification)
- Multiple loss terms for improved prediction quality

## Requirements

- Python 3.x
- PyTorch
- NumPy
- Other dependencies listed in config files

## Results

The model outputs:
- Per-frame position errors
- Right hand specific errors
- Quality metrics (% of predictions under various error thresholds)
- Intention classification metrics (accuracy, F1 scores) when enabled
```

This README provides a comprehensive overview of your project, including structure, usage, and key features. You may want to add additional sections like "Installation" or "Citation" if needed for your specific use case.