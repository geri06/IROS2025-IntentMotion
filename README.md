Here's your revised `README.md` in proper Markdown format, optimized for GitHub rendering:

```markdown
# Handover Motion Prediction with siMLPe

This repository contains code for predicting human motion during handover interactions using the **siMLPe** (Simple MLP-based) model. The system predicts future human poses based on past motion and intention information.

---

## 📁 Project Structure

```plaintext
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

---

## 📊 Dataset

The dataset consists of motion capture data from handover interactions.

- **Training Subjects**: `S3`, `S4`, `S5`, `S6`, `S8`, `S9`, `S10`  
- **Test Subject**: `S7`

---

## ⚙️ Configuration

The main configuration file [`config.py`](exps/baseline_handover/config.py) includes settings for:

- Model architecture (MLP layers, normalization, etc.)
- Training parameters (batch size, learning rate, etc.)
- Loss functions (MPJPE, velocity loss, intention classification, etc.)
- Data preprocessing (DCT transformations, etc.)

Most model settings can be overridden via command-line arguments when running the training script.

---

## 🏋️‍♀️ Training

To train the model, run:

```bash
python train.py --exp-name [experiment_name] --seed [random_seed] [other_options]
```

### Available Options:
- `--temporal-only`: Use temporal-only layers  
- `--layer-norm-axis`: Set layer normalization axis  
- `--with-normalization`: Enable layer normalization  
- `--spatial-fc`: Use only spatial fully-connected layers  
- `--num`: Number of MLP blocks  
- `--weight`: Loss weight  

Training features include:
- Cosine learning rate scheduling  
- Multiple loss functions (position, velocity, intention classification)  
- Periodic evaluation on test set  
- Model checkpointing

---

## ✅ Evaluation

To evaluate a trained model:

```bash
python test.py --model-pth [path_to_model_weights]
```

### Metrics Computed:
- MPJPE (Mean Per Joint Position Error) – full body  
- Right-hand-specific MPJPE  
- Percentage of predictions under various error thresholds (10cm, 15cm, etc.)  
- Intention classification accuracy and F1 score (if enabled)

---

## 🧠 Model Architecture

The **siMLPe** model features:

- MLP-based architecture with configurable depth  
- Optional DCT transformations for input/output  
- Configurable normalization layers  
- Multi-task learning (motion prediction + intention classification)  
- Multi-term loss for enhanced prediction quality

---

## 📦 Requirements

- Python 3.x  
- PyTorch  
- NumPy  
- Other dependencies as specified in the config files

---

## 📈 Results

The model provides:

- Per-frame position errors  
- Right-hand-specific errors  
- Quality metrics (e.g., % under 10cm, 15cm thresholds)  
- Intention classification metrics (accuracy, F1 score, if applicable)

---

## 📝 Notes

Feel free to add the following sections as needed:
- **Installation** instructions  
- **Citation** for academic use  
- **License** and contributing guidelines  

---


```