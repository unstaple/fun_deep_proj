# CIFAR-10 Image Classification with a Convolutional Neural Network

A from-scratch PyTorch CNN project for the **Fundamentals of Deep Learning** capstone. The project trains and compares convolutional neural networks on CIFAR-10, selects the final model using validation accuracy, and reports performance once on the held-out test set.

[![Open in Google Colab](https://colab.research.google.com/github/Nat-D/kmitl-dl-notebooks/blob/main/06-capstone-starter.ipynb)

## Project overview

The task is 10-class image classification on **CIFAR-10**. Each example is a 32 × 32 RGB image. The model is implemented and trained in PyTorch; it does not rely on a pre-trained image classifier.

The notebook contains the complete workflow:

1. Download and inspect the dataset.
2. Create separate training, validation, and test sets.
3. Train a baseline CNN and compare architecture/training changes.
4. Inspect training curves and validation errors.
5. Select the model with the best validation accuracy.
6. Evaluate that selected model on the held-out test set.

## Dataset

- **Dataset:** [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)
- **Source:** The official CIFAR-10 dataset, loaded through `torchvision.datasets.CIFAR10`.
- **Size:** 60,000 colour images (50,000 official training images and 10,000 official test images).
- **Input shape:** 3 × 32 × 32.
- **Classes:** airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.
- **Download location:** `./data` (downloaded automatically by Torchvision; the dataset is not committed to this repository).

### Data split and leakage prevention

The 50,000-image official training portion is split using the fixed random seed into:

| Split      | Images | Purpose                                        |
| ---------- | -----: | ---------------------------------------------- |
| Training   | 40,000 | Fit model parameters                           |
| Validation | 10,000 | Compare experiments and select the final model |
| Test       | 10,000 | Final held-out evaluation only                 |

## Model architecture

The notebook compares a two-block baseline CNN with a deeper model, **BetterCNN**. BetterCNN was selected using validation accuracy.

```text
Input: 3 × 32 × 32 RGB image
        │
        ▼
Conv2D (3 → 64, 3×3) → BatchNorm → ReLU → MaxPool 2×2
        │
        ▼
Conv2D (64 → 128, 3×3) → BatchNorm → ReLU → MaxPool 2×2
        │
        ▼
Conv2D (128 → 256, 3×3) → BatchNorm → ReLU → MaxPool 2×2
        │
        ▼
Flatten (256 × 4 × 4)
        │
        ▼
Linear (4096 → 128) → ReLU → Dropout (p = 0.25)
        │
        ▼
Linear (128 → 10 class logits)
```

The three convolutional blocks progressively increase the channel count (64, 128, and 256). Batch normalization is used after each convolution, max pooling reduces spatial dimensions, and dropout regularizes the classifier head. The selected model has **897,418 trainable parameters**.

## Training configuration

The shared training loop and the model definitions are in [`06_capstone.ipynb`](./06_capstone.ipynb).

| Setting              | Value                              |
| -------------------- | ---------------------------------- |
| Framework            | PyTorch / Torchvision              |
| Model initialization | Trained from scratch               |
| Loss                 | Cross-entropy                      |
| Optimizer            | Adam                               |
| Epochs               | 8 per experiment                   |
| Batch size           | 128                                |
| Learning rate        | `1e-3`                             |
| Weight decay         | `1e-4`                             |
| Dropout              | `0.25`                             |
| Random seed          | `0`                                |
| Device               | CUDA when available; otherwise CPU |

## Results

The following experiments are recorded in the notebook. Validation accuracy is used for model selection; the test set is not used to compare these experiments.

| Experiment                                            | Validation accuracy |
| ----------------------------------------------------- | ------------------: |
| Baseline CNN                                          |              70.07% |
| Baseline CNN with dropout increased to `0.5`          |              67.05% |
| Baseline CNN with learning rate increased to `3e-3`   |              51.28% |
| **BetterCNN (BatchNorm + third convolutional block)** |          **75.94%** |

The validation-selected BetterCNN was then evaluated on the held-out test set:

| Metric                       |     Result |
| ---------------------------- | ---------: |
| Test cross-entropy loss      |      0.670 |
| **Test accuracy**            | **76.84%** |
| Majority-class test baseline |     10.00% |

_Results above are the outputs recorded in the committed notebook. Re-running training can produce small differences depending on the software and hardware environment._

## Reproducing the project

### Option 1: Google Colab (recommended)

1. Open the notebook using the **Open in Google Colab** badge above.
2. In Colab, choose **Runtime → Change runtime type → T4 GPU** (or another available GPU).
3. Run the notebook from the first cell to the last cell.
4. The CIFAR-10 files are downloaded automatically to `./data`.
5. Review the training curves and validation comparison. The notebook selects the best model by validation accuracy and then runs the final held-out test evaluation in the last section.

### Option 2: Run locally

Clone the repository and install the notebook dependencies:

```bash
git clone https://github.com/unstaple/fun_deep_proj.git
cd fun_deep_proj

python -m venv .venv
```

Activate the environment:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

Install the required packages and start Jupyter:

```bash
python -m pip install --upgrade pip
python -m pip install torch torchvision numpy matplotlib jupyter
jupyter lab
```

Open `06_capstone.ipynb` and run the cells in order. The notebook automatically uses CUDA when PyTorch detects an available CUDA device; otherwise, it runs on CPU (which may take longer).

### Evaluation entry point

The training and evaluation code currently lives in the notebook rather than in separate `train.py` / `eval.py` scripts:

- `train(...)` implements the training loop and records per-epoch train/validation metrics.
- `evaluate(...)` computes mean loss and accuracy for a supplied data loader.
- The final section of `06_capstone.ipynb` selects the best run by **validation accuracy** and evaluates it on `test_loader`.

For the reported test score, do not use the test set to tune the model or choose hyperparameters. The dataset pointer is the [official CIFAR-10 page](https://www.cs.toronto.edu/~kriz/cifar.html); Torchvision handles downloading it when the notebook is run.

## Repository structure

```text
fun_deep_proj/
├── 06_capstone.ipynb   # Data loading, model definitions, training, experiments, evaluation
├── figure/             # Project figures
├── report/             # Project report files
├── LICENSE             # Apache-2.0 license
├── .gitignore
└── README.md
```

## References

- [CIFAR-10 dataset — Alex Krizhevsky / University of Toronto](https://www.cs.toronto.edu/~kriz/cifar.html)
- [PyTorch documentation](https://pytorch.org/docs/stable/index.html)
- [Torchvision CIFAR-10 dataset documentation](https://pytorch.org/vision/stable/generated/torchvision.datasets.CIFAR10.html)

## License

This project is distributed under the [Apache License 2.0](./LICENSE).
