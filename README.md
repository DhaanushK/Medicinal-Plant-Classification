# Medicinal Plant Image Classifier

A reproducible image-classification project that combines a pretrained Vision Transformer (ViT) with a convolutional neural network (CNN). The two branches process the same RGB image independently; a learned fusion layer combines their class scores.

## What is included

- A training pipeline with a stratified validation split, training-only augmentation, class-weighted loss, early stopping, and a fixed random seed.
- Reproducible dataset analysis and plots for class balance, sample images, image dimensions, and RGB statistics.
- A final held-out test evaluation with accuracy, per-class precision/recall/F1, and a confusion matrix in `artifacts/metrics.json`.
- A checkpoint containing model weights, class names, preprocessing settings, and the pretrained model ID.
- A Streamlit app for uploading an image and viewing the top three predictions.

## Requirements

- Python 3.10–3.12
- Internet access on first run to download the Hugging Face dataset and pretrained ViT weights
- A CUDA GPU is recommended for training. CPU training is supported but may take a long time.
- Enough free disk space for the dataset cache, preprocessed dataset, and model weights. The Hugging Face cache can be redirected with `MEDICINAL_PLANT_CACHE`.

Create an isolated environment from the repository root:

```cmd
py -3.12 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install torch torchvision
python -m pip install -r requirements-lock.txt
```

PyTorch's standard package index installs the CPU build, which is enough to run inference but can make training very slow. If you have an NVIDIA GPU, use the [official PyTorch install selector](https://pytorch.org/get-started/locally/) to choose **Windows + Pip + Python + the supported CUDA platform**, and run its command in the virtual environment before installing `requirements-lock.txt`. The lock file pins the other dependencies and leaves PyTorch for this hardware-specific selection. Confirm GPU access with `python -c "import torch; print(torch.cuda.is_available())"`.

## Train and evaluate

If your Windows user cache on `C:` is short on space, point Hugging Face's caches to a folder on a drive with room before training. For example, in Command Prompt:

```cmd
set MEDICINAL_PLANT_CACHE=D:\College Projects\HF-cache\Medicinal-Plant-Classification
```

The training code places the Hugging Face downloads, dataset cache, and Xet cache under this folder. Set the variable again in each new Command Prompt session before running the project.

```cmd
python -m src.train
```

Generate dataset summaries and plots with:

```cmd
python -m src.eda
```

The JSON summary and PNG charts are written under `artifacts/eda/`.

The defaults use five maximum epochs, batch size 4, seed 42, and a 15% validation split from the dataset's training split. The dataset's test split is used only once, after selecting the checkpoint by validation accuracy. Training stops early if validation accuracy does not improve. The best validation checkpoint and final test metrics are written to the ignored `artifacts/` directory.

You can adjust training settings, for example:

```cmd
python -m src.train --epochs 8 --batch-size 4 --seed 42
```

The actual split sizes and class names are read from the dataset and saved with the run metrics. This avoids relying on the inconsistent sample totals in the original report.

## Run the portfolio demo

After training creates `artifacts/medicinal_plant_hybrid.pt`:

```cmd
streamlit run app.py
```

Upload a JPG or PNG image to see the top predictions and confidence scores. The demo is an educational prototype; it does not establish that a plant is safe to consume or use medicinally.

## Verify installation

```cmd
python -m compileall src app.py
python -m src.train --help
```

The training results described in the April 2025 report are historical results, not results produced by this cleaned-up pipeline. Re-run training to generate comparable metrics. Because the model architecture and evaluation process have been made explicit and the test split is no longer evaluated after every epoch, the previous 93.16% figure should not be presented as verified for this version unless a new run reproduces it.

## Project structure

```text
app.py                 Streamlit upload and prediction demo
src/data.py            Dataset loading and preprocessing
src/eda.py             Dataset counts and visual exploration
src/model.py           CNN + ViT model and fusion head
src/predict.py         Checkpoint loading and image prediction
src/train.py           Training, validation, checkpointing, evaluation
requirements.txt       Supported Python dependencies
requirements-lock.txt  Exact Python 3.12 dependency versions (PyTorch selected separately)
```

## License

MIT. See [LICENSE](LICENSE).
