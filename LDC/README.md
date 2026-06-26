# Lid Driven Cavity

## Prerequisites
*   **Python 3.12**
*   NVIDIA GPUs with CUDA 12 drivers

## Environmental Setup and Code Execution

### 1. Create Virtual Environment
Ensure you are using Python 3.12 to create the environment.

```bash
python3.12 -m venv torch_env
source torch_env/bin/activate
```

### 2. Install Dependencies
We install PyTorch and all dependencies together, pointing to the CUDA 12.4 wheel repository. This ensures all versions are compatible.

```bash
pip install --upgrade pip
pip install torch torchvision torchaudio matplotlib pytorch-optimizer scikit-optimize tqdm
```

### 3. Running the Code
```bash
python LDC_module_square.py
```