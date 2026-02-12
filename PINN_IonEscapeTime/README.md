# JONTA ion guiding center module

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
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124
```

### 3. Modify SciPy routines
Two SciPy files need to be modified inside the environment to allow SSBroyden to be used
```bash
cp _optimize.py torch_env/lib/python3.12/site-packages/scipy/optimize/.
cp _minimize.py torch_env/lib/python3.12/site-packages/scipy/optimize/.
```

### 4. Running the Code
Before running the simulation, create the required output directories inside the project folder:

```bash
mkdir figures
mkdir model
```

Run the script:

```bash
python EscapeTime.py
```