# Constant Source

## Description
This describes the first model, which assumes there is a constant source and that the temperature is constant. We then assume electrons follow a Boltzmann distribution. The network is only solving for the electric potential and ion density.

## Prerequisites
* **Python 3.10 or 3.11** (tested with Python 3.11.4)
* NVIDIA GPU with a driver compatible with CUDA 12.8

## Environment Setup and Code Execution

### 1. Create Virtual Environment
Create an isolated environment so the modified SciPy routines do not affect other Python environments.

```bash
python3.11 -m venv env
source env/bin/activate
```

If your system uses Python 3.10, replace `python3.11` with `python3.10`.

### 2. Install Dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The requirements file installs DeepXDE 1.10.0, NumPy 1.24.4, SciPy 1.12.0, Matplotlib 3.6.3, and PyTorch 2.7.0 with CUDA 12.8.

### 3. Install the SSBroyden SciPy Routines
The repository includes modified `_optimize.py` and `_minimize.py` files required for the SSBroyden optimizer. Copy them into the SciPy installation inside the active virtual environment:

```bash
SCIPY_OPT=$(python -c "import scipy.optimize, os; print(os.path.dirname(scipy.optimize.__file__))")
cp _optimize.py "$SCIPY_OPT/_optimize.py"
cp _minimize.py "$SCIPY_OPT/_minimize.py"
```

This avoids hard-coding a Python-version-specific `site-packages` path.

### 4. Run the Model
No path editing is required. Outputs are created automatically in the local `models/` directory.

```bash
python sheathsource.py
```

Training uses 5000 Adam iterations followed by 10 SSBroyden rounds. 

### 5. Plot Results
The plotting script automatically finds the latest `model_SSBroyden_*.pt` checkpoint and saves figures in `models/figures/`.

```bash
python Plotsheathsource.py
```

The full plotting scan is huge: massratio values × 10 temperature-ratio values × 50 Knudsen values = 20,000 sheath-entrance cases. This can take a long time. Progress is printed every 100 cases so the calculation does not appear frozen.

## Output Layout

```text
models/
├── data/
│   ├── solution0_fine.dat
│   ├── solution0_coarse.dat
│   ├── loss.dat
│   ├── train.dat
│   └── test.dat
├── figures/
├── model.pt-5000.pt
├── model_SSBroyden_1.pt
├── ...
└── model_SSBroyden_10.pt
```

## Notes
* The modified SciPy files should only be installed inside the dedicated virtual environment.
* `sheathsource.py` forces the DeepXDE backend to PyTorch, so no PaddlePaddle installation is needed.
* The plotting script uses a non-interactive Matplotlib backend so it can run on headless/HPC nodes.

