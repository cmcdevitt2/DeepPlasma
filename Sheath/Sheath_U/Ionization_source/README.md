# Ionization Source

## Description

This describes the second model, which assumes there is an ionization source with a constant neutral population and that the temperature is constant. We then assume electrons follow a Boltzmann distribution. The network also includes a trainable parameter for the electron wall density, so the outputs are potential, ion density, electron velocity, and electron density at the wall.

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

The requirements file installs NumPy 1.26.4, SciPy 1.12.0, Matplotlib 3.6.3, PyTorch 2.7.0, pytorch-optimizer 3.10.1, and tqdm 4.66.2.

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
python Ionization_source.py
```

Training uses 500 SOAP iterations followed by 5 SSBroyden rounds. Internally, the SSBroyden optimizer uses the `SSBroyden2` SciPy variant.

### 5. Plot Results

The plotting script loads `model_final.pt` and saves figures in `models/figures/`.

```bash
python PlotIonization_source.py
```

## Output Layout

```text
models/
├── figures/
├── loss_history.txt
├── test_history.txt
├── model_final.pt
└── model*.pt
```

## Notes

* The modified SciPy files should only be installed inside the dedicated virtual environment.
* NumPy 1.26.4 is required because `pytorch_optimizer==3.10.1` requires NumPy newer than 1.24.4, while SciPy 1.12.0 requires NumPy below 1.29.
* The code automatically uses CUDA when an NVIDIA GPU is available.
