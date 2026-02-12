# JAX Particle Pusher (JONTA Ion GC)

## Prerequisites
*   **Python 3.12**
*   NVIDIA GPUs with CUDA 12 drivers

## Environment Setup

### 1. Create Virtual Environment
Ensure you are using Python 3.12 to create the environment.

```bash
python3.12 -m venv jax_env
source jax_env/bin/activate

### 2. Install Dependencies
Once the environment is activated, upgrade pip and install JAX with CUDA support.

```bash
pip install --upgrade pip
pip install "jax[cuda12]"
pip install -r requirements.txt

### 3. Running the Code
Before running the simulation, create the required output directories inside the project folder:

```bash
mkdir figures
mkdir data

Run the main particle pushing script:

```bash
python JONTA_IonGC.py
