# Constant Source 

## Description
This describes the first model, which assumes there is a constant source and that the temperature is constant. We then assume electrons follow a Boltzmann distribution. The network is only solving for the electric potential and ion density

## Prerequisites
* **Python 3.10**
* NVIDIA GPUs with CUDA 12 drivers

## Environment Setup and Code Execution

### 1. Create Virtual Environment
Ensure you are using Python 3.10 to create the environment

'''bash
python3.10 -m venv env
source env/bin/activate
'''

### 2. Install Dependencies
We install DeepXDE and all dependencies together. This ensures version compatibility.

'''bash
pip install --upgrade pip
pip install -r requirements.txt
'''

### 3. Modify the SciPy Routines
Two SciPy files need to be modified inside the environment to enable SSBroyden

'''bash
cp _optimize.py env/lib/python3.10/site-packages/scipy/optimize/
cp _minimize.py env/lib/python3.10/site-packages/scipy/optimize/
'''

### 4. Running the Code
Before running, update the "save_path" variable in both sheathsource.py and Plotsheathsource.py to reflect desired output directories.

Run the script:

'''bash
python sheathsource.py
'''

Plot results:

'''bash
python Plotsheashsource.py
'''