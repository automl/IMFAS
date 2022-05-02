# AlgoSelectionMF

## Installation

```
git clone https://github.com/automl-private/AlgoSelectionMF/AlgoSelectionMF.git
cd AlgoSelectionMF
conda create -n mf_gravitas python=3.8
conda activate mf_gravitas

# Install for usage
pip install .

# Install for development
make install-dev
```

Documentaiton at https://automl-private/AlgoSelectionMF.github.io/AlgoSelectionMF/main

## Minimal Example

## Hydra Smac Sweeper Plugin:

since hydra smac sweeper is still experimental, please follow this structure:

# TODO hydra smac sweeper is not public yet: rerunning the experiments will require this though!

```
# install hydra smac sweeper
conda activate <algoselectionmf_env>
cd /path/to/hydra_smac_sweeper
git clone https://github.com/automl-private/hydra-smac-sweeper.git
pip install -e .

# install smac
conda install gxx_linux-64 gcc_linux-64 swig
pip install smac

```
