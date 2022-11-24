# margret: Implicit Multi-Fidelity Algorithm Selection 

## Preparation

If you want to use LCBench, you have to download the dataset first:

```bash
bash scripts/download_lcbench.sh
```

## Installation
```bash
cd margret
conda create -n margret python=3.9.7
conda activate margret

# Install for usage
pip install -e .

# Install for development
make install-dev
```


## Start experiments

An example command is fiven below
```bash
python main.py '+experiment=margret_h'+model.model_opts=['reduce','pe_g','d_meta_guided']
```

The project extensively uses [hydra](https://hydra.cc/docs/intro/) for configurations and [Weights and Biases](https://wandb.ai/site) for tracking experiments. Please set-up the project and account on this and then update ```configs/base.yaml``` with the ```entity``` and ```project_name``` fields for running full tests. 

The complete tests can be run using

```bash
bash scripts/margret_tests.sh
```
