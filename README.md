# IMFAS: Implicit Multi-Fidelity Algorithm Selection 


Automatically selecting the best performing algorithm for a given dataset or ranking
multiple of them by their expected performance supports users in developing new machine
learning applications. Most approaches for this problem rely on dataset meta-features
and landmarking performances to capture the salient topology of the datasets and those
topologies that the algorithms attend to. Landmarking usually exploits cheap algorithms
not necessarily in the pool of candidate algorithms to get inexpensive approximations of the topology. While somewhat indicative, hand-crafted dataset meta-features and landmarks
are likely insufficient descriptors, strongly depending on the alignment of the geometries
the landmarks and candidates search for. We propose IMFAS, a method to exploit multi-
fidelity landmarking information directly from the candidate algorithms in the form of
non-parametrically non-myopic meta-learned learning curves via LSTMs in a few-shot setting during testing. Using this mechanism, IMFAS jointly learns a dataset's topology and algorithms' inductive biases, without the need to expensively train them to convergence. Our approach produces informative landmarks, easily enriched by arbitrary meta-features at a low computational cost, capable of producing the desired ranking using cheaper fidelities. We additionally show that IMFAS is able to beat Successive Halving with at most 50% of the fidelity sequence during test time.


## Preparation

If you want to use LCBench, you have to download the dataset first:

```bash
bash scripts/download_lcbench.sh
```

## Installation
```bash
git clone https://github.com/automl/IMFAS.git
cd IMFAS
conda create -n imfas python=3.9.7
conda activate imfas

# Install for usage
pip install -e .

# Install for development
make install-dev
```

Documentation at https://automl.github.io/IMFAS/main.


## Start experiments

An example command is fiven below
```bash
python main.py '+experiment=lcbench'
```

The project extensively uses [hydra](https://hydra.cc/docs/intro/) for configurations and [Weights and Biases](https://wandb.ai/site) for tracking experiments. Please set-up the project and account on this and then update ```configs/base.yaml``` with the ```entity``` and ```project_name``` fields for running full tests. 

The complete tests can be run using

```bash
bash scripts/imfas_tests.sh
```
