# Offline Reinforcement Learning at Multiple Frequencies
Code for reproducing the results of **Offline RL at Multiple Frequencies** ([arXiv](), [website](https://sites.google.com/stanford.edu/adaptive-nstep-returns/)). 

Offline data was collected from replay buffers during training with the [DAU repository](https://github.com/ctallec/continuous-rl) or this repository and can be downloaded [here]().

This repository builds off of [Young Geng's implementation of CQL](https://github.com/young-geng/CQL).

## Installation

1. Install and use the included Ananconda environment
```
$ conda env create -f environment.yml
$ source activate 
```

2. Add this repo directory to your `PYTHONPATH` environment variable.
```
export PYTHONPATH="$PYTHONPATH:$(pwd)"
```

## Run Experiments
We provide example run scripts for [pendulum](), [door](), and [kitchen]().

For example, to run the **adaptive n-step** algorithm:
```
./run_kitchen.sh 120 101 .99 500
```

To run the **naive mixing** baseline:
```
./run_kitchen.sh 0 101 .99 500
```

The **max n-step** baseline can be run by setting the `all_same_N` flag to `True` and the **individual training** baselines can be run by commenting out the data loaders.


## Experiment Tracking with Weights and Biases
By default, the scripts log to [W&B](https://wandb.ai/site). To log to W&B, set your W&B API key environment variable:
```
export WANDB_API_KEY='YOUR W&B API KEY HERE'
```

