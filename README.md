# Differentiable Random Partition Models
This is the official codebae for the Neurips 2023 paper [Differentiable Random Partition Models](https://arxiv.org/abs/2305.16841).

In this paper, we introduce the *Differentiable Random Partition Model (DRPM)*, a differentiable relaxation to incorporate RPMs into gradient-based learning frameworks.

## Two-stage Process
![Two-stage process](drpm/twostage.png)

## Installation

To install the differentiable random partition module and run experiments from the paper *Differentiable Random Partition Models*, make sure to add the path to the repository for the MVHG distribution to line 4 in `setup.py`.
After that, you can install the drpm using 
```
pip install .[pt]
```
In order to run the multi-task experiment or the clustering experiment, please consult the `README.md` files in their respective experiment folder under `experiments/`.
