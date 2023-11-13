# Differentiable Random Partition Models
This is the official codebae for the Neurips 2023 paper [Differentiable Random Partition Models](https://arxiv.org/abs/2305.16841).

The code and repository is still work in progress.

The code for the experiment "partitioning of generative factors" is not yet ready to be released, but we are working on it. Apologies for that!

## Two-stage Process
In this paper, we introduce the *Differentiable Random Partition Model (DRPM)*, a differentiable relaxation to incorporate RPMs into gradient-based learning frameworks.

The DRPM is based on the following two-stage process:

![Two-stage process](files/twostage.png)

## Installation

To install the differentiable random partition module and run experiments from the paper *Differentiable Random Partition Models*, make sure to add the path to the repository for the MVHG distribution to line 4 in `setup.py`.

The code for the MVHG model can be found [here](https://github.com/thomassutter/mvhg).

After that, you can install the drpm using 
```
pip install .[pt]
```
In order to run the multi-task experiment or the clustering experiment, please consult the `README.md` files in their respective experiment folder under `experiments/`.

## Citation
If you use our model in your own, please cite us using the following citation
```
@inproceedings{sutterryser2023drpm,
  title={Differentiable Random Partition Models},
  author={Sutter, Thomas M and Ryser, Alain and Liebeskind, Joram and Vogt, Julia E},
  year = {2023},
  booktitle = {Advances in Neural Information Processing Systems},
}
```

and also

```
@inproceedings{sutter2023mvhg,
  title={Learning Group Importance using the Differentiable Hypergeometric Distribution},
  author={Sutter, Thomas M and Manduchi, Laura and Ryser, Alain and Vogt, Julia E},
  year = {2023},
  booktitle = {International Conference on Learning Representations},
}
```

## Questions
For any questions or requests, please reach out to:

[Thomas Sutter](https://thomassutter.github.io/) [(thomas.sutter@inf.ethz.ch)](mailto:thomas.sutter@inf.ethz.ch)


[Alain Ryser](https://mds.inf.ethz.ch/team/detail/alain-ryser) [(alain.ryser@inf.ethz.ch)](mailto:alain.ryser@inf.ethz.ch)

