# IB-VONNs: an automatic framework to learn internal variables and their dynamics

We propose a machine learning approach to learn internal variables and the evolution equations from data, in a way that is consistent with the principles of statistical mechanics and thermodynamics. The proposed approach leverages the following techniques: (i) the information bottleneck (IB) method to ensure that the learned internal variables are functions of the microstates and are capable of capturing the salient feature of the microscopic distribution; (ii) conditional normalizing flows (CNFs) to represent arbitrary probability distributions of the microscopic states as functions of the state variables; and (iii) Variational Onsager Neural Networks (VONNs) to guarantee thermodynamic consistency of the learned evolution equations and that the state variables are sufficient to predict the future state of the system in the absence of memory effects. The proposed framework is tested on two problems of colloidal systems, governed at the microscale by overdamped Langevin dynamics. The first one is a prototypical model for a colloidal particle in an optical trap, and the second problem is a one-dimensional phase-transforming system.

## Required Packages

- [NumPy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/)
- [JAX](https://github.com/google/jax)
- [Flax](https://github.com/google/flax)
- [Optax](https://github.com/deepmind/optax)
- [tqdm](https://github.com/tqdm/tqdm)


## Citation
Please cite us if you find our work useful for your research:

```bibtex
@article{QIU2025106211,
title = {Bridging statistical mechanics and thermodynamics away from equilibrium: A data-driven approach for learning internal variables and their dynamics},
journal = {Journal of the Mechanics and Physics of Solids},
volume = {203},
pages = {106211},
year = {2025},
issn = {0022-5096},
doi = {https://doi.org/10.1016/j.jmps.2025.106211},
url = {https://www.sciencedirect.com/science/article/pii/S0022509625001875},
author = {Weilun Qiu and Shenglin Huang and Celia Reina}
}
