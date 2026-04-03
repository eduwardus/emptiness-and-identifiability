# emptiness-and-identifiability
Formalizing the transition from non-identifiability to dominance in latent  graph models. A mathematical exploration of how structure emerges without  inherent identity.
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19376217.svg)](https://doi.org/10.5281/zenodo.19376217)

## Overview

This repository accompanies the paper:

> **Identifiability Without Uniqueness: Emergence of Dominance in Generative Graph Models**

We study structural non-identifiability in latent graph models and show that a dominant solution emerges in an intermediate regime where Euclidean and radial contributions compete.

## Key findings

- Identifiability peaks at λ* ≈ 0.4–0.5
- The mixed regime maximizes the likelihood gap
- Degeneracy collapses as dominance emerges
- Results are stable across N=100,200,500

## Reproducibility

See `src/` for all experiment scripts. Run in order:

1. `experiment_M4.py`
2. `experiment_M4_N200.py`
3. `experiment_M4_N500.py`
4. `baseline_comparison.py`
5. `convergence_test.py`
6. `final_analysis.py`

## License

MIT
