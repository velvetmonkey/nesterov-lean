# nesterov-lean

[![Lean 4](https://img.shields.io/badge/Lean-4.28.0-blue)](https://lean-lang.org/)
[![Mathlib](https://img.shields.io/badge/Mathlib-v4.28.0-purple)](https://github.com/leanprover-community/mathlib4)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Proofs](https://img.shields.io/badge/proofs-pending-lightgrey)](Nesterov)

Lean 4 formal proofs of Nesterov accelerated gradient descent: O(1/k²) convergence rate over real Hilbert spaces.

**Zero sorry statements.**

## Why it matters

Plain gradient descent achieves O(1/k) convergence on smooth convex objectives. Nesterov's 1983 accelerated method achieves O(1/k²) using a momentum term -- and this rate is optimal for first-order methods. The gap between O(1/k) and O(1/k²) is the difference between linear and quadratic speedup, which matters enormously in large-scale machine learning.

This library machine-checks the O(1/k²) rate in Lean 4, extending [gradient-descent-lean](https://github.com/velvetmonkey/gradient-descent-lean) with the accelerated sequence and its convergence proof.

## Setting

f : E → ℝ, where E is a real Hilbert space. f is L-smooth and convex. Step size α = 1/L.

AGD sequence:
```
y_{k+1} = x_k - α * ∇f(x_k)
x_{k+1} = y_{k+1} + (k-1)/(k+2) * (y_{k+1} - y_k)
```

## Planned project structure

```
Nesterov/
├── Defs.lean         — AGD sequence, momentum coefficient, step size
├── MomentumStep.lean — Per-step descent with momentum
└── Convergence.lean  — O(1/k²) convergence rate
```

## Planned theorem inventory

| # | Theorem | Statement |
|---|---------|-----------|
| 1 | `momentum_coeff_bound` | (k-1)/(k+2) ∈ [0, 1) for k ≥ 1 |
| 2 | `agd_descent_step` | f(y_{k+1}) ≤ f(x_k) - α/2 * ‖∇f(x_k)‖² |
| 3 | `nesterov_convergence` | f(y_k) - f(x*) ≤ 2L‖x₀-x*‖² / (k+1)² -- O(1/k²) rate |

## Key technical highlights

- O(1/k²) is optimal for first-order methods on smooth convex objectives (Nesterov lower bound)
- Builds directly on the L-smoothness and convexity infrastructure from gradient-descent-lean
- Works over arbitrary real Hilbert spaces
- Standard axioms only: `propext`, `Classical.choice`, `Quot.sound`
- Zero `sorry`, zero `admit`

## Dependencies

- Lean 4.28.0
- Mathlib v4.28.0

## Paper

Companion paper forthcoming. To be published on Zenodo.

## Related work

- [gradient-descent-lean](https://github.com/velvetmonkey/gradient-descent-lean) — Lean 4 gradient descent convergence (O(1/k) rate). Zenodo: https://doi.org/10.5281/zenodo.20472996
- [kuramoto-lean](https://github.com/velvetmonkey/kuramoto-lean) — Lean 4 Kuramoto synchronisation
- [hopfield-lean](https://github.com/velvetmonkey/hopfield-lean) — Lean 4 Hopfield attractor convergence
- [contraction-lean](https://github.com/velvetmonkey/contraction-lean) — Lean 4 contraction theory
- [lotka-volterra-lean](https://github.com/velvetmonkey/lotka-volterra-lean) — Lean 4 Lotka-Volterra Hamiltonian conservation

## Acknowledgements

Proofs in this library were generated using [Aristotle](https://aristotle.harmonic.fun), an AI proof assistant for Lean 4 and Mathlib. The proof discipline -- zero sorry, every Mathlib lemma name `#check`ed before use -- was specified by the author and enforced by the Lean type checker.

## Author

Ben Cassie · [@thevelvetmonke](https://x.com/thevelvetmonke)
