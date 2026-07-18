# nesterov-lean

[![thread](https://img.shields.io/badge/%F0%9F%A7%B5-how%20it%20works-1DA1F2)](https://x.com/thevelvetmonke)
[![Lean 4](https://img.shields.io/badge/Lean-4.28.0-blue)](https://lean-lang.org/)
[![Mathlib](https://img.shields.io/badge/Mathlib-v4.28.0-purple)](https://github.com/leanprover-community/mathlib4)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Proofs](https://img.shields.io/badge/proofs-proven%20%2F%200%20sorry-brightgreen)](NesterovLean)
[![Paper](https://img.shields.io/badge/Zenodo-20474481-blue)](https://zenodo.org/records/20474481)

Lean 4 formal proofs of Nesterov accelerated gradient descent: O(1/k²) convergence rate via Lyapunov potential argument.

**Zero sorry statements.** Zero new axioms.

## What this is, and why it matters

This library formalizes an accelerated gradient sequence and its O(1/k^2) convergence rate. Its headline theorem, `convergence_rate`, bounds the function-value gap at `y_k` by `2*L*||x0-xstar||^2/k^2` for every positive iteration index.

The machine-checked proof uses a Lyapunov potential. A mixing identity relates the accelerated variables, convexity and smoothness give the per-step inequality, and an auxiliary sequence makes the troublesome gradient and inner-product terms cancel. Nonincrease of the potential then exposes the quadratic denominator.

The smoothness and first-order convexity inequalities are hypotheses stated for a supplied gradient map, and a global minimizer is also supplied. The library does not prove that the map is a derivative of a concrete function, establish the assumptions for an application, or account for numerical and finite-precision effects.

## Background and motivation

Plain gradient descent achieves O(1/k) convergence on smooth convex objectives. Nesterov's 1983 accelerated method achieves O(1/k²) using a momentum term -- and this rate is optimal for first-order methods. The gap between O(1/k) and O(1/k²) is the difference between linear and quadratic speedup, which matters enormously in large-scale machine learning.

This library machine-checks the O(1/k²) rate in Lean 4 via a Lyapunov potential argument, extending [gradient-descent-lean](https://github.com/velvetmonkey/gradient-descent-lean) with the accelerated sequence and its convergence proof.

## Setting

f : E → ℝ, L-smooth and convex. Step size α = 1/L. Momentum coefficient βₖ = (k−1)/(k+2).

AGD sequences:
```
y_{k+1} = x_k - (1/L) * ∇f(x_k)
x_{k+1} = y_{k+1} + β_k * (y_{k+1} - y_k)
```

Lyapunov potential: Ψₖ = (k²/4)(f(yₖ) − f★) + (L/2)‖vₖ − x★‖²

## Project structure

```
NesterovLean/
├── Defs.lean          — LSmooth, ConvexFirstOrder, IsGlobalMin, momentumCoeff, agdState,
│                        agd_x, agd_y, theta, v_seq, lyapunov (Ψ_k)
├── MomentumStep.lean  — β_k ∈ [0,1), descent lemma
└── Convergence.lean   — Lyapunov non-increase, O(1/k²) rate
NesterovLean.lean      — Root module
```

## Theorem inventory

### Layer 1 — Momentum properties

| # | Name | Statement |
|---|------|-----------|
| 1 | `momentumCoeff_nonneg` | βₖ ≥ 0 for k ≥ 1 |
| 2 | `momentumCoeff_lt_one` | βₖ < 1 for k ≥ 1 |
| 3 | `momentumCoeff_mem_Ico` | βₖ ∈ [0, 1) for k ≥ 1 |
| 4 | `descent_lemma` | f(y_{k+1}) ≤ f(x_k) − (1/2L)‖∇f(x_k)‖² |

### Layer 2 — Convergence

| # | Name | Statement |
|---|------|-----------|
| 5 | `x_eq_combo` | xₖ = (1−θₖ)·yₖ + θₖ·vₖ (mixing identity) |
| 6 | `convex_interp_bound` | Convex interpolation bound via mixing identity |
| 7 | `key_inequality` | Per-step inequality: descent + convexity combined |
| 8 | `v_norm_sq_expand` | Norm-squared expansion for auxiliary sequence |
| 9 | `lyapunov_nonincrease` | Ψₖ₊₁ ≤ Ψₖ |
| 10 | `convergence_rate` | f(yₖ) − f(x★) ≤ 2L‖x₀−x★‖² / k² for k ≥ 1 |

## Key technical highlights

- Convergence proved via Lyapunov potential Ψₖ = (k²/4)(f(yₖ)−f★) + (L/2)‖vₖ−x★‖²
- `lyapunov_nonincrease` is the heart: inner product and gradient terms cancel exactly
- Bound is 2L/k² (natural from the Lyapunov analysis); same O(1/k²) asymptotic as the classical 2L/(k+1)² form
- O(1/k²) is optimal for first-order methods on smooth convex objectives
- Standard axioms only: `propext`, `Classical.choice`, `Quot.sound`
- Zero `sorry`, zero `admit`

## Dependencies

- Lean 4.28.0
- Mathlib v4.28.0

## Paper

**nesterov-lean: Formal Proofs of Nesterov Accelerated Gradient Descent in Lean 4**  
Ben Cassie (2026). Zenodo.  
https://zenodo.org/records/20474481

## Cite

Cassie, B. (2026). *nesterov-lean: Formal Proofs of Nesterov Accelerated Gradient Descent in Lean 4*. Zenodo. https://zenodo.org/records/20474481

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
## Part of the Lean proof corpus

One of a family of small, machine-checked Lean 4 developments. Index: [velvetmonkey/lean](https://github.com/velvetmonkey/lean) ([live index](https://velvetmonkey.github.io/lean)).
