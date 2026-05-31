/-
Copyright (c) 2024. All rights reserved.
Nesterov's Accelerated Gradient Descent — Definitions

Setting: f : E → ℝ on a real Hilbert space E,
  f is L-smooth (quadratic upper bound) and convex (first-order condition).
  Step size α = 1/L.

AGD sequence:
  y_{k+1} = x_k − (1/L) • ∇f(x_k)
  x_{k+1} = y_{k+1} + momentumCoeff k • (y_{k+1} − y_k)

where momentumCoeff k = (k − 1) / (k + 2).
-/
import Mathlib

noncomputable section

open scoped InnerProductSpace

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-! ## Core hypotheses -/

/-- L-smoothness via the quadratic upper bound:
  f(y) ≤ f(x) + ⟪∇f(x), y − x⟫ + (L/2)‖y − x‖². -/
def LSmooth (f : E → ℝ) (grad_f : E → E) (L : ℝ) : Prop :=
  ∀ x y : E, f y ≤ f x + @inner ℝ E _ (grad_f x) (y - x) + L / 2 * ‖y - x‖ ^ 2

/-- First-order convexity condition:
  f(y) ≥ f(x) + ⟪∇f(x), y − x⟫. -/
def ConvexFirstOrder (f : E → ℝ) (grad_f : E → E) : Prop :=
  ∀ x y : E, f y ≥ f x + @inner ℝ E _ (grad_f x) (y - x)

/-- Global minimiser. -/
def IsGlobalMin (f : E → ℝ) (x_star : E) : Prop :=
  ∀ x : E, f x_star ≤ f x

/-! ## AGD iterates -/

/-- The momentum coefficient β_k = (k − 1) / (k + 2). -/
def momentumCoeff (k : ℕ) : ℝ := ((k : ℝ) - 1) / ((k : ℝ) + 2)

/-- Joint AGD state (x_k, y_k). -/
def agdState (grad_f : E → E) (L : ℝ) (x₀ : E) : ℕ → E × E
  | 0 => (x₀, x₀)
  | k + 1 =>
    let s := agdState grad_f L x₀ k
    let yk1 := s.1 - (1 / L) • grad_f s.1
    let xk1 := yk1 + momentumCoeff k • (yk1 - s.2)
    (xk1, yk1)

/-- The x_k sequence of AGD. -/
def agd_x (grad_f : E → E) (L : ℝ) (x₀ : E) (k : ℕ) : E :=
  (agdState grad_f L x₀ k).1

/-- The y_k sequence of AGD. -/
def agd_y (grad_f : E → E) (L : ℝ) (x₀ : E) (k : ℕ) : E :=
  (agdState grad_f L x₀ k).2

/-! ## Basic unfolding lemmas -/

@[simp] lemma agd_x_zero (grad_f : E → E) (L : ℝ) (x₀ : E) :
    agd_x grad_f L x₀ 0 = x₀ := rfl

@[simp] lemma agd_y_zero (grad_f : E → E) (L : ℝ) (x₀ : E) :
    agd_y grad_f L x₀ 0 = x₀ := rfl

lemma agd_y_succ (grad_f : E → E) (L : ℝ) (x₀ : E) (k : ℕ) :
    agd_y grad_f L x₀ (k + 1) =
      agd_x grad_f L x₀ k - (1 / L) • grad_f (agd_x grad_f L x₀ k) := rfl

lemma agd_x_succ (grad_f : E → E) (L : ℝ) (x₀ : E) (k : ℕ) :
    agd_x grad_f L x₀ (k + 1) =
      agd_y grad_f L x₀ (k + 1) +
        momentumCoeff k • (agd_y grad_f L x₀ (k + 1) - agd_y grad_f L x₀ k) := rfl

/-! ## Auxiliary sequences for convergence proof -/

/-- θ_k = 2 / (k + 1), the coupling parameter. -/
def theta (k : ℕ) : ℝ := 2 / ((k : ℝ) + 1)

/-- Auxiliary sequence v_k for the Lyapunov argument.
  v₀ = x₀, v_{k+1} = v_k − ((k+1)/(2L)) • ∇f(x_k). -/
def v_seq (grad_f : E → E) (L : ℝ) (x₀ : E) : ℕ → E
  | 0 => x₀
  | k + 1 => v_seq grad_f L x₀ k -
      (((k : ℝ) + 1) / (2 * L)) • grad_f (agd_x grad_f L x₀ k)

@[simp] lemma v_seq_zero (grad_f : E → E) (L : ℝ) (x₀ : E) :
    v_seq grad_f L x₀ 0 = x₀ := rfl

lemma v_seq_succ (grad_f : E → E) (L : ℝ) (x₀ : E) (k : ℕ) :
    v_seq grad_f L x₀ (k + 1) =
      v_seq grad_f L x₀ k -
        (((k : ℝ) + 1) / (2 * L)) • grad_f (agd_x grad_f L x₀ k) := rfl

/-! ## Lyapunov function -/

/-- The Lyapunov / potential function:
  Ψ_k = (k²/4)(f(y_k) − f★) + (L/2)‖v_k − x★‖². -/
def lyapunov (f : E → ℝ) (grad_f : E → E) (L : ℝ) (x₀ x_star : E) (k : ℕ) : ℝ :=
  (k : ℝ) ^ 2 / 4 * (f (agd_y grad_f L x₀ k) - f x_star) +
    L / 2 * ‖v_seq grad_f L x₀ k - x_star‖ ^ 2

end
