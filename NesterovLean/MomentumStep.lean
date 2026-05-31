/-
Copyright (c) 2024. All rights reserved.
Nesterov's AGD — Momentum coefficient bounds & descent lemma.
-/
import NesterovLean.Defs

noncomputable section

open scoped InnerProductSpace

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-! ## Theorem 1: momentum coefficient ∈ [0, 1) for k ≥ 1 -/

theorem momentumCoeff_nonneg {k : ℕ} (hk : 1 ≤ k) : 0 ≤ momentumCoeff k := by
  exact div_nonneg ( sub_nonneg_of_le ( mod_cast hk ) ) ( by positivity )

theorem momentumCoeff_lt_one (k : ℕ) : momentumCoeff k < 1 := by
  exact div_lt_one ( by linarith ) |>.2 ( by linarith )

/-- The momentum coefficient (k−1)/(k+2) lies in [0, 1) for k ≥ 1. -/
theorem momentumCoeff_mem_Ico {k : ℕ} (hk : 1 ≤ k) :
    momentumCoeff k ∈ Set.Ico (0 : ℝ) 1 := by
  exact ⟨momentumCoeff_nonneg hk, momentumCoeff_lt_one k⟩

/-! ## Theorem 2: descent lemma -/

/-
**Descent lemma**: one gradient step with step size 1/L decreases f by at least
  (1/(2L))‖∇f(x_k)‖². That is,
    f(y_{k+1}) ≤ f(x_k) − (1/(2L))‖∇f(x_k)‖².
-/
theorem descent_lemma
    (f : E → ℝ) (grad_f : E → E) (L : ℝ) (x₀ : E)
    (hL : 0 < L)
    (hSmooth : LSmooth f grad_f L)
    (k : ℕ) :
    f (agd_y grad_f L x₀ (k + 1)) ≤
      f (agd_x grad_f L x₀ k) -
        1 / (2 * L) * ‖grad_f (agd_x grad_f L x₀ k)‖ ^ 2 := by
  convert hSmooth ( agd_x grad_f L x₀ k ) ( agd_y grad_f L x₀ ( k + 1 ) ) using 1 ; norm_num [ agd_y_succ ] ; ring;
  simp +decide [ norm_smul, inner_smul_right, hL.ne' ] ; ring;
  grind

end