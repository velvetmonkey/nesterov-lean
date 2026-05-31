/-
Copyright (c) 2024. All rights reserved.
Nesterov's AGD — O(1/k²) convergence rate.

Main result: for L-smooth convex f with minimiser x★,
  f(y_k) − f(x★) ≤ 2L ‖x₀ − x★‖² / k²   for k ≥ 1.
-/
import NesterovLean.MomentumStep

noncomputable section

open scoped InnerProductSpace

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-! ## Helper: f(y_k) − f★ ≥ 0 -/

lemma f_sub_fstar_nonneg
    (f : E → ℝ) (grad_f : E → E) (L : ℝ) (x₀ x_star : E)
    (hMin : IsGlobalMin f x_star) (k : ℕ) :
    0 ≤ f (agd_y grad_f L x₀ k) - f x_star := by
  exact sub_nonneg_of_le ( hMin _ )

/-! ## Mixing identity: x_k = (1 − θ_k) • y_k + θ_k • v_k -/

/-
The AGD iterates satisfy x_k = (1 − θ_k) • y_k + θ_k • v_k for all k.
-/
lemma x_eq_combo
    (grad_f : E → E) (L : ℝ) (x₀ : E) (hL : 0 < L) (k : ℕ) :
    agd_x grad_f L x₀ k =
      (1 - theta k) • agd_y grad_f L x₀ k + theta k • v_seq grad_f L x₀ k := by
  induction' k with k ih' <;> simp_all +decide [ theta, momentumCoeff ];
  · norm_num [ two_smul ];
  · rw [ agd_x_succ, v_seq_succ, agd_y_succ ];
    rw [ ih' ];
    simp +decide [ momentumCoeff, smul_sub, smul_add, sub_smul, add_smul, div_eq_mul_inv, mul_assoc, mul_left_comm, hL.ne' ] ; ring;
    rw [ show ( k : ℝ ) * ( 2 + k : ℝ ) ⁻¹ - ( 2 + k : ℝ ) ⁻¹ = ( 2 + k : ℝ ) ⁻¹ * ( k - 1 ) by ring ] ; norm_num [ ← smul_assoc ] ; ring;
    field_simp;
    rw [ show ( 2 * ( k - 1 ) : ℝ ) / ( ( 1 + k ) * ( 2 + k ) ) = ( 2 / ( 2 + k ) ) - ( 4 / ( ( 1 + k ) * ( 2 + k ) ) ) by rw [ div_sub_div, div_eq_div_iff ] <;> ring <;> positivity ] ;
    rw [ show ( k - 1 : ℝ ) / ( L * ( 2 + k ) ) = ( k + 1 ) / ( L * ( 2 + k ) ) - 2 / ( L * ( 2 + k ) ) by ring ] ; norm_num [ sub_smul, add_smul ] ; ring;
    abel1

/-! ## Convex interpolation bound -/

/-
Using convexity and the mixing identity:
  f(x_k) − f★ ≤ (1 − θ_k)(f(y_k) − f★) + θ_k ⟪∇f(x_k), v_k − x★⟫.
-/
lemma convex_interp_bound
    (f : E → ℝ) (grad_f : E → E) (L : ℝ) (x₀ x_star : E)
    (hL : 0 < L)
    (hConvex : ConvexFirstOrder f grad_f)
    (hMin : IsGlobalMin f x_star)
    (k : ℕ) :
    f (agd_x grad_f L x₀ k) - f x_star ≤
      (1 - theta k) * (f (agd_y grad_f L x₀ k) - f x_star) +
        theta k * @inner ℝ E _ (grad_f (agd_x grad_f L x₀ k))
          (v_seq grad_f L x₀ k - x_star) := by
  obtain heq := x_eq_combo grad_f L x₀ hL k;
  have h1 := hConvex (agd_x grad_f L x₀ k) (agd_y grad_f L x₀ k)
  have h2 := hConvex (agd_x grad_f L x₀ k) x_star;
  rw [ heq ] at *;
  norm_num [ inner_sub_left, inner_sub_right, inner_smul_left, inner_smul_right ] at *;
  by_cases hk : 1 ≤ k <;> simp_all +decide [ theta ];
  · norm_num [ inner_add_left, inner_add_right, inner_smul_left, inner_smul_right ] at *;
    nlinarith [ show ( 0 : ℝ ) ≤ 2 / ( k + 1 ) by positivity, show ( 2 : ℝ ) / ( k + 1 ) ≤ 1 by rw [ div_le_iff₀ ] <;> norm_cast <;> linarith ];
  · norm_num [ two_smul ] at *;
    linarith [ hMin x₀ ]

/-! ## Key per-step inequality -/

/-
The key per-step inequality combining descent and convex interpolation:
  f(y_{k+1}) − f★ ≤ (1 − θ_k)(f(y_k) − f★)
                    + θ_k ⟪∇f(x_k), v_k − x★⟫ − (1/(2L))‖∇f(x_k)‖².
-/
lemma key_inequality
    (f : E → ℝ) (grad_f : E → E) (L : ℝ) (x₀ x_star : E)
    (hL : 0 < L)
    (hSmooth : LSmooth f grad_f L)
    (hConvex : ConvexFirstOrder f grad_f)
    (hMin : IsGlobalMin f x_star)
    (k : ℕ) :
    f (agd_y grad_f L x₀ (k + 1)) - f x_star ≤
      (1 - theta k) * (f (agd_y grad_f L x₀ k) - f x_star) +
        theta k * @inner ℝ E _ (grad_f (agd_x grad_f L x₀ k))
          (v_seq grad_f L x₀ k - x_star) -
        1 / (2 * L) * ‖grad_f (agd_x grad_f L x₀ k)‖ ^ 2 := by
  linarith [ descent_lemma f grad_f L x₀ hL hSmooth k, convex_interp_bound f grad_f L x₀ x_star hL hConvex hMin k ]

/-! ## Norm expansion for v_{k+1} -/

/-
Expanding ‖v_{k+1} − x★‖² using the v_k recurrence.
-/
lemma v_norm_sq_expand
    (grad_f : E → E) (L : ℝ) (x₀ x_star : E)
    (hL : 0 < L)
    (k : ℕ) :
    L / 2 * ‖v_seq grad_f L x₀ (k + 1) - x_star‖ ^ 2 =
      L / 2 * ‖v_seq grad_f L x₀ k - x_star‖ ^ 2 -
        ((k : ℝ) + 1) / 2 * @inner ℝ E _ (grad_f (agd_x grad_f L x₀ k))
          (v_seq grad_f L x₀ k - x_star) +
        ((k : ℝ) + 1) ^ 2 / (8 * L) * ‖grad_f (agd_x grad_f L x₀ k)‖ ^ 2 := by
  -- Expand using the properties of the inner product and the definition of $c$.
  have h_expand : ‖v_seq grad_f L x₀ (k + 1) - x_star‖^2 = ‖v_seq grad_f L x₀ k - x_star‖^2 - 2 * (k + 1) * (1 / (2 * L)) * ⟪grad_f (agd_x grad_f L x₀ k), v_seq grad_f L x₀ k - x_star⟫_ℝ + ((k + 1) / (2 * L))^2 * ‖grad_f (agd_x grad_f L x₀ k)‖^2 := by
    rw [ show v_seq grad_f L x₀ ( k + 1 ) - x_star = ( v_seq grad_f L x₀ k - x_star ) - ( ( k + 1 ) / ( 2 * L ) ) • grad_f ( agd_x grad_f L x₀ k ) by
          rw [ v_seq_succ ] ; abel_nf ] ; rw [ @norm_sub_sq ℝ ] ; ring;
    norm_num [ norm_smul, inner_smul_right ] ; ring;
    rw [ abs_of_nonneg ( by positivity ) ] ; rw [ real_inner_comm ] ; ring;
  grind +revert

/-! ## Lyapunov non-increase -/

/-
The Lyapunov function Ψ_k = (k²/4)(f(y_k) − f★) + (L/2)‖v_k − x★‖²
  is non-increasing.
-/
theorem lyapunov_nonincrease
    (f : E → ℝ) (grad_f : E → E) (L : ℝ) (x₀ x_star : E)
    (hL : 0 < L)
    (hSmooth : LSmooth f grad_f L)
    (hConvex : ConvexFirstOrder f grad_f)
    (hMin : IsGlobalMin f x_star)
    (k : ℕ) :
    lyapunov f grad_f L x₀ x_star (k + 1) ≤
      lyapunov f grad_f L x₀ x_star k := by
  unfold lyapunov;
  -- Apply the key_inequality and v_norm_sq_expand lemmas.
  have h1 := key_inequality f grad_f L x₀ x_star hL hSmooth hConvex hMin k
  have h2 := v_norm_sq_expand grad_f L x₀ x_star hL k
  have h3 := f_sub_fstar_nonneg f grad_f L x₀ x_star hMin k;
  unfold theta at *;
  field_simp at *;
  norm_num at * ; nlinarith [ mul_le_mul_of_nonneg_left h3 hL.le ]

/-! ## Lyapunov at step 0 -/

lemma lyapunov_zero
    (f : E → ℝ) (grad_f : E → E) (L : ℝ) (x₀ x_star : E) :
    lyapunov f grad_f L x₀ x_star 0 = L / 2 * ‖x₀ - x_star‖ ^ 2 := by
  unfold lyapunov; aesop;

/-! ## Theorem 3: O(1/k²) convergence rate -/

/-
**Nesterov's O(1/k²) convergence rate.**
  For L-smooth convex f with global minimiser x★ and AGD step size 1/L:
    f(y_k) − f(x★) ≤ 2L ‖x₀ − x★‖² / k²   for every k ≥ 1.
-/
theorem convergence_rate
    (f : E → ℝ) (grad_f : E → E) (L : ℝ) (x₀ x_star : E)
    (hL : 0 < L)
    (hSmooth : LSmooth f grad_f L)
    (hConvex : ConvexFirstOrder f grad_f)
    (hMin : IsGlobalMin f x_star)
    {k : ℕ} (hk : 1 ≤ k) :
    f (agd_y grad_f L x₀ k) - f x_star ≤
      2 * L * ‖x₀ - x_star‖ ^ 2 / k ^ 2 := by
  have := lyapunov_nonincrease f grad_f L x₀ x_star hL hSmooth hConvex hMin;
  -- By induction on $k$, we can show that $\Psi_k \leq \Psi_0$ for all $k \geq 0$.
  have h_ind : ∀ k, lyapunov f grad_f L x₀ x_star k ≤ lyapunov f grad_f L x₀ x_star 0 := by
    exact fun k => Nat.recOn k le_rfl fun n ih => le_trans ( this n ) ih;
  -- By definition of lyapunov, we have:
  have h_lyapunov_def : lyapunov f grad_f L x₀ x_star k = (k : ℝ) ^ 2 / 4 * (f (agd_y grad_f L x₀ k) - f x_star) + L / 2 * ‖v_seq grad_f L x₀ k - x_star‖ ^ 2 := by
    rfl;
  have := h_ind k; rw [ lyapunov_zero ] at this; rw [ le_div_iff₀ ( by positivity ) ] ; nlinarith [ show ( 0 : ℝ ) ≤ L / 2 * ‖v_seq grad_f L x₀ k - x_star‖ ^ 2 by positivity ] ;

end