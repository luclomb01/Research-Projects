# Valutazione pointwise per il caso di taglio semplice.
#   - Energia Ω(F, b_L) = W_mech(F) + (1/(2 μ0 μr J)) b_L · (C b_L),  C = F^T F
#   - Variabile di input per il campo: B spaziale, con b_L = J F^{-1} B.
#   - Output: trazioni superficiali (lagrangiane e spaziali) e campi esterni
#             coerenti con le condizioni di continuità elettromagnetica.

from jax import config
config.update("jax_enable_x64", True)  # più precisione

import jax
import jax.numpy as jnp
import numpy as np
from dataclasses import dataclass

# ---------------- Parametri ----------------
@dataclass
class Params:
    G: float   = 4e6            # shear modulus [Pa]
    K: float   = 8e8            # bulk modulus [Pa]
    mu0: float = 4*jnp.pi*1e-7  # vacuum permeability [H/m]
    mu_r: float = 1.5           # relative permeability [-]

# ---------------- Kinematica ----------------

# ---------------- Energia ----------------
def _mechanical_energy(F, p: Params):
    """Neo-Hookean comprimibile."""
    J  = jnp.linalg.det(F)
    C  = F.T @ F
    I1 = jnp.trace(C)
    return 0.5*p.G*(I1 - 3 - 2*jnp.log(J)) + 0.5*p.K*(J - 1)**2


def _magnetic_energy(F, bL, p: Params):
    """Termine magnetico espresso con b_L."""
    J = jnp.linalg.det(F)
    C = F.T @ F
    return (1.0 / (2.0*p.mu0*p.mu_r*J)) * (bL @ (C @ bL))


def energy_density_jax(F, bL, p: Params):
    """Energia totale Ω = W_mech + W_mag."""
    return _mechanical_energy(F, p) + _magnetic_energy(F, bL, p)

def sigma_from_F_B(F, B, p: Params):
    """Cauchy σ a partire da F e B spaziale."""
    J   = jnp.linalg.det(F)
    Finv = jnp.linalg.inv(F)
    bL  = J * (Finv @ B)                      # B (spaziale) -> b_L (lagrangiano)
    P = jax.jacfwd(lambda F_: energy_density_jax(F_, bL, p))(F)
    # σ = (1/J) P F^T
    return (1.0/J) * (P @ F.T)


def magnetic_field_from_F_B(F, B, p: Params):
    """Campo magnetico h (spaziale) ottenuto da ∂Ω/∂b_L e push-forward."""
    J   = jnp.linalg.det(F)
    Finv = jnp.linalg.inv(F)
    bL  = J * (Finv @ B)
    grad_bL = jax.grad(lambda bL_: energy_density_jax(F, bL_, p))
    H_L = grad_bL(bL)
    # F^T h = H_L => h = F^{-T} H_L
    return jnp.linalg.solve(F.T, H_L)


def lagrangian_magnetic_fields(F, B, p: Params):
    """Restituisce (b_L, h_L)."""
    J   = jnp.linalg.det(F)
    Finv = jnp.linalg.inv(F)
    bL  = J * (Finv @ B)
    grad_bL = jax.grad(lambda bL_: energy_density_jax(F, bL_, p))
    hL = grad_bL(bL)
    return bL, hL


def first_piola_total(F, B, p: Params):
    """Prima Piola totale (incluso contributo magnetico)."""
    J   = jnp.linalg.det(F)
    Finv = jnp.linalg.inv(F)
    bL  = J * (Finv @ B)
    return jax.jacfwd(lambda F_: energy_density_jax(F_, bL, p))(F)


def first_piola_mechanical(F, p: Params):
    """Contributo meccanico puro della prima Piola."""
    return jax.jacfwd(lambda F_: _mechanical_energy(F_, p))(F)


def lagrangian_maxwell_tensor(F, B, p: Params):
    """Tm = J τ_m F^{-T} con τ_m = B⊗h - 0.5*mu0*(h·h)I (campi interni)."""
    J = jnp.linalg.det(F)
    FinvT = jnp.linalg.inv(F).T
    h = magnetic_field_from_F_B(F, B, p)
    tau_m = jnp.outer(B, h) - 0.5 * p.mu0 * (h @ h) * jnp.eye(3)
    return J * (tau_m @ FinvT)


def lagrangian_maxwell_tensor_external(F, B_ext, h_ext, p: Params):
    """Tm = J τ_m F^{-T} con τ_m = B_ext⊗h_ext - 0.5*mu0*(h_ext·h_ext)I (vuoto)."""
    J = jnp.linalg.det(F)
    FinvT = jnp.linalg.inv(F).T
    tau_m = jnp.outer(B_ext, h_ext) - 0.5 * p.mu0 * (h_ext @ h_ext) * jnp.eye(3)
    return J * (tau_m @ FinvT)


def surface_traction(F, B, n, p: Params):
    """Trazione t = [σ - (b⊗h) + 0.5*mu0*(h·h)I] n."""
    sigma = sigma_from_F_B(F, B, p)
    h = magnetic_field_from_F_B(F, B, p)
    traction_tensor = sigma - jnp.outer(B, h) + 0.5 * p.mu0 * (h @ h) * jnp.eye(3)
    return traction_tensor @ n, sigma, h


def lagrangian_surface_tractions(F, B, N, p: Params):
    """
    Restituisce PN totale, Maxwell (vuoto) e meccanico in configurazione di riferimento.
    Usa il tensore di Maxwell costruito con i campi esterni (vuoto).
    """
    P_mech = first_piola_mechanical(F, p)
    PN_mech = P_mech @ N
    h_int = magnetic_field_from_F_B(F, B, p)
    n_current = current_normal(F, N)
    B_ext, h_ext, _, _ = external_spatial_fields(B, h_int, n_current, p)
    Tm_ext = lagrangian_maxwell_tensor_external(F, B_ext, h_ext, p)
    PN_maxwell = Tm_ext @ N
    PN_total = PN_mech + PN_maxwell
    return PN_total, PN_mech, PN_maxwell, P_mech, Tm_ext


def current_normal(F, N):
    """Spinge la normale referenziale N con F^{-T} e la normalizza."""
    FinvT = jnp.linalg.inv(F).T  # F^{-T}
    n_unnormalized = FinvT @ N
    norm_n = jnp.linalg.norm(n_unnormalized)
    return n_unnormalized / norm_n


def lagrangian_boundary_data(bL, hL, N):
    """Componenti normale di bL e tangenziale di hL rispetto a N (referenziale)."""
    norm_N = jnp.linalg.norm(N)
    N_hat = N / norm_N
    b_normal = (bL @ N_hat) * N_hat
    h_tangent = hL - (hL @ N_hat) * N_hat
    return N_hat, b_normal, h_tangent


def _normalize_vector(v):
    """Restituisce v/||v|| gestendo il caso norm=0."""
    norm_v = jnp.linalg.norm(v)
    return jnp.where(norm_v > 0, v / norm_v, jnp.zeros_like(v))


def _decompose_with_normal(vec, n_hat):
    """Scompone un vettore in parti normale e tangenziale rispetto a n_hat."""
    normal = (vec @ n_hat) * n_hat
    tangential = vec - normal
    return normal, tangential


def external_spatial_fields(B_int, h_int, n, p: Params):
    """
    Calcola (B_ext, h_ext) in vacuum imponendo:
      - continuità di h tangenziale,
      - continuità di b normale,
      - B_ext = mu0 h_ext.
    """
    n_hat = _normalize_vector(n)
    B_normal, _ = _decompose_with_normal(B_int, n_hat)
    _, h_tangent = _decompose_with_normal(h_int, n_hat)
    h_normal = (1.0 / p.mu0) * B_normal
    B_tangent = p.mu0 * h_tangent
    h_ext = h_tangent + h_normal
    B_ext = B_tangent + B_normal
    return B_ext, h_ext, B_normal, h_tangent


def external_lagrangian_fields(F, h_ext, p: Params):
    """
    Versione lagrangiana dei campi esterni usando b_L_ext = mu0 * C^{-1} * h_L_ext
    con h_L_ext = F^T h_ext e C = F^T F.
    """
    hL_ext = F.T @ h_ext
    C = F.T @ F
    bL_ext = p.mu0 * (jnp.linalg.inv(C) @ hL_ext)
    return bL_ext, hL_ext

# ---------------- Relazioni da verificare ----------------
# (5.4) simple shear in x1 (k2=0), B = (0,B2,0):
#   k1 μ3^2 (τ11 - τ33) = τ13 [ μ1^2 + μ3^2 (k1^2 - 1) ]
def simple_shear_F(k1):
    """Gradiente di deformazione per taglio semplice (x1 scorre su x2)."""
    return jnp.array([[1.0, k1, 0.0],
                      [0.0, 1.0, 0.0],
                      [0.0, 0.0, 1.0]], dtype=float)


def _format_vector(vec):
    """Helper per una stampa compatta."""
    return np.array2string(np.array(vec), precision=4)


def face_diagnostics(F, B, N, p: Params, bL, hL, h):
    """Dati completi per una faccia con normale referenziale N."""
    n_current = current_normal(F, N)
    B_ext, h_ext, B_normal_ext, h_tangent_ext = external_spatial_fields(B, h, n_current, p)
    bL_ext, hL_ext = external_lagrangian_fields(F, h_ext, p)
    PN_total, PN_mech, PN_maxwell, _, _ = lagrangian_surface_tractions(F, B, N, p)
    _, b_normal, h_tangent = lagrangian_boundary_data(bL, hL, N)
    return {
        "PN_total": PN_total,
        "PN_mech": PN_mech,
        "PN_maxwell": PN_maxwell,
        "b_normal": b_normal,
        "h_tangent": h_tangent,
        "n_current": n_current,
        "B_ext": B_ext,
        "B_ext_normal": B_normal_ext,
        "h_ext": h_ext,
        "h_ext_tangent": h_tangent_ext,
        "bL_ext": bL_ext,
        "hL_ext": hL_ext,
    }


def run_simple_shear_case(shear_angle_deg: float, B2: float, p: Params):
    """Setta il caso di taglio semplice e stampa i risultati."""
    k1 = np.tan(np.deg2rad(shear_angle_deg))
    F = simple_shear_F(k1)
    B = jnp.array([0.0, B2, 0.0], dtype=float)
    h = magnetic_field_from_F_B(F, B, p)
    bL, hL = lagrangian_magnetic_fields(F, B, p)

    faces = {
        "upper": jnp.array([0.0, 1.0, 0.0], dtype=float),   # N = e2
        "lower": jnp.array([0.0, -1.0, 0.0], dtype=float),  # N = -e2
        "right": jnp.array([1.0, 0.0, 0.0], dtype=float),   # N = e1
        "left":  jnp.array([-1.0, 0.0, 0.0], dtype=float),  # N = -e1
    }

    print("\n--- Taglio semplice: trazioni e campi ---")
    print(f"shear angle = {shear_angle_deg}° (k1 = {k1:.4f})")
    print("B =", _format_vector(B))
    print("b_L =", _format_vector(bL), "| h_L =", _format_vector(hL))

    for name, N in faces.items():
        diag = face_diagnostics(F, B, N, p, bL, hL, h)
        print(f"\nFace {name} (N = {np.array(N)}):")
        print("  PN_total     =", _format_vector(diag['PN_total']), "[Pa]")
        print("  PN_mech      =", _format_vector(diag['PN_mech']))
        print("  PN_maxwell   =", _format_vector(diag['PN_maxwell']))
        print("  b_L · N      =", _format_vector(diag['b_normal']))
        print("  h_L_tang     =", _format_vector(diag['h_tangent']))
        print("  n_current    =", _format_vector(diag['n_current']))
        print("  B_ext        =", _format_vector(diag['B_ext']))
        print("    normal     =", _format_vector(diag['B_ext_normal']))
        print("  h_ext        =", _format_vector(diag['h_ext']))
        print("    tangential =", _format_vector(diag['h_ext_tangent']))
        print("  b_L_ext      =", _format_vector(diag['bL_ext']))
        print("  h_L_ext      =", _format_vector(diag['hL_ext']))


# ---------------- Entrypoint ----------------
if __name__ == "__main__":
    params = Params()
    shear_angle = 3.0  # [deg]
    B2_value = 0.2     # Intensità del campo B lungo e2
    run_simple_shear_case(shear_angle, B2_value, params)
