# ====== MAGNETO-ELASTO (senza vuoto): impongo b (spaziale) ======
from dolfinx import mesh, fem, log, default_scalar_type
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from basix.ufl import element, mixed_element
from mpi4py import MPI
import numpy as np
import ufl
from ufl import replace
import pyvista as pv
from dolfinx.plot import vtk_mesh
import os

# -----------------------
# Geometria e mesh
# -----------------------
Lx, Ly = 0.20, 0.12
nx, ny = 100, 60

domain = mesh.create_rectangle(
    MPI.COMM_WORLD,
    [[0.0, 0.0], [Lx, Ly]],
    [nx, ny],
    cell_type=mesh.CellType.triangle
)
tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)
domain.topology.create_connectivity(tdim, 0)

# -----------------------
# Marcatori di bordo
# -----------------------
def left(x):   return np.isclose(x[0], 0.0)
def right(x):  return np.isclose(x[0], Lx)
def bottom(x): return np.isclose(x[1], 0.0)
def top(x):    return np.isclose(x[1], Ly)

facet_blocks, value_blocks = [], []
for tag, locator in ((1, left), (2, right), (3, bottom), (4, top)):
    f = mesh.locate_entities_boundary(domain, fdim, locator)
    facet_blocks.append(f)
    value_blocks.append(np.full(f.shape, tag, dtype=np.int32))
facets = np.hstack(facet_blocks)
values = np.hstack(value_blocks)
order = np.argsort(facets)
facet_tag = mesh.meshtags(domain, fdim, facets[order], values[order])

dx = ufl.Measure("dx", domain=domain)
ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tag)

# -----------------------
# Spazi e incognite
# -----------------------
mesh_gdim = domain.geometry.dim  # 2
phys_dim = 3                     # estensione 3D su mesh 2D (deformazione piana)
cellname = domain.ufl_cell().cellname()

Ve_u = element("Lagrange", cellname, 1, shape=(phys_dim,))  # spostamenti 3D
Ve_a = element("Lagrange", cellname, 1, shape=(phys_dim,))  # potenziale vettore 3D
W = fem.functionspace(domain, mixed_element([Ve_u, Ve_a]))

U  = fem.Function(W)              # incognite (u, a)
dU = ufl.TrialFunction(W)
Z  = ufl.TestFunction(W)
u_, a_ = ufl.split(U)
w_, z_ = ufl.split(Z)

# -----------------------
# Costanti/materiali
# -----------------------
mu0  = fem.Constant(domain, default_scalar_type(1.25663706212e-6))
mu_r = fem.Constant(domain, default_scalar_type(1.5))
G    = fem.Constant(domain, default_scalar_type(4e6))
K    = fem.Constant(domain, default_scalar_type(200*G.value))

# campo b spaziale che vogliamo imporre (uniforme, 3 componenti)
b0_vec = fem.Constant(domain, default_scalar_type((0.0, 0.0, 0.2)))  # [T]

# trazione al bordo superiore (taglio orizzontale)
tau_1 = fem.Constant(domain, default_scalar_type(240e3))  # [Pa]
tau_2 = fem.Constant(domain, default_scalar_type(10.61e3))
t_shear_top = ufl.as_vector((tau_1, tau_2, 0.0))
t_shear_right = ufl.as_vector((-tau_2, tau_1, 0.0))
t_shear_bottom = -t_shear_top
t_shear_left = -t_shear_right

# -----------------------
# Kinematica e operatori
# -----------------------
def plane_grad3(v):
    grad_v = ufl.grad(v)
    return ufl.as_tensor(
        (
            (grad_v[0, 0], grad_v[0, 1], 0.0),
            (grad_v[1, 0], grad_v[1, 1], 0.0),
            (grad_v[2, 0], grad_v[2, 1], 0.0),
        )
    )


def curl_plane(a_vec):
    grad_a = ufl.grad(a_vec)
    da_dx = grad_a[:, 0]
    da_dy = grad_a[:, 1]
    curl_x = da_dy[2]
    curl_y = -da_dx[2]
    curl_z = da_dx[1] - da_dy[0]
    return ufl.as_vector((curl_x, curl_y, curl_z))


I = ufl.Identity(phys_dim)
F = ufl.variable(I + plane_grad3(u_))
J = ufl.variable(ufl.det(F))
C = ufl.variable(F.T * F)
I1 = ufl.tr(C)

b_l = ufl.variable(curl_plane(a_))   # induzione referenziale 3D
b    = (1.0/J) * F * b_l             # induzione spaziale (Euleriana)

# -----------------------
# Energia
# -----------------------
# Meccanica: neo-Hooke compressibile
Omega_mech = 0.5*G*(I1 - 3 - 2*ufl.ln(J)) + 0.5*K*(J - 1)**2

# Magnetica (come da tua formula a mano): (1/(2 J mu0 mu_r)) b_l · C b_l
Omega_mag  = 0.5/(J*mu0*mu_r) * ufl.dot(b_l, C * b_l)

# Energia totale
Omega = Omega_mech + Omega_mag

# -----------------------
# Derivate costitutive
# -----------------------
sigma = (1 / J) * ufl.diff(Omega, F) * F.T                            # 1° Piola
# h efficace (quello che moltiplica curl(z) nella forma debole)
h_eff = ufl.inv(F).T * ufl.diff(Omega, b_l)       # F^{-T} ∂Ω/∂b_l

# -----------------------
# Forma debole e Jacobiano
# -----------------------
F_mech = (
    ufl.inner(sigma, plane_grad3(w_)) * dx
    - ufl.inner(t_shear_top, w_) * ds(4)
    - ufl.inner(t_shear_bottom, w_) * ds(3)
    - ufl.inner(t_shear_right, w_) * ds(2)
    - ufl.inner(t_shear_left, w_) * ds(1)
)
F_mag  = ufl.inner(h_eff, curl_plane(z_)) * dx
F_tot  = F_mech + F_mag
J_tot  = ufl.derivative(F_tot, U, dU)

# -----------------------
# Condizioni al contorno
# -----------------------
# Vincoli puntuali per bloccare i moti rigidi
V_u_c, _ = W.sub(0).collapse()
zero_vec = fem.Function(V_u_c)
zero_vec.x.array[:] = 0.0

def bottom_left_point(x):
    return np.logical_and(np.isclose(x[0], 0.0), np.isclose(x[1], 0.0))

def bottom_right_point(x):
    return np.logical_and(np.isclose(x[0], Lx), np.isclose(x[1], 0.0))

corner_left_dofs = fem.locate_dofs_geometrical((W.sub(0), V_u_c), bottom_left_point)
bc_u_corner_left = fem.dirichletbc(zero_vec, corner_left_dofs, W.sub(0))

V_uy_c, _ = W.sub(0).sub(1).collapse()
zero_scalar = fem.Function(V_uy_c)
zero_scalar.x.array[:] = 0.0
corner_right_dofs_uy = fem.locate_dofs_geometrical((W.sub(0).sub(1), V_uy_c), bottom_right_point)
bc_u_corner_right_uy = fem.dirichletbc(zero_scalar, corner_right_dofs_uy, W.sub(0).sub(1))

# Spazio per visualizzazione e diagnostica
V_vis = fem.functionspace(domain, ("Lagrange", 1, (phys_dim,)))
bottom_dofs_vis = fem.locate_dofs_topological(V_vis, fdim, facet_tag.find(3))

V_a_c, _ = W.sub(1).collapse()
Bx_target = float(b0_vec.value[0])
By_target = float(b0_vec.value[1])
Bz_target = float(b0_vec.value[2])

def a_target_expr(x):
    X, Y = x[0], x[1]
    a_x = -0.5 * Bz_target * Y
    a_y = 0.5 * Bz_target * X
    a_z = Bx_target * Y - By_target * X
    return np.vstack((a_x, a_y, a_z))

a_bc_fun = fem.Function(V_a_c)
a_bc_fun.interpolate(a_target_expr)
a_bc_fun.x.scatter_forward()

def everywhere(x):
    return np.ones(x.shape[1], dtype=bool)

all_dofs_a = fem.locate_dofs_geometrical((W.sub(1), V_a_c), everywhere)
bc_a_dirichlet = fem.dirichletbc(a_bc_fun, all_dofs_a, W.sub(1))

bcs = [bc_u_corner_left, bc_u_corner_right_uy, bc_a_dirichlet]

# -----------------------
# Solver nonlineare
# -----------------------
problem = NonlinearProblem(F_tot, U, bcs=bcs, J=J_tot)
solver = NewtonSolver(domain.comm, problem)
solver.rtol = 1e-8
solver.atol = 1e-8
solver.max_it = 50
solver.convergence_criterion = "incremental"
solver.line_search = "bt"
log.set_log_level(log.LogLevel.INFO)

its, converged = solver.solve(U)
assert converged, "Newton non converge"
U.x.scatter_forward()
if domain.comm.rank == 0:
    coords_xy = domain.geometry.x[:, :mesh_gdim]
    print(f"Convergenza Newton: sì, iterazioni = {its}")

# -----------------------
# Post-processing essenziale
# -----------------------
# Collassa e separa u, a
dofs_u = W.sub(0).collapse()[1]
dofs_a = W.sub(1).collapse()[1]
u = fem.Function(V_u_c); u.x.array[:] = U.x.array[dofs_u]; u.x.scatter_forward()
u_vis = fem.Function(V_vis); u_vis.interpolate(u)
a = fem.Function(V_a_c); a.x.array[:] = U.x.array[dofs_a]; a.x.scatter_forward()

# Campi b_l, b e J
interp_pts_vec = V_u_c.element.interpolation_points()
b_l_eval = curl_plane(a)
b_l_expr = fem.Expression(b_l_eval, interp_pts_vec)
b_l_fun  = fem.Function(V_u_c); b_l_fun.interpolate(b_l_expr); b_l_fun.x.scatter_forward()

F_eval = ufl.variable(I + plane_grad3(u_vis))
J_eval = ufl.variable(ufl.det(F_eval))
b_sp   = (1.0/J_eval) * F_eval * b_l_eval
replacement_map = {
    u_: u_vis,
    a_: a,
    F: F_eval,
    J: J_eval,
    C: F_eval.T * F_eval,
    b_l: b_l_eval,
    b: b_sp,
}

b_expr = fem.Expression(b_sp, interp_pts_vec)
b_fun  = fem.Function(V_u_c); b_fun.interpolate(b_expr); b_fun.x.scatter_forward()

V_tensor = fem.functionspace(domain, ("Lagrange", 1, (phys_dim, phys_dim)))
F_expr = fem.Expression(F_eval, V_tensor.element.interpolation_points())
F_fun = fem.Function(V_tensor); F_fun.interpolate(F_expr); F_fun.x.scatter_forward()
sigma_eval = replace(sigma, replacement_map)
sigma_expr = fem.Expression(sigma_eval, V_tensor.element.interpolation_points())
sigma_fun = fem.Function(V_tensor); sigma_fun.interpolate(sigma_expr); sigma_fun.x.scatter_forward()

# Stampa errore rispetto a b0
b_vals = b_fun.x.array.reshape((-1, phys_dim))
b0 = np.array(b0_vec.value, dtype=b_vals.dtype)
err_inf = np.linalg.norm(b_vals - b0, ord=np.inf)
err_l2  = np.sqrt(np.sum((b_vals - b0)**2)/b_vals.shape[0])
tol_b = 1e-6
if domain.comm.rank == 0:
    print(f"||b - b0||_inf ≈ {err_inf:.3e},  ||b - b0||_rms ≈ {err_l2:.3e}")
    if err_inf < tol_b:
        print(f"b coincide con b0 entro {tol_b:.1e} (controllo passato).")
    else:
        print(f"Attenzione: b differisce da b0 più di {tol_b:.1e}.")

# -----------------------
# Visualizzazione 2D deformata
# -----------------------
if domain.comm.rank == 0:
    if not os.environ.get("DISPLAY"):
        pv.OFF_SCREEN = True

    topology_data, cell_types, x = vtk_mesh(domain, tdim)
    grid = pv.UnstructuredGrid(topology_data, cell_types, x)

    u_vec = u_vis.x.array.reshape((-1, phys_dim))
    b_vec = b_fun.x.array.reshape((-1, phys_dim))
    u_mag = np.linalg.norm(u_vec, axis=1)

    b_target = np.array(b0_vec.value, dtype=b_vec.dtype)
    dev = np.linalg.norm(b_vec - b_target, axis=1)
    b0_norm = np.linalg.norm(b_target)
    rel_dev = dev / b0_norm if b0_norm > 0 else dev
    max_dev = dev.max()
    max_rel = rel_dev.max()
    print(f"Range |u|: [{u_mag.min():.3e}, {u_mag.max():.3e}]")
    print(f"max |b - b0| = {max_dev:.3e}  ({max_rel*100:.3f}% di |b0|)")

    bottom_vals = np.linalg.norm(u_vec[bottom_dofs_vis], axis=1)
    print(
        f"|u| bordo inferiore -> max: {bottom_vals.max():.3e}, "
        f"mediana: {np.median(bottom_vals):.3e}"
    )

    F_vals_all = F_fun.x.array.reshape((-1, phys_dim, phys_dim))
    F_mean = F_vals_all.mean(axis=0)
    diff = F_vals_all - F_mean
    frob = np.linalg.norm(diff.reshape(diff.shape[0], -1), axis=1)
    rms = np.sqrt(np.mean(frob**2))
    max_dev_F = frob.max()
    print(
        "Uniformità di F: max dev = "
        f"{max_dev_F:.3e}, rms = {rms:.3e} (|F - <F>| in norma di Frobenius)"
    )

    top_mask = np.isclose(coords_xy[:, 1], Ly)
    if np.any(top_mask):
        sigma_vals = sigma_fun.x.array.reshape((-1, phys_dim, phys_dim))
        F_vals = F_fun.x.array.reshape((-1, phys_dim, phys_dim))
        top_indices = np.where(top_mask)[0]
        if top_indices.size > 0:
            samples = np.unique(
                np.clip(
                    np.array([0, top_indices.size // 2, top_indices.size - 1]),
                    0,
                    top_indices.size - 1,
                )
            )
            for idx in samples:
                global_idx = top_indices[idx]
                print(
                    f"F sul top boundary (indice {idx}, coord {coords_xy[global_idx]}):"
                )
                print(F_vals[global_idx])
        sigma11 = sigma_vals[top_mask, 0, 0]
        sigma22 = sigma_vals[top_mask, 1, 1]
        sigma33 = sigma_vals[top_mask, 2, 2]
        sigma12 = sigma_vals[top_mask, 0, 1]
        sigma13 = sigma_vals[top_mask, 0, 2]
        sigma23 = sigma_vals[top_mask, 1, 2]
        kappa1 = F_vals[top_mask, 0, 1]
        kappa2 = 0.0
        tau13 = sigma13
        tau23 = sigma23
        tau12 = sigma12
        B1 = b_vals[top_mask, 0]
        B2 = b_vals[top_mask, 1]
        B3 = b_vals[top_mask, 2]
        mu1_sq = mu2_sq = 1.0
        mu3_sq = 0.0

        term = (
            tau13 * kappa1 * mu3_sq
            + tau23 * (mu2_sq + mu3_sq * (kappa2**2 - 1))
            + (sigma33 - sigma22) * kappa2 * mu3_sq
            - tau12 * kappa1 * mu3_sq
        ) * B1
        term += (
            (sigma11 - sigma33) * kappa1 * mu3_sq
            + tau12 * kappa2 * mu3_sq
            - tau13 * (mu1_sq + mu3_sq * (kappa2**2 - 1))
            - tau23 * kappa1 * mu3_sq
        ) * B2
        term += (
            tau12 * (mu1_sq - mu2_sq + mu3_sq * (kappa1**2 - kappa2**2))
            + mu3_sq * ((sigma22 - sigma11) * kappa1 * kappa2
            + tau23 * kappa1
            - tau13 * kappa2)
        ) * B3
        residual = term
        print(
            f"Relazione universale (top boundary, versione completa): "
            f"max|res| = {np.max(np.abs(residual)):.3e}, "
            f"rms = {np.sqrt(np.mean(residual**2)):.3e}"
        )

    grid.point_data["u"] = u_vec
    grid.point_data["|u|"] = u_mag

    scale_vis = 1.0
    grid_def = grid.copy(deep=True)
    grid_def.points = grid.points + scale_vis * u_vec

    plotter = pv.Plotter(window_size=(900, 600))
    plotter.add_text(f"Deformata (vista xy, scala {scale_vis:.0f}×)", font_size=12)
    plotter.add_mesh(
        grid_def,
        scalars="|u|",
        cmap="viridis",
        show_edges=False,
        scalar_bar_args={"title": "|u| [m]"},
    )
    plotter.add_mesh(grid, color="w", opacity=0.15, show_edges=True)
    plotter.view_xy()
    plotter.enable_parallel_projection()
    plotter.show()
