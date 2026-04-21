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
Lx, Ly = 0.80, 0.12  # piastra più lunga per ridurre effetti di bordo
nx, ny = 300, 60     # mantieni passo in x simile aumentando le celle

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

Ve_u = element("Lagrange", cellname, 1, shape=(phys_dim,))
Ve_a = element("Lagrange", cellname, 1, shape=(phys_dim,))
W = fem.functionspace(domain, mixed_element([Ve_u, Ve_a]))

U  = fem.Function(W)
dU = ufl.TrialFunction(W)
Z  = ufl.TestFunction(W)
u_, a_ = ufl.split(U)
w_, z_ = ufl.split(Z)

# -----------------------
# Costanti/materiali
# -----------------------
MU0_VALUE = 1.25663706212e-6
MU_R_VALUE = 1.5
G_VALUE = 4e6
K_VALUE = 200 * G_VALUE

mu0  = fem.Constant(domain, default_scalar_type(MU0_VALUE))
mu_r = fem.Constant(domain, default_scalar_type(MU_R_VALUE))
G    = fem.Constant(domain, default_scalar_type(G_VALUE))
K    = fem.Constant(domain, default_scalar_type(K_VALUE))

# Le traction/magnetic BC si impostano manualmente (Lagrangiane -> convertite all'esterno).
T_1 = fem.Constant(domain, default_scalar_type(210e3))
#T_2 = fem.Constant(domain, default_scalar_type(10.6e3))
T_top = ufl.as_vector((T_1, 0.0, 0.0))
#T_right = ufl.as_vector((-T_2, T_1, 0.0))
#T_bottom = -T_top
#T_left = -T_right

B_target = np.array([0.0, 0.2, 0.0], dtype=float)  # target euleriano per i check globali
SHEAR_ANGLE_TARGET_DEG = 3.0
k_target = np.tan(np.deg2rad(SHEAR_ANGLE_TARGET_DEG))
F_target = np.array(
    [
        [1.0, k_target, 0.0],
        [0.0, 1.0,      0.0],
        [0.0, 0.0,      1.0],
    ],
    dtype=float,
)
J_target = np.linalg.det(F_target)
Finv_target = np.linalg.inv(F_target)
C_target = F_target.T @ F_target
bL_target = J_target * (Finv_target @ B_target)
hL_target = (1.0 / (MU0_VALUE * MU_R_VALUE * J_target)) * (C_target @ bL_target)

# h_l_ext e b_l_ext di riferimento (lagrangiani) – modifica liberamente
h_l_ext_data = {
    # left/right disabilitati temporaneamente
    3: fem.Constant(domain, default_scalar_type((0.0, 1.59e5, 0.0))),  # bottom
    4: fem.Constant(domain, default_scalar_type((0.0, 1.59e5, 0.0))),  # top
}

b_l_ext_data = {
    # left/right disabilitati temporaneamente
    3: (-0.0105, 0.2, 0.0),   # bottom (lagrangiano)
    4: (-0.0105, 0.2, 0.0),   # top
}


# b_euleriano di riferimento: uniforme su tutto il dominio
B_boundary_target = {tag: tuple(B_target) for tag in (1, 2, 3, 4)}
B_target_monitor = B_target.copy()
SHEAR_ANGLE_MONITOR_DEG = SHEAR_ANGLE_TARGET_DEG
F_target_monitor = F_target.copy()

def a_boundary_expr(x):
    X = x[0]
    Y = x[1]
    Ax = np.zeros_like(X)
    Ay = np.zeros_like(X)
    Az = np.zeros_like(X)
    counts = np.zeros_like(X)
    # left/right disabilitati temporaneamente
    boundary_masks = {
        3: np.isclose(Y, 0.0),
        4: np.isclose(Y, Ly),
    }
    for tag, mask in boundary_masks.items():
        if not np.any(mask):
            continue
        Bx, By, Bz = b_l_ext_data[tag]
        Ax[mask] += -0.5 * Bz * Y[mask]
        Ay[mask] += 0.5 * Bz * X[mask]
        Az[mask] += Bx * Y[mask] - By * X[mask]
        counts[mask] += 1.0
    boundary_points = counts > 0.0
    Ax[boundary_points] /= counts[boundary_points]
    Ay[boundary_points] /= counts[boundary_points]
    Az[boundary_points] /= counts[boundary_points]
    return np.vstack((Ax, Ay, Az))

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
F_inv = ufl.inv(F)
C = ufl.variable(F.T * F)
I1 = ufl.tr(C)

b_l = ufl.variable(curl_plane(a_))                      # b_L = curl a
b = ufl.variable((1.0 / J) * (F * b_l))                 # push-forward: b = J^{-1} F b_L

# -----------------------
# Energia
# -----------------------
Omega_mech = 0.5*G*(I1 - 3 - 2*ufl.ln(J)) + 0.5*K*(J - 1)**2
Omega_mag  = 0.5/(J*mu0*mu_r) * ufl.dot(b_l, C * b_l)
Omega = Omega_mech + Omega_mag

# -----------------------
# Derivate costitutive
# -----------------------
P = ufl.diff(Omega, F)
h_l = ufl.diff(Omega, b_l)
sigma = (1.0 / J) * P * F.T

# -----------------------
# Forma debole e Jacobiano
# -----------------------
F_mech = (
    ufl.inner(P, plane_grad3(w_)) * dx
)
F_mag  = ufl.inner(h_l, curl_plane(z_)) * dx

# contributo di Neumann: -∫ h_ext × n · z_ sui lati taggati
n_plane = ufl.FacetNormal(domain)
n_vec = ufl.as_vector((n_plane[0], n_plane[1], 0.0))

# Traction con contributo magnetico esterno sul top
b_l_ext_top = ufl.as_vector(b_l_ext_data[4])
h_l_ext_top = h_l_ext_data[4]
bext_top = (1.0 / J) * (F * b_l_ext_top)
hext_top = F_inv.T * h_l_ext_top
tau_m_top = ufl.outer(bext_top, hext_top) - 0.5 * mu0 * ufl.dot(hext_top, hext_top) * I
Tm_top = J * tau_m_top * F_inv.T
F_mech -= ufl.inner(T_top + Tm_top * n_vec, w_) * ds(4)

for tag, h_ext in h_l_ext_data.items():
    F_mag -= ufl.inner(ufl.cross(h_ext, n_vec), z_) * ds(tag)

F_tot = F_mech + F_mag
J_tot = ufl.derivative(F_tot, U, dU)

# -----------------------
# Condizioni al contorno
# -----------------------
V_u_c, _ = W.sub(0).collapse()
zero_vec = fem.Function(V_u_c)
zero_vec.x.array[:] = 0.0

bottom_facets = facet_tag.find(3)
boundary_dofs_bottom = fem.locate_dofs_topological((W.sub(0), V_u_c), fdim, bottom_facets)
bc_u_bottom = fem.dirichletbc(zero_vec, boundary_dofs_bottom, W.sub(0))

V_a_c, _ = W.sub(1).collapse()
a_bc_fun = fem.Function(V_a_c)
a_bc_fun.interpolate(a_boundary_expr)
a_bc_fun.x.scatter_forward()
boundary_facets = np.hstack([facet_tag.find(tag) for tag in (3, 4)])
boundary_dofs_a = fem.locate_dofs_topological((W.sub(1), V_a_c), fdim, boundary_facets)
bc_a_dirichlet = fem.dirichletbc(a_bc_fun, boundary_dofs_a, W.sub(1))

bcs = [bc_u_bottom, bc_a_dirichlet]

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
dofs_u = W.sub(0).collapse()[1]
dofs_a = W.sub(1).collapse()[1]
u = fem.Function(V_u_c); u.x.array[:] = U.x.array[dofs_u]; u.x.scatter_forward()
V_vis = fem.functionspace(domain, ("Lagrange", 1, (phys_dim,)))
u_vis = fem.Function(V_vis); u_vis.interpolate(u)
a = fem.Function(V_a_c); a.x.array[:] = U.x.array[dofs_a]; a.x.scatter_forward()

interp_pts_vec = V_u_c.element.interpolation_points()
F_eval = ufl.variable(I + plane_grad3(u_vis))
J_eval = ufl.variable(ufl.det(F_eval))

b_l_eval = curl_plane(a)
b_eval = (1.0 / J_eval) * (F_eval * b_l_eval)
b_l_expr = fem.Expression(b_l_eval, interp_pts_vec)
b_l_fun  = fem.Function(V_u_c); b_l_fun.interpolate(b_l_expr); b_l_fun.x.scatter_forward()

replacement_map = {
    u_: u_vis,
    a_: a,
    F: F_eval,
    J: J_eval,
    C: F_eval.T * F_eval,
    b_l: b_l_eval,
    b: b_eval,
}

b_expr = fem.Expression(b_eval, interp_pts_vec)
b_fun  = fem.Function(V_u_c); b_fun.interpolate(b_expr); b_fun.x.scatter_forward()

b_vals = b_fun.x.array.reshape((-1, phys_dim))
if domain.comm.rank == 0:
    coords_xy = domain.geometry.x[:, :mesh_gdim]

    # finestra centrale in x basata sulla lunghezza iniziale 0.8 (per confronti aumentando Lx)
    x_min = 0.24  # 0.3 * 0.8
    x_max = 0.56  # 0.7 * 0.8
    mask_central = (coords_xy[:, 0] >= x_min) & (coords_xy[:, 0] <= x_max)

    b_sel = b_vals[mask_central]
    b_mean = b_sel.mean(axis=0)
    delta_b = b_sel - B_target_monitor
    delta_b_rms = np.sqrt(np.mean(delta_b**2, axis=0))
    print(f"[zona centrale] b mean = {b_mean}")
    print(f"[zona centrale] b delta vs target RMS = {delta_b_rms}")

V_tensor = fem.functionspace(domain, ("Lagrange", 1, (phys_dim, phys_dim)))
F_expr = fem.Expression(F_eval, V_tensor.element.interpolation_points())
F_fun = fem.Function(V_tensor); F_fun.interpolate(F_expr); F_fun.x.scatter_forward()
sigma_eval = replace(sigma, replacement_map)
sigma_expr = fem.Expression(sigma_eval, V_tensor.element.interpolation_points())
sigma_fun = fem.Function(V_tensor); sigma_fun.interpolate(sigma_expr); sigma_fun.x.scatter_forward()

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
    dev = np.linalg.norm(b_vec, axis=1)
    max_dev = dev.max()

    bottom_dofs_vis = fem.locate_dofs_topological(V_vis, fdim, facet_tag.find(3))
    bottom_vals = np.linalg.norm(u_vec[bottom_dofs_vis], axis=1)

    print(f"Range |u|: [{u_mag.min():.3e}, {u_mag.max():.3e}]")
    print(f"max |b| = {max_dev:.3e}")
    print(
        f"|u| bordo inferiore -> max: {bottom_vals.max():.3e}, "
        f"mediana: {np.median(bottom_vals):.3e}"
    )

    F_vals_all = F_fun.x.array.reshape((-1, phys_dim, phys_dim))
    F_sel = F_vals_all[mask_central]
    F_mean = F_sel.mean(axis=0)
    diff_target_rms = np.sqrt(np.mean((F_sel - F_target_monitor)**2, axis=0))
    shear_mean = np.degrees(np.arctan(F_mean[0, 1]))
    shear_target = SHEAR_ANGLE_MONITOR_DEG
    print("[zona centrale] F medio =")
    print(F_mean)
    print(f"[zona centrale] RMS(F - F_target) component-wise =\n{diff_target_rms}")
    print(
        f"[zona centrale] Shear angle medio (deg) = {shear_mean:.3f}, "
        f"target = {shear_target:.3f}, "
        f"Δ = {abs(shear_mean - shear_target):.3e}"
    )

    grid.point_data["u"] = u_vec
    grid.point_data["|u|"] = u_mag
    grid.point_data["b_y"] = b_vec[:, 1]

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

    # Mappa della componente y di b sul dominio deformato
    plotter_b = pv.Plotter(window_size=(900, 600))
    plotter_b.add_text("Componente b_3", font_size=12)
    plotter_b.add_mesh(
        grid_def,
        scalars="b_y",
        cmap="coolwarm",
        show_edges=False,
        scalar_bar_args={"title": "b_3 [T]"},
    )
    plotter_b.add_mesh(grid, color="k", opacity=0.1, show_edges=True)
    plotter_b.view_xy()
    plotter_b.enable_parallel_projection()
    plotter_b.show()
