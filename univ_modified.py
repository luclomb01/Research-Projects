from dolfinx import fem, mesh, log, default_scalar_type
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from mpi4py import MPI
import numpy as np
import ufl

# --- Parametri geometrici ---
Lx, Ly = 0.20, 0.12
L_sx, L_sy = 0.06, 0.04   # rettangolo magnetoelastico interno
cx, cy = Lx/2, Ly/3
nx, ny = 120, 80

domain = mesh.create_rectangle(MPI.COMM_WORLD, [[0, 0], [Lx, Ly]], [nx, ny], mesh.CellType.triangle)
topology = domain.topology
tdim = domain.topology.dim
fdim = tdim - 1
gdim = domain.geometry.dim
topology.create_connectivity(fdim, tdim)
topology.create_connectivity(tdim, 0)

# --- Definizione sottodomini ---
def in_rect(x):
    return np.logical_and.reduce((
        x[0] >= cx - L_sx/2, x[0] <= cx + L_sx/2,
        x[1] >= cy - L_sy/2, x[1] <= cy + L_sy/2,
    ))

cells_solid = mesh.locate_entities(domain, tdim, in_rect)
index_map = topology.index_map(tdim)
num_cells_local = index_map.size_local + index_map.num_ghosts
all_cells = np.arange(num_cells_local, dtype=np.int32)
cells_air = np.setdiff1d(all_cells, cells_solid)

cell_indices = np.hstack([cells_solid, cells_air])
cell_values = np.hstack([
    np.full(cells_solid.shape, 1, dtype=np.int32),
    np.full(cells_air.shape, 2, dtype=np.int32),
])
order = np.argsort(cell_indices)
cell_tag = mesh.meshtags(domain, tdim, cell_indices[order], cell_values[order])

# --- Bordo esterno (per Neumann magnetico e BC meccaniche) ---
def left(x): return np.isclose(x[0], 0)
def right(x): return np.isclose(x[0], Lx)
def bottom(x): return np.isclose(x[1], 0)
def top(x): return np.isclose(x[1], Ly)

facet_blocks, value_blocks = [], []
for tag, locator in ((1, left), (2, right), (3, bottom), (4, top)):
    f = mesh.locate_entities_boundary(domain, fdim, locator)
    facet_blocks.append(f)
    value_blocks.append(np.full(f.shape, tag, dtype=np.int32))
facets = np.hstack(facet_blocks)
values = np.hstack(value_blocks)
order = np.argsort(facets)
facet_tag = mesh.meshtags(domain, fdim, facets[order], values[order])

# --- Spazi funzionali ---
from basix.ufl import element, mixed_element
phys_dim = 3
cellname = domain.ufl_cell().cellname()
Ve_u = element("Lagrange", cellname, 1, shape=(phys_dim,))
Ve_a = element("Lagrange", cellname, 1, shape=(phys_dim,))
W = fem.functionspace(domain, mixed_element([Ve_u, Ve_a]))

U = fem.Function(W)
dU = ufl.TrialFunction(W)
Z = ufl.TestFunction(W)
u_, a_ = ufl.split(U)
w_, z_ = ufl.split(Z)

# --- Materiali ---
mu0 = fem.Constant(domain, default_scalar_type(4*np.pi*1e-7))
mu_r_s = 1.5
mu_r_v = 1.0
G_s = 4e6
K_s = 200*G_s
G_v = 1e-2*G_s
K_v = 1e-2*K_s

# --- Kinematica e operatori ---
def plane_grad3(v):
    g = ufl.grad(v)
    return ufl.as_tensor(
        ((g[0, 0], g[0, 1], 0.0),
         (g[1, 0], g[1, 1], 0.0),
         (g[2, 0], g[2, 1], 0.0))
    )

def curl_plane(a):
    g = ufl.grad(a)
    da_dx, da_dy = g[:, 0], g[:, 1]
    curl_x = da_dy[2]
    curl_y = -da_dx[2]
    curl_z = da_dx[1] - da_dy[0]
    return ufl.as_vector((curl_x, curl_y, curl_z))

I = ufl.Identity(phys_dim)
F = ufl.variable(I + plane_grad3(u_))
J = ufl.variable(ufl.det(F))
C = ufl.variable(F.T * F)
b = ufl.variable(curl_plane(a_))
b_l = ufl.variable(J * ufl.inv(F) * b)

# --- Energie per solido e vuoto ---
def energy_density(mu_r, G, K):
    I1 = ufl.tr(C)
    mech = 0.5*G*(I1 - 3 - 2*ufl.ln(J)) + 0.5*K*(J - 1)**2
    mag = 0.5/(J*mu0*mu_r) * ufl.dot(b_l, C*b_l)
    return mech + mag

Omega_s = energy_density(mu_r_s, G_s, K_s)
Omega_v = energy_density(mu_r_v, G_v, K_v)

# --- Derivate costitutive ---
sigma_s = (1/J) * ufl.diff(Omega_s, F) * F.T
sigma_v = (1/J) * ufl.diff(Omega_v, F) * F.T
h_eff_s = ufl.inv(F).T * ufl.diff(Omega_s, b_l)
h_eff_v = ufl.inv(F).T * ufl.diff(Omega_v, b_l)

# --- Forme deboli ---
dx = ufl.Measure("dx", domain=domain, subdomain_data=cell_tag)
ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tag)
facet_normal_2d = ufl.FacetNormal(domain)
n_vec = ufl.as_vector((facet_normal_2d[0], facet_normal_2d[1], 0.0))

F_mech = (
    ufl.inner(sigma_s, plane_grad3(w_)) * dx(1)
  + ufl.inner(sigma_v, plane_grad3(w_)) * dx(2)
)

F_mag = (
    ufl.inner(h_eff_s, curl_plane(z_)) * dx(1)
  + ufl.inner(h_eff_v, curl_plane(z_)) * dx(2)
)

# --- Condizioni di Neumann all'infinito (bordo esterno del vuoto) ---
H0 = 2e3  # A/m
h_inf = fem.Constant(domain, default_scalar_type((0.0, 0.0, H0)))
for tag in (1, 2, 3, 4):
    F_mag -= ufl.inner(ufl.cross(h_inf, n_vec), z_) * ds(tag)

F_tot = F_mech + F_mag
J_tot = ufl.derivative(F_tot, U, dU)

# --- Condizioni Dirichlet meccaniche (blocca moti rigidi) ---
V_u_c, _ = W.sub(0).collapse()
zero_vec = fem.Function(V_u_c)
zero_vec.x.array[:] = 0.0
zero_vec.x.scatter_forward()

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

bcs = [bc_u_corner_left, bc_u_corner_right_uy]

# --- Risoluzione Newton ---
problem = NonlinearProblem(F_tot, U, bcs=bcs, J=J_tot)
solver = NewtonSolver(domain.comm, problem)
solver.rtol = 1e-8
solver.atol = 1e-8
solver.max_it = 50
solver.convergence_criterion = "incremental"
solver.line_search = "bt"
log.set_log_level(log.LogLevel.INFO)

nits, converged = solver.solve(U)
assert converged, "Newton non converge"
U.x.scatter_forward()
if domain.comm.rank == 0:
    print(f"Convergenza Newton sì, iterazioni = {nits}")

# --- Post-processing: campo b e controllo ---
V_u_c, dofs_u = W.sub(0).collapse()
V_a_c, dofs_a = W.sub(1).collapse()
u = fem.Function(V_u_c); u.x.array[:] = U.x.array[dofs_u]; u.x.scatter_forward()
a = fem.Function(V_a_c); a.x.array[:] = U.x.array[dofs_a]; a.x.scatter_forward()

b_fun = fem.Function(V_u_c)
b_expr = fem.Expression((1/ufl.det(I+plane_grad3(u)))*((I+plane_grad3(u))*(curl_plane(a))), V_u_c.element.interpolation_points())
b_fun.interpolate(b_expr); b_fun.x.scatter_forward()

if domain.comm.rank == 0:
    b_vals = b_fun.x.array.reshape((-1,3))
    b_mean = np.mean(b_vals, axis=0)
    print("Campo medio B = ", b_mean)


