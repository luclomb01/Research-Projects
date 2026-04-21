from dolfinx import fem, mesh, log, default_scalar_type
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from mpi4py import MPI
import numpy as np
import ufl
import os
 

# --- geometria ---
Lx, Ly = 0.80, 0.90
inclusion_w, inclusion_h = 0.08, 0.30
ax = inclusion_w / 2.0
ay = inclusion_h / 2.0
cx, cy = Lx / 2.0, Ly / 2.0
nx, ny = 220, 300
hx, hy = Lx / nx, Ly / ny
mesh_size = min(hx, hy)

domain = mesh.create_rectangle(
    MPI.COMM_WORLD,
    [[0.0, 0.0], [Lx, Ly]],
    [nx, ny],
    cell_type=mesh.CellType.triangle,
)


def in_rectangle(x):
    return np.logical_and.reduce(
        (
            x[0] >= (cx - ax),
            x[0] <= (cx + ax),
            x[1] >= (cy - ay),
            x[1] <= (cy + ay),
        )
    )


cells_B = mesh.locate_entities(domain, domain.topology.dim, in_rectangle)

topology = domain.topology
tdim = topology.dim
fdim = tdim - 1
gdim = domain.geometry.dim

index_map = topology.index_map(tdim)
num_cells_local = index_map.size_local + index_map.num_ghosts
all_cells = np.arange(num_cells_local, dtype=np.int32)
cells_ext = np.setdiff1d(all_cells, cells_B)

cell_indices = np.hstack([cells_B, cells_ext])
cell_values = np.hstack(
    [
        np.full(cells_B.shape, 1, dtype=np.int32),
        np.full(cells_ext.shape, 2, dtype=np.int32),
    ]
)
order_cells = np.argsort(cell_indices)
cell_tag = mesh.meshtags(domain, tdim, cell_indices[order_cells], cell_values[order_cells])

# --- identificazione dei bordi esterni ---
def left(x):
    return np.isclose(x[0], 0.0)


def right(x):
    return np.isclose(x[0], Lx)


def bottom(x):
    return np.isclose(x[1], 0.0)


def top(x):
    return np.isclose(x[1], Ly)


facet_blocks = []
value_blocks = []
boundary_locators = {
    1: left,
    2: right,
    3: bottom,
    4: top,
}
for marker, locator in boundary_locators.items():
    facets = mesh.locate_entities_boundary(domain, fdim, locator)
    facet_blocks.append(facets)
    value_blocks.append(np.full(facets.shape, marker, dtype=np.int32))

marked_facets_ext = np.hstack(facet_blocks)
marked_values_ext = np.hstack(value_blocks)

# --- connettività necessarie ---
def ensure_connectivity(pairs):
    """
    Pre-build the connectivities DolfinX needs. Creating them once upfront
    avoids scattering identical calls along the script at runtime.
    """
    for entity_pair in pairs:
        topology.create_connectivity(*entity_pair)


ensure_connectivity(((fdim, tdim), (fdim, 0), (tdim, 0), (tdim, tdim)))


# --- interfaccia tra inclusione e matrice ---
f2c = topology.connectivity(fdim, tdim)

cell_values_arr = np.zeros(index_map.size_local + index_map.num_ghosts, dtype=np.int32)
cell_values_arr[cell_tag.indices] = cell_tag.values

interface_facets = []
for f in range(topology.index_map(fdim).size_local):
    cids = f2c.links(f)
    if len(cids) == 2:
        t1 = cell_values_arr[cids[0]]
        t2 = cell_values_arr[cids[1]]
        if {t1, t2} == {1, 2}:
            interface_facets.append(f)
interface_facets = np.array(interface_facets, dtype=np.int32)

# --- tag per i lati superiore/inferiore dell'inclusione ---
coords = domain.geometry.x
f2v = topology.connectivity(fdim, 0)

tol_iface = max(0.3 * mesh_size, 1e-4)
bottom_square_facets = []
top_square_facets = []

for f in interface_facets:
    vertex_ids = f2v.links(f)
    if len(vertex_ids) == 0:
        continue
    bary = coords[vertex_ids].mean(axis=0)
    rel = bary[:2] - np.array([cx, cy])
    # |rel_y| >= |rel_x| -> facet belongs to top/bottom sides of the inclusion
    if abs(rel[1]) >= abs(rel[0]) - tol_iface:
        if rel[1] < 0:
            bottom_square_facets.append(f)
        else:
            top_square_facets.append(f)

bottom_square_facets = np.asarray(bottom_square_facets, dtype=np.int32)
top_square_facets = np.asarray(top_square_facets, dtype=np.int32)

# --- meshtags dei bordi ---
marked_facets_all = np.hstack([marked_facets_ext, bottom_square_facets, top_square_facets])
marked_values_all = np.hstack(
    [
        marked_values_ext,
        np.full(bottom_square_facets.shape, 5, dtype=np.int32),
        np.full(top_square_facets.shape, 6, dtype=np.int32),
    ]
)
order = np.argsort(marked_facets_all)
facet_tag = mesh.meshtags(domain, fdim, marked_facets_all[order], marked_values_all[order])
# ==========================
# SOLVE ACCOPPIATO (u, phi)
# ==========================

from basix.ufl import element, mixed_element

# Spazi funzionali
cellname = domain.ufl_cell().cellname()
Ve_u = element("Lagrange", cellname, 1, shape=(gdim,))  # Grado ridotto
Ve_m = element("Lagrange", cellname, 1)
W = fem.functionspace(domain, mixed_element([Ve_u, Ve_m]))

# Spazi collassati per vincoli e post-processing
V_u_c, dofs_u = W.sub(0).collapse()
V_phi_c, dofs_phi = W.sub(1).collapse()
interp_pts_vec = V_u_c.element.interpolation_points()

# Incognite / test / trial
U = fem.Function(W)
dU = ufl.TrialFunction(W)
Z = ufl.TestFunction(W)

u_, phi_ = ufl.split(U)
w_, v_ = ufl.split(Z)
du_, dphi_ = ufl.split(dU)

# Misure
metadata = {"quadrature_degree": 8}  # Grado di quadratura ridotto
dx = ufl.Measure("dx", domain=domain, subdomain_data=cell_tag, metadata=metadata)
dS = ufl.Measure("dS", domain=domain, subdomain_data=facet_tag, metadata=metadata)

# Costanti/materiali 
mu0 = fem.Constant(domain, default_scalar_type(1.2566e-6))
mu = fem.Constant(domain, default_scalar_type(1.5))
G = fem.Constant(domain, default_scalar_type(4e6))  # shear modulus gomma

# Trazione (solo quelle effettivamente usate)

# Cinematica
d = gdim
I = ufl.Identity(d)
F = ufl.variable(I + ufl.grad(u_))
c = ufl.variable(F.T * F)
B = ufl.variable(F * F.T)
J = ufl.variable(ufl.det(F))

hl = ufl.variable(-ufl.grad(phi_))


I1 = ufl.tr(c)

# Parametri meccanici
K  = fem.Constant(domain, default_scalar_type(G.value))   # bulk solido (esempio robusto)
mu_v = fem.Constant(domain, default_scalar_type(1e-2*G.value))  # shear vuoto molto piccolo
K_v  = fem.Constant(domain, default_scalar_type(1e-2*K.value))  # bulk vuoto molto piccolo
# Energie
W_solid_mech = 0.5*G*(I1 - 3 - 2*ufl.ln(J)) + 0.5*K*(J - 1)**2
W_vac_mech   = 0.5*mu_v*(I1 - 3 - 2*ufl.ln(J)) + 0.5*K_v*(J - 1)**2

Omega   = W_solid_mech - 0.5 * J * mu0*mu*ufl.dot(ufl.inv(F.T)*hl, ufl.inv(F.T)*hl)
OmegaOut= W_vac_mech   - 0.5 * J * mu0*ufl.dot(ufl.inv(F.T)*hl, ufl.inv(F.T)*hl)

# Relazioni costitutive
Pin = ufl.diff(Omega, F)
blin = -ufl.diff(Omega, hl)
Pout = ufl.diff(OmegaOut, F)
blout = -ufl.diff(OmegaOut, hl)

sigma = (1 / J) * Pin * F.T


 


# --- Indicatore DG0 dell’inclusione: chi=1 in B, 0 altrove
Q0 = fem.functionspace(domain, ("DG", 0))
chi = fem.Function(Q0)
chi.x.array[:] = 0.0

dofs_inc = fem.locate_dofs_topological(Q0, tdim, cell_tag.find(1))  # 1 = tag celle del quadrato B
chi.x.array[dofs_inc] = 1.0
chi.x.scatter_forward()

# Trazione verticale sul TOP dell’inclusione (tag 6)
tau = -50e3  # intensità (modulo invariato rispetto al caso precedente)
t_surface = fem.Constant(domain, default_scalar_type((0.0, -tau)))

dS_top = dS(6)

F_u = (
    ufl.inner(Pin,  ufl.grad(w_)) * dx(1)
  + ufl.inner(Pout, ufl.grad(w_)) * dx(2)
  # lavoro esterno: SOLO lato vuoto, grazie a chi('+')/chi('-')
  - (chi('-')*ufl.inner(t_surface, w_('+')) + chi('+')*ufl.inner(t_surface, w_('-'))) * dS_top
)

F_phi = ufl.inner(blout, ufl.grad(v_)) * dx(2) + ufl.inner(blin, ufl.grad(v_)) * dx(1)
F_tot = F_u + F_phi
#J_tot = ufl.derivative(F_tot, U, dU)

# --- BC semplici con Constant ---
zero = fem.Constant(domain, default_scalar_type(0.0))

# Impone hl = (0, 1e7 A/m) sui bordi orizzontali: phi varia linearmente in y
H_target = 2e5
phi_bottom_val = 0.5 * H_target * Ly
phi_top_val = -0.5 * H_target * Ly
phitilde_bottom = fem.Constant(domain, default_scalar_type(phi_bottom_val))
phitilde_top = fem.Constant(domain, default_scalar_type(phi_top_val))

bc_list = []

# Vincoli meccanici: solo uy=0 sul bordo inferiore dell'inclusione (tag 5)
bottom_facets_inc = facet_tag.find(5)
if bottom_facets_inc.size > 0:
    bottom_dofs_uy = fem.locate_dofs_topological(W.sub(0).sub(1), fdim, bottom_facets_inc)
    bc_uy_bottom = fem.dirichletbc(zero, bottom_dofs_uy, W.sub(0).sub(1))
    bc_list.append(bc_uy_bottom)
else:
    print("Attenzione: nessun bordo inferiore dell'inclusione (tag 5) trovato per applicare uy=0.")

# Vincoli vuoto: bordo esterno sinistra/destra/alto/basso incastrati (ux=uy=0)
external_tags_full = [1, 2, 3, 4]  # incastro completo
for tag in external_tags_full:
    facets_ext = facet_tag.find(tag)
    if facets_ext.size == 0:
        continue
    dofs_ext_ux = fem.locate_dofs_topological(W.sub(0).sub(0), fdim, facets_ext)
    dofs_ext_uy = fem.locate_dofs_topological(W.sub(0).sub(1), fdim, facets_ext)
    bc_ext_ux = fem.dirichletbc(zero, dofs_ext_ux, W.sub(0).sub(0))
    bc_ext_uy = fem.dirichletbc(zero, dofs_ext_uy, W.sub(0).sub(1))
    bc_list.extend([bc_ext_ux, bc_ext_uy])

# BC per φ ai bordi superiore/inferiore esterni (tag 4 e 3)
top_dofs_phi = fem.locate_dofs_topological(W.sub(1), fdim, facet_tag.find(4))
bc_phi_top = fem.dirichletbc(phitilde_top, top_dofs_phi, W.sub(1))
bottom_dofs_phi = fem.locate_dofs_topological(W.sub(1), fdim, facet_tag.find(3))
bc_phi_bottom = fem.dirichletbc(phitilde_bottom, bottom_dofs_phi, W.sub(1))

bc_list.extend([bc_phi_top, bc_phi_bottom])
bcs_mixed = bc_list

 


# Problema nonlineare e solver
problem = NonlinearProblem(F_tot, U, bcs=bcs_mixed) #J=J_tot)
solver = NewtonSolver(domain.comm, problem)

solver.atol = 1e-8
solver.rtol = 1e-8
solver.convergence_criterion = "incremental" 
solver.max_it = 50
solver.line_search = "bt"


log.set_log_level(log.LogLevel.INFO)

num_its, converged = solver.solve(U)
assert converged, "Il solutore non è convergente"
U.x.scatter_forward()

# # --- check: max |u| sui nodi a y=0 (bordo inferiore) ---
# V_u_chk, dofs_u_chk = W.sub(0).collapse()
# embed_dim_chk = domain.geometry.x.shape[1]  # = 3 in 2D (con z=0)
# coords_chk = V_u_chk.tabulate_dof_coordinates().reshape((-1, embed_dim_chk))
# u_chk = U.x.array[dofs_u_chk].reshape((-1, gdim))  # qui resta 2 (ux, uy)
# mask_bottom = np.isclose(coords_chk[:, 1], 0.0, atol=1e-14)
# max_u_bottom = 0.0 if not np.any(mask_bottom) else float(np.linalg.norm(u_chk[mask_bottom], axis=1).max())
# print("max |u| su y=0 =", max_u_bottom)


print(f"Convergenza: sì, Iterazioni: {num_its}")


# Collassa gli spazi per avere funzioni standalone
u = fem.Function(V_u_c)
phi = fem.Function(V_phi_c)
u.x.array[:]   = U.x.array[dofs_u]
phi.x.array[:] = U.x.array[dofs_phi]
u.x.scatter_forward()
phi.x.scatter_forward()

# Campo scalare J = det(F) e statistiche globali
J_det_expr = ufl.det(ufl.Identity(gdim) + ufl.grad(u))
J_expr = fem.Expression(J_det_expr, V_phi_c.element.interpolation_points())
J_func = fem.Function(V_phi_c)
J_func.interpolate(J_expr)
J_func.x.scatter_forward()

# Valori di J (det F) per visualizzazione nella configurazione deformata
J_values = np.array(J_func.x.array, copy=True)

# Campo magnetico bl: inside usa blin, outside usa blout
bl_expr = fem.Expression(chi * blin + (1.0 - chi) * blout, interp_pts_vec)
bl = fem.Function(V_u_c)
bl.interpolate(bl_expr)
bl.x.scatter_forward()

# Campo magnetico dal potenziale: hl = -∇φ
hl_expr = fem.Expression(-ufl.grad(phi), interp_pts_vec)
hl = fem.Function(V_u_c)
hl.interpolate(hl_expr)
hl.x.scatter_forward()

# =========================
# STRESS: sigma (Cauchy) nel solido e valutazione su un punto fissato della frontiera superiore del quadrato
# =========================
d = gdim

F_eval = ufl.variable(ufl.Identity(d) + ufl.grad(u))
c_eval = ufl.variable(F_eval.T * F_eval)
J_eval = ufl.variable(ufl.det(F_eval))
hl_eval = ufl.variable(-ufl.grad(phi))
I1_eval = ufl.tr(c_eval)
b = (1.0 / J_eval) * F_eval * bl
b_expr = fem.Expression(b, interp_pts_vec)
b_func = fem.Function(V_u_c)
b_func.interpolate(b_expr)
b_func.x.scatter_forward()

h_field = ufl.inv(F_eval).T * hl
h_expr = fem.Expression(h_field, interp_pts_vec)
h_func = fem.Function(V_u_c)
h_func.interpolate(h_expr)
h_func.x.scatter_forward()

Omega_solid_eval = (
    0.5 * G * (I1_eval - 2 - 2 * ufl.ln(J_eval))
    + 0.5 * K * (J_eval - 1) ** 2
    + J_eval * mu0 * mu * ufl.dot(ufl.inv(F_eval.T) * hl_eval, ufl.inv(F_eval.T) * hl_eval)
)
Pin_solid_eval = ufl.diff(Omega_solid_eval, F_eval)
sigma_solid = (1.0 / J_eval) * Pin_solid_eval * F_eval.T

V_sigma = fem.functionspace(domain, ("Lagrange", 1, (d, d)))
sigma_func = fem.Function(V_sigma)
sigma_expr = fem.Expression(sigma_solid, V_sigma.element.interpolation_points())
sigma_func.interpolate(sigma_expr)
sigma_func.x.scatter_forward()

B_expr = fem.Expression(F_eval * F_eval.T, V_sigma.element.interpolation_points())
B_tensor = fem.Function(V_sigma)
B_tensor.interpolate(B_expr)
B_tensor.x.scatter_forward()

# Valutazione commutatore sigma–B disattivata (non più necessaria)
# if top_square_facets.size == 0:
#     print("Attenzione: nessuna faccia marcata sul lato superiore del quadrato; sigma e commutatore non valutati.")
# else:
#     facet_id = int(top_square_facets[0])
#     vertex_ids = f2v.links(facet_id)
#
#     edge_pts = domain.geometry.x[vertex_ids]
#     if edge_pts.shape[0] < 2:
#         print("Attenzione: faccia del lato superiore con meno di due vertici; sigma e commutatore non valutati.")
#     else:
#         boundary_point = edge_pts.mean(axis=0)
#
#         cell_candidates = f2c.links(facet_id)
#         cell_on_square = None
#         for cid in cell_candidates:
#             if cell_values_arr[cid] == 1:
#                 cell_on_square = cid
#                 break
#         if cell_on_square is None and len(cell_candidates) > 0:
#             cell_on_square = cell_candidates[0]
#
#         if cell_on_square is None:
#             print("Attenzione: nessuna cella adiacente trovata per la faccia selezionata; sigma e commutatore non valutati.")
#         else:
#             c2v = topology.connectivity(tdim, 0)
#             cell_vertices = c2v.links(cell_on_square)
#             cell_coords = domain.geometry.x[cell_vertices]
#             cell_bary = cell_coords.mean(axis=0)
#             interior_point = boundary_point + 1e-6 * (cell_bary - boundary_point)
#
#             point_eval = np.zeros(3, dtype=boundary_point.dtype)
#             point_eval[:domain.geometry.x.shape[1]] = interior_point[:domain.geometry.x.shape[1]]
#
#             sigma_val = sigma_func.eval(point_eval, cell_on_square).reshape((d, d))
#             B_val = B_tensor.eval(point_eval, cell_on_square).reshape((d, d))
#             comm = sigma_val @ B_val - B_val @ sigma_val
#             print(f"Commutatore (sigma*B - B*sigma) @ {interior_point[:2]} =\n{comm}")
#
#             comm3 = np.zeros((3, 3), dtype=comm.dtype)
#             comm3[:d, :d] = comm
#
#             b_vec = b_func.eval(point_eval, cell_on_square).reshape(-1)
#             b_vec3 = np.zeros(3, dtype=comm.dtype)
#             b_vec3[:b_vec.size] = b_vec.astype(comm.dtype, copy=False)
#
#             skew_b = np.array(
#                 [
#                     [0.0, -b_vec3[2],  b_vec3[1]],
#                     [b_vec3[2], 0.0,  -b_vec3[0]],
#                     [-b_vec3[1], b_vec3[0], 0.0],
#                 ],
#                 dtype=comm.dtype,
#             )
#
#             comm_dot_skew = float(np.sum(comm3 * skew_b))
#             print(f"sigma @ {interior_point[:2]} (cella {cell_on_square}) =\n{sigma_val}")
#             print(f"(sigma*B - B*sigma) : skew(b) @ {interior_point[:2]} = {comm_dot_skew}")

# =========================
# VISUALIZZAZIONE con PyVista (2D)
# =========================
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from dolfinx.plot import vtk_mesh

if not os.environ.get("DISPLAY"):
    # Abilita rendering off-screen quando non c'è un server X disponibile
    pv.OFF_SCREEN = True

# --- Griglia VTK dal dominio dolfinx ---
topology_data, cell_types, x = vtk_mesh(domain, tdim)
grid = pv.UnstructuredGrid(topology_data, cell_types, x)
embed_dim = domain.geometry.x.shape[1]  # di solito 3 (z=0 in 2D)

def pad_to_three(vec):
    """Embed 2D vectors into 3D arrays required by PyVista."""
    vec3 = np.zeros((vec.shape[0], 3), dtype=vec.dtype)
    vec3[:, :gdim] = vec
    return vec3

# --- Dati puntuali (P1: ai vertici) ---
num_pts = u.x.array.size // gdim
u_vec = u.x.array.reshape((num_pts, gdim))
bl_vec = bl.x.array.reshape((num_pts, gdim))
hl_vec = hl.x.array.reshape((num_pts, gdim))
b_vec = b_func.x.array.reshape((num_pts, gdim))
h_vec = h_func.x.array.reshape((num_pts, gdim))
sigma_vals = sigma_func.x.array.reshape((num_pts, d, d))
sigma_yy = sigma_vals[:, 1, 1]

# PyVista vuole vettori 3D
u_vec3 = pad_to_three(u_vec)
bl_vec3 = pad_to_three(bl_vec)
hl_vec3 = pad_to_three(hl_vec)
b_vec3 = pad_to_three(b_vec)
h_vec3 = pad_to_three(h_vec)

phi_pt = np.asarray(phi.x.array)
u_mag = np.linalg.norm(u_vec, axis=1)
bl_mag = np.linalg.norm(bl_vec, axis=1)
hl_mag = np.linalg.norm(hl_vec, axis=1)
b_mag = np.linalg.norm(b_vec, axis=1)
h_mag = np.linalg.norm(h_vec, axis=1)

# Attacca i campi come point_data
grid.point_data["u"] = u_vec3
grid.point_data["|u|"] = u_mag
grid.point_data["phi"] = phi_pt
grid.point_data["bl"] = bl_vec3
grid.point_data["|bl|"] = bl_mag
grid.point_data["hl"] = hl_vec3
grid.point_data["|hl|"] = hl_mag
grid.point_data["b"] = b_vec3
grid.point_data["|b|"] = b_mag
grid.point_data["h"] = h_vec3
grid.point_data["|h|"] = h_mag
grid.point_data["J"] = J_values
grid.point_data["sigma_yy"] = sigma_yy

# Informazione per celle: 1 = inclusione, 2 = esterno
cell_regions = np.full(grid.n_cells, 2, dtype=np.int32)
owned_cells = cell_tag.indices < grid.n_cells
cell_regions[cell_tag.indices[owned_cells]] = cell_tag.values[owned_cells]
grid.cell_data["region"] = cell_regions

# --- Polilinea dell'interfaccia B–B' (utile per i pannelli 1 e 3) ---
interface_points, interface_lines = [], []
scale_def = 10.0

for facet in interface_facets:
    vertex_ids = f2v.links(facet)
    if len(vertex_ids) != 2:
        continue
    pts = np.zeros((2, 3), dtype=x.dtype)
    pts[:, :embed_dim] = domain.geometry.x[vertex_ids]
    start_idx = len(interface_points)
    interface_points.extend(pts)
    interface_lines.extend([2, start_idx, start_idx + 1])

interface_poly = None
if interface_points:
    interface_poly = pv.PolyData(np.asarray(interface_points), np.asarray(interface_lines, dtype=np.int32))

grid_def = grid.copy(deep=True)
grid_def.points = grid.points + scale_def * u_vec3
grid_def.point_data["|u|"] = u_mag
grid_def.point_data["b"] = b_vec3
grid_def.point_data["|b|"] = b_mag
grid_def.point_data["J"] = J_values
grid_def.point_data["sigma_yy"] = sigma_yy
grid_def.cell_data["region"] = cell_regions

cells_inclusion = np.where(cell_regions == 1)[0]
inclusion_orig = grid.extract_cells(cells_inclusion) if cells_inclusion.size > 0 else None

region_colors = ["#2ca02c", "#1f77b4"]
cmap_scalar = "magma"

if domain.comm.rank == 0:
    if not os.environ.get("DISPLAY"):
        pv.OFF_SCREEN = True

    # Deformazione complessiva del dominio
    if cells_inclusion.size > 0:
        inclusion_def = grid_def.extract_cells(cells_inclusion)
        p_def = pv.Plotter(window_size=(1000, 600))
        p_def.enable_parallel_projection()
        p_def.add_text(f"Deformata inclusione (scale={scale_def})", font_size=12)
        p_def.add_mesh(
            inclusion_def,
            scalars="|u|",
            cmap="viridis",
            show_edges=False,
            scalar_bar_args={"title": "|u| [m]"},
        )
        if inclusion_orig is not None:
            # Contorno della configurazione di riferimento
            inclusion_outline = inclusion_orig.extract_feature_edges(boundary_edges=True, manifold_edges=False)
            p_def.add_mesh(
                inclusion_outline,
                color="black",
                line_width=0.8,
            )
        p_def.view_xy()
        p_def.show_axes()
        p_def.show()

        # Campo J (solo inclusione) sulla configurazione deformata
        p_J = pv.Plotter(window_size=(1000, 600))
        p_J.enable_parallel_projection()
        p_J.add_text("J = det(F) sull'inclusione deformata", font_size=12)
        p_J.add_mesh(
            inclusion_def,
            scalars="J",
            cmap="plasma",
            show_edges=False,
            scalar_bar_args={"title": "J [-]"},
        )
        p_J.view_xy()
        p_J.show_axes()
        p_J.show()

        # Sigma_yy sulla configurazione deformata (solo inclusione)
        p_sigma11 = pv.Plotter(window_size=(1000, 600))
        p_sigma11.enable_parallel_projection()
        p_sigma11.add_text(r"Distribuzione $\sigma_{yy}$ (inclusione deformata)", font_size=12)
        p_sigma11.add_mesh(
            inclusion_def,
            scalars="sigma_yy",
            cmap="rainbow",
            show_edges=False,
            scalar_bar_args={"title": "sigma_yy [Pa]"},
        )
        p_sigma11.view_xy()
        p_sigma11.show_axes()
        p_sigma11.show()

    # Streamlines di bl sull'intero dominio
    n_seed = 16
    x_seed = np.linspace(0.0, Lx, n_seed)
    y_seed = np.linspace(0.0, Ly, n_seed)
    xx_seed, yy_seed = np.meshgrid(x_seed, y_seed)
    seed_points = np.column_stack([xx_seed.ravel(), yy_seed.ravel(), np.zeros(xx_seed.size)])
    seed_source = pv.PolyData(seed_points)

    p_stream = pv.Plotter(window_size=(1000, 600))
    p_stream.enable_parallel_projection()
    p_stream.add_text("Streamlines bl", font_size=12)
    p_stream.add_mesh(
        grid,
        scalars="region",
        cmap=["#2ca02c", "#1f77b4"],
        show_edges=False,
        clim=(1, 2),
        opacity=0.25,
        show_scalar_bar=False,
    )
    if interface_poly is not None:
        p_stream.add_mesh(interface_poly, color="black", line_width=2)
    stream = grid.streamlines_from_source(
        seed_source,
        vectors="bl",
        max_time=5 * min(Lx, Ly),
        initial_step_length=0.02 * min(Lx, Ly),
        terminal_speed=1e-8,
        integrator_type=45,
    )
    if stream.n_cells > 0:
        p_stream.add_mesh(stream, scalars="|bl|", cmap="magma", line_width=2, scalar_bar_args={"title": "|bl| [T]"})
    p_stream.view_xy()
    p_stream.show_axes()
    p_stream.show()

# =========================
# STREAMLINES di bl, hl, b e h
# =========================
# =========================
# STREAMLINES di bl
# =========================
n_seed = 12
x_seed = np.linspace(cx - ax, cx + ax, n_seed)
y_seed = np.linspace(cy - ay, cy + ay, n_seed)
xx_seed, yy_seed = np.meshgrid(x_seed, y_seed)
seed_points = np.column_stack([xx_seed.ravel(), yy_seed.ravel(), np.zeros(xx_seed.size)])
seed_source = pv.PolyData(seed_points)

# p_stream = pv.Plotter(shape=(2, 2), window_size=(1500, 900))
p_stream = pv.Plotter(shape=(1, 1), window_size=(1500, 900))
p_stream.enable_parallel_projection()

stream_configs = [
    ("Streamlines campo bl", "bl", "|bl|", "|bl| [T]"),
    # ("Streamlines campo hl", "hl", "|hl|", "|hl| [A/m]"),
    # ("Streamlines campo b", "b", "|b|", "|b| [T]"),
    # ("Streamlines campo h", "h", "|h|", "|h| [A/m]"),
]

for idx, (title, vec_name, mag_name, bar_title) in enumerate(stream_configs):
    row, col = divmod(idx, 2)
    p_stream.subplot(row, col)
    p_stream.add_text(title, font_size=12)
    p_stream.add_mesh(
        grid,
        scalars="region",
        cmap=region_colors,
        show_edges=False,
        clim=(1, 2),
        opacity=0.3,
        show_scalar_bar=False,
    )
    if interface_poly is not None:
        p_stream.add_mesh(interface_poly, color="black", line_width=2)
    stream = grid.streamlines_from_source(
        seed_source,
        vectors=vec_name,
        max_time=5 * min(Lx, Ly),
        initial_step_length=0.02 * min(Lx, Ly),
        terminal_speed=1e-8,
        integrator_type=45,
    )
    if stream.n_cells > 0:
        p_stream.add_mesh(stream, scalars=mag_name, cmap=cmap_scalar, line_width=2)
        p_stream.add_scalar_bar(title=bar_title, n_labels=3)
    p_stream.view_xy()
    p_stream.show_axes()

p_stream.link_views()
p_stream.view_xy()
p_stream.show()

# =========================
# GRAFICO 1D: componenti di bl e hl_y su x = cx
# =========================
dof_coords = V_u_c.tabulate_dof_coordinates().reshape((-1, embed_dim))
bl_dofs = bl.x.array.reshape((-1, gdim))
hl_dofs = hl.x.array.reshape((-1, gdim))
y_coords = dof_coords[:, 1]
x_coords = dof_coords[:, 0]
line_tol = max(0.5 * mesh_size, 5e-4)
mask_vertical = np.abs(x_coords - cx) <= line_tol
y_vals = y_coords[mask_vertical]
bl_vals = bl_dofs[mask_vertical]
hl_vals = hl_dofs[mask_vertical]

if y_vals.size > 0:
    dy_group = max(0.25 * mesh_size, 1e-5)

    def average_along_axis(coord_raw, values_raw):
        if coord_raw.size == 0:
            return coord_raw, values_raw
        bins = np.round(coord_raw / dy_group).astype(np.int64)
        unique_bins, inverse = np.unique(bins, return_inverse=True)
        grouped_coord = np.zeros(unique_bins.size, dtype=coord_raw.dtype)
        grouped_vals = np.zeros((unique_bins.size, values_raw.shape[1]), dtype=values_raw.dtype)
        counts = np.zeros(unique_bins.size, dtype=np.int64)
        for i, b_idx in enumerate(inverse):
            grouped_coord[b_idx] += coord_raw[i]
            grouped_vals[b_idx] += values_raw[i]
            counts[b_idx] += 1
        grouped_coord /= counts
        grouped_vals /= counts[:, None]
        order = np.argsort(grouped_coord)
        return grouped_coord[order], grouped_vals[order]

    y_line, bl_line = average_along_axis(y_vals, bl_vals)
    _, hl_line = average_along_axis(y_vals, hl_vals)
    y_norm = y_line

    band_tol = max(dy_group, 1e-3)
    inside_mask = np.logical_and(y_line >= (cy - ay) - band_tol, y_line <= (cy + ay) + band_tol)
    outside_mask = ~inside_mask

    def jump_report(field_line, component_index, label):
        lower_band = np.abs(y_line - (cy - ay)) <= band_tol
        upper_band = np.abs(y_line - (cy + ay)) <= band_tol
        diffs = []
        for band_mask in (lower_band, upper_band):
            inside_vals = field_line[np.logical_and(inside_mask, band_mask), component_index]
            outside_vals = field_line[np.logical_and(outside_mask, band_mask), component_index]
            if inside_vals.size and outside_vals.size:
                diffs.append(float(np.mean(inside_vals) - np.mean(outside_vals)))
        if diffs:
            avg_jump = np.mean(np.abs(diffs))
            print("Salto medio {} lungo x = {:.3f}: {:.3e}".format(label, cx, avg_jump))
        else:
            print("Salto medio {} non valutabile: pochi punti vicino all'interfaccia.".format(label))

    plt.figure()
    plt.plot(
        y_norm,
        bl_line[:, 1],
        color="crimson",
        linewidth=2,
        label="bl_y",
    )
    for y_sep in (cy - ay, cy + ay):
        plt.axvline(y_sep, color="black", linestyle="--", linewidth=1)
    plt.xlabel("y [m]")
    plt.ylabel("bl_y [T]")
    plt.title("Componente y di bl lungo x = {:.3f}".format(cx))
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    jump_report(bl_line, 1, "bl_y")

    plt.figure()
    plt.plot(
        y_norm,
        hl_line[:, 1],
        color="darkgreen",
        linewidth=2,
        label="hl_y",
    )
    for y_sep in (cy - ay, cy + ay):
        plt.axvline(y_sep, color="black", linestyle="--", linewidth=1)
    plt.xlabel("y [m]")
    plt.ylabel("hl_y [A/m]")
    plt.title("Componente y di hl lungo x = {:.3f}".format(cx))
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    jump_report(hl_line, 1, "hl_y")

    # Profilo hl_x su una retta verticale spostata a destra del centro
    x_line_offset = cx + 0.25 * Lx
    mask_vertical_offset = np.abs(x_coords - x_line_offset) <= line_tol
    y_vals_off = y_coords[mask_vertical_offset]
    hl_vals_off = hl_dofs[mask_vertical_offset]

    if y_vals_off.size > 0:
        y_line_off, hl_line_off = average_along_axis(y_vals_off, hl_vals_off)
        plt.figure()
        plt.plot(
            y_line_off,
            hl_line_off[:, 0],
            color="slateblue",
            linewidth=2,
            label="hl_x",
        )
        for y_sep in (cy - ay, cy + ay):
            plt.axvline(y_sep, color="black", linestyle="--", linewidth=1)
        plt.xlabel("y [m]")
        plt.ylabel("hl_x [A/m]")
        plt.title("Componente x di hl lungo x = {:.3f}".format(x_line_offset))
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
    else:
        print(f"Nessun punto trovato vicino a x = {x_line_offset:.3f} per tracciare hl_x.")

    # Analisi parametrica di bl al variare della trazione superficiale
    tau_scale_factors = [0.0, 1.0, 5.0, 10.0]
    tau_values_sweep = [tau * scale for scale in tau_scale_factors]
    bl_profiles_sweep = [(tau_values_sweep[0], bl_line[:, 1].copy())]

    if len(tau_values_sweep) > 1:
        U_reference = U.x.array.copy()
        traction_reference = np.array(t_surface.value, copy=True)

        for tau_val in tau_values_sweep[1:]:
            t_surface.value[:] = (0.0, tau_val)
            try:
                num_it_tau, converged_tau = solver.solve(U)
            except RuntimeError as err:
                print(f"Avviso: il solver non converge per τ = {tau_val:.0f} Pa ({err}).")
                continue

            if not converged_tau:
                print(f"Avviso: solver non convergente per τ = {tau_val:.0f} Pa (Newton non convergente).")
                continue

            U.x.scatter_forward()
            u.x.array[:] = U.x.array[dofs_u]
            u.x.scatter_forward()
            phi.x.array[:] = U.x.array[dofs_phi]
            phi.x.scatter_forward()
            bl.interpolate(bl_expr)
            bl.x.scatter_forward()

            bl_dofs_tau = bl.x.array.reshape((-1, gdim))
            bl_vals_tau = bl_dofs_tau[mask_vertical]
            if bl_vals_tau.size == 0:
                print(f"Avviso: nessun dof trovato sulla retta x = {cx:.3f} per τ = {tau_val:.0f} Pa.")
                continue

            _, bl_line_tau = average_along_axis(y_vals, bl_vals_tau)
            bl_profiles_sweep.append((tau_val, bl_line_tau[:, 1].copy()))
        # Ripristina soluzione e trazione originali
        t_surface.value[:] = traction_reference
        U.x.array[:] = U_reference
        U.x.scatter_forward()
        u.x.array[:] = U.x.array[dofs_u]
        u.x.scatter_forward()
        phi.x.array[:] = U.x.array[dofs_phi]
        phi.x.scatter_forward()
        bl.interpolate(bl_expr)
        bl.x.scatter_forward()
        bl_dofs = bl.x.array.reshape((-1, gdim))
        bl_vals = bl_dofs[mask_vertical]
        y_line, bl_line = average_along_axis(y_vals, bl_vals)
        y_norm = y_line

        if len(bl_profiles_sweep) > 1:
            plt.figure()
            for tau_val, profile_y in bl_profiles_sweep:
                plt.plot(y_norm, profile_y, linewidth=2, label=f"τ = {tau_val:.0f} Pa")
            for y_sep in (cy - ay, cy + ay):
                plt.axvline(y_sep, color="black", linestyle="--", linewidth=1)
            plt.xlabel("y [m]")
            plt.ylabel("bl_y [T]")
            plt.title("Profilo di bl_y lungo x = {:.3f} vs trazione superficiale".format(cx))
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.show()
else:
    print("Avviso: nessun dof trovato sulla retta x = {:.3f} per i campi bl".format(cx))
# -------------------------
# (Opzionale) Salvataggi su disco per ParaView/VTK
# -------------------------
grid.save("solution_original.vtu")
grid_def.save("solution_deformed.vtu")
if inclusion_def is not None:
    inclusion_def.save("solution_inclusion_deformed.vtu")
if inclusion_orig is not None:
    inclusion_orig.save("solution_inclusion_original.vtu")
print(
    "Salvati: solution_original.vtu, solution_deformed.vtu"
    + (", solution_inclusion_original.vtu, solution_inclusion_deformed.vtu" if inclusion_def is not None else "")
)
