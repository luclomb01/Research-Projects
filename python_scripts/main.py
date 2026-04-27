import numpy as np
from mpi4py import MPI
from dolfinx import fem, log, default_scalar_type
from geometry_build import Geometry
from mesh_building import build_mesh
from compute_meshsize import compute_hmin
from pyvistamesh import show_mesh_pyvista, SCALAR_BAR_VERTICAL
from dolfinx.fem.petsc import NewtonSolverNonlinearProblem
from dolfinx.nls.petsc import NewtonSolver

geom = Geometry()
L1 = geom.L1
L2 = geom.L2
l1 = geom.l1
l2 = geom.l2
domain, cell_tag, facet_tag = build_mesh(MPI.COMM_WORLD, geom)
show_mesh_pyvista("mesh.msh")
cx, cy = geom.center
cx_quarter_right = cx + 0.25 * geom.l1
_, y0 = geom.inclusion_origin
y1 = y0 + geom.l2
ax = geom.l1 / 2.0
ay = geom.l2 / 2.0

# check on facet tags
topology = domain.topology
tdim = topology.dim
fdim = tdim - 1
gdim = domain.geometry.dim
topology.create_connectivity(tdim, tdim)
topology.create_connectivity(tdim, 0)
topology.create_connectivity(fdim, tdim)
topology.create_connectivity(fdim, 0)
mesh_size = compute_hmin(domain)

if domain.comm.rank == 0:
    unique_facets = np.unique(facet_tag.values)
    print(f"Unique facet tags: {unique_facets}")
    for tag_id in (5, 6, 1, 2, 3, 4):
        count = facet_tag.find(tag_id).size
        print(f"  tag {tag_id}: {count} facets")


from basix.ufl import element, mixed_element
import ufl


cellname = domain.ufl_cell().cellname() # get the name of the cell type
Ve_u = element("Lagrange", cellname, 2, shape=(gdim,)) # linear shape functions
Ve_m = element("Lagrange", cellname, 2)
W = fem.functionspace(domain, mixed_element([Ve_u, Ve_m]))

# collapsed function spaces
V_u_c, dofs_u = W.sub(0).collapse()
V_phi_c, dofs_phi = W.sub(1).collapse()
interp_pts_vec = V_u_c.element.interpolation_points
# print(interp_pts_vec)


# test and trial functions
U = fem.Function(W)
dU = ufl.TrialFunction(W)
Z = ufl.TestFunction(W)

u_, phi_ = ufl.split(U)
w_, v_ = ufl.split(Z)
du_, dphi_ = ufl.split(dU)
U.name = "U"

# measures
dx = ufl.Measure("dx", domain=domain, subdomain_data=cell_tag)
dS = ufl.Measure("dS", domain=domain, subdomain_data=facet_tag)

# magnetic material constants
mu0 = fem.Constant(domain, default_scalar_type(1.2566e-6))
mu = fem.Constant(domain, default_scalar_type(1.5))

# variables
d = gdim
I = ufl.Identity(d)
F = ufl.variable(I + ufl.grad(u_))
C = ufl.variable(F.T * F)
B = ufl.variable(F * F.T)
J = ufl.variable(ufl.det(F))

hl = ufl.variable(-ufl.grad(phi_))

I1 = ufl.tr(C)

# mechanical material constants
G = fem.Constant(domain, default_scalar_type(1e6)) # solid shear modulus
K = fem.Constant(domain, default_scalar_type(G.value)) # solid bulk modulus
mu_v = fem.Constant(domain, default_scalar_type(4e4)) # void shear modulus
K_v = fem.Constant(domain, default_scalar_type(4e4)) # void bulk modulus

# energies
W_solid_mech = 0.5 * G * (I1 - 2 - 2*ufl.ln(J)) + 0.5 * K * (J - 1)**2
W_vac_mech = 0.5 * mu_v * (I1 - 2 - 2*ufl.ln(J)) + 0.5 * K_v * (J - 1)**2

Omega = W_solid_mech - 0.5 * J * mu0 * mu * ufl.dot(ufl.inv(F.T)*hl, ufl.inv(F.T)*hl)
OmegaOut = W_vac_mech - 0.5 * J * mu0 * ufl.dot(ufl.inv(F.T)*hl, ufl.inv(F.T)*hl)

# constitutive relations
Pin = ufl.diff(Omega, F)
blin = -ufl.diff(Omega, hl)
Pout = ufl.diff(OmegaOut, F)
blout = -ufl.diff(OmegaOut, hl)

sigma = (1 / J) * Pin * F.T

# DG0 indicator
Q0 = fem.functionspace(domain, ("DG", 0))
chi = fem.Function(Q0)
chi.x.array[:] = 0.0 # 

dofs_inc = fem.locate_dofs_topological(Q0, tdim, cell_tag.find(1)) # find the inclusion cells
chi.x.array[dofs_inc] = 1.0
chi.x.scatter_forward()


# surface traction
tau = -50e3 # surface traction absolute value
t_surface = fem.Constant(domain, default_scalar_type((0.0, -tau)))

dS_top = dS(6) # top interface tag as input

###
# weak formulation
###
# the surface traction is applied to the inclusion. the sign solution covers both the cases ('-', '+' may indicate either the matrix or the inclusion)
F_u = (
    ufl.inner(Pin, ufl.grad(w_)) * dx(1)
  + ufl.inner(Pout, ufl.grad(w_)) * dx(2)
  - (chi('-') * ufl.inner(t_surface, w_('+')) + chi('+') * ufl.inner(t_surface, w_('-'))) * dS_top
)

F_phi = ufl.inner(blout, ufl.grad(v_)) * dx(2) + ufl.inner(blin, ufl.grad(v_)) * dx(1)
F_tot = F_u + F_phi
J_tot = ufl.derivative(F_tot, U, dU)

# boundary conditions
zero = fem.Constant(domain, default_scalar_type(0.0))

H_target = 1e6
phi_bottom_val = 0.5 * H_target * L2
phi_top_val = -0.5 * H_target * L2
phitilde_bottom = fem.Constant(domain, default_scalar_type(phi_bottom_val))
phitilde_top = fem.Constant(domain, default_scalar_type(phi_top_val))

bc_list = []

# mechanical constraints: uy = 0 on inclusion bottom interface
bottom_facets_inc = facet_tag.find(5)
if bottom_facets_inc.size > 0:
    bottom_dofs_uy = fem.locate_dofs_topological(W.sub(0).sub(1), fdim, bottom_facets_inc)
    bc_uy_bottom = fem.dirichletbc(zero, bottom_dofs_uy, W.sub(0).sub(1))
    bc_list.append(bc_uy_bottom)
else:
    print("Attenzione: nessun bordo inferiore dell'inclusione (tag 5) trovato per applicare uy=0.")

# void constraints (external boundaries fixed)
external_tags_full = [1, 2, 3, 4]
for tag in external_tags_full:
    facets_ext = facet_tag.find(tag)
    if facets_ext.size == 0:
        continue
    dofs_ext_ux = fem.locate_dofs_topological(W.sub(0).sub(0), fdim, facets_ext)
    dofs_ext_uy = fem.locate_dofs_topological(W.sub(0).sub(1), fdim, facets_ext)
    bc_ext_ux = fem.dirichletbc(zero, dofs_ext_ux, W.sub(0).sub(0))
    bc_ext_uy = fem.dirichletbc(zero, dofs_ext_uy, W.sub(0).sub(1))
    bc_list.extend([bc_ext_ux, bc_ext_uy])

# magnetic potential boundary conditions
top_dofs_phi = fem.locate_dofs_topological(W.sub(1), fdim, facet_tag.find(4))
bc_phi_top = fem.dirichletbc(phitilde_top, top_dofs_phi, W.sub(1))
bottom_dofs_phi = fem.locate_dofs_topological(W.sub(1), fdim, facet_tag.find(3))
bc_phi_bottom = fem.dirichletbc(phitilde_bottom, bottom_dofs_phi, W.sub(1))

bc_list.extend([bc_phi_top, bc_phi_bottom])
bcs_mixed = bc_list

# nonlinear problem and solver

problem = NewtonSolverNonlinearProblem(F_tot, U, bcs=bcs_mixed, J=J_tot)
solver = NewtonSolver(domain.comm, problem)

# --- Configurazione fondamentale per problemi misti (saddle-point) ---
ksp = solver.krylov_solver
ksp.setType("preonly") # Disabilita i metodi iterativi
pc = ksp.getPC()
pc.setType("lu")       # Forza la fattorizzazione diretta LU
pc.setFactorSolverType("mumps") # Usa MUMPS (molto robusto per zeri sulla diagonale)
# --------------------------------------------------------------------

solver.atol = 1e-8
solver.rtol = 1e-8
solver.convergence_criterion = "incremental"
solver.max_it = 50
solver.line_search = "bt" # damped newton method

log.set_log_level(log.LogLevel.INFO)

num_its, converged = solver.solve(U)
assert converged, "Solutore non convergente"
U.x.scatter_forward()

print(f"Convergenza: sì, Iterazioni: {num_its}")

# results
u = fem.Function(V_u_c)
phi = fem.Function(V_phi_c)
u.name = "u"
phi.name = "phi"
u.x.array[:] = U.x.array[dofs_u]
phi.x.array[:] = U.x.array[dofs_phi]
u.x.scatter_forward()
phi.x.scatter_forward()

# lagrangian quantities

J_det_expr = ufl.det(ufl.Identity(d) + ufl.grad(u))
J_expr = fem.Expression(J_det_expr, V_phi_c.element.interpolation_points)
J_func = fem.Function(V_phi_c)
J_func.interpolate(J_expr)
J_func.x.scatter_forward()

bl_expr = fem.Expression(chi * blin + (1.0 - chi) * blout, interp_pts_vec)
bl = fem.Function(V_u_c)
bl.interpolate(bl_expr)
bl.x.scatter_forward()

hl_expr = fem.Expression(-ufl.grad(phi), interp_pts_vec)
hl = fem.Function(V_u_c)
hl.interpolate(hl_expr)
hl.x.scatter_forward()


# eulerian quantities

F_eval = ufl.variable(ufl.Identity(d) + ufl.grad(u))
C_eval = ufl.variable(F_eval.T * F_eval)
J_eval = ufl.variable(ufl.det(F_eval))
hl_eval = ufl.variable(-ufl.grad(phi))
I1_eval = ufl.variable(ufl.tr(C_eval))
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
    - J_eval * mu0 * mu * ufl.dot(ufl.inv(F_eval.T) * hl_eval, ufl.inv(F_eval.T) * hl_eval)
)
Pin_solid_eval = ufl.diff(Omega_solid_eval, F_eval)
sigma_solid = (1.0 / J_eval) * Pin_solid_eval * F_eval.T

V_sigma = fem.functionspace(domain, ("Lagrange", 2, (d, d)))
sigma_func = fem.Function(V_sigma)
sigma_expr = fem.Expression(sigma_solid, V_sigma.element.interpolation_points)
sigma_func.interpolate(sigma_expr)
sigma_func.x.scatter_forward()


# =========================
# VISUALIZZAZIONE con PyVista (2D) – versione robusta
# =========================
import pyvista as pv
from dolfinx.plot import vtk_mesh

# --- 1. Spazio scalare P1 per post-processing (coerente con la mesh di dominio)
V1 = fem.functionspace(domain, ("Lagrange", 1))

# Funzione di comodo: crea e riempie un Function in V1 da una espressione UFL
def interp_to_V1(expr_ufl, name=None):
    expr = fem.Expression(expr_ufl, V1.element.interpolation_points)
    f = fem.Function(V1)
    if name is not None:
        f.name = name
    f.interpolate(expr)
    f.x.scatter_forward()
    return f

# --- 2. Ricostruisci i campi in V1

# u: componenti ux, uy
ux_V1 = interp_to_V1(u[0], name="ux")
uy_V1 = interp_to_V1(u[1], name="uy")

# bl: componenti
blx_V1 = interp_to_V1(bl[0], name="blx")
bly_V1 = interp_to_V1(bl[1], name="bly")

# hl
hlx_V1 = interp_to_V1(hl[0], name="hlx")
hly_V1 = interp_to_V1(hl[1], name="hly")

# b
bx_V1 = interp_to_V1(b_func[0], name="bx")
by_V1 = interp_to_V1(b_func[1], name="by")

# h
hx_V1 = interp_to_V1(h_func[0], name="hx")
hy_V1 = interp_to_V1(h_func[1], name="hy")

# φ (semplicemente reinterpolata)
phi_V1 = interp_to_V1(phi, name="phi")

# J
J_V1 = interp_to_V1(J_det_expr, name="J")   # oppure interp_to_V1(J_func)

# sigma_yy direttamente da sigma_solid (UFL)
sigma_yy_V1 = interp_to_V1(sigma_solid[1, 1], name="sigma_yy")
# sigma_yy cell-based (DG0) per evitare smearing ai bordi
sigma_yy_expr = fem.Expression(sigma_solid[1, 1], Q0.element.interpolation_points)
sigma_yy_cell = fem.Function(Q0)
sigma_yy_cell.name = "sigma_yy_cell"
sigma_yy_cell.interpolate(sigma_yy_expr)
sigma_yy_cell.x.scatter_forward()

# Altri campi cell-based (DG0) per plot su inclusione
u_mag_expr = fem.Expression(ufl.sqrt(ufl.inner(u, u)), Q0.element.interpolation_points)
u_mag_cell = fem.Function(Q0)
u_mag_cell.name = "|u|_cell"
u_mag_cell.interpolate(u_mag_expr)
u_mag_cell.x.scatter_forward()

J_cell = fem.Function(Q0)
J_cell.name = "J_cell"
J_cell.interpolate(fem.Expression(J_det_expr, Q0.element.interpolation_points))
J_cell.x.scatter_forward()

hx_cell = fem.Function(Q0)
hx_cell.name = "hx_cell"
hx_cell.interpolate(fem.Expression(chi * h_field[0], Q0.element.interpolation_points))
hx_cell.x.scatter_forward()

hy_cell = fem.Function(Q0)
hy_cell.name = "hy_cell"
hy_cell.interpolate(fem.Expression(chi * h_field[1], Q0.element.interpolation_points))
hy_cell.x.scatter_forward()

h_mag_cell = fem.Function(Q0)
h_mag_cell.name = "|h|_cell"
h_mag_cell.interpolate(fem.Expression(chi * ufl.sqrt(ufl.inner(h_field, h_field)), Q0.element.interpolation_points))
h_mag_cell.x.scatter_forward()

# Annulla (o pone a NaN) i valori cell-based fuori dall'inclusione per evitare bleed dal vuoto
dofs_inclusion_cells = fem.locate_dofs_topological(Q0, tdim, cell_tag.find(1))
all_cell_dofs = np.arange(Q0.dofmap.index_map.size_local, dtype=np.int32)
mask_outside = np.ones_like(all_cell_dofs, dtype=bool)
mask_outside[dofs_inclusion_cells] = False
for f_cell in (sigma_yy_cell, u_mag_cell, J_cell, hx_cell, hy_cell, h_mag_cell):
    arr = f_cell.x.array
    arr[mask_outside] = np.nan
    f_cell.x.array[:] = arr
    f_cell.x.scatter_forward()

# --- 3. Mesh VTK dal dominio (punti = vertici, dof V1 = quei vertici)
topology_data, cell_types, x = vtk_mesh(domain, tdim)
grid = pv.UnstructuredGrid(topology_data, cell_types, x)

embed_dim = x.shape[1]     # in 2D di solito = 3
num_pts = x.shape[0]

# --- 4. Costruisci i vettori a partire dai valori nodali in V1

def make_vec2(fx, fy):
    vx = fx.x.array
    vy = fy.x.array
    vec = np.zeros((num_pts, gdim), dtype=default_scalar_type)
    vec[:, 0] = vx
    vec[:, 1] = vy
    return vec

u_vec  = make_vec2(ux_V1,  uy_V1)
bl_vec = make_vec2(blx_V1, bly_V1)
hl_vec = make_vec2(hlx_V1, hly_V1)
b_vec  = make_vec2(bx_V1,  by_V1)
h_vec  = make_vec2(hx_V1,  hy_V1)

phi_pt = phi_V1.x.array.copy()
J_vals = J_V1.x.array.copy()
sigma_yy_vals = sigma_yy_V1.x.array.copy()

# Pad in 3D per PyVista
def pad3(v):
    v3 = np.zeros((v.shape[0], 3), dtype=v.dtype)
    v3[:, :gdim] = v
    return v3

u_vec3  = pad3(u_vec)
bl_vec3 = pad3(bl_vec)
hl_vec3 = pad3(hl_vec)
b_vec3  = pad3(b_vec)
h_vec3  = pad3(h_vec)

# Magnitudini
u_mag  = np.linalg.norm(u_vec, axis=1)
bl_mag = np.linalg.norm(bl_vec, axis=1)
hl_mag = np.linalg.norm(hl_vec, axis=1)
b_mag  = np.linalg.norm(b_vec, axis=1)
h_mag  = np.linalg.norm(h_vec, axis=1)

# --- 5. Attacca i campi alla griglia
grid.point_data["u"]        = u_vec3
grid.point_data["|u|"]      = u_mag
grid.point_data["bl"]       = bl_vec3
grid.point_data["|bl|"]     = bl_mag
grid.point_data["hl"]       = hl_vec3
grid.point_data["|hl|"]     = hl_mag
grid.point_data["hx"]       = h_vec[:, 0]
grid.point_data["hy"]       = h_vec[:, 1]
grid.point_data["b"]        = b_vec3
grid.point_data["|b|"]      = b_mag
grid.point_data["h"]        = h_vec3
grid.point_data["|h|"]      = h_mag
grid.point_data["phi"]      = phi_pt
grid.point_data["J"]        = J_vals
grid.point_data["sigma_yy"] = sigma_yy_vals

# --- 6. Tag di cella (1 = inclusione, 2 = esterno)
cell_regions = np.full(grid.n_cells, 2, dtype=np.int32)
owned_cells = cell_tag.indices < grid.n_cells
cell_regions[cell_tag.indices[owned_cells]] = cell_tag.values[owned_cells]
grid.cell_data["region"] = cell_regions
# Aggiungi sigma_yy su base cella (DG0) per plot senza smearing ai bordi
grid.cell_data["sigma_yy_cell"] = sigma_yy_cell.x.array.copy()
grid.cell_data["|u|_cell"]      = u_mag_cell.x.array.copy()
grid.cell_data["J_cell"]        = J_cell.x.array.copy()
grid.cell_data["hx_cell"]       = hx_cell.x.array.copy()
grid.cell_data["hy_cell"]       = hy_cell.x.array.copy()
grid.cell_data["|h|_cell"]      = h_mag_cell.x.array.copy()

# --- 7. Griglia deformata
scale_def = 1.0
grid_def = grid.copy()
grid_def.points = grid.points + scale_def * u_vec3

# --- 8. Estrai inclusione (celle con region == 1)
cells_inclusion = np.where(cell_regions == 1)[0]
inclusion_orig = grid.extract_cells(cells_inclusion)
inclusion_def  = grid_def.extract_cells(cells_inclusion)

# --- 9. Plot deformata dell’inclusione
if domain.comm.rank == 0:
    p = pv.Plotter(window_size=(900, 600))
    p.enable_parallel_projection()
    p.add_text(f"Spostamento u", font_size=12)
    p.add_mesh(
        inclusion_def,
        scalars="|u|_cell",
        cmap="viridis",
        show_edges=False,
        scalar_bar_args={"title": "|u| [m]", **SCALAR_BAR_VERTICAL},
    )
    # Contorno della configurazione indeformata
    outline = inclusion_orig.extract_feature_edges(boundary_edges=True, manifold_edges=False)
    p.add_mesh(outline, color="black", line_width=1)
    p.view_xy()
    p.show()

    # Deformata inclusione colorata con J
    pJ = pv.Plotter(window_size=(900, 700))
    pJ.enable_parallel_projection()
    pJ.add_text(f"J = det(F)", font_size=12)
    pJ.add_mesh(
        inclusion_def,
        scalars="J_cell",
        cmap="plasma",
        show_edges=False,
        scalar_bar_args={"title": "J", **SCALAR_BAR_VERTICAL},
    )
    pJ.add_mesh(outline, color="black", line_width=1)
    pJ.view_xy()
    pJ.show()

    # Deformata inclusione colorata con sigma_yy
    pSig = pv.Plotter(window_size=(900, 700))
    pSig.enable_parallel_projection()
    pSig.add_text(f"Stress totale tau22", font_size=12)
    pSig.add_mesh(
        inclusion_def,
        scalars="sigma_yy_cell",
        cmap="coolwarm",
        show_edges=False,
        scalar_bar_args={"title": "[Pa]", **SCALAR_BAR_VERTICAL},
    )
    pSig.add_mesh(outline, color="black", line_width=1)
    pSig.view_xy()
    pSig.show()

    # Deformata inclusione colorata con h_x (h = F^{-T} hl)
    pHx = pv.Plotter(window_size=(900, 700))
    pHx.enable_parallel_projection()
    pHx.add_text(f"Campo magnetico h_x", font_size=12)
    pHx.add_mesh(
        inclusion_def,
        scalars="hx_cell",
        cmap="rainbow",
        show_edges=False,
        scalar_bar_args={"title": "h_x [A/m]", **SCALAR_BAR_VERTICAL},
    )
    pHx.add_mesh(outline, color="black", line_width=1)
    pHx.view_xy()
    pHx.show()

    # Deformata inclusione colorata con h_y
    pHy = pv.Plotter(window_size=(900, 700))
    pHy.enable_parallel_projection()
    pHy.add_text(f"Campo magnetico h_y", font_size=12)
    hy_vals = inclusion_def.cell_data["hy_cell"]
    finite_hy = hy_vals[np.isfinite(hy_vals)]
    clim_hy = None
    if finite_hy.size > 0:
        lo, hi = np.percentile(finite_hy, [1.0, 99.0])
        if lo < hi:
            clim_hy = (lo, hi)
    pHy.add_mesh(
        inclusion_def,
        scalars="hy_cell",
        cmap="rainbow",
        show_edges=False,
        scalar_bar_args={"title": "h_y [A/m]", **SCALAR_BAR_VERTICAL},
        clim=clim_hy,
    )
    pHy.add_mesh(outline, color="black", line_width=1)
    pHy.view_xy()
    pHy.show()

    # Deformata inclusione colorata con |h|
    pHmag = pv.Plotter(window_size=(900, 700))
    pHmag.enable_parallel_projection()
    pHmag.add_text(f"Inclusione deformata (scale={scale_def}) - |h|", font_size=12)
    hmag_vals = inclusion_def.cell_data["|h|_cell"]
    finite_hmag = hmag_vals[np.isfinite(hmag_vals)]
    clim_hmag = None
    if finite_hmag.size > 0:
        lo, hi = np.percentile(finite_hmag, [1.0, 99.0])
        if lo < hi:
            clim_hmag = (lo, hi)
    pHmag.add_mesh(
        inclusion_def,
        scalars="|h|_cell",
        cmap="viridis",
        show_edges=False,
        scalar_bar_args={"title": "|h|", **SCALAR_BAR_VERTICAL},
        clim=clim_hmag,
    )
    pHmag.add_mesh(outline, color="black", line_width=1)
    pHmag.view_xy()
    pHmag.show()

    # Streamlines di bl sulla configurazione indeformata
    n_seed = 16
    x_seed = np.linspace(0.0, L1, n_seed)
    y_seed = np.linspace(0.0, L2, n_seed)
    xx, yy = np.meshgrid(x_seed, y_seed)
    seeds = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)])
    seed_src = pv.PolyData(seeds)

    p_stream = pv.Plotter(window_size=(900, 900))
    p_stream.enable_parallel_projection()
    p_stream.add_text("Streamlines induzione magnetica lagrangiana b_l", font_size=12)
    p_stream.add_mesh(
        grid,
        scalars="region",
        cmap=["#f28e2b", "#4e79a7"],
        show_edges=False,
        opacity=0.25,
        clim=(1, 2),
        show_scalar_bar=False,
    )
    stream = grid.streamlines_from_source(
        seed_src,
        vectors="bl",
        max_time=5 * min(L1, L2),
        initial_step_length=0.02 * min(L1, L2),
        terminal_speed=1e-8,
        integrator_type=45,
    )
    if stream.n_cells > 0:
        p_stream.add_mesh(
            stream,
            scalars="|bl|",
            cmap="magma",
            line_width=2,
            scalar_bar_args={"title": "|b_l| [T] ", **SCALAR_BAR_VERTICAL},
        )
    p_stream.view_xy()
    p_stream.show()

    # Profilo 1D: componente y di bl lungo la retta y=cy
    try:
        import matplotlib.pyplot as plt
    except Exception as err:
        print(f"Plot 1D bl_y non disponibile (matplotlib): {err}")
    else:
        # Profilo su retta verticale x = cx, y in [0, Ly]
        sampled = grid.sample_over_line(pointa=(cx, 0.0, 0.0), pointb=(cx, L2, 0.0), resolution=400)
        bl_line = sampled["bl"]
        if bl_line is not None:
            y_line = sampled.points[:, 1]
            bly_line = bl_line[:, 1]  # componente y
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(y_line, bly_line, lw=1.2)
            ax.axvline(y0, color="k", ls="--", lw=0.8)
            ax.axvline(y1, color="k", ls="--", lw=0.8)
            ax.set_xlabel("y [m]")
            ax.set_ylabel("b_ly [T]")
            ax.set_title(f"b_ly lungo x = {cx}")
            ax.grid(True, ls="--", alpha=0.5)
            plt.tight_layout()
            plt.show()
        else:
            print("Plot 1D bl_y: campo 'bl' non trovato sulla griglia.")

        # Profilo 1D: componente y di hl lungo la stessa retta
        hl_line = sampled["hl"]
        if hl_line is not None:
            hly_line = hl_line[:, 1]
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(y_line, hly_line, lw=1.2, color="tab:orange")
            ax.axvline(y0, color="k", ls="--", lw=0.8)
            ax.axvline(y1, color="k", ls="--", lw=0.8)
            ax.set_xlabel("y [m]")
            ax.set_ylabel("h_ly [A/m]")
            ax.set_title(f"h_ly lungo x = {cx}")
            ax.grid(True, ls="--", alpha=0.5)
            plt.tight_layout()
            plt.show()
        else:
            print("Plot 1D hl_y: campo 'hl' non trovato sulla griglia.")

        # Profilo 1D: componente x di hl lungo x = cx + w_incl/4 (verso destra)
        sampled_hlx = grid.sample_over_line(
            pointa=(cx_quarter_right, 0.0, 0.0), pointb=(cx_quarter_right, L2, 0.0), resolution=400
        )
        hl_line_q = sampled_hlx["hl"]
        if hl_line_q is not None:
            y_line_q = sampled_hlx.points[:, 1]
            hlx_line = hl_line_q[:, 0]
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(y_line_q, hlx_line, lw=1.2, color="tab:green")
            ax.axvline(y0, color="k", ls="--", lw=0.8)
            ax.axvline(y1, color="k", ls="--", lw=0.8)
            ax.set_xlabel("y [m]")
            ax.set_ylabel("h_lx [A/m]")
            ax.set_title(f"h_lx lungo x = {cx_quarter_right:.4f}")
            ax.grid(True, ls="--", alpha=0.5)
            plt.tight_layout()
            plt.show()
        else:
            print("Plot 1D hl_x (linea dx) non disponibile: campo 'hl' mancante.")

# --- 10. Salvataggi VTK
grid.save("solution_original.vtu")
grid_def.save("solution_deformed.vtu")
inclusion_orig.save("solution_inclusion_original.vtu")
inclusion_def.save("solution_inclusion_deformed.vtu")

print(
    "Salvati: solution_original.vtu, solution_deformed.vtu, "
    "solution_inclusion_original.vtu, solution_inclusion_deformed.vtu"
)




