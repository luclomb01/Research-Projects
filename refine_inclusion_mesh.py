import os
from dataclasses import dataclass
from typing import List, Tuple

import gmsh
import meshio
import numpy as np
import ufl
from basix.ufl import element, mixed_element
from dolfinx import fem, mesh, log, default_scalar_type
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.io import XDMFFile, gmshio
from dolfinx.nls.petsc import NewtonSolver
from mpi4py import MPI

# Impostazioni condivise per posizionare le barre dei colori a destra e in verticale
SCALAR_BAR_VERTICAL = {
    "vertical": True,
    "height": 0.8,
    "width": 0.08,
    "position_x": 0.88,
    "position_y": 0.12,
}
 

# --- geometria gmsh con angoli raccordati e raffittimento vicino all'inclusione ---
@dataclass
class Geometry:
    Lx: float = 0.80
    Ly: float = 0.90
    inclusion_w: float = 0.08
    inclusion_h: float = 0.30
    fillet_radius: float = 0.001  # raggio di raccordo

    @property
    def center(self) -> Tuple[float, float]:
        return self.Lx / 2.0, self.Ly / 2.0

    @property
    def inclusion_origin(self) -> Tuple[float, float]:
        return (self.Lx - self.inclusion_w) / 2.0, (self.Ly - self.inclusion_h) / 2.0

    @property
    def inclusion_bounds(self) -> Tuple[float, float, float, float]:
        x0, y0 = self.inclusion_origin
        return x0, y0, x0 + self.inclusion_w, y0 + self.inclusion_h


def rounded_rectangle(x0: float, y0: float, w: float, h: float, r: float) -> Tuple[int, List[int], List[int], List[int]]:
    r = min(r, 0.5 * min(w, h) * 0.999)
    x1, y1 = x0 + w, y0 + h
    p1 = gmsh.model.occ.addPoint(x0 + r, y0, 0)
    p2 = gmsh.model.occ.addPoint(x1 - r, y0, 0)
    p3 = gmsh.model.occ.addPoint(x1, y0 + r, 0)
    p4 = gmsh.model.occ.addPoint(x1, y1 - r, 0)
    p5 = gmsh.model.occ.addPoint(x1 - r, y1, 0)
    p6 = gmsh.model.occ.addPoint(x0 + r, y1, 0)
    p7 = gmsh.model.occ.addPoint(x0, y1 - r, 0)
    p8 = gmsh.model.occ.addPoint(x0, y0 + r, 0)
    c_br = gmsh.model.occ.addPoint(x1 - r, y0 + r, 0)
    c_tr = gmsh.model.occ.addPoint(x1 - r, y1 - r, 0)
    c_tl = gmsh.model.occ.addPoint(x0 + r, y1 - r, 0)
    c_bl = gmsh.model.occ.addPoint(x0 + r, y0 + r, 0)
    l1 = gmsh.model.occ.addLine(p1, p2)  # bottom
    a1 = gmsh.model.occ.addCircleArc(p2, c_br, p3)  # bottom-right
    l2 = gmsh.model.occ.addLine(p3, p4)  # right
    a2 = gmsh.model.occ.addCircleArc(p4, c_tr, p5)  # top-right
    l3 = gmsh.model.occ.addLine(p5, p6)  # top
    a3 = gmsh.model.occ.addCircleArc(p6, c_tl, p7)  # top-left
    l4 = gmsh.model.occ.addLine(p7, p8)  # left
    a4 = gmsh.model.occ.addCircleArc(p8, c_bl, p1)  # bottom-left
    curves = [l1, a1, l2, a2, l3, a3, l4, a4]
    loop = gmsh.model.occ.addCurveLoop(curves)
    surface = gmsh.model.occ.addPlaneSurface([loop])
    bottom_curves = [l1, a1, a4]
    top_curves = [l3, a2, a3]
    return surface, curves, bottom_curves, top_curves


def build_mesh(comm: MPI.Comm, geom: Geometry) -> Tuple[mesh.Mesh, mesh.meshtags, mesh.meshtags]:
    if comm.rank == 0:
        gmsh.initialize()
        gmsh.model.add("refined_inclusion")
        outer = gmsh.model.occ.addRectangle(0.0, 0.0, 0.0, geom.Lx, geom.Ly)
        inc_surface, inc_curves, bottom_curves, top_curves = rounded_rectangle(
            *geom.inclusion_origin, geom.inclusion_w, geom.inclusion_h, geom.fillet_radius
        )
        gmsh.model.occ.fragment([(2, outer)], [(2, inc_surface)])
        gmsh.model.occ.synchronize()

        # Physical groups celle
        entities_2d = gmsh.model.getEntities(dim=2)
        areas = []
        for _, tag in entities_2d:
            xmin, ymin, _, xmax, ymax, _ = gmsh.model.getBoundingBox(2, tag)
            areas.append(((xmax - xmin) * (ymax - ymin), tag))
        areas.sort(key=lambda x: x[0])
        if len(areas) < 2:
            raise RuntimeError("Geometria non valida: attese 2 superfici (matrice+inclusione)")
        inclusion, matrix = areas[0][1], areas[1][1]
        gmsh.model.addPhysicalGroup(2, [inclusion], 1)
        gmsh.model.setPhysicalName(2, 1, "inclusion")
        gmsh.model.addPhysicalGroup(2, [matrix], 2)
        gmsh.model.setPhysicalName(2, 2, "matrix")

        # Physical groups bordi
        boundary_all = [abs(c[1]) for c in gmsh.model.getBoundary([(2, matrix)], oriented=False)]
        def boundary_curves(coord: str, target: float, tol: float = 1e-9):
            out = []
            for tag in boundary_all:
                xmin, ymin, _, xmax, ymax, _ = gmsh.model.getBoundingBox(1, tag)
                if coord == "x" and abs(xmin - target) < tol and abs(xmax - target) < tol:
                    out.append(tag)
                if coord == "y" and abs(ymin - target) < tol and abs(ymax - target) < tol:
                    out.append(tag)
            return out
        left = boundary_curves("x", 0.0)
        right = boundary_curves("x", geom.Lx)
        bottom = boundary_curves("y", 0.0)
        top = boundary_curves("y", geom.Ly)
        if left:
            gmsh.model.addPhysicalGroup(1, left, 1)
        if right:
            gmsh.model.addPhysicalGroup(1, right, 2)
        if bottom:
            gmsh.model.addPhysicalGroup(1, bottom, 3)
        if top:
            gmsh.model.addPhysicalGroup(1, top, 4)
        # Interfaccia inclusione: 5=bottom, 6=top
        gmsh.model.addPhysicalGroup(1, bottom_curves, 5)
        gmsh.model.addPhysicalGroup(1, top_curves, 6)

        # Campi di misura: più fitto vicino all'interfaccia
        interface_curves = bottom_curves + top_curves
        dist_field = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(dist_field, "CurvesList", interface_curves)
        gmsh.model.mesh.field.setNumber(dist_field, "Sampling", 200)

        thresh = gmsh.model.mesh.field.add("Threshold")
        base = min(geom.inclusion_w, geom.inclusion_h)
        # Mesh più fine (ancora) vicino ai bordi interni
        h_interface = base / 90.0    # bordo inclusione molto fitto
        h_inclusion = base / 65.0    # dentro l'inclusione
        h_far = min(geom.Lx, geom.Ly) / 32.0  # esterno leggermente raffinato
        band = base * 0.55           # banda di raffinamento attorno all'inclusione
        gmsh.model.mesh.field.setNumber(thresh, "InField", dist_field)
        gmsh.model.mesh.field.setNumber(thresh, "SizeMin", h_interface)
        gmsh.model.mesh.field.setNumber(thresh, "SizeMax", h_far)
        gmsh.model.mesh.field.setNumber(thresh, "DistMin", 0.0)
        gmsh.model.mesh.field.setNumber(thresh, "DistMax", band)

        x0, y0, x1, y1 = geom.inclusion_bounds
        box = gmsh.model.mesh.field.add("Box")
        gmsh.model.mesh.field.setNumber(box, "VIn", h_inclusion)
        gmsh.model.mesh.field.setNumber(box, "VOut", h_far)
        gmsh.model.mesh.field.setNumber(box, "XMin", x0 - 0.5 * band)
        gmsh.model.mesh.field.setNumber(box, "XMax", x1 + 0.5 * band)
        gmsh.model.mesh.field.setNumber(box, "YMin", y0 - 0.5 * band)
        gmsh.model.mesh.field.setNumber(box, "YMax", y1 + 0.5 * band)
        gmsh.model.mesh.field.setNumber(box, "Thickness", 1e-6)

        min_field = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(min_field, "FieldsList", [thresh, box])
        gmsh.model.mesh.field.setAsBackgroundMesh(min_field)
        gmsh.option.setNumber("Mesh.Algorithm", 6)
        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.optimize("Netgen")

        gmsh.write("refined_inclusion.msh")
    else:
        gmsh.initialize()
    gmsh.finalize()
    domain, cell_tags, facet_tags = gmshio.read_from_msh("refined_inclusion.msh", comm=comm, rank=0, gdim=2)
    return domain, cell_tags, facet_tags


def compute_hmin(domain: mesh.Mesh) -> float:
    """Calcola l'hmin senza usare helper non disponibili in alcune versioni di dolfinx."""
    topo = domain.topology
    tdim = topo.dim
    topo.create_connectivity(tdim, 0)
    c2v = topo.connectivity(tdim, 0)
    coords = domain.geometry.x
    local_min = np.inf
    for cell in range(topo.index_map(tdim).size_local):
        verts = c2v.links(cell)
        pts = coords[verts]
        for i in range(len(verts)):
            for j in range(i + 1, len(verts)):
                dist = np.linalg.norm(pts[i] - pts[j])
                if dist < local_min:
                    local_min = dist
    global_min = domain.comm.allreduce(local_min, op=MPI.MIN)
    return float(global_min)


def save_mesh_preview(msh_path: str, out_path: str = "refined_inclusion_mesh.png") -> None:
    """Salva un'immagine rapida della mesh per controllo visivo."""
    if MPI.COMM_WORLD.rank != 0:
        return
    try:
        import matplotlib.pyplot as plt
        import matplotlib.tri as mtri
    except Exception as err:  # ImportError o backend non disponibile
        print(f"Plot mesh non disponibile (matplotlib non utilizzabile): {err}")
        return

    try:
        msh = meshio.read(msh_path)
    except Exception as err:
        print(f"Impossibile leggere {msh_path} per il plot: {err}")
        return

    tri_cells = None
    for cell_block in msh.cells:
        if cell_block.type in ("triangle", "tri"):
            tri_cells = cell_block.data
            break
    if tri_cells is None:
        print("Nessun elemento triangolare trovato per il plot della mesh.")
        return

    pts = msh.points
    tri = mtri.Triangulation(pts[:, 0], pts[:, 1], tri_cells)
    fig, ax = plt.subplots(figsize=(5.5, 6))
    ax.triplot(tri, lw=0.4, color="tab:gray")
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Refined inclusion mesh")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Mesh plot salvato in {out_path}")


def show_mesh_pyvista(msh_path: str) -> None:
    """Mostra la mesh con PyVista colorando inclusione e matrice."""
    if MPI.COMM_WORLD.rank != 0:
        return
    try:
        import pyvista as pv
    except Exception as err:
        print(f"Plot interattivo PyVista non disponibile: {err}")
        return

    try:
        msh = meshio.read(msh_path)
    except Exception as err:
        print(f"Impossibile leggere {msh_path} per il plot PyVista: {err}")
        return

    tri_blocks = []
    tri_regions = []
    for i, cell_block in enumerate(msh.cells):
        if cell_block.type not in ("triangle", "tri"):
            continue
        tri_blocks.append(cell_block.data)

        region_i = None
        for _, data_list in msh.cell_data.items():
            if len(data_list) > i and data_list[i] is not None:
                cand = np.asarray(data_list[i])
                if cand.shape[0] == cell_block.data.shape[0]:
                    region_i = cand
                    break
        if region_i is None:
            region_i = np.full(cell_block.data.shape[0], 0, dtype=int)
        tri_regions.append(region_i)

    if not tri_blocks:
        print("Nessun elemento triangolare trovato per il plot PyVista.")
        return

    pts = msh.points
    all_tri = np.vstack(tri_blocks)
    num_tri = all_tri.shape[0]
    cells = np.hstack([np.full((num_tri, 1), 3, dtype=np.int64), all_tri]).ravel()
    cell_types = np.full(num_tri, pv.CellType.TRIANGLE, dtype=np.uint8)
    grid = pv.UnstructuredGrid(cells, cell_types, pts)

    region = np.concatenate(tri_regions) if tri_regions else np.full(num_tri, 0, dtype=int)
    grid.cell_data["region"] = region

    cmap = ["#d8d8d8", "#f28e2b", "#4e79a7"]  # 0 dummy, 1 inclusione, 2 matrice
    p = pv.Plotter(window_size=(720, 820))
    p.add_mesh(
        grid,
        scalars="region",
        show_edges=True,
        line_width=0.6,
        cmap=cmap,
        clim=(0, 2),
        categories=True,
        nan_color="white",
        scalar_bar_args={"title": "region (1=inclusion, 2=matrix)", **SCALAR_BAR_VERTICAL},
    )
    p.view_xy()
    p.show(title="Refined inclusion mesh")


def show_inclusion_deformation(grid_base: "pv.UnstructuredGrid", cells_idx: np.ndarray, scale: float = 10.0) -> None:
    """Mostra solo l'inclusione con deformata (warp by vector) colorata per |u|."""
    if MPI.COMM_WORLD.rank != 0:
        return
    try:
        import pyvista as pv
    except Exception as err:
        print(f"Preview deformata non disponibile (PyVista): {err}")
        return

    if cells_idx.size == 0:
        print("Preview deformata: nessuna cella di inclusione trovata.")
        return

    incl = grid_base.extract_cells(cells_idx)
    if incl.n_points == 0:
        print("Preview deformata: inclusione vuota dopo estrazione.")
        return
    if "u" not in incl.point_data:
        print("Preview deformata: campo 'u' mancante, non posso fare warp.")
        return

    warped = incl.warp_by_vector("u", factor=scale)
    outline = incl.extract_feature_edges(boundary_edges=True, manifold_edges=False)
    cmap = "viridis"
    p = pv.Plotter(window_size=(600, 900))
    p.add_mesh(
        warped,
        scalars="|u|" if "|u|" in warped.point_data else None,
        cmap=cmap,
        show_edges=False,
        scalar_bar_args={"title": "|u|", "fmt": "%.2e", **SCALAR_BAR_VERTICAL},
    )
    if outline.n_cells > 0:
        p.add_mesh(outline, color="black", line_width=1)
    p.add_text(f"inclusione, scale={scale}", font_size=10)
    p.view_xy()
    p.show(title="Deformata inclusione")


def show_inclusion_resampled(grid_base: "pv.UnstructuredGrid", cells_idx: np.ndarray, field: str = "|u|") -> None:
    """Mostra l'inclusione come heatmap liscia, campionando scalari su griglia regolare."""
    if MPI.COMM_WORLD.rank != 0:
        return
    try:
        import pyvista as pv
    except Exception as err:
        print(f"Preview liscia non disponibile (PyVista): {err}")
        return
    if cells_idx.size == 0:
        print("Preview liscia: nessuna cella di inclusione trovata.")
        return
    incl = grid_base.extract_cells(cells_idx)
    if incl.n_points == 0:
        print("Preview liscia: inclusione vuota dopo estrazione.")
        return
    if field not in incl.point_data:
        print(f"Preview liscia: campo '{field}' non trovato.")
        return

    # bounding box e griglia regolare fine
    xmin, xmax, ymin, ymax, zmin, zmax = incl.bounds
    nx = 250
    ny = 800
    grid_img = pv.ImageData(spacing=((xmax - xmin) / nx, (ymax - ymin) / ny, 1e-3))
    grid_img.origin = (xmin, ymin, 0.0)
    grid_img.dimensions = (nx + 1, ny + 1, 1)

    sampled = grid_img.sample(incl)
    p = pv.Plotter(window_size=(500, 900))
    p.add_mesh(
        sampled,
        scalars=field,
        cmap="viridis",
        show_edges=False,
        scalar_bar_args={"title": field, **SCALAR_BAR_VERTICAL},
    )
    p.view_xy()
    p.show(title=f"Heatmap {field} (inclusione, resampled)")


gemom = Geometry()
domain, cell_tag, facet_tag = build_mesh(MPI.COMM_WORLD, gemom)
show_mesh_pyvista("refined_inclusion.msh")
cx, cy = gemom.center
cx_quarter_right = cx + 0.25 * gemom.inclusion_w
Lx, Ly = gemom.Lx, gemom.Ly
_, y0 = gemom.inclusion_origin
y1 = y0 + gemom.inclusion_h
ax = gemom.inclusion_w / 2.0
ay = gemom.inclusion_h / 2.0


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
    print("Facet tags disponibili:", unique_facets)
    for tag_id in (5, 6, 1, 2, 3, 4):
        count = facet_tag.find(tag_id).size
        print(f"  tag {tag_id}: {count} facets")

# Se mancano i tag esterni (1-4) li ricostruiamo geometricamente
missing_ext = all(facet_tag.find(t).size == 0 for t in (1, 2, 3, 4))
if missing_ext:
    def left(x): return np.isclose(x[0], 0.0)
    def right(x): return np.isclose(x[0], Lx)
    def bottom(x): return np.isclose(x[1], 0.0)
    def top(x): return np.isclose(x[1], Ly)

    boundary_locators = {1: left, 2: right, 3: bottom, 4: top}
    facet_blocks = [facet_tag.indices]
    value_blocks = [facet_tag.values]
    for marker, locator in boundary_locators.items():
        facets = mesh.locate_entities_boundary(domain, fdim, locator)
        facet_blocks.append(facets)
        value_blocks.append(np.full(facets.shape, marker, dtype=np.int32))

    facets_all = np.hstack(facet_blocks)
    values_all = np.hstack(value_blocks)
    order = np.argsort(facets_all)
    facets_all = facets_all[order]
    values_all = values_all[order]
    unique_facets, unique_idx = np.unique(facets_all, return_index=True)
    facet_tag = mesh.meshtags(domain, fdim, unique_facets, values_all[unique_idx])
    if domain.comm.rank == 0:
        print("Ricostruiti tag 1-4 in base alla geometria.")

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
U.name = "U"

# Misure
metadata = {"quadrature_degree": 8}  # Grado di quadratura ridotto
dx = ufl.Measure("dx", domain=domain, subdomain_data=cell_tag, metadata=metadata)
dS = ufl.Measure("dS", domain=domain, subdomain_data=facet_tag, metadata=metadata)

# Costanti/materiali 
mu0 = fem.Constant(domain, default_scalar_type(1.2566e-6))
mu = fem.Constant(domain, default_scalar_type(1.5))
G = fem.Constant(domain, default_scalar_type(1e6))  # shear modulus gomma

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
mu_v = fem.Constant(domain, default_scalar_type(4e4))  # shear vuoto molto piccolo
K_v  = fem.Constant(domain, default_scalar_type(4e4))  # bulk vuoto molto piccolo
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
tau = 0.0  # intensità (modulo invariato rispetto al caso precedente)
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
H_target = 1e6
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

# --- check: max |u| sui nodi a y=0 (bordo inferiore) ---
V_u_chk, dofs_u_chk = W.sub(0).collapse()
embed_dim_chk = domain.geometry.x.shape[1]  # = 3 in 2D (con z=0)
coords_chk = V_u_chk.tabulate_dof_coordinates().reshape((-1, embed_dim_chk))
u_chk = U.x.array[dofs_u_chk].reshape((-1, gdim))  # qui resta 2 (ux, uy)
mask_bottom = np.isclose(coords_chk[:, 1], 0.0, atol=1e-14)
max_u_bottom = 0.0 if not np.any(mask_bottom) else float(np.linalg.norm(u_chk[mask_bottom], axis=1).max())
print("max |u| su y=0 =", max_u_bottom)


print(f"Convergenza: sì, Iterazioni: {num_its}")


# Collassa gli spazi per avere funzioni standalone
u = fem.Function(V_u_c)
phi = fem.Function(V_phi_c)
u.name = "u"
phi.name = "phi"
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
# VISUALIZZAZIONE con PyVista (2D) – versione robusta
# =========================
import pyvista as pv
from dolfinx.plot import vtk_mesh

if not os.environ.get("DISPLAY"):
    pv.OFF_SCREEN = True

# --- 1. Spazio scalare P1 per post-processing (coerente con la mesh di dominio)
V1 = fem.functionspace(domain, ("Lagrange", 1))

# Funzione di comodo: crea e riempie un Function in V1 da una espressione UFL
def interp_to_V1(expr_ufl, name=None):
    expr = fem.Expression(expr_ufl, V1.element.interpolation_points())
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
sigma_yy_expr = fem.Expression(sigma_solid[1, 1], Q0.element.interpolation_points())
sigma_yy_cell = fem.Function(Q0)
sigma_yy_cell.name = "sigma_yy_cell"
sigma_yy_cell.interpolate(sigma_yy_expr)
sigma_yy_cell.x.scatter_forward()
sigma_yy_neg_cell = fem.Function(Q0)
sigma_yy_neg_cell.name = "sigma_yy_neg_cell"
sigma_yy_neg_cell.interpolate(fem.Expression(-sigma_solid[1, 1], Q0.element.interpolation_points()))
sigma_yy_neg_cell.x.scatter_forward()

# Altri campi cell-based (DG0) per plot su inclusione
u_mag_expr = fem.Expression(ufl.sqrt(ufl.inner(u, u)), Q0.element.interpolation_points())
u_mag_cell = fem.Function(Q0)
u_mag_cell.name = "|u|_cell"
u_mag_cell.interpolate(u_mag_expr)
u_mag_cell.x.scatter_forward()

J_cell = fem.Function(Q0)
J_cell.name = "J_cell"
J_cell.interpolate(fem.Expression(J_det_expr, Q0.element.interpolation_points()))
J_cell.x.scatter_forward()

hx_cell = fem.Function(Q0)
hx_cell.name = "hx_cell"
hx_cell.interpolate(fem.Expression(chi * h_field[0], Q0.element.interpolation_points()))
hx_cell.x.scatter_forward()

hy_cell = fem.Function(Q0)
hy_cell.name = "hy_cell"
hy_cell.interpolate(fem.Expression(chi * h_field[1], Q0.element.interpolation_points()))
hy_cell.x.scatter_forward()

h_mag_cell = fem.Function(Q0)
h_mag_cell.name = "|h|_cell"
h_mag_cell.interpolate(fem.Expression(chi * ufl.sqrt(ufl.inner(h_field, h_field)), Q0.element.interpolation_points()))
h_mag_cell.x.scatter_forward()

# Annulla (o pone a NaN) i valori cell-based fuori dall'inclusione per evitare bleed dal vuoto
dofs_inclusion_cells = fem.locate_dofs_topological(Q0, tdim, cell_tag.find(1))
all_cell_dofs = np.arange(Q0.dofmap.index_map.size_local, dtype=np.int32)
mask_outside = np.ones_like(all_cell_dofs, dtype=bool)
mask_outside[dofs_inclusion_cells] = False
for f_cell in (sigma_yy_cell, u_mag_cell, J_cell, hx_cell, hy_cell, h_mag_cell, sigma_yy_neg_cell):
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
grid.cell_data["-sigma_yy_cell"] = sigma_yy_neg_cell.x.array.copy()
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
        scalars="-sigma_yy_cell",
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
    x_seed = np.linspace(0.0, Lx, n_seed)
    y_seed = np.linspace(0.0, Ly, n_seed)
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
        max_time=5 * min(Lx, Ly),
        initial_step_length=0.02 * min(Lx, Ly),
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
        sampled = grid.sample_over_line(pointa=(cx, 0.0, 0.0), pointb=(cx, Ly, 0.0), resolution=400)
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
            pointa=(cx_quarter_right, 0.0, 0.0), pointb=(cx_quarter_right, Ly, 0.0), resolution=400
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
