import gmsh
from dolfinx.io import gmsh as gmshio
from mpi4py import MPI
from typing import Tuple
from dolfinx import mesh
from geometry_build import Geometry, rounded_rectangle


def build_mesh(comm: MPI.Comm, geometry: Geometry) -> Tuple[mesh.Mesh, mesh.meshtags, mesh.meshtags]:
    if comm.rank == 0:
        gmsh.initialize()
        gmsh.model.add("refined_inclusion")
        outer = gmsh.model.occ.addRectangle(0.0, 0.0, 0.0, geometry.L1, geometry.L2) # origin at (0, 0) and dimensions L1 x L2
        inc_surface, inc_curves, bottom_curves, top_curves = rounded_rectangle(
            *geometry.inclusion_origin, geometry.l1, geometry.l2, geometry.r
        )
        gmsh.model.occ.fragment([(2, outer)], [(2, inc_surface)])
        gmsh.model.occ.synchronize()

        entities2d = gmsh.model.getEntities(dim=2)
        areas = []
        for _, tag in entities2d:
            xmin, ymin, _, xmax, ymax, _ = gmsh.model.getBoundingBox(2, tag)
            areas.append(((xmax - xmin) * (ymax - ymin), tag))
        areas.sort(key=lambda x: x[0])
        if len(areas) < 2:
            raise ValueError("Non sono riuscito a identificare correttamente la matrice e l'inclusione nella geometria, assicurati che siano state definite correttamente e che abbiano aree distinte.")
        inclusion, matrix = areas[0][1], areas[1][1] # extract the tags
        gmsh.model.addPhysicalGroup(2, [inclusion], 1)
        gmsh.model.setPhysicalName(2, 1, "inclusion")
        gmsh.model.addPhysicalGroup(2, [matrix], 2)
        gmsh.model.setPhysicalName(2, 2, "matrix")

        # boundaries
        # --- Identificazione robusta dei bordi post-fragment ---
        eps = 1e-5  # Tolleranza più rilassata per assorbire il rumore numerico di OpenCASCADE
        L1, L2 = geometry.L1, geometry.L2

        # Sintassi: getEntitiesInBoundingBox(xmin, ymin, zmin, xmax, ymax, zmax, dim)
        left = [tag for d, tag in gmsh.model.getEntitiesInBoundingBox(-eps, -eps, -eps, eps, L2+eps, eps, dim=1)]
        right = [tag for d, tag in gmsh.model.getEntitiesInBoundingBox(L1-eps, -eps, -eps, L1+eps, L2+eps, eps, dim=1)]
        bottom = [tag for d, tag in gmsh.model.getEntitiesInBoundingBox(-eps, -eps, -eps, L1+eps, eps, eps, dim=1)]
        top = [tag for d, tag in gmsh.model.getEntitiesInBoundingBox(-eps, L2-eps, -eps, L1+eps, L2+eps, eps, dim=1)]

        if left: 
            gmsh.model.addPhysicalGroup(1, left, 1)
        if right: 
            gmsh.model.addPhysicalGroup(1, right, 2)
        if bottom:
            gmsh.model.addPhysicalGroup(1, bottom, 3)
        if top:
            gmsh.model.addPhysicalGroup(1, top, 4)
        
        gmsh.model.addPhysicalGroup(1, bottom_curves, 5)
        gmsh.model.addPhysicalGroup(1, top_curves, 6)

        # measure fields
        interface_curves = bottom_curves + top_curves
        distance_field = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(distance_field, 'CurvesList', interface_curves)
        gmsh.model.mesh.field.setNumber(distance_field, "Sampling", 200)

        threshold_field = gmsh.model.mesh.field.add("Threshold")
        base = min(geometry.l1, geometry.l2) 
        h_interface = base / 90.0
        h_inclusion  = base / 65.0
        h_far = min(geometry.L1, geometry.L2) / 32.0
        band = base * 0.55
        gmsh.model.mesh.field.setNumber(threshold_field, "InField", distance_field)
        gmsh.model.mesh.field.setNumber(threshold_field, "SizeMin", h_interface)
        gmsh.model.mesh.field.setNumber(threshold_field, "SizeMax", h_far)
        gmsh.model.mesh.field.setNumber(threshold_field, "DistMin", 0.0)
        gmsh.model.mesh.field.setNumber(threshold_field, "DistMax", band)

        x0, y0, x1, y1 = geometry.inclusion_bounds
        box = gmsh.model.mesh.field.add("Box")
        gmsh.model.mesh.field.setNumber(box, "VIn", h_inclusion)
        gmsh.model.mesh.field.setNumber(box, "VOut", h_far)
        gmsh.model.mesh.field.setNumber(box, "XMin", x0 - 0.5 * band)
        gmsh.model.mesh.field.setNumber(box, "XMax", x1 + 0.5 * band)
        gmsh.model.mesh.field.setNumber(box, "YMin", y0 - 0.5 * band)
        gmsh.model.mesh.field.setNumber(box, "YMax", y1 + 0.5 * band)
        gmsh.model.mesh.field.setNumber(box, "Thickness", 1e-6)

        min_field = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(min_field, "FieldsList", [threshold_field, box])
        gmsh.model.mesh.field.setAsBackgroundMesh(min_field)
        gmsh.option.setNumber("Mesh.Algorithm", 6) # use the frontal-Delaunay algorithm for better mesh quality with size fields
        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.optimize("Netgen") # optimize the mesh to improve element quality, this can help reduce numerical errors in the simulation; we use the Netgen optimization algorithm which is effective for 2D meshes
        gmsh.write("mesh.msh")
    
    else: 
        gmsh.initialize()
    gmsh.finalize()
    mesh_data = gmshio.read_from_msh("mesh.msh", comm=comm, rank=0, gdim=2)
    domain = mesh_data[0]
    cell_tags = mesh_data[1]
    facet_tags = mesh_data[2]
    return domain, cell_tags, facet_tags
