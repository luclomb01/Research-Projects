import numpy as np
import meshio
from mpi4py import MPI


SCALAR_BAR_VERTICAL = {
    "vertical": True,
    "height": 0.8,
    "width": 0.08,
    "position_x": 0.88,
    "position_y": 0.12,
}

def show_mesh_pyvista(msh_path: str) -> None:

    if MPI.COMM_WORLD.rank != 0:
        return
    try: 
        import pyvista as pv
    except Exception as err:
        print(f"Plot interattivo PyVista non disponibile: {err}")
        return
    
    try: 
        msh = meshio.read(msh_path) # read the mesh file using meshio, this gives us a mesh object that contains the points, cells, and cell data (e.g., physical tags)
    except Exception as err:
        print(f"Errore nella lettura del file mesh: {err}")
        return
    
    tri_blocks = []
    tri_regions = []

    for i, cell_block in enumerate(msh.cells):
        if cell_block.type not in ("triangle", "tri"):
            continue
        tri_blocks.append(cell_block.data) # add the triangular cell data (i.e., the vertex indices for each triangle)
        # remember that the index i corresponds to the block index in cell_data
        
        region_i = None
        for _, data_list in msh.cell_data.items():
            if len(data_list) > i and data_list[i] is not None:
                cand = np.asarray(data_list[i])
                # check if the number of cells matches
                if cand.shape[0] == cell_block.data.shape[0]:
                    region_i = cand
                    break
        if region_i is None:
            region_i = np.full(cell_block.data.shape[0], 0, dtype=int)
        tri_regions.append(region_i)
        
    if not tri_blocks:
        print("Nessun blocco di celle triangolari trovato nel file mesh.")
        return
    
    pts = msh.points
    all_tri = np.vstack(tri_blocks) # concatenate all the triangular cell blocks into a single array of vertex indices for the triangles
    num_tri = all_tri.shape[0] # number of triangular cells
    cells = np.hstack([np.full((num_tri, 1), 3, dtype=np.int64), all_tri]).ravel()
    cell_types = np.full(num_tri, pv.CellType.TRIANGLE, dtype=np.uint8)
    grid = pv.UnstructuredGrid(cells, cell_types, pts)

    region = np.concatenate(tri_regions) if tri_regions else np.full(num_tri, 0, dtype=int)
    grid.cell_data["region"] = region

    cmap = ["#d8d8d8", "#f28e2b", "#4e79a7"]
    p = pv.Plotter(window_size=(800, 600))
    p.add_mesh(
        grid,
        scalars="region",
        show_edges=True,
        line_width=0.6,
        cmap=cmap,
        clim=(0, 2),
        categories=True,
        nan_color="white",
        scalar_bar_args={"title": "region (1=inclusion, 2=matrix)", **SCALAR_BAR_VERTICAL}
    )
    p.view_xy()
    p.show(title="Mesh Visualization with PyVista")
