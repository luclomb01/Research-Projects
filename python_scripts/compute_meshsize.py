from mpi4py import MPI
from dolfinx import mesh
import numpy as np

def compute_hmin(domain: mesh.Mesh) -> float:
    topo = domain.topology
    tdim = topo.dim
    topo.create_connectivity(tdim, 0)
    c2v = topo.connectivity(tdim, 0)
    coords = domain.geometry.x
    local_min = np.inf

    for cell in range(topo.index_map(tdim).size_local):
        vertices = c2v.links(cell)
        pts = coords[vertices]
        for i in range(len(vertices)):
            for j in range(i+1, len(vertices)):
                dist = np.linalg.norm(pts[i] - pts[j])
                if dist < local_min:
                    local_min = dist
    global_min = domain.comm.allreduce(local_min, op=MPI.MIN)
    return float(global_min)
