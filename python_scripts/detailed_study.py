from dataclasses import dataclass
import numpy as np
import pyvista as pv
import gmsh
import meshio
from dolfinx.io import gmsh as gmshio
from mpi4py import MPI

# create a geometry class
@dataclass
class Geometry:
    L1: float = 1.0
    L2: float = 1.0
    l1: float = 0.5
    l2: float = 0.5
    r: float = 0.001

    @property
    def center(self):
        return self.L1 / 2, self.L2 / 2
    
    # bottom left corner of the inclusion
    @property
    def inclusion_origin(self):
        return (self.L1 - self.l1) / 2, (self.L2 - self.l2) / 2
    
    @property
    def inclusion_bounds(self):
        x0, y0 = self.inclusion_origin
        return x0, y0, x0 + self.l1, y0 + self.L2

def rounded_rectangle(x0, y0, w, h, r):
    r = min(r, w/2, h/2)
    # top right corner (x1, y1)
    x1 = x0 + w
    y1 = y0 + h
    # define the points for the rounded rectangle
    p1 = gmsh.model.occ.addPoint(x0 +  r, y0, 0)
    p2 = gmsh.model.occ.addPoint(x1 -  r, y0, 0)
    p3 = gmsh.model.occ.addPoint(x1, y0 +  r, 0) 
    p4 = gmsh.model.occ.addPoint(x1, y1 -  r, 0)
    p5 = gmsh.model.occ.addPoint(x1 -  r, y1, 0)
    p6 = gmsh.model.occ.addPoint(x0 +  r, y1, 0)
    p7 = gmsh.model.occ.addPoint(x0, y1 -  r, 0)
    p8 = gmsh.model.occ.addPoint(x0, y0 +  r, 0)
    c_br = gmsh.model.occ.addPoint(x1 - r, y0 + r, 0)
    c_tr = gmsh.model.occ.addPoint(x1 - r, y1 - r, 0)
    c_tl = gmsh.model.occ.addPoint(x0 + r, y1 - r, 0)
    c_bl = gmsh.model.occ.addPoint(x0 + r, y0 + r, 0)
    # define the lines for the rounded rectangle
    l1 = gmsh.model.occ.addLine(p1, p2)
    a1 = gmsh.model.occ.addCircleArc(p2, c_br, p3)
    l2 = gmsh.model.occ.addLine(p3, p4)
    a2 = gmsh.model.occ.addCircleArc(p4, c_tr, p5)
    l3 = gmsh.model.occ.addLine(p5, p6)
    a3 = gmsh.model.occ.addCircleArc(p6, c_tl, p7)
    l4 = gmsh.model.occ.addLine(p7, p8)
    a4 = gmsh.model.occ.addCircleArc(p8, c_bl, p1)
    curves = [l1, a1, l2, a2, l3, a3, l4, a4]
    loop = gmsh.model.occ.addCurveLoop(curves)
    surface = gmsh.model.occ.addPlaneSurface([loop])
    bottom_curves = [l1, a1, a4]
    top_curves = [l3, a2, a3]
    return surface, curves, bottom_curves, top_curves
