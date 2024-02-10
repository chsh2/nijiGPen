import numpy as np
import bmesh
from scipy.optimize import minimize

class MeshDepthSolver:
    """
    Calculating the depth of each vertex of a generated mesh.
    Most math expressions are reduced for higher efficiency. Please refer to the documentation site for the original formula:
        https://chsh2.github.io/nijigp/docs/developer_notes/algorithms/#mesh-generation
    """
    N: int             # Number of vertices, i.e. dimension of the problem
    z0: np.array       # depth of each vertex
    bounds: list
    norm_z: np.array   # Z value of vertex normal
    graph_coef: map    # edge map recording vertex connectivity

    def initialize_from_bmesh(self, bm: bmesh.types.BMesh, scale_factor, boundary_map, is_depth_negative):
        self.N = len(bm.verts)
        self.z0 = np.zeros(self.N)
        self.norm_z = np.zeros(self.N)
        self.bounds = []
        self.graph_coef = {}
        
        depth_layer = bm.verts.layers.float.get('Depth')
        normal_map_layer = bm.verts.layers.float_vector.get('NormalMap')
        
        for i,vert in enumerate(bm.verts):
            self.z0[i] = vert[depth_layer]
            self.norm_z[i] = vert[normal_map_layer].z * 2 - 1
            if (int(vert.co.x), int(vert.co.y)) in boundary_map:
                self.bounds.append((0,0))
            elif is_depth_negative:
                self.bounds.append((None,0))
            else:
                self.bounds.append((0,None))
            
        for i,edge in enumerate(bm.edges):
            v0 = edge.verts[0]
            v1 = edge.verts[1]
            # Frequently used constants that do not change in each iteration
            self.graph_coef[(v0.index, v1.index)] = ((v0.co.x - v1.co.x) * (v0[normal_map_layer].x * 2 - 1) 
                                                    + (v0.co.y - v1.co.y) * (v0[normal_map_layer].y * 2 - 1)) / scale_factor
            self.graph_coef[(v1.index, v0.index)] = ((v1.co.x - v0.co.x) * (v1[normal_map_layer].x * 2 - 1) 
                                                    + (v1.co.y - v0.co.y) * (v1[normal_map_layer].y * 2 - 1)) / scale_factor

    def solve(self, max_iter: int):
        def cost(z):
            cost = 0
            for edge in self.graph_coef:
                i0, i1 = edge[0], edge[1]
                # Squared production of edge tangent and vertex normal
                cost += (self.graph_coef[edge] + (z[i0] - z[i1]) * self.norm_z[i0]) ** 2
            return cost
        
        def grad(z):
            der = np.zeros(self.N)
            for edge in self.graph_coef:
                i0, i1 = edge[0], edge[1]
                der[i0] += (2 * (self.norm_z[i0]**2) * (z[i0] - z[i1]) + 2 * self.norm_z[i0] * self.graph_coef[edge])
                der[i1] += (2 * (self.norm_z[i0]**2) * (z[i1] - z[i0]) - 2 * self.norm_z[i0] * self.graph_coef[edge])
            return der

        res = minimize(cost, self.z0, method='L-BFGS-B', jac=grad, bounds=self.bounds, options={'maxiter':max_iter})
        self.z0 = res.x
    
    def write_back(self, bm: bmesh.types.BMesh):
        depth_layer = bm.verts.layers.float.get('Depth')
        for i,vert in enumerate(bm.verts):
            if not np.isclose(vert[depth_layer], 0):
                vert[depth_layer] = self.z0[i]