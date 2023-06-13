import math
import bpy
import numpy as np
from mathutils import *

SCALE_CONSTANT = 8192
LINE_WIDTH_FACTOR = 2000.0

#region [Color Utilities]
def linear_to_srgb(color):
    """
    Convert a Linear RGB value to an sRGB one. Can be replaced by from_scene_linear_to_srgb() if Blender version >= 3.2
    """
    s_color = 0
    if color < 0.0031308:
        s_color = 12.92 * color
    else:
        s_color = 1.055 * math.pow(color, 1/2.4) - 0.055
    return s_color

def srgb_to_linear(color):
    '''
    Can be replaced by from_srgb_to_scene_linear() if Blender version >= 3.2
    '''
    if color<0:
        return 0
    elif color<0.04045:
        return color/12.92
    else:
        return ((color+0.055)/1.055)**2.4

def hex_to_rgb(h, to_linear = False) -> Color:
    r = (h & 0xff0000) >> 16 
    g = (h & 0x00ff00) >> 8
    b = (h & 0x0000ff)
    if to_linear:
        return Color((srgb_to_linear(r/255.0), srgb_to_linear(g/255.0), srgb_to_linear(b/255.0)))
    else:
        return Color((r/255.0, g/255.0, b/255.0))
    
def rgb_to_hex_code(color:Color) -> str:
    r,g,b = int(color[0]*255), int(color[1]*255), int(color[2]*255)
    hex_code = f"#{r:02x}{g:02x}{b:02x}"
    return hex_code
#endregion

#region [Math Utilities]
def smoothstep(x):
    if x<0:
        return 0
    if x>1:
        return 1
    return 3*(x**2) - 2*(x**3)

def get_concyclic_info(x0,y0,x1,y1,x2,y2):
     """
     Given three 2D points, calcualte their concyclic center and reciprocal radius
     """
     # Coefficients of the formula
     a = 2*(x1-x0)
     b = 2*(y1-y0)
     c = x1**2 - x0**2 + y1**2 - y0**2
     d = 2*(x2-x0)
     e = 2*(y2-y0)
     f = x2**2 - x0**2 + y2**2 - y0**2
     det = a*e - b*d
     
     # Case of colinear points
     if math.isclose( det, 0 ):
         return 0, None
     
     # Solution of the formula
     xc = (c*e-f*b)/det
     yc = (a*f-c*d)/det
     center = Vector(( xc, yc ))
     radius = math.sqrt((x0-xc)**2 + (y0-yc)**2)
     
     return 1.0/radius, center

def get_2d_squared_distance(co1, scale_factor1, co2, scale_factor2):
    """Euclidean distance that takes the scale factors into consideration"""
    delta = [co1[0]/scale_factor1 - co2[0]/scale_factor2, co1[1]/scale_factor1 - co2[1]/scale_factor2]
    return delta[0]*delta[0] + delta[1]*delta[1]

def is_poly_in_poly(poly1, poly2):
    """False if either at least one point is outside, or all points are on the boundary"""
    import pyclipper
    inner_count = 0
    for co in poly1:
        res = pyclipper.PointInPolygon(co, poly2)
        if res == 0:
            return False
        if res == 1:
            inner_count += 1
    return inner_count > 0

def get_an_inside_co(poly):
    """Return a coordinate that is inside an integer polygon as part of the triangle input"""
    import pyclipper
    delta = 64
    while delta > 1:
        for co in poly:
            for co_ in [(co[0]-delta, co[1]-delta), (co[0]-delta, co[1]+delta), 
                        (co[0]+delta, co[1]-delta), (co[0]+delta, co[1]+delta)]:
                if pyclipper.PointInPolygon(co_, poly):
                    return co_
        delta /= 2
    return None

def stroke_bound_box_overlapping(s1, s2, t_mat):
    """Judge if bound boxes of two strokes overlap in any given 2D plane"""
    
    # Bpy provides two bound box points for each stroke, however we need all 8 points
    bound_points = []
    for s in (s1,s2):
        bound_points.append([])
        for x in (s.bound_box_min.x, s.bound_box_max.x):
            for y in (s.bound_box_min.y, s.bound_box_max.y):
                for z in (s.bound_box_min.z, s.bound_box_max.z):
                    bound_points[-1].append(np.array([x,y,z]))
    for i in range(2):
        for j in range(8):
            co_2d = bound_points[i][j].dot(t_mat)
            bound_points[i][j] = co_2d
    # Check first two axes if strokes overlap on them
    bound_points = np.array(bound_points)
    for i in range(2):
        pa_min, pa_max = min(bound_points[0,:,i]), max(bound_points[0,:,i])
        pb_min, pb_max = min(bound_points[1,:,i]), max(bound_points[1,:,i])
        if pa_min > pb_max or pb_min > pa_max:
            return False
    return True        

#endregion

#region [Bpy Utilities]
def is_stroke_line(stroke, gp_obj):
    """
    Check if a stroke does not have fill material
    """
    mat_idx = stroke.material_index
    material = gp_obj.material_slots[mat_idx].material
    return not material.grease_pencil.show_fill

def is_stroke_locked(stroke, gp_obj):
    """
    Check if a stroke has the material that is being locked or invisible
    """
    mat_idx = stroke.material_index
    material = gp_obj.material_slots[mat_idx].material
    return material.grease_pencil.lock or material.grease_pencil.hide

def is_stroke_hole(stroke, gp_obj):
    """
    Check if a stroke has a material with fill holdout
    """
    mat_idx = stroke.material_index
    material = gp_obj.material_slots[mat_idx].material
    return material.grease_pencil.use_fill_holdout

def is_layer_locked(layer: bpy.types.GPencilLayer):
    """
    Check if a layer should be edited
    """
    return layer.lock or layer.hide

def get_stroke_length(stroke: bpy.types.GPencilStroke = None, co_list = None):
    """Calculate the total length of a stroke"""
    res = 0
    if stroke:
        for i,point in enumerate(stroke.points):
            point0 = stroke.points[i-1]
            if i==0 and not stroke.use_cyclic:
                continue
            res += (point.co - point0.co).length
    if co_list:
        for i,co in enumerate(co_list):
            co0 = co_list[i-1]
            if i>0:
                res += math.sqrt(get_2d_squared_distance(co,1,co0,1))
    return max(res,1e-9)
#endregion

#region [3D<->2D Utilities]
def pca(data):
    """
    Calculate the matrix that projects 3D shapes to 2D in the optimal way
    """
    mean = np.mean(data, axis=0)
    centered_data = data - mean
    covariance_matrix = np.cov(centered_data, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

    idx = eigenvalues.argsort()[::-1]  
    values =  eigenvalues[idx]
    mat = eigenvectors[:,idx]
    return mat, values

def get_transformation_mat(mode='VIEW', gp_obj=None, strokes=[], operator=None):
    """
    Get the transformation matrix and its inverse matrix given a 2D working plane.
    The x and y values of transformed coordinates will be used for 2D operators.
    NumPy arrays are preferred compared with the Bpy Matrix class
    """
    presets = {'X-Z': np.array([[1,0,0],
                                [0,0,-1],
                                [0,1,0]]),
               'Y-Z': np.array([[0,0,1],
                                [1,0,0],
                                [0,1,0]]),
               'X-Y': np.array([[1,0,0],
                                [0,1,0],
                                [0,0,1]])}
    view_matrix = np.array(bpy.context.space_data.region_3d.view_matrix.to_3x3())
    obj_rotation = np.array(gp_obj.matrix_world.to_3x3().normalized())
    view_matrix = view_matrix @ obj_rotation

    # Use orthogonal planes
    if mode in presets:
        mat = presets[mode]
        return mat, np.linalg.pinv(mat)
    
    # Auto mode: use PCA combined with the view vector to determine
    for _ in range(1):
        if mode == 'AUTO':
            data = []
            for stroke in strokes:
                for point in stroke.points:
                    data.append(point.co)
            if len(data) < 2:                           # PCA cannot handle. Use view plane instead.
                break
            
            mat, eigenvalues = pca(np.array(data))
            
            # Cases that result has 1 or 3 dimensions
            if eigenvalues[1] < 1e-6 and eigenvalues[2] < 1e-6:
                break
            if eigenvalues[-1] > 1e-6 and operator:
                operator.report({"INFO"}, "More than one 2D plane detected. The result may be inaccurate.")
            
            # Align the result with the current view
            if mat[:,2].dot(view_matrix[2,:]) < 0:  
                mat[:,2] *= -1                          # 1. Make the depth axis facing the screen
            if np.cross(mat[:,0], mat[:,1]).dot(mat[:,2]) < 0:
                mat[:,1] *= -1                          # 2. Ensure the rule of calculating normals is consistent
            # 3. Rotate the first two axes to align the UV/tangent
            target_up_axis = (np.linalg.pinv(view_matrix) @ np.array((0,1,0))).dot(mat)
            delta = Vector((0,1)).angle_signed(target_up_axis[:2])
            rotation = np.array(Matrix.Rotation(-delta,3,mat[:,2]))
            mat = rotation @ mat
            
            return mat, np.linalg.pinv(mat)
        
    # Use view plane
    mat = np.transpose(view_matrix)
    if np.cross(mat[:,0], mat[:,1]).dot(mat[:,2]) < 0:
        mat[:,1] *= -1 
    return mat, np.linalg.pinv(mat)

def get_2d_co_from_strokes(stroke_list, t_mat, scale = False, correct_orientation = False, scale_factor=None):
    """
    Convert points from a list of strokes to 2D coordinates. Scale them to be compatible with Clipper.
    Return a 2D coordinate list, a z-depth list and the scale factor 
    """
    poly_list = []
    poly_depth_list = []
    w_bound = [math.inf, -math.inf]
    h_bound = [math.inf, -math.inf]

    for stroke in stroke_list:
        co_list = []
        depth_list = []
        for point in stroke.points:
            transformed_co = np.array(point.co).dot(t_mat)
            co_list.append([transformed_co[0],transformed_co[1]])
            depth_list.append(transformed_co[2])
            w_bound[0] = min(w_bound[0], co_list[-1][0])
            w_bound[1] = max(w_bound[1], co_list[-1][0])
            h_bound[0] = min(h_bound[0], co_list[-1][1])
            h_bound[1] = max(h_bound[1], co_list[-1][1])
        poly_list.append(co_list)
        poly_depth_list.append(depth_list)

    if scale and not scale_factor:
        poly_W = w_bound[1] - w_bound[0]
        poly_H = h_bound[1] - h_bound[0]
        
        if math.isclose(poly_W, 0) and math.isclose(poly_H, 0):
            scale_factor = 1
        elif math.isclose(poly_W, 0):
            scale_factor = SCALE_CONSTANT / min(poly_H, SCALE_CONSTANT)
        elif math.isclose(poly_H, 0):
            scale_factor = SCALE_CONSTANT / min(poly_W, SCALE_CONSTANT)
        else:
            scale_factor = SCALE_CONSTANT / min(poly_W, poly_H, SCALE_CONSTANT)

    if scale_factor:
        for co_list in poly_list:
            for co in co_list:
                co[0] *= scale_factor
                co[1] *= scale_factor

    # Since Grease Pencil does not care whether the sequence of points is clockwise,
    # Clipper may regard some strokes as negative polygons, which needs a fix 
    if correct_orientation:
        import pyclipper
        for co_list in poly_list:
            if not pyclipper.Orientation(co_list):
                co_list.reverse()

    return poly_list, poly_depth_list, scale_factor

def xy0(vec, depth=0):
    """Empty the depth for 2D lookup, e.g., KDTree search"""
    return Vector((vec[0],vec[1],depth))

class DepthLookupTree:
    """
    Data structure based on KDTree to get depth value from 2D coordinates.
    Some operators have more complicated rules to determine the depth and do not use this class
    """
    def __init__(self, poly_list, depth_list):
        self.co2d = []
        self.depth = []
        self.count = 0
        for i,co_list in enumerate(poly_list):
            for j,co in enumerate(co_list):
                self.co2d.append(xy0(co))
                self.depth.append(depth_list[i][j])
                self.count += 1
        self.kdtree = kdtree.KDTree(self.count)
        for i in range(self.count):
            self.kdtree.insert(self.co2d[i], i)
        self.kdtree.balance()

    def get_depth(self, co):
        _, i, _ = self.kdtree.find(xy0(co))
        return self.depth[i]

def restore_3d_co(co, depth, inv_mat, scale_factor=1):
    """Perform inverse transformation on 2D coordinates"""
    vec = np.array([co[0]/ scale_factor,
                    co[1]/ scale_factor,
                    depth]).dot(inv_mat)
    return vec
#endregion
