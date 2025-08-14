import math
import bpy
import numpy as np
from mathutils import *
from .api_router import *

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
    """
    Can be replaced by from_srgb_to_scene_linear() if Blender version >= 3.2
    """
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

def mix_rgb(c1, c2, factor, op='REGULAR'):
    c = c2
    if op == 'OVERLAY':
        c = (2*c1*c2) if c1<0.5 else (1 - 2*(1-c1)*(1-c2))
    elif op == 'HARDLIGHT':
        c = (2*c1*c2) if c2<0.5 else (1 - 2*(1-c1)*(1-c2))
    elif op == 'SOFTLIGHT':
        c = c1 - (1 - 2*c2) * c1 * (1 - c1) if c2 <= 0.5 \
            else c1 + (2*c2 - 1) * (16*(c1**3) - 12*(c1**2) + 3*c1) if c1 <= 0.25 \
            else c1 + (2*c2 - 1) * (np.sqrt(c1) - c1)
    elif op == 'ADD':
        c = c1 + c2
    elif op == 'SUBTRACT':
        c = c1 - c2
    elif op == 'MULTIPLY':
        c = c1 * c2
    elif op == 'DIVIDE':
        c = c1 / c2 if c2!=0 else 1
    elif op == 'SCREEN':
        c = 1 - (1 - c1) * (1 - c2)
    c = max(0, min(1, c))
    return c * factor + c1 * (1 - factor)

def mix_hsv(rgb1, rgb2, factor, ops = {}):
    from colorsys import rgb_to_hsv, hsv_to_rgb
    h1, s1, v1 = rgb_to_hsv(rgb1[0], rgb1[1], rgb1[2])
    h2, s2, v2 = rgb_to_hsv(rgb2[0], rgb2[1], rgb2[2])
    h = h2 if 'HUE' in ops else h1
    s = s2 if 'SATURATION' in ops else s1
    v = v2 if 'BRIGHTNESS' in ops else v1
    rgb = hsv_to_rgb(h, s, v)
    return [(rgb1[i]*(1-factor) + rgb[i]*factor) for i in range(3)]

def get_mixed_color(gp_obj, stroke, point_idx = None, to_linear = False):
    """Get the displayed color by jointly considering the material and vertex colors"""
    res = [0,0,0,1]
    mat = gp_obj.data.materials[stroke.material_index]

    if not mat:
        return res
    mat_gp = mat.grease_pencil

    if point_idx is None:
        # Case of fill color
        if gp_obj.data.materials[stroke.material_index].grease_pencil.show_fill:
            for i in range(4):
                res[i] = mat_gp.fill_color[i]
        if hasattr(stroke,'vertex_color_fill'):
            alpha = stroke.vertex_color_fill[3]
            for i in range(3):
                res[i] = res[i] * (1-alpha) + alpha * stroke.vertex_color_fill[i]
                if not to_linear:
                    res[i] = linear_to_srgb(res[i])
        return res
    else:
        # Case of line point color
        point = stroke.points[point_idx]
        if gp_obj.data.materials[stroke.material_index].grease_pencil.show_stroke:
            for i in range(4):
                res[i] = mat_gp.color[i]
        if hasattr(point,'vertex_color'):
            alpha = point.vertex_color[3]
            for i in range(3):
                res[i] = res[i] * (1-alpha) + alpha * point.vertex_color[i]
                if not to_linear:
                    res[i] = linear_to_srgb(res[i])
        return res

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

def get_an_inside_co(poly, max_distance = 64):
    """Return a coordinate that is inside an integer polygon as part of the triangle input"""
    import pyclipper
    delta = max_distance
    while delta > 1:
        for co in poly:
            for co_ in [(co[0]-delta, co[1]-delta), (co[0]-delta, co[1]+delta), 
                        (co[0]+delta, co[1]-delta), (co[0]+delta, co[1]+delta)]:
                if pyclipper.PointInPolygon(co_, poly):
                    return co_
        delta /= 2
    return None

def get_full_bound_box(s):
    """Bpy provides two bound box points for each stroke, but we need all 8 if transformed"""
    bound_points = []
    for x in (s.bound_box_min.x, s.bound_box_max.x):
        for y in (s.bound_box_min.y, s.bound_box_max.y):
            for z in (s.bound_box_min.z, s.bound_box_max.z):
                bound_points.append(Vector([x,y,z]))
    return bound_points

def get_2d_bound_box(strokes, t_mat):
    """Get 2D bounds [u_min, v_min, u_max, v_max] of a list of strokes given a 3D-to-2D transformation matrix"""
    corners = [None, None, None, None]
    for s in strokes:
        if len(s.points) < 1:
            continue
        bound_points = get_full_bound_box(s)
        for co in bound_points:
            co_2d = t_mat @ co
            u, v = co_2d[0], co_2d[1]
            corners[0] = u if (not corners[0] or u<corners[0]) else corners[0]
            corners[1] = v if (not corners[1] or v<corners[1]) else corners[1]
            corners[2] = u if (not corners[2] or u>corners[2]) else corners[2]
            corners[3] = v if (not corners[3] or v>corners[3]) else corners[3]
    return corners

def pad_2d_box(corners, ratio, return_bounds = False):
    """Scale a 2D box by a given ratio from its center. Return either 4 points or 2 bounds"""
    bound_W, bound_H = corners[2]-corners[0], corners[3]-corners[1]
    if return_bounds:
        return [corners[0] - ratio * bound_W, corners[1] - ratio * bound_H, corners[2] + ratio * bound_W, corners[3] + ratio * bound_H]
    else:
        return [(corners[0] - ratio * bound_W, corners[1] - ratio * bound_H),
                (corners[0] - ratio * bound_W, corners[3] + ratio * bound_H),
                (corners[2] + ratio * bound_W, corners[1] - ratio * bound_H),
                (corners[2] + ratio * bound_W, corners[3] + ratio * bound_H)]

def stroke_bound_box_overlapping(s1, s2, t_mat):
    """Judge if bound boxes of two strokes overlap in any given 2D plane"""
    bound_points = [get_full_bound_box(s1), get_full_bound_box(s2)]
    for i in range(2):
        for j in range(8):
            bound_points[i][j] = t_mat @ bound_points[i][j]
    # Check first two axes if strokes overlap on them
    bound_points = np.array(bound_points)
    for i in range(2):
        pa_min, pa_max = min(bound_points[0,:,i]), max(bound_points[0,:,i])
        pb_min, pb_max = min(bound_points[1,:,i]), max(bound_points[1,:,i])
        if pa_min > pb_max or pb_min > pa_max:
            return False
    return True        

def get_2d_stroke_outline(path_2d, stroke, scale_factor = 1, path_inverted = False, resolution = 32):
    """
    Given the 2D path and point radius data of a stroke, return its outline polygon.
    Different from Blender native operator, this one uses boolean to avoid generating a self-overlapping path.
    """
    path = list(reversed(path_2d)) if path_inverted else path_2d
    if len(path) < 2:
        return []
    import pyclipper
    clipper = pyclipper.Pyclipper()
    clipper.PreserveCollinear = True
    clipper.StrictlySimple = True

    radius_profile = [get_point_radius(p, stroke.line_width) * scale_factor for p in stroke.points]
    length_from_start = [0 for _ in range(len(stroke.points))]
    forwarding_normals = [None for _ in range(len(stroke.points) - 1)]

    # Get normal direction of each point
    for i in range(len(stroke.points) - 1):
        direction = Vector(path[i+1]) - Vector(path[i])
        length_from_start[i+1] = length_from_start[i] + direction.length
        forwarding_normals[i] = Vector((-direction[1], direction[0])).normalized()

    # Draw each segment as a quad
    for i in range(len(stroke.points) - 1):
        quad = [Vector(path[i]) - radius_profile[i] * forwarding_normals[i],
                Vector(path[i]) + radius_profile[i] * forwarding_normals[i],
                Vector(path[i+1]) + radius_profile[i+1] * forwarding_normals[i],
                Vector(path[i+1]) - radius_profile[i+1] * forwarding_normals[i]]
        area = pyclipper.Area(quad)
        if area < 0:
            quad.reverse()
        if area > 1 or area < -1:
            clipper.AddPath(quad, pyclipper.PT_SUBJECT, True)
            
        if i != len(stroke.points) - 2:
            joint = [Vector(path[i+1]) + radius_profile[i+1] * forwarding_normals[i],
                     Vector(path[i+1]) + radius_profile[i+1] * forwarding_normals[i+1],
                     Vector(path[i+1]) - radius_profile[i+1] * forwarding_normals[i],
                     Vector(path[i+1]) - radius_profile[i+1] * forwarding_normals[i+1]]
            area = pyclipper.Area(joint)
            if area < 0:
                joint.reverse()
            if area > 1 or area < -1:
                clipper.AddPath(joint, pyclipper.PT_SUBJECT, True)   
        
    # Draw a circle at specific joints
    for i in range(len(stroke.points)):
        if radius_profile[i] < 1:
            continue
        if stroke.start_cap_mode == 'FLAT' and length_from_start[i] < radius_profile[0]:
            continue
        elif stroke.end_cap_mode == 'FLAT' and (length_from_start[-1] - length_from_start[i]) < radius_profile[-1]:
            continue
        else:     
            angles = np.linspace(0, 2* np.pi, resolution, endpoint=False)
            circle = [ [path[i][0] + radius_profile[i] * math.cos(angle), 
                        path[i][1] + radius_profile[i] * math.sin(angle)] for angle in angles ]
        if not pyclipper.Orientation(circle):
            circle.reverse()
        clipper.AddPath(circle, pyclipper.PT_SUBJECT, True)
    res = clipper.Execute(pyclipper.CT_UNION, pyclipper.PFT_NONZERO, pyclipper.PFT_NONZERO)
    return res

#endregion

#region [Bpy Utilities]
def is_stroke_line(stroke, gp_obj):
    """
    Check if a stroke does not have fill material
    """
    mat_idx = stroke.material_index
    material = gp_obj.material_slots[mat_idx].material
    if not material:
        return False
    return not material.grease_pencil.show_fill

def is_stroke_fill(stroke, gp_obj):
    """
    Check if a stroke does not have line material
    """
    mat_idx = stroke.material_index
    material = gp_obj.material_slots[mat_idx].material
    if not material:
        return False
    return not material.grease_pencil.show_stroke

def is_stroke_protected(stroke, gp_obj):
    """
    Check if a stroke has the material that is being locked or invisible
    """
    mat_idx = stroke.material_index
    material = gp_obj.material_slots[mat_idx].material
    if not material:
        return False
    return material.grease_pencil.lock or material.grease_pencil.hide

def is_stroke_hole(stroke, gp_obj):
    """
    Check if a stroke has a material with fill holdout
    """
    mat_idx = stroke.material_index
    material = gp_obj.material_slots[mat_idx].material
    if not material:
        return False
    return material.grease_pencil.use_fill_holdout

def is_layer_protected(layer):
    """
    Check if a layer should be edited
    """
    return layer_locked(layer) or layer_hidden(layer)

def is_layer_inside_group(layer, group):
    """Check if a GPv3 layer belongs to a layer group"""
    if not is_gpv3():
        return False
    p = layer
    while p.parent_group is not None:
        if p.parent_group == group:
            return True
        p = p.parent_group
    return False

def multilayer_search_func(self, context, edit_text):
    """For operators that take multiple layers as input, generate the UI to allow the user add one layer at a time"""
    # Allow the user to reselect
    candidates = [layer.info for layer in context.object.data.layers]
    if is_gpv3():
        candidates += [group.name for group in context.object.data.layer_groups]

    # Allow the user to select one more layer/group
    existing_layers, existing_groups = multilayer_search_decode(edit_text)
    if len(existing_layers) > 0 or len(existing_groups) > 0:
        available_layers = [layer.info for layer in context.object.data.layers if layer not in existing_layers]
        candidates += [edit_text + " + " + layer_name for layer_name in available_layers]
        if is_gpv3():
            available_groups = [group.name for group in context.object.data.layer_groups if group not in existing_groups]
            candidates += [edit_text + " + " + group_name for group_name in available_groups]      
    return candidates

def multilayer_search_decode(edit_text):
    """Decode the input string into a list of layers"""
    names = edit_text.split(" + ")
    gp_obj = bpy.context.object
    groups = set()
    if is_gpv3():
        groups = [group for group in gp_obj.data.layer_groups if group.name in names]
        # Also add groups that are inside any input group
        sub_groups = []
        for group in gp_obj.data.layer_groups:
            for ancestor_group in groups:
                if is_layer_inside_group(group, ancestor_group):
                    sub_groups.append(group)
                    break
        groups = set(groups + sub_groups)

    layers = [layer for layer in gp_obj.data.layers if layer.info in names]
    if is_gpv3():
        for layer in gp_obj.data.layers:
            for ancestor_group in groups:
                if is_layer_inside_group(layer, ancestor_group):
                    layers.append(layer)
                    break
    layers = set(layers)
    return layers, groups

def get_layer_latest_frame(layer, current_frame):
    """Find the latest frame of a layer before a given frame"""
    frame = None
    for f in layer.frames:
        if f.frame_number > current_frame:
            break
        frame = f
    return frame

def get_stroke_length(stroke = None, co_list = None):
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

def get_stroke_center(stroke = None, co_list = None):
    """Calculate the 3D or 2D center of a stroke"""
    if stroke:
        res = Vector((0,0,0))
        for i,point in enumerate(stroke.points):
            res += point.co
        res /= len(stroke.points)
    if co_list:
        res = Vector((0,0))
        for i,co in enumerate(co_list):
            res += Vector((co[0],co[1]))
        res /= len(co_list)
    return res
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

def get_transformation_mat(mode='VIEW', gp_obj=None, strokes=[], operator=None, requires_layer=True):
    """
    Get the transformation matrix and its inverse matrix given a 2D working plane.
    The x and y values of transformed coordinates will be used for 2D operators.
    """
    # This is usually the first function in an operator to read from active layer, therefore check the condition here
    if requires_layer and not gp_obj.data.layers.active:
        operator.report({"WARNING"}, "Please select a layer.")
        raise Exception("Please select a Grease Pencil layer to perform the operation.")

    presets = {'X-Z': Matrix([[1,0,0],
                              [0,0,1],
                              [0,-1,0]]),
               'Y-Z': Matrix([[0,1,0],
                              [0,0,1],
                              [1,0,0]]),
               'X-Y': Matrix([[1,0,0],
                              [0,1,0],
                              [0,0,1]])}
      
    view_matrix = bpy.context.space_data.region_3d.view_matrix.to_3x3()
    if gp_obj:
        obj_rotation = gp_obj.matrix_world.to_3x3().normalized()
        view_matrix = view_matrix @ obj_rotation
        if gp_obj.data.layers.active and bpy.context.scene.nijigp_working_plane_layer_transform:
            layer_rotation = gp_obj.data.layers.active.matrix_layer.to_3x3().normalized()
            view_matrix = view_matrix @ layer_rotation

    # Use orthogonal planes
    if mode in presets:
        mat = presets[mode]
        return mat, mat.inverted_safe()
    
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
            
            mat = Matrix(mat).transposed()

            # Align the result with the current view
            if mat[2].dot(view_matrix[2]) < 0:  
                mat[2] *= -1                          # 1. Make the depth axis facing the screen
            if mat[0].cross(mat[1]).dot(mat[2]) < 0:
                mat[1] *= -1                          # 2. Ensure the rule of calculating normals is consistent
            # 3. Rotate the first two axes to align the UV/tangent
            target_up_axis = mat @ view_matrix.inverted_safe() @ Vector((0,1,0))
            delta = Vector((0,1)).angle_signed(target_up_axis[:2])
            rotation = Matrix.Rotation(delta,3,mat[2])
            mat = mat @ rotation
            return mat, mat.inverted_safe()
        
    # Use view plane
    mat = view_matrix
    if mat[0].cross(mat[1]).dot(mat[2]) < 0:
        mat[1] *= -1 
    return mat, mat.inverted_safe()

def get_2d_co_from_strokes(stroke_list, t_mat, scale = False, correct_orientation = False, scale_factor=None, return_orientation = False):
    """
    Convert points from a list of strokes to 2D coordinates. Scale them to be compatible with Clipper.
    Return a 2D coordinate list, a z-depth list and the scale factor 
    """
    poly_list = []
    poly_depth_list = []
    poly_inverted = []
    w_bound = [math.inf, -math.inf]
    h_bound = [math.inf, -math.inf]

    for stroke in stroke_list:
        co_list = []
        depth_list = []
        for point in stroke.points:
            transformed_co = t_mat @ point.co
            co_list.append([transformed_co[0],transformed_co[1]])
            depth_list.append(transformed_co[2])
            w_bound[0] = min(w_bound[0], co_list[-1][0])
            w_bound[1] = max(w_bound[1], co_list[-1][0])
            h_bound[0] = min(h_bound[0], co_list[-1][1])
            h_bound[1] = max(h_bound[1], co_list[-1][1])
        poly_list.append(co_list)
        poly_depth_list.append(depth_list)
        poly_inverted.append(False)

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
        for i,co_list in enumerate(poly_list):
            if not pyclipper.Orientation(co_list):
                co_list.reverse()
                poly_inverted[i] = True

    if return_orientation:
        return poly_list, poly_depth_list, poly_inverted, scale_factor
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
        self.indices = []
        for i,co_list in enumerate(poly_list):
            for j,co in enumerate(co_list):
                self.co2d.append(xy0(co))
                self.depth.append(depth_list[i][j])
                self.indices.append((i,j))
                self.count += 1
        self.kdtree = kdtree.KDTree(self.count)
        for i in range(self.count):
            self.kdtree.insert(self.co2d[i], i)
        self.kdtree.balance()

    def get_depth(self, co):
        _, i, _ = self.kdtree.find(xy0(co))
        return self.depth[i]

    def get_info(self, co):
        """Get stroke index, point index and the distance"""
        _, i, dist = self.kdtree.find(xy0(co))
        return self.indices[i][0], self.indices[i][1], dist       

def restore_3d_co(co, depth, inv_mat, scale_factor=1):
    """Perform inverse transformation on 2D coordinates"""
    vec = inv_mat @ Vector([co[0]/ scale_factor,
                            co[1]/ scale_factor,
                            depth])
    return vec
#endregion
