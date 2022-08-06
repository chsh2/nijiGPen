import math
import bpy
from mathutils import *

SCALE_CONSTANT = 8192

def get_2d_projection(point):
    """Currently, the addon only works in XZ plane"""
    return Vector([ point.co[0], -point.co[2] ])

def set_2d_projection(point, co, scale_factor = 1):
    """Currently, the addon only works in XZ plane"""
    point.co[0] = co[0] / scale_factor
    point.co[2] = -co[1] / scale_factor

def get_orthogonal_position(point):
    """Currently, the orthogonal axis is the Y axis"""
    return point.co[1]

def set_orthogonal_position(point, depth):
    """Currently, the orthogonal axis is the Y axis"""
    point.co[1] = depth

def get_2d_squared_distance(co1, scale_factor1, co2, scale_factor2):
    """Euclidean distance that takes the scale factors into consideration"""
    delta = [co1[0]/scale_factor1 - co2[0]/scale_factor2, co1[1]/scale_factor1 - co2[1]/scale_factor2]
    return delta[0]*delta[0] + delta[1]*delta[1]
    
def is_2d_point_on_segment(xA, yA, xB, yB, xC, yC):
    """
    Check if the point C is on the line segment AB
    """
    vAB = (xB-xA, yB-yA)
    vAC = (xC-xA, yC-yA)
    
    # Parallelism
    if not math.isclose(vAB[1]*vAC[0], vAC[1]*vAB[0]):
        return False

    # Whether C is between A and B
    if xC > max(xA, xB) or xC < min(xA, xB):
        return False
    if yC > max(yA, yB) or yC < min(yA, yB):
        return False  
    return True

def intersecting_segments(x1,y1,x2,y2,x3,y3,x4,y4):
    """
    Check if two 2D line segments (x1,y1)->(x2,y2) and (x3,y3)->(x4,y4) intersect with each other.
    More information:
    https://mathworld.wolfram.com/Line-LineIntersection.html
    """
    divisor = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
    
    # Case of parallel lines
    if math.isclose(divisor, 0):
        if is_2d_point_on_segment(x1,y1,x2,y2,x3,y3) or is_2d_point_on_segment(x1,y1,x2,y2,x4,y4):
            return True
        if is_2d_point_on_segment(x3,y3,x4,y4,x1,y1) or is_2d_point_on_segment(x3,y3,x4,y4,x2,y2):
            return True
        return False
    # Case of non-parallel lines
    else:
        x_inter = ((x1*y2-y1*x2)*(x3-x4) - (x1-x2)*(x3*y4-y3*x4)) / divisor
        y_inter = ((x1*y2-y1*x2)*(y3-y4) - (y1-y2)*(x3*y4-y3*x4)) / divisor
        if is_2d_point_on_segment(x1,y1,x2,y2,x_inter,y_inter) and is_2d_point_on_segment(x3,y3,x4,y4,x_inter,y_inter):
            return True
        return False
    
def overlapping_strokes(s1, s2):
    """
    Check if two strokes overlap with each other. Ignore the cases involving holes
    """
    
    # First, check if bounding boxes overlap
    if s1.bound_box_max[0] < s2.bound_box_min[0] or s1.bound_box_max[2] < s2.bound_box_min[2]:
        return False
    if s2.bound_box_max[0] < s1.bound_box_min[0] or s2.bound_box_max[2] < s1.bound_box_min[2]:
        return False
    
    # Then check every pair of edge
    N1 = len(s1.points)
    N2 = len(s2.points)
    for i in range(N1):
        for j in range(N2):
            p1 = get_2d_projection(s1.points[i])
            p2 = get_2d_projection(s1.points[(i+1)%N1])
            p3 = get_2d_projection(s2.points[j])
            p4 = get_2d_projection(s2.points[(j+1)%N2])
            # May use mathutils.geometry.intersect_line_line_2d instead?
            if intersecting_segments(p1[0],p1[1],p2[0],p2[1],p3[0],p3[1],p4[0],p4[1]):
                return True
    return False

def is_stroke_locked(stroke, gp_obj):
    """
    Check if a stroke has the material that is being locked or invisible
    """
    mat_idx = stroke.material_index
    material = gp_obj.material_slots[mat_idx].material
    return material.grease_pencil.lock or material.grease_pencil.hide

def stroke_to_poly(stroke_list, scale = False, correct_orientation = False):
    """
    Convert Blender strokes to a list of 2D coordinates compatible with Clipper.
    Scaling can be applied instead of Clipper's built-in method
    """

    import pyclipper
    scale_factor = [1, 1]
    poly_list = []
    w_bound = [math.inf, -math.inf]
    h_bound = [math.inf, -math.inf]

    for stroke in stroke_list:
        co_list = []
        for point in stroke.points:
            co_list.append(get_2d_projection(point))
            w_bound[0] = min(w_bound[0], co_list[-1][0])
            w_bound[1] = max(w_bound[1], co_list[-1][0])
            h_bound[0] = min(h_bound[0], co_list[-1][1])
            h_bound[1] = max(h_bound[1], co_list[-1][1])
        poly_list.append(co_list)

    if scale:
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

        for co_list in poly_list:
            for co in co_list:
                co[0] *= scale_factor
                co[1] *= scale_factor

    # Since Grease Pencil does not care whether the sequence of points is clockwise,
    # Clipper may regard some strokes as negative polygons, which needs a fix 
    if correct_orientation:
        for co_list in poly_list:
            if not pyclipper.Orientation(co_list):
                co_list.reverse()

    return poly_list, scale_factor

def poly_to_stroke(co_list, stroke_info, gp_obj, scale_factor, rearrange = True, arrange_offset = 0, fast_mode = False, max_error = 0, ref_stroke_mask = {}):
    """
    Generate a new stroke according to 2D polygon data. Point and stroke attributes will be copied from a list of reference strokes.
    stroke_info: A list of [stroke, layer_index, stroke_index]
    """

    # Find closest reference point and corresponding stroke
    # TODO: Speed-up this process with data structures like KDTree
    ref_stroke_index_list = []
    ref_point_index_list = []
    ref_stroke_count = {}
    i_offset = 0
    j_offset = 0
    for co in co_list:
        ref_stroke = i_offset
        ref_point = j_offset
        min_distance = math.inf

        for i,info in enumerate(stroke_info):
            true_i = (i + i_offset) % len(stroke_info)
            for j,point in enumerate(stroke_info[true_i][0].points):
                true_j = (j + j_offset) % len(stroke_info[true_i][0].points)
                true_point = stroke_info[true_i][0].points[true_j]
                distance = get_2d_squared_distance(co, scale_factor, get_2d_projection(true_point), 1)
                
                # Standard mode: search all points to find the closest one
                if not fast_mode and distance < min_distance:
                    ref_stroke = true_i
                    ref_point = true_j
                    min_distance = distance

                # Fast mode: stop once the error can be tolerated
                if fast_mode and distance < max_error:
                    ref_stroke = true_i
                    ref_point = true_j
                    # Start the next search from the current point 
                    i_offset = true_i
                    j_offset = true_j
                    break
            else:
                continue
            break


        ref_stroke_index_list.append(ref_stroke)
        ref_point_index_list.append(ref_point)
        if ref_stroke in ref_stroke_count:
            ref_stroke_count[ref_stroke] += 1
        elif ref_stroke not in ref_stroke_mask:
            ref_stroke_count[ref_stroke] = 1

    # Determine the reference stroke either by calculating the majority or manual assignment
    if len(ref_stroke_count) > 0:
        ref_stroke_index = max(ref_stroke_count, key=ref_stroke_count.get)
    else:
        for i in range(len(stroke_info)):
            if i not in ref_stroke_mask:
                ref_stroke_index = i
    layer_index = stroke_info[ref_stroke_index][1]
    stroke_index = stroke_info[ref_stroke_index][2]
    src_stroke = stroke_info[ref_stroke_index][0]

    # Making the starting point of the new stroke close to the existing one
    min_distance = None
    index_offset = None
    for i,co in enumerate(co_list):
        distance = get_2d_squared_distance(co, scale_factor, get_2d_projection(src_stroke.points[0]), 1)
        if min_distance == None or min_distance > distance:
            min_distance = distance
            index_offset = i

    # Create a new stroke
    new_stroke = gp_obj.data.layers[layer_index].active_frame.strokes.new()
    N = len(co_list)
    new_stroke.points.add(N)
    for i in range(N):
        new_i = (i + index_offset) % N
        set_2d_projection(new_stroke.points[i], co_list[new_i], scale_factor)

    # Copy stroke properties
    new_stroke.hardness = src_stroke.hardness
    new_stroke.line_width = src_stroke.line_width
    new_stroke.material_index = src_stroke.material_index
    new_stroke.start_cap_mode = src_stroke.start_cap_mode
    new_stroke.end_cap_mode = src_stroke.end_cap_mode
    new_stroke.use_cyclic = src_stroke.use_cyclic
    new_stroke.uv_rotation = src_stroke.uv_rotation
    new_stroke.uv_scale = src_stroke.uv_scale
    new_stroke.uv_translation = src_stroke.uv_translation
    new_stroke.vertex_color_fill = src_stroke.vertex_color_fill

    # Copy point properties
    for i in range(N):
        new_i = (i + index_offset) % N
        src_point = stroke_info[ref_stroke_index_list[new_i]][0].points[ref_point_index_list[new_i]]
        dst_point = new_stroke.points[i]
        dst_point.pressure = src_point.pressure
        dst_point.strength = src_point.strength
        dst_point.uv_factor = src_point.uv_factor
        dst_point.uv_fill = src_point.uv_fill
        dst_point.uv_rotation = src_point.uv_rotation
        dst_point.vertex_color = src_point.vertex_color
        set_orthogonal_position(dst_point, get_orthogonal_position(src_point))

    # Rearrange the new stroke
    current_index = len(gp_obj.data.layers[layer_index].active_frame.strokes) - 1
    new_index = current_index
    if rearrange:
        new_index = stroke_index + 1 - arrange_offset
        bpy.ops.gpencil.select_all(action='DESELECT')
        new_stroke.select = True
        for i in range(current_index - new_index):
            bpy.ops.gpencil.stroke_arrange("EXEC_DEFAULT", direction='DOWN')

    return new_stroke, new_index
