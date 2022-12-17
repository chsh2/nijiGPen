import math
import bpy
from mathutils import *

SCALE_CONSTANT = 8192

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

def smoothstep(x):
    if x<0:
        return 0
    if x>1:
        return 1
    return 3*(x**2) - 2*(x**3)

def save_stroke_selection(gp_obj):
    """
    Record the selection state of a Grease Pencil object to a map
    """
    select_map = {}
    for layer in gp_obj.data.layers:
        select_map[layer] = {}
        for frame in layer.frames:
            select_map[layer][frame] = {}
            for stroke in frame.strokes:
                select_map[layer][frame][stroke] = stroke.select_index
    return select_map

def load_stroke_selection(gp_obj, select_map):
    """
    Apply selection to strokes according to a map saved by save_stroke_selection()
    """
    for layer in gp_obj.data.layers:
        for frame in layer.frames:
            for stroke in frame.strokes:
                if stroke in select_map[layer][frame]:
                    stroke.select = (select_map[layer][frame][stroke] > 0)
                else:
                    stroke.select = False

def vec3_to_vec2(co) -> Vector:
    """Convert 3D coordinates into 2D"""
    scene = bpy.context.scene
    if scene.nijigp_working_plane == 'X-Z':
        return Vector([co.x, -co.z])
    if scene.nijigp_working_plane == 'Y-Z':
        return Vector([co.y, -co.z])
    if scene.nijigp_working_plane == 'X-Y':
        return Vector([co.x, -co.y])

def vec2_to_vec3(co, depth, scale_factor) -> Vector:
    """Convert 2D coordinates into 3D"""
    scene = bpy.context.scene
    if scene.nijigp_working_plane == 'X-Z':
        return Vector([co[0] / scale_factor, -depth, -co[1] / scale_factor])
    if scene.nijigp_working_plane == 'Y-Z':
        return Vector([depth, co[0] / scale_factor, -co[1] / scale_factor])
    if scene.nijigp_working_plane == 'X-Y':
        return Vector([co[0] / scale_factor, -co[1] / scale_factor, depth])

def set_vec2(point, co, scale_factor = 1):
    """Set 2D coordinates to a GP point"""
    scene = bpy.context.scene
    if scene.nijigp_working_plane == 'X-Z':    
        point.co.x = co[0] / scale_factor
        point.co.z = -co[1] / scale_factor
    if scene.nijigp_working_plane == 'Y-Z':    
        point.co.y = co[0] / scale_factor
        point.co.z = -co[1] / scale_factor
    if scene.nijigp_working_plane == 'X-Y':    
        point.co.x = co[0] / scale_factor
        point.co.y = -co[1] / scale_factor

def vec3_to_depth(co):
    """Get depth value from 3D coordinates"""
    scene = bpy.context.scene
    if scene.nijigp_working_plane == 'X-Z':    
        return -co.y
    if scene.nijigp_working_plane == 'Y-Z':    
        return co.x
    if scene.nijigp_working_plane == 'X-Y':    
        return co.z

def set_depth(point, depth):
    """Set depth to a GP point"""
    scene = bpy.context.scene
    if hasattr(point, 'co'):
        target = point.co
    else:
        target = point
    if scene.nijigp_working_plane == 'X-Z':    
        target.y = -depth
    if scene.nijigp_working_plane == 'Y-Z':    
        target.x = depth
    if scene.nijigp_working_plane == 'X-Y':    
        target.z = depth

def get_depth_direction() -> Vector:
    """Return a vector pointing to the positive side of the depth dimension"""
    scene = bpy.context.scene
    if scene.nijigp_working_plane == 'X-Z':    
        return Vector((0,-1,0))
    if scene.nijigp_working_plane == 'Y-Z':    
        return Vector((1,0,0))
    if scene.nijigp_working_plane == 'X-Y':    
        return Vector((0,0,1))

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

def raycast_2d_up(x0, y0, x1, y1, x2, y2):
    """
    Emit ray from (x0, y0) in +y direction and check if it intersects with the line segment (x1,y1)->(x2,y2)
    mathutils.geometry.intersect_line_line_2d sometimes has inconsistent results, therefore is not adopted here
    """
    if x1>x0 and x2>x0:
        return False
    if x1<x0 and x2<x0:
        return False
    
    # Several special cases to consider
    if math.isclose(x1,x2):
        return False
    if x1<x2 and math.isclose(x1, x0):
        return y1 > y0
    if x1>x2 and math.isclose(x2, x0):
        return y2 > y0

    # The normal case
    ratio = (x0 - x1) / (x2 - x1)
    h_intersect = y1 + ratio * (y2 - y1)
    return h_intersect > y0

def crossing_number_2d_up(stroke, x, y, scale_factor):
    """
    Raycast to every point of the stroke to calculate the winding number
    """
    res = 0
    point_num = len(stroke.points)
    for i in range(point_num):
        co1 = vec3_to_vec2(stroke.points[i].co)
        co2 = vec3_to_vec2(stroke.points[(i+1)%point_num].co)
        res += raycast_2d_up(x/scale_factor, y/scale_factor, co1[0], co1[1], co2[0], co2[1])    
    return res

def overlapping_strokes(s1, s2):
    """
    Check if two strokes overlap with each other. Ignore the cases involving holes
    """
    
    # First, check if bounding boxes overlap
    scene = bpy.context.scene
    if scene.nijigp_working_plane == 'X-Z':
        if s1.bound_box_max[0] < s2.bound_box_min[0] or s1.bound_box_max[2] < s2.bound_box_min[2]:
            return False
        if s2.bound_box_max[0] < s1.bound_box_min[0] or s2.bound_box_max[2] < s1.bound_box_min[2]:
            return False
    if scene.nijigp_working_plane == 'Y-Z':
        if s1.bound_box_max[1] < s2.bound_box_min[1] or s1.bound_box_max[2] < s2.bound_box_min[2]:
            return False
        if s2.bound_box_max[1] < s1.bound_box_min[1] or s2.bound_box_max[2] < s1.bound_box_min[2]:
            return False
    if scene.nijigp_working_plane == 'X-Y':
        if s1.bound_box_max[0] < s2.bound_box_min[0] or s1.bound_box_max[1] < s2.bound_box_min[1]:
            return False
        if s2.bound_box_max[0] < s1.bound_box_min[0] or s2.bound_box_max[1] < s1.bound_box_min[1]:
            return False
    
    # Then check every pair of edge
    N1 = len(s1.points)
    N2 = len(s2.points)
    for i in range(N1):
        for j in range(N2):
            p1 = vec3_to_vec2(s1.points[i].co)
            p2 = vec3_to_vec2(s1.points[(i+1)%N1].co)
            p3 = vec3_to_vec2(s2.points[j].co)
            p4 = vec3_to_vec2(s2.points[(j+1)%N2].co)
            if geometry.intersect_line_line_2d(p1,p2,p3,p4):
                return True

    return False

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
            co_list.append(vec3_to_vec2(point.co))
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

def poly_to_stroke(co_list, stroke_info, gp_obj, scale_factor, rearrange = True, arrange_offset = 0, ref_stroke_mask = {}):
    """
    Generate a new stroke according to 2D polygon data. Point and stroke attributes will be copied from a list of reference strokes.
    stroke_info: A list of [stroke, layer_index, stroke_index, frame]
    """

    # Find closest reference point and corresponding stroke
    ref_stroke_index_list = []
    ref_point_index_list = []
    ref_stroke_count = {}

    # Setup a KDTree for point lookup
    total_point_count = 0
    for i,info in enumerate(stroke_info):
        total_point_count += len(info[0].points)
    kdt = kdtree.KDTree(total_point_count)

    kdtree_indices = []
    for i,info in enumerate(stroke_info):
        for j,point in enumerate(info[0].points):
            kdtree_indices.append( (i,j) )
            # Ignore the 3rd dimension
            kdt.insert( vec3_to_vec2(point.co).to_3d(), len(kdtree_indices)-1 )
    kdt.balance()

    # Search every new generated point in the KDTree
    for co in co_list:
        vec_, kdt_idx, dist_ = kdt.find([co[0]/scale_factor, co[1]/scale_factor, 0])

        ref_stroke = kdtree_indices[kdt_idx][0]
        ref_point = kdtree_indices[kdt_idx][1]
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
    frame = gp_obj.data.layers[layer_index].active_frame
    if len(stroke_info[ref_stroke_index]) > 3:
        frame = stroke_info[ref_stroke_index][3]

    # Making the starting point of the new stroke close to the existing one
    min_distance = None
    index_offset = None
    for i,co in enumerate(co_list):
        distance = get_2d_squared_distance(co, scale_factor, vec3_to_vec2(src_stroke.points[0].co), 1)
        if min_distance == None or min_distance > distance:
            min_distance = distance
            index_offset = i

    # Create a new stroke
    new_stroke = frame.strokes.new()
    N = len(co_list)
    new_stroke.points.add(N)
    for i in range(N):
        new_i = (i + index_offset) % N
        set_vec2(new_stroke.points[i], co_list[new_i], scale_factor)

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
        set_depth(dst_point, vec3_to_depth(src_point.co))

    # Rearrange the new stroke
    current_index = len(frame.strokes) - 1
    new_index = current_index
    if rearrange:
        new_index = stroke_index + 1 - arrange_offset
        bpy.ops.gpencil.select_all(action='DESELECT')
        new_stroke.select = True
        for i in range(current_index - new_index):
            bpy.ops.gpencil.stroke_arrange("EXEC_DEFAULT", direction='DOWN')

    return new_stroke, new_index, layer_index
