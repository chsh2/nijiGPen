import bpy
from ..utils import *
from ..api_router import *

class NoiseConfig:
    """
    Options for adding noise to stroke/point attributes
    """
    random_seed: bpy.props.IntProperty(
            name='Seed', default=1, min=0, max=65535
    )
    random_scale: bpy.props.FloatProperty(
            name='Scale', default=1, min=0.01, max=100
    )
    random_factor: bpy.props.FloatProperty(
            name='Factor', default=0, min=0, max=1
    )

class ColorTintConfig:
    """
    Options of applying a color tint, shard by multiple operators
    """
    tint_color: bpy.props.FloatVectorProperty(
            name = "Tint Color",
            subtype = "COLOR",
            default = (.0,.0,.0,1.0),
            min = 0.0, max = 1.0, size = 4,
            description='Change the vertex color by blending with a new color',
    )
    tint_color_factor: bpy.props.FloatProperty(
            name='Tint Factor',
            default=0, min=0, max=1
    )    
    tint_mode: bpy.props.EnumProperty(
            name='Mode',
            items=[('BOTH', 'Stroke & Fill', ''),
                    ('FILL', 'Fill', ''),
                    ('LINE', 'Stroke', '')],
            default='FILL'
    ) 
    blend_mode: bpy.props.EnumProperty(
            name='Blend',
            items=[('REGULAR', 'Regular', ''),
                    ('SCREEN', 'Screen', ''),
                    ('OVERLAY', 'Overlay', ''),
                    ('HARDLIGHT', 'Hard Light', ''),
                    ('SOFTLIGHT', 'Soft Light', ''),
                    ('ADD', 'Add', ''),
                    ('SUBTRACT', 'Subtract', ''),
                    ('MULTIPLY', 'Multiply', ''),
                    ('DIVIDE', 'Divide', ''),
                    ('HUE', 'Hue', ''),
                    ('SATURATION', 'Saturation', ''),
                    ('BRIGHTNESS', 'Brightness', '')],
            default='REGULAR'
    ) 

class ColorTintPropertyGroup(ColorTintConfig, bpy.types.PropertyGroup):
    """
    Used when an operator needs more than one type of color tint
    """

def save_stroke_selection(gp_obj):
    """
    Record the selection state of a Grease Pencil object to a map
    """
    select_map = {}
    for layer in gp_obj.data.layers:
        select_map[layer] = {}
        for frame in layer.frames:
            select_map[layer][frame] = {}
            for stroke in frame.nijigp_strokes:
                select_map[layer][frame][stroke] = stroke.select_index
    return select_map

def load_stroke_selection(gp_obj, select_map):
    """
    Apply selection to strokes according to a map saved by save_stroke_selection()
    """
    for layer in gp_obj.data.layers:
        for frame in layer.frames:
            if frame in select_map[layer]:
                for stroke in frame.nijigp_strokes:
                    if stroke in select_map[layer][frame]:
                        stroke.select = (select_map[layer][frame][stroke] > 0)
                    else:
                        stroke.select = False

def get_input_frames(gp_obj, multiframe=False, return_map=False, layers=None):
    """
    Get either active frames or all selected frames depending on the edit mode.
    Return either a list of frames, or a detailed map with the following format:
        {frame_number: {layer_index: [frame, (start_frame_num, end_frame_num)]}}
    """
    layers_to_process = gp_obj.data.layers if layers == None else layers
    frames_to_process = []
    frame_number_layer_map = {}
    
    layer_active_frames_number = []
    for i,layer in enumerate(layers_to_process):
        if layer.active_frame:
            layer_active_frames_number.append(layer.active_frame.frame_number)
        else:
            layer_active_frames_number.append(None)            

    # Process every selected frame
    for i,layer in enumerate(layers_to_process):
        if not is_layer_protected(layer):
            for j,frame in enumerate(layer.frames):
                f_num = frame.frame_number
                if multiframe and not frame.select:
                    continue
                if not multiframe and f_num != layer_active_frames_number[i]:
                    continue
                if len(frame.nijigp_strokes) < 1:
                    continue
                frames_to_process.append(frame)
                if f_num not in frame_number_layer_map:
                    frame_number_layer_map[f_num] = {}
                frame_number_layer_map[f_num][i] = [frame , None]
                
                # Get frame number range
                if j != len(layer.frames)-1:
                    frame_number_layer_map[f_num][i][1] = (f_num, layer.frames[j+1].frame_number)
                else:
                    frame_number_layer_map[f_num][i][1] = (f_num, bpy.context.scene.frame_end + 1)
    if return_map:
        return frame_number_layer_map
    else:
        return frames_to_process
               
def get_input_strokes(gp_obj, frame, select_all = False):
    """
    Check each stroke in a frame if it belongs to the input
    """
    res = []
    if is_frame_valid(frame):
        for stroke in frame.nijigp_strokes:
            if not is_stroke_protected(stroke, gp_obj) and (select_all or stroke.select):
                res.append(stroke)
    return res

def refresh_strokes(gp_obj, frame_numbers = None):
    """
    When generating new strokes via scripting, sometimes the strokes do not have correct bound boxes and are not displayed correctly.
    This function recalculates the geometry data
    """
    if is_gpv3():
        return
    
    current_mode = bpy.context.mode
    current_frame = bpy.context.scene.frame_current

    bpy.ops.object.mode_set(mode='EDIT_GPENCIL')
    if frame_numbers == None:
        bpy.ops.gpencil.recalc_geometry()
    else:
        for f in frame_numbers:
            bpy.context.scene.frame_set(f)
            bpy.ops.gpencil.recalc_geometry()
    
    bpy.ops.object.mode_set(mode=current_mode)
    bpy.context.scene.frame_set(current_frame)

def copy_stroke_attributes(dst, srcs,
                           copy_hardness = False,
                           copy_linewidth = False,
                           copy_cap = False,
                           copy_cyclic = False,
                           copy_uv = False,
                           copy_material = False,
                           copy_color = False):
    """
    Set the attributes of a stroke by copying from other stroke(s)
    """
    # Single values
    src = srcs[0]
    if copy_cap:
        dst.start_cap_mode = src.start_cap_mode
        dst.end_cap_mode = src.end_cap_mode
    if copy_cyclic:
        dst.use_cyclic = src.use_cyclic
    if copy_material:
        dst.material_index = src.material_index
    if copy_uv:
        dst.uv_scale = src.uv_scale
        
    # Average values
    color_fill = Vector([.0,.0,.0,.0])
    uv_translation = Vector([.0,.0])
    uv_rotation = .0
    linewidth = 0
    hardness = .0
    for src in srcs:
        if copy_hardness:
            hardness += src.hardness
        if copy_linewidth:
            linewidth += src.line_width
        if copy_uv:
            uv_rotation += src.uv_rotation
            uv_translation += Vector(src.uv_translation)
        if copy_color:
            color_fill += Vector(src.vertex_color_fill)
    n = len(srcs)
    if copy_hardness:
        dst.hardness = hardness / n
    if copy_linewidth:
        dst.line_width = int(linewidth // n)
    if copy_uv:
        dst.uv_rotation = uv_rotation / n
        dst.uv_translation = uv_translation / n
    if copy_color:
        dst.vertex_color_fill = color_fill / n
    # Check this attribute separately because it exists in GPv3 only
    fill_opacity = .0        
    if hasattr(dst, 'fill_opacity') and copy_color:
        dst.fill_opacity = sum([src.fill_opacity for src in srcs]) / n
    
def smooth_stroke_attributes(stroke, smooth_level, attr_map = {'co':3, 'strength':1, 'pressure':1}):
    """
    Calculate average values of a stroke's vertex attributes along its path.
    attr_map: {attribute: dimension}
    """
    import numpy as np
    kernel = np.array([1.0/3, 1.0/3, 1.0/3])
    num_point = len(stroke.points)
    if num_point < 2:
        return
    for name in attr_map:
        attr_values = np.zeros( num_point * attr_map[name] )
        stroke.points.foreach_get(name, attr_values)
        for _ in range(smooth_level):
            step = attr_map[name]
            for dim in range(step):
                # Since convolution is not periodic, calculate the first and last points manually
                if stroke.use_cyclic:
                    new_start = np.dot(kernel, (attr_values[dim-step], attr_values[dim], attr_values[dim+step]))
                    new_end = np.dot(kernel, (attr_values[dim-2*step], attr_values[dim-step], attr_values[dim]))
                    
                attr_values[dim::step][1:-1] = np.convolve(attr_values[dim::step], kernel, mode='same')[1:-1]
                
                if stroke.use_cyclic:
                    attr_values[dim] = new_start
                    attr_values[dim-step] = new_end
        stroke.points.foreach_set(name, attr_values)
    
def get_generated_meshes(gp_obj):
    """Get a list of meshes generated from the given object"""
    generated_objects = set()
    for obj in gp_obj.children:
        if 'nijigp_mesh' in obj:
            generated_objects.add(obj)
    for obj in bpy.context.scene.objects:
        if ('nijigp_mesh' in obj and 
            'nijigp_parent' in obj and
            obj['nijigp_parent'] == gp_obj):
            generated_objects.add(obj)
    return list(generated_objects)