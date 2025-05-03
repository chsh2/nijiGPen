import bpy
import math
import numpy as np
from .common import *
from ..utils import *
from ..api_router import *

def generate_stroke_from_2d(new_co_list, inv_mat, 
                            poly_list, depth_list, 
                            stroke_info, gp_obj, scale_factor, 
                            rearrange = True, arrange_offset = 0, ref_stroke_mask = {}):
    """
    Given a list of 2D coordinates (new_co_list), transform it back to 3D and generate a stroke.
    Copy stroke and point attributes from input strokes (poly_list and stroke_info).
    Format of stroke_info: A list of [stroke, layer_index, stroke_index, frame]
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
    for i,co_list in enumerate(poly_list):
        for j,co in enumerate(co_list):
            kdtree_indices.append( (i,j) )
            kdt.insert( xy0(co), len(kdtree_indices)-1 )
    kdt.balance()
    depth_lookup_tree = DepthLookupTree(poly_list, depth_list)

    # Search every new generated point in the KDTree
    for co in new_co_list:
        _vec, kdt_idx, _dist = kdt.find(xy0(co))
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
    starting_point_co = poly_list[ref_stroke_index][0]
    for i,co in enumerate(new_co_list):
        distance = get_2d_squared_distance(co, scale_factor, starting_point_co, 1)
        if not min_distance or min_distance > distance:
            min_distance = distance
            index_offset = i
            
    # Create a new stroke
    new_stroke = frame.nijigp_strokes.new()
    N = len(new_co_list)
    new_stroke.points.add(N)
    for i in range(N):
        new_i = (i + index_offset) % N
        new_stroke.points[i].co = restore_3d_co(new_co_list[new_i], depth_lookup_tree.get_depth(new_co_list[new_i]), inv_mat, scale_factor)
    # Copy stroke properties
    copy_stroke_attributes(new_stroke, [src_stroke],
                           copy_hardness=True, copy_linewidth=True,
                           copy_cap=True, copy_cyclic=True,
                           copy_uv=True, copy_material=True, copy_color=True)        
    # Copy point properties
    for i in range(N):
        new_i = (i + index_offset) % N
        src_stroke = stroke_info[ref_stroke_index_list[new_i]][0]
        src_point = src_stroke.points[ref_point_index_list[new_i]]
        dst_point = new_stroke.points[i]
        dst_point.pressure = src_point.pressure * src_stroke.line_width / new_stroke.line_width
        dst_point.strength = src_point.strength
        dst_point.uv_factor = src_point.uv_factor
        dst_point.uv_fill = src_point.uv_fill
        dst_point.uv_rotation = src_point.uv_rotation
        dst_point.vertex_color = src_point.vertex_color
    # Rearrange the new stroke
    current_index = len(frame.nijigp_strokes) - 1
    new_index = current_index
    if rearrange:
        new_index = stroke_index + 1 - arrange_offset
        op_deselect()
        new_stroke.select = True
        for i in range(current_index - new_index):
            op_arrange_stroke(direction='DOWN')

    return new_stroke, new_index, layer_index

def overlapping_strokes(s1, s2_outline, s2_bound_box, t_mat, scale_factor=1):
    """
    Check if a stroke overlap with another projected to given 2D plane.
    Ignore the cases involving holes
    """
    # First, check if bound boxes overlap
    s1_bound_box = get_2d_bound_box([s1], t_mat)
    if (s1_bound_box[0]*scale_factor > s2_bound_box[2] or s2_bound_box[0] > s1_bound_box[2]*scale_factor or
        s1_bound_box[1]*scale_factor > s2_bound_box[3] or s2_bound_box[1] > s1_bound_box[3]*scale_factor):
        return False
    
    # Convert the first stroke to 2D path
    path1 = []
    N1 = len(s1.points)
    for i in range(N1):
        path1.append(t_mat @ s1.points[i].co)
        
    path2 = []
    for p in s2_outline:
        path2.append(Vector(p) / scale_factor)
    N2 = len(path2)
            
    # Check each pair of edges
    for i in range(N1):
        for j in range(N2):
            if geometry.intersect_line_line_2d(path1[i][:2],path1[(i+1)%N1][:2],path2[j][:2],path2[(j+1)%N2][:2]):
                return True
    return False

class HoleProcessingOperator(bpy.types.Operator):
    """Reorder strokes and assign holdout materials to holes inside another stroke"""
    bl_idname = "gpencil.nijigp_hole_processing"
    bl_label = "Hole Processing"
    bl_category = 'View'
    bl_options = {'REGISTER', 'UNDO'}

    rearrange: bpy.props.BoolProperty(
            name='Rearrange Strokes',
            default=True,
            description='Move holes to the top, which may be useful for handling some imported SVG shapes'
    )
    separate_colors: bpy.props.BoolProperty(
            name='Separate Colors',
            default=False,
            description='Detect holes separately for each vertex fill color'
    )
    apply_holdout: bpy.props.BoolProperty(default=True)

    def draw(self, context):
        layout = self.layout
        row = layout.row()
        row.prop(self, "rearrange")
        row = layout.row()
        row.prop(self, "separate_colors")

    def execute(self, context):
        try:
            import pyclipper
        except ImportError:
            self.report({"ERROR"}, "Please install PyClipper in the Preferences panel.")
            return {'FINISHED'}
        
        gp_obj: bpy.types.Object = context.object
        frames_to_process = get_input_frames(gp_obj, get_multiedit(gp_obj))
        material_idx_map = {}
        def change_material(stroke):
            '''
            Duplicate the stroke's material but enable holdout for it. Reuse existing material if possible.
            '''
            src_mat_idx = stroke.material_index
            src_mat = gp_obj.material_slots[src_mat_idx].material
            src_mat_name = gp_obj.material_slots[src_mat_idx].name

            if not src_mat or src_mat.grease_pencil.use_fill_holdout:
                return

            # Case 1: holdout material available in cache
            if src_mat_idx in material_idx_map:
                stroke.material_index = material_idx_map[src_mat_idx]
                return

            # Case 2: holdout material has been added to this object
            dst_mat_name = src_mat_name + '_Holdout'
            for i,material_slot in enumerate(gp_obj.material_slots):
                if dst_mat_name == material_slot.name:
                    stroke.material_index = i
                    material_idx_map[src_mat_idx] = i
                    return

            # Case 3: create a new material
            dst_mat: bpy.types.Material = src_mat.copy()
            dst_mat.name = dst_mat_name
            dst_mat['original_material_index'] = stroke.material_index
            dst_mat.grease_pencil.fill_style = 'SOLID'
            dst_mat.grease_pencil.fill_color = (0,0,0,1)
            dst_mat.grease_pencil.use_fill_holdout = True
            gp_obj.data.materials.append(dst_mat)
            dst_mat_idx = len(gp_obj.data.materials)-1
            material_idx_map[src_mat_idx] = dst_mat_idx
            stroke.material_index = dst_mat_idx

        def process_one_frame(frame):
            select_map = save_stroke_selection(gp_obj)
            to_process = get_input_strokes(gp_obj, frame)
            t_mat, inv_mat = get_transformation_mat(mode=context.scene.nijigp_working_plane,
                                                    gp_obj=gp_obj, strokes=to_process, operator=self,
                                                    requires_layer=False)
            # Initialize the relationship matrix
            poly_list, _, _ = get_2d_co_from_strokes(to_process, t_mat, scale=True)
            relation_mat = np.zeros((len(to_process),len(to_process)))
            for i in range(len(to_process)):
                for j in range(len(to_process)):
                    if i!=j and is_poly_in_poly(poly_list[i], poly_list[j]) and not is_stroke_line(to_process[j], gp_obj):
                        relation_mat[i][j] = 1

            # Record each vertex color
            is_hole_map = {0: False}
            if self.separate_colors:
                for stroke in to_process:
                    is_hole_map[rgb_to_hex_code(stroke.vertex_color_fill)] = False

            # Iteratively process and exclude outmost strokes
            processed = set()
            while len(processed) < len(to_process):
                op_deselect()
                idx_list = []
                color_modified = set()
                for i in range(len(to_process)):
                    if np.sum(relation_mat[i]) == 0 and i not in processed:
                        idx_list.append(i)
                for i in idx_list:
                    processed.add(i)
                    relation_mat[:,i] = 0
                    to_process[i].select = True
                    key = rgb_to_hex_code(to_process[i].vertex_color_fill) if self.separate_colors else 0
                    if self.apply_holdout and is_hole_map[key] and not is_stroke_line(to_process[i], gp_obj):
                        if hasattr(to_process[i], 'fill_opacity'):
                            to_process[i].fill_opacity = 1.0
                        change_material(to_process[i])
                    color_modified.add(key)
                if self.rearrange:
                    op_arrange_stroke(direction='TOP')

                for color in color_modified:
                    is_hole_map[color] = not is_hole_map[color]
                if len(idx_list)==0:
                    break

            load_stroke_selection(gp_obj, select_map)

        for frame in frames_to_process:
            process_one_frame(frame)

        return {'FINISHED'}

class OffsetSelectedOperator(bpy.types.Operator, ColorTintConfig):
    """Offset or inset the selected strokes"""
    bl_idname = "gpencil.nijigp_offset_selected"
    bl_label = "Offset Selected"
    bl_category = 'View'
    bl_options = {'REGISTER', 'UNDO'}

    offset_amount: bpy.props.FloatProperty(
            name='Offset',
            default=0, unit='LENGTH',
            description='Offset length'
    )
    multiframe_falloff: bpy.props.FloatProperty(
            name='Multiframe Falloff',
            default=0, min=0, max=1,
            description='The ratio of offset length falloff per frame in the multiframe editing mode',
    )
    corner_shape: bpy.props.EnumProperty(
            name='Corner Shape',
            items=[('JT_ROUND', 'Round', ''),
                    ('JT_SQUARE', 'Square', ''),
                    ('JT_MITER', 'Miter', '')],
            default='JT_ROUND',
            description='Shape of corners generated by offsetting'
    )
    end_type_mode: bpy.props.EnumProperty(
            name='End Type',
            items=[('ET_CLOSE', 'Fill', ''),
                    ('ET_OPEN', 'Line', ''),
                    ('ET_CLOSE_CORNER', 'Corner', '')],
            default='ET_CLOSE',
            description='Offset a stroke as either an open line or a closed fill shape'
    )
    keep_original: bpy.props.BoolProperty(
            name='Keep Original',
            default=False,
            description='Do not delete the original stroke'
    )
    invert_holdout: bpy.props.BoolProperty(
            name='Invert Offset for Holdout',
            default=True,
            description='If a stroke has its fill holdout, invert the offset value'
    )

    def draw(self, context):
        layout = self.layout
        layout.label(text = "Geometry Options:")
        box1 = layout.box()
        box1.prop(self, "offset_amount", text = "Offset Amount")
        if get_multiedit(context.object):
            box1.prop(self, "multiframe_falloff", text = "Multiframe Falloff")
        box1.label(text = "Corner Shape")
        box1.prop(self, "corner_shape", text = "")
        box1.label(text = "Offset Mode")
        box1.prop(self, "end_type_mode", text = "")
        box1.prop(self, "keep_original", text = "Keep Original")
        box1.prop(self, "invert_holdout", text = "Invert Offset for Holdout")

        layout.label(text = "Post-Processing Options:")
        box2 = layout.box()
        box2.prop(self, "tint_color")
        box2.prop(self, "tint_color_factor")
        box2.prop(self, "tint_mode")
        box2.prop(self, "blend_mode")

    def execute(self, context):

        # Import and configure Clipper
        try:
            import pyclipper
        except ImportError:
            self.report({"ERROR"}, "Please install PyClipper in the Preferences panel.")
            return {'FINISHED'}
        clipper = pyclipper.PyclipperOffset()
        clipper.MiterLimit = 10     # TODO: What is the best value?

        jt = pyclipper.JT_ROUND
        if self.corner_shape == "JT_SQUARE":
            jt = pyclipper.JT_SQUARE
        elif self.corner_shape == "JT_MITER":
            jt = pyclipper.JT_MITER

        et = pyclipper.ET_CLOSEDPOLYGON
        if self.end_type_mode == "ET_OPEN":
            if self.corner_shape == "JT_ROUND":
                et = pyclipper.ET_OPENROUND
            else:
                et = pyclipper.ET_OPENBUTT

        # Get a list of layers / frames to process
        current_gp_obj = context.object
        frames_to_process = get_input_frames(current_gp_obj,
                                             multiframe = get_multiedit(current_gp_obj),
                                             return_map = True)

        select_map = save_stroke_selection(current_gp_obj)
        generated_strokes = []
        for frame_number, layer_frame_map in frames_to_process.items():
            stroke_info = []
            stroke_list = []
            load_stroke_selection(current_gp_obj, select_map)

            # Convert selected strokes to 2D polygon point lists
            for i,item in layer_frame_map.items():
                frame = item[0]
                # Consider the case where active_frame is None
                if is_frame_valid(frame):
                    for j,stroke in enumerate(frame.nijigp_strokes):
                        if stroke.select and not is_stroke_protected(stroke, current_gp_obj):
                            stroke_info.append([stroke, i, j, frame])
                            stroke_list.append(stroke)
            t_mat, inv_mat = get_transformation_mat(mode=context.scene.nijigp_working_plane,
                                                    gp_obj=current_gp_obj, 
                                                    strokes=stroke_list, operator=self)
            poly_list, depth_list, scale_factor = get_2d_co_from_strokes(stroke_list, t_mat, scale=True)

            # Call Clipper to execute offset on each stroke
            for j,co_list in enumerate(poly_list):

                # Judge if the stroke has holdout fill
                invert_offset = 1
                if self.invert_holdout and is_stroke_hole(stroke_list[j], current_gp_obj):
                    invert_offset = -1

                # Offset amount calculation
                falloff_factor = 1
                if get_multiedit(current_gp_obj):
                    frame_gap = abs(context.scene.frame_current - frame_number) 
                    falloff_factor = max(0, 1 - frame_gap * self.multiframe_falloff)

                # Execute offset
                clipper.Clear()
                clipper.AddPath(co_list, join_type = jt, end_type = et)
                poly_results = clipper.Execute(self.offset_amount * invert_offset * scale_factor * falloff_factor)

                # For corner mode, execute another offset in the opposite direction
                if self.end_type_mode == 'ET_CLOSE_CORNER':
                    clipper.Clear()
                    clipper.AddPaths(poly_results, join_type = jt, end_type = et)
                    poly_results = clipper.Execute(-self.offset_amount * invert_offset * scale_factor * falloff_factor)

                # If the new stroke is larger, arrange it behind the original one
                arrange_offset = (self.offset_amount * invert_offset) > 0

                if len(poly_results) > 0:
                    for result in poly_results:
                        new_stroke, new_index, new_layer_index = generate_stroke_from_2d(result, inv_mat, [poly_list[j]], [depth_list[j]],
                                                                                         [stroke_info[j]], current_gp_obj, scale_factor,    
                                                                                         rearrange = True, arrange_offset = arrange_offset)
                        generated_strokes.append(new_stroke)
                        new_stroke.use_cyclic = True

                        # Update the stroke index
                        for info in stroke_info:
                            if new_index <= info[2] and new_layer_index == info[1]:
                                info[2] += 1

                if not self.keep_original:
                    stroke_info[j][3].nijigp_strokes.remove(stroke_list[j])

        # Post-processing: change colors
        for stroke in generated_strokes:
            stroke.select = True
        bpy.ops.gpencil.nijigp_color_tint(tint_color=self.tint_color,
                                          tint_color_factor=self.tint_color_factor,
                                          tint_mode=self.tint_mode,
                                          blend_mode=self.blend_mode)
        refresh_strokes(current_gp_obj, list(frames_to_process.keys()))

        return {'FINISHED'}


class FractureSelectedOperator(bpy.types.Operator):
    """Split the paths of multiple selected polygons into non-overlapping parts"""
    bl_idname = "gpencil.nijigp_fracture_selected"
    bl_label = "Fracture Selected"
    bl_category = 'View'
    bl_options = {'REGISTER', 'UNDO'}

    keep_original: bpy.props.BoolProperty(
            name='Keep Original',
            default=False,
            description='Do not delete the original strokes'
    )

    def draw(self, context):
        layout = self.layout
        row = layout.row()
        row.prop(self, "keep_original")

    def execute(self, context):

        # Import and configure Clipper
        try:
            import pyclipper
        except ImportError:
            self.report({"ERROR"}, "Please install PyClipper in the Preferences panel.")
            return {'FINISHED'}
        clipper = pyclipper.Pyclipper()
        clipper.PreserveCollinear = True
        clipper.StrictlySimple = True
        
        # Get a list of layers / frames to process
        current_gp_obj = context.object
        frames_to_process = get_input_frames(current_gp_obj,
                                             multiframe = get_multiedit(current_gp_obj),
                                             return_map = True)

        select_map = save_stroke_selection(current_gp_obj)
        generated_strokes = []
        for frame_number, layer_frame_map in frames_to_process.items():
            load_stroke_selection(current_gp_obj, select_map)
            stroke_info = []
            stroke_list = []
            select_seq_map = {}

            # Convert selected strokes to 2D polygon point lists
            for i,item in layer_frame_map.items():
                frame = item[0]
                layer = current_gp_obj.data.layers[i]
                if is_frame_valid(frame):
                    for j,stroke in enumerate(frame.nijigp_strokes):
                        if stroke.select and not is_stroke_protected(stroke, current_gp_obj):
                            stroke_info.append([stroke, i, j, frame])
                            stroke_list.append(stroke)
                            select_seq_map[len(stroke_list) - 1] = select_map[layer][frame][stroke]
                            
            t_mat, inv_mat = get_transformation_mat(mode=context.scene.nijigp_working_plane,
                                                    gp_obj=current_gp_obj, 
                                                    strokes=stroke_list, operator=self)
            poly_list, depth_list, scale_factor = get_2d_co_from_strokes(stroke_list, t_mat,
                                                                         scale=True,
                                                                         correct_orientation = True)

            # Boolean operation requires at least two shapes
            if len(stroke_info) < 2:
                continue

            poly_results = []
            def generate_new_strokes():
                if len(poly_results) > 0:
                    for result in poly_results:
                        if len(result) == 0:
                            continue
                        new_stroke, new_index, new_layer_index = generate_stroke_from_2d(result, inv_mat, poly_list, depth_list,
                                                                                         stroke_info, current_gp_obj, scale_factor,    
                                                                                         rearrange = True)
                        generated_strokes.append(new_stroke)
                        new_stroke.use_cyclic = True
                        # Update the stroke index
                        for info in stroke_info:
                            if new_index <= info[2] and new_layer_index == info[1]:
                                info[2] += 1

            def fracture_pair(poly1, poly2):
                """
                Split two polygon paths by returning A-B and A*B
                """
                clipper.Clear()
                clipper.AddPath(poly1, pyclipper.PT_SUBJECT, True)
                clipper.AddPath(poly2, pyclipper.PT_CLIP, True)
                part1 = clipper.Execute(pyclipper.CT_DIFFERENCE, pyclipper.PFT_NONZERO, pyclipper.PFT_NONZERO)
                clipper.Clear()
                clipper.AddPath(poly1, pyclipper.PT_SUBJECT, True)
                clipper.AddPath(poly2, pyclipper.PT_CLIP, True)
                part2 = clipper.Execute(pyclipper.CT_INTERSECTION, pyclipper.PFT_NONZERO, pyclipper.PFT_NONZERO)                       
                return part1 + part2
            
            # Starting from 2 polygons, calculate A-B, B-A, A*B as initial paths. Also calculate A+B for future usage
            clipper.Clear()
            clipper.AddPath(poly_list[0], pyclipper.PT_SUBJECT, True)
            clipper.AddPath(poly_list[1], pyclipper.PT_CLIP, True)
            total_union = clipper.Execute(pyclipper.CT_UNION, pyclipper.PFT_NONZERO, pyclipper.PFT_NONZERO)[0]
            poly_results = fracture_pair(poly_list[0], poly_list[1])
            clipper.Clear()
            clipper.AddPath(poly_list[1], pyclipper.PT_SUBJECT, True)
            clipper.AddPath(poly_list[0], pyclipper.PT_CLIP, True)
            poly_results += clipper.Execute(pyclipper.CT_DIFFERENCE, pyclipper.PFT_NONZERO, pyclipper.PFT_NONZERO)
            poly_results = [_ for _ in poly_results if len(_) > 0]
            # Execute Boolean operations for each remaining polygon
            for i in range(2, len(poly_list)):
                tmp_res = []
                # Do fracture with each previously generated path
                for poly in poly_results:
                    tmp_res += fracture_pair(poly, poly_list[i])
                # Do fracture with the union path
                clipper.Clear()
                clipper.AddPath(poly_list[i], pyclipper.PT_SUBJECT, True)
                clipper.AddPath(total_union, pyclipper.PT_CLIP, True)
                tmp_res += clipper.Execute(pyclipper.CT_DIFFERENCE, pyclipper.PFT_NONZERO, pyclipper.PFT_NONZERO)                
                poly_results = [_ for _ in tmp_res if len(_) > 0]
                # Update the union path
                clipper.Clear()
                clipper.AddPath(total_union, pyclipper.PT_SUBJECT, True)
                clipper.AddPath(poly_list[i], pyclipper.PT_CLIP, True)
                total_union = clipper.Execute(pyclipper.CT_UNION, pyclipper.PFT_NONZERO, pyclipper.PFT_NONZERO)[0]
            generate_new_strokes()
            
            # Delete old strokes
            for i,info in enumerate(stroke_info):
                if not self.keep_original:
                    stroke_info[i][3].nijigp_strokes.remove(info[0])

        # Post-processing
        refresh_strokes(current_gp_obj, list(frames_to_process.keys()))
        op_deselect()
        for stroke in generated_strokes:
            stroke.select = True

        return {'FINISHED'}

class BoolSelectedOperator(bpy.types.Operator):
    """Execute Boolean operations on selected strokes"""
    bl_idname = "gpencil.nijigp_bool_selected"
    bl_label = "Boolean of Selected"
    bl_category = 'View'
    bl_options = {'REGISTER', 'UNDO'}

    operation_type: bpy.props.EnumProperty(
            name='Operation',
            items=[('UNION', 'Union', ''),
                    ('INTERSECTION', 'Intersection', ''),
                    ('DIFFERENCE', 'Difference', ''),
                    ('XOR', 'Exclusive', '')],
            default='UNION',
            description='Type of the Boolean operation'
    )
    stroke_inherited: bpy.props.EnumProperty(
            name='Stroke Properties',
            items=[('AUTO', 'Auto', ''),
                    ('SUBJECT', 'Subjects', ''),
                    ('CLIP', 'Clips', '')],
            default='SUBJECT',
            description='Where to get the properties for the new generated strokes'
    )
    processing_seq: bpy.props.EnumProperty(
            name='Processing Sequence',
            items=[('ALL', 'Together', ''),
                    ('EACH', 'One By One', '')],
            default='ALL',
            description='Regard all subjects as a whole or as separate polygons'
    )
    num_clips: bpy.props.IntProperty(
            name='Number of Clips',
            min=1, default=1,
            description='The last one or more strokes are clips, and the rest are subjects. '
                        'In Blender 4.3 or later, the sequence refers to the display order. In earlier versions, it refers to the selection order'
            )
    keep_subjects: bpy.props.BoolProperty(
            name='Keep Subjects',
            default=False,
            description='Do not delete the original subject strokes'
    )
    keep_clips: bpy.props.BoolProperty(
            name='Keep Clips',
            default=False,
            description='Do not delete the original clip strokes'
    )
    reversed_input_order: bpy.props.BoolProperty(
            name='Swap Subject and Clip',
            default=False,
            description='If false, the first strokes are subjects and the last ones are clips, vice versa otherwise. '
                        'In Blender 4.3 or later, the sequence refers refers to the display order. In earlier versions, it refers to the selection order'
    )

    def draw(self, context):
        layout = self.layout
        row = layout.row()
        row.prop(self, "operation_type", text = "Operation")
        row = layout.row()
        row.prop(self, "reversed_input_order")
        layout.label(text = "Inherit stroke properties from:")
        row = layout.row()
        row.prop(self, "stroke_inherited", text = "")
        row = layout.row()
        row.prop(self, "keep_subjects")
        row = layout.row()
        row.prop(self, "keep_clips")

        layout.label(text = "When More Than 2 Strokes Selected:")
        box = layout.box()
        box.prop(self, "num_clips", text = "Number of Clips")
        box.label(text = "Process Each Subject:")
        box.prop(self, "processing_seq", text = "")


    def execute(self, context):

        # Import and configure Clipper
        try:
            import pyclipper
        except ImportError:
            self.report({"ERROR"}, "Please install PyClipper in the Preferences panel.")
            return {'FINISHED'}
        clipper = pyclipper.Pyclipper()
        clipper.PreserveCollinear = True
        clipper.StrictlySimple = True

        op = pyclipper.CT_UNION
        if self.operation_type == 'INTERSECTION':
            op = pyclipper.CT_INTERSECTION
        elif self.operation_type == 'DIFFERENCE':
            op = pyclipper.CT_DIFFERENCE
        elif self.operation_type == 'XOR':
            op = pyclipper.CT_XOR
        
        # Get a list of layers / frames to process
        current_gp_obj = context.object
        frames_to_process = get_input_frames(current_gp_obj,
                                             multiframe = get_multiedit(current_gp_obj),
                                             return_map = True)

        select_map = save_stroke_selection(current_gp_obj)
        generated_strokes = []
        for frame_number, layer_frame_map in frames_to_process.items():
            load_stroke_selection(current_gp_obj, select_map)
            stroke_info = []
            stroke_list = []
            select_seq_map = {}

            # Convert selected strokes to 2D polygon point lists
            for i,item in layer_frame_map.items():
                frame = item[0]
                layer = current_gp_obj.data.layers[i]
                if is_frame_valid(frame):
                    for j,stroke in enumerate(frame.nijigp_strokes):
                        if stroke.select and not is_stroke_protected(stroke, current_gp_obj):
                            stroke_info.append([stroke, i, j, frame])
                            stroke_list.append(stroke)
                            select_seq_map[len(stroke_list) - 1] = select_map[layer][frame][stroke]
                            
            t_mat, inv_mat = get_transformation_mat(mode=context.scene.nijigp_working_plane,
                                                    gp_obj=current_gp_obj, 
                                                    strokes=stroke_list, operator=self)
            poly_list, depth_list, scale_factor = get_2d_co_from_strokes(stroke_list, t_mat,
                                                                         scale=True,
                                                                         correct_orientation = True)

            # Boolean operation requires at least two shapes
            if len(stroke_info) < 2:
                continue

            # Divide strokes into subjects and clips according to their select_index
            # Their should be at least one subject
            true_num_clips = min(self.num_clips, len(stroke_list)-1)
            select_seq = [_ for _ in range(len(stroke_list))]
            select_seq.sort(key = lambda x: select_seq_map[x])
            if self.reversed_input_order:
                select_seq.reverse()
            subject_set = set(select_seq[:len(stroke_list) - true_num_clips])
            clip_set = set(select_seq[-true_num_clips:])

            # Prepare to generate new strokes
            ref_stroke_mask = {}
            if self.stroke_inherited == 'SUBJECT':
                ref_stroke_mask = clip_set
            elif self.stroke_inherited == 'CLIP':
                ref_stroke_mask = subject_set

            poly_results = []
            def generate_new_strokes():
                """
                Depending on the operator option, this function may be called once or multiple times
                """
                if len(poly_results) > 0:
                    for result in poly_results:
                        new_stroke, new_index, new_layer_index = generate_stroke_from_2d(result, inv_mat, poly_list, depth_list,
                                                                                         stroke_info, current_gp_obj, scale_factor,    
                                                                                         rearrange = True, ref_stroke_mask = ref_stroke_mask)
                        generated_strokes.append(new_stroke)
                        if self.operation_type == 'INTERSECTION':
                            new_stroke.use_cyclic = True
                        # Update the stroke index
                        for info in stroke_info:
                            if new_index <= info[2] and new_layer_index == info[1]:
                                info[2] += 1

            # Call Clipper functions
            if self.processing_seq == 'ALL':
                # Add all paths at once
                clipper.Clear()
                for i,co_list in enumerate(poly_list):
                    if i in subject_set:
                        clipper.AddPath(co_list, pyclipper.PT_SUBJECT, True)
                    else:
                        clipper.AddPath(co_list, pyclipper.PT_CLIP, True)
                poly_results = clipper.Execute(op, pyclipper.PFT_NONZERO, pyclipper.PFT_NONZERO)
                generate_new_strokes()
            else:
                # Add one subject path per operation
                for i in subject_set:
                    # Both mask and paths need to be reset after processing each stroke in this mode
                    if self.stroke_inherited == 'SUBJECT':
                        ref_stroke_mask = set(select_seq)
                        ref_stroke_mask.remove(i)
                    clipper.Clear()
                    clipper.AddPath(poly_list[i], pyclipper.PT_SUBJECT, True)
                    for j in clip_set:
                        clipper.AddPath(poly_list[j], pyclipper.PT_CLIP, True)
                    poly_results = clipper.Execute(op, pyclipper.PFT_NONZERO, pyclipper.PFT_NONZERO)
                    generate_new_strokes()

            # Delete old strokes
            for i,info in enumerate(stroke_info):
                if not self.keep_subjects and i in subject_set:
                    stroke_info[i][3].nijigp_strokes.remove(info[0])
                elif not self.keep_clips and i in clip_set:
                    stroke_info[i][3].nijigp_strokes.remove(info[0])

        # Post-processing
        refresh_strokes(current_gp_obj, list(frames_to_process.keys()))
        op_deselect()
        for stroke in generated_strokes:
            stroke.select = True

        return {'FINISHED'}

class BoolLastOperator(bpy.types.Operator):
    """Execute Boolean operations with the latest drawn strokes"""
    bl_idname = "gpencil.nijigp_bool_last"
    bl_label = "Boolean with Last Stroke"
    bl_category = 'View'
    bl_options = {'REGISTER', 'UNDO'}

    operation_type: bpy.props.EnumProperty(
            name='Operation',
            items=[('UNION', 'Union', ''),
                    ('INTERSECTION', 'Intersection', ''),
                    ('DIFFERENCE', 'Difference', ''),
                    ('XOR', 'Exclusive', '')],
            default='UNION',
            description='Type of the Boolean operation'
    )
    clip_mode: bpy.props.EnumProperty(
            name='Boolean Using',
            items=[('FILL', 'Fill', ''),
                    ('LINE', 'Line Radius', '')],
            default='FILL',
            description='Using either the line radius or the fill to calculate the Boolean shapes'
    )
    inherit_clip: bpy.props.BoolProperty(
            name='Inherit Clip Attributes',
            default=False,
            description='Use the stroke attributes of the latest drawn stroke'
    )
    keep_subjects: bpy.props.BoolProperty(
            name='Keep Subjects',
            default=False,
            description='Do not delete the strokes affected by the latest drawn one'
    )
    keep_clips: bpy.props.BoolProperty(
            name='Keep Clips',
            default=False,
            description='Do not delete the latest drawn stroke'
    )

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "operation_type", text = "Operation")
        layout.prop(self, "clip_mode", text = "Type")
        layout.prop(self, "inherit_clip", text = "Use Drawn Stroke's Attributes")
        layout.prop(self, "keep_subjects", text = "Keep Affected Strokes")

    def execute(self, context):
        # Import and configure Clipper
        try:
            import pyclipper
        except ImportError:
            self.report({"ERROR"}, "Please install PyClipper in the Preferences panel.")
            return {'FINISHED'}
        clipper = pyclipper.Pyclipper()
        clipper.PreserveCollinear = True
        clipper.StrictlySimple = True
        
        op = pyclipper.CT_UNION
        if self.operation_type == 'INTERSECTION':
            op = pyclipper.CT_INTERSECTION
        elif self.operation_type == 'DIFFERENCE':
            op = pyclipper.CT_DIFFERENCE
        elif self.operation_type == 'XOR':
            op = pyclipper.CT_XOR

        # Search operation targets only in active layer
        current_gp_obj = context.object
        if not current_gp_obj.data.layers.active:
            self.report({"INFO"}, "Please select a layer.")
            return {'FINISHED'}
        bpy.ops.object.mode_set(mode=get_obj_mode_str('EDIT'))
        layer_index = get_active_layer_index(current_gp_obj)
        layer = current_gp_obj.data.layers[layer_index]

        if layer_locked(layer):
            self.report({"INFO"}, "Please select an unlocked layer.")
            bpy.ops.object.mode_set(mode='PAINT_GPENCIL')
            return {'FINISHED'}
        if len(layer.active_frame.nijigp_strokes) < 1:
            self.report({"INFO"}, "Please select a non-empty layer.")
            bpy.ops.object.mode_set(mode='PAINT_GPENCIL')
            return {'FINISHED'}   
        
        stroke_index = 0 if context.scene.tool_settings.use_gpencil_draw_onback else (len(layer.active_frame.nijigp_strokes) - 1)
        clip_stroke = layer.active_frame.nijigp_strokes[stroke_index]
        stroke_info = [[clip_stroke, layer_index, stroke_index]]
        stroke_list = [clip_stroke]
        selection_map = []
        t_mat, inv_mat = get_transformation_mat(mode=context.scene.nijigp_working_plane,
                                                strokes=stroke_list,
                                                gp_obj=current_gp_obj, operator=self)
        # Determine the clip shape
        clip_polys, _, poly_inverted, scale_factor = get_2d_co_from_strokes(stroke_list, t_mat,
                                                                     scale=True, correct_orientation=True, return_orientation=True)
        if self.clip_mode == 'LINE':
            clip_polys = get_2d_stroke_outline(clip_polys[0], clip_stroke, scale_factor, poly_inverted[0])
            clip_polys = [poly for poly in clip_polys if pyclipper.Area(poly) > 1]
        if len(clip_polys) < 1:
            bpy.ops.object.mode_set(mode=get_obj_mode_str('PAINT'))
            return {'FINISHED'}
            
        clip_points = []
        for poly in clip_polys:
            clip_points += poly
        clip_bound_box = [min(clip_points, key=lambda p: p[0])[0], min(clip_points, key=lambda p: p[1])[1],
                          max(clip_points, key=lambda p: p[0])[0], max(clip_points, key=lambda p: p[1])[1]]

        # Check each existing stroke if it can be operated
        for j,stroke in enumerate(layer.active_frame.nijigp_strokes):
            if j == stroke_index:
                continue
            selection_map.append((stroke, stroke.select))
            if is_stroke_protected(stroke, current_gp_obj):
                continue
            # Check the condition of each filter
            if context.scene.nijigp_draw_bool_material_constraint == 'SAME' and stroke.material_index != clip_stroke.material_index:
                continue
            if context.scene.nijigp_draw_bool_material_constraint == 'DIFF' and stroke.material_index == clip_stroke.material_index:
                continue
            if context.scene.nijigp_draw_bool_fill_constraint == 'FILL' and (not current_gp_obj.data.materials[stroke.material_index].grease_pencil.show_fill):
                continue
            if context.scene.nijigp_draw_bool_selection_constraint and not stroke.select:
                continue
            if not overlapping_strokes(stroke, clip_points, clip_bound_box, t_mat, scale_factor):
                continue
            stroke_list.append(stroke)
            stroke_info.append([stroke, layer_index, j])

        if len(stroke_list) == 1:
            bpy.ops.object.mode_set(mode=get_obj_mode_str('PAINT'))
            return {'FINISHED'}
        op_deselect()
        
        # Use two transform matrices: 
        #   - The first is based on the newly drawn stroke for viewport projection
        #   - The second is based on existing strokes for depth correction
        t_mat_mod, inv_mat_mod = get_transformation_mat(mode=context.scene.nijigp_working_plane,
                                                        strokes=stroke_list[1:],
                                                        gp_obj=current_gp_obj, operator=self)   
        poly_list_tmp, depth_list_tmp, _ = get_2d_co_from_strokes(stroke_list[1:], t_mat_mod, scale=False, correct_orientation=False)
        depth_correction_lookup = DepthLookupTree(poly_list_tmp, depth_list_tmp)

        poly_list, depth_list, poly_inverted, _ = get_2d_co_from_strokes(stroke_list, t_mat,
                                                                     scale=True, correct_orientation=True, scale_factor=scale_factor, return_orientation=True)

        # Operate on the last stroke with any other stroke one by one
        generated_strokes = []
        for j in range(1, len(stroke_list)):
            clipper.Clear()
            clipper.AddPath(poly_list[j], pyclipper.PT_SUBJECT, True)
            clipper.AddPaths(clip_polys, pyclipper.PT_CLIP, True)
            poly_results = clipper.Execute(op, pyclipper.PFT_NONZERO, pyclipper.PFT_NONZERO)

            if len(poly_results) > 0:
                for result in poly_results:
                    new_stroke, new_index, _ = generate_stroke_from_2d(result, inv_mat, 
                                                                       [poly_list[j]] if self.clip_mode == 'LINE' else [poly_list[j], poly_list[0]],
                                                                       [depth_list[j]] if self.clip_mode == 'LINE' else [depth_list[j], depth_list[0]],
                                                                        [stroke_info[j], stroke_info[0]], current_gp_obj, scale_factor,    
                                                                        rearrange = True, ref_stroke_mask = {not self.inherit_clip})
                    if self.operation_type == 'INTERSECTION':
                        new_stroke.use_cyclic = True
                    # Depth correction
                    for point in new_stroke.points:
                        co_2d = t_mat_mod @ point.co
                        new_depth = depth_correction_lookup.get_depth(co_2d)
                        point.co = inv_mat_mod @ Vector((co_2d[0], co_2d[1], new_depth))
                    # Update the stroke index
                    for info in stroke_info:
                        if new_index <= info[2]:
                            info[2] += 1
                    generated_strokes.append(new_stroke)

        # Select either the existing strokes or the new stroke, depending on options
        for stroke in generated_strokes:
            if self.keep_subjects and context.scene.nijigp_draw_bool_selection_constraint:
                stroke.select = False
            else:
                stroke.select = True
        if context.scene.nijigp_draw_bool_selection_constraint:
            for stroke, selected in selection_map:
                if stroke:
                    stroke.select = selected
        # Delete old strokes
        for i,info in enumerate(stroke_info):
            if (i==0 and self.keep_clips) or (i>0 and self.keep_subjects):
                continue
            layer_index = info[1]
            current_gp_obj.data.layers[layer_index].active_frame.nijigp_strokes.remove(info[0])

        refresh_strokes(current_gp_obj)
        bpy.ops.object.mode_set(mode=get_obj_mode_str('PAINT'))
        return {'FINISHED'}
    
class SweepSelectedOperator(bpy.types.Operator, ColorTintConfig):
    """Sweep selected shapes along a path"""
    bl_idname = "gpencil.nijigp_sweep_selected"
    bl_label = "Sweep Selected"
    bl_category = 'View'
    bl_options = {'REGISTER', 'UNDO'}

    multiframe_falloff: bpy.props.FloatProperty(
            name='Multiframe Falloff',
            default=0, min=0, max=1,
            description='The ratio of sweep length falloff per frame in the multiframe editing mode',
    )
    path_type: bpy.props.EnumProperty(
            name='Path Type',
            items=[('STROKE', 'Last Selected Stroke', ''),
                    ('VEC', 'Vector', '')],
            default='VEC'
    )
    path_vector: bpy.props.FloatVectorProperty(
            name='Path Vector',
            default=(0.0, -0.1), size=2, soft_min=-5, soft_max=5, unit='LENGTH',
            description='2D vector of the sweeping path'
    )
    style: bpy.props.EnumProperty(
            name='Style',
            items=[('EXTRUDE', 'Extrude', ''),
                    ('OUTER', 'Outer Shadow', ''),
                    ('INNER', 'Inner Shadow', '')],
            default='EXTRUDE'
    )
    invert_holdout: bpy.props.BoolProperty(
            name='Process Holdout Separately',
            default=True,
            description='When generating outer shadow, use inner shadow instead for shapes with their fill holdout; when generating inner shadow, ignore such shapes'
    )
    keep_original: bpy.props.BoolProperty(
            name='Keep Original',
            default=True,
            description='Do not delete the original stroke'
    )

    def draw(self, context):
        layout = self.layout
        layout.label(text = "Geometry Options:")
        box1 = layout.box()
        box1.prop(self, "style")
        box1.prop(self, "path_type")
        if self.path_type == 'VEC':
            box1.prop(self, "path_vector")
        if get_multiedit(context.object):
            box1.prop(self, "multiframe_falloff")
        box1.prop(self, "invert_holdout")
        box1.prop(self, "keep_original")

        layout.label(text = "Post-Processing Options:")
        box2 = layout.box()
        box2.prop(self, "tint_color")
        box2.prop(self, "tint_color_factor")
        box2.prop(self, "tint_mode")
        box2.prop(self, "blend_mode")


    def execute(self, context):
        MIN_AREA = 4                # Eliminate too small shapes which may be errors of Boolean operations
        CONSTANT_SWEEP_DIST = 64    # A large enough constant, for inner shadow style only

        # Import and configure Clipper
        try:
            import pyclipper
        except ImportError:
            self.report({"ERROR"}, "Please install PyClipper in the Preferences panel.")
            return {'FINISHED'}
        clipper = pyclipper.Pyclipper()
        clipper.PreserveCollinear = True
     
        # Get a list of layers / frames to process
        current_gp_obj = context.object
        frames_to_process = get_input_frames(current_gp_obj,
                                             multiframe = get_multiedit(current_gp_obj),
                                             return_map = True)

        select_map = save_stroke_selection(current_gp_obj)
        generated_strokes = []
        for frame_number, layer_frame_map in frames_to_process.items():
            load_stroke_selection(current_gp_obj, select_map)
            stroke_info = []
            stroke_list = []
            select_seq_map = {}

            # Convert selected strokes to 2D polygon point lists
            for i,item in layer_frame_map.items():
                frame = item[0]
                layer = current_gp_obj.data.layers[i]
                if is_frame_valid(frame):
                    for j,stroke in enumerate(frame.nijigp_strokes):
                        if stroke.select and not is_stroke_protected(stroke, current_gp_obj):
                            stroke_info.append([stroke, i, j, frame])
                            stroke_list.append(stroke)
                            select_seq_map[len(stroke_list) - 1] = select_map[layer][frame][stroke]
                            
            t_mat, inv_mat = get_transformation_mat(mode=context.scene.nijigp_working_plane,
                                                    gp_obj=current_gp_obj, 
                                                    strokes=stroke_list, operator=self)
            poly_list, depth_list, scale_factor = get_2d_co_from_strokes(stroke_list, t_mat,
                                                                         scale=True,
                                                                         correct_orientation = False)

            if len(stroke_info) < 1:
                continue

            # Multi-frame falloff factor
            falloff_factor = 1
            if get_multiedit(current_gp_obj):
                frame_gap = abs(context.scene.frame_current - frame_number) 
                falloff_factor = max(0, 1 - frame_gap * self.multiframe_falloff)

            # Get the sweep path
            path = {}
            path_is_closed = False
            if self.path_type == 'VEC':
                # For inner shadow, offset the starting point
                path['INNER'] = [(self.path_vector[0]*scale_factor*falloff_factor, 
                                    self.path_vector[1]*scale_factor*falloff_factor)]
                end_direction = Vector(self.path_vector).normalized()
                path['INNER'].append((path['INNER'][0][0]+end_direction[0]*CONSTANT_SWEEP_DIST*scale_factor, 
                                        path['INNER'][0][1]+end_direction[1]*CONSTANT_SWEEP_DIST*scale_factor))
                # For outer shadow, starting from the origin
                path['OUTER'] = [(0,0), 
                                (self.path_vector[0]*scale_factor*falloff_factor, 
                                self.path_vector[1]*scale_factor*falloff_factor)]
            else:
                # Find the last selected stroke and get its information
                path_idx = max(range(len(stroke_list)), key=lambda k: select_seq_map[k])
                path_is_closed = stroke_list[path_idx].use_cyclic
                if len(poly_list[path_idx]) < 2:
                    continue
                starting_point = poly_list[path_idx][0]
                
                end_point = max(1, falloff_factor*(len(poly_list[path_idx])-1))
                path['INNER'] = [(poly_list[path_idx][end_point][0]-starting_point[0],
                                poly_list[path_idx][end_point][1]-starting_point[1])]
                end_direction = (Vector(poly_list[path_idx][end_point]) - Vector(poly_list[path_idx][end_point-1])).normalized()
                path['INNER'].append((path['INNER'][0][0]+end_direction[0]*CONSTANT_SWEEP_DIST*scale_factor, 
                                    path['INNER'][0][1]+end_direction[1]*CONSTANT_SWEEP_DIST*scale_factor))
                path['OUTER'] = []
                for co in poly_list[path_idx]:
                    path['OUTER'].append((co[0] - starting_point[0], co[1] - starting_point[1]))
                    if len(path['OUTER'])>1 and len(path['OUTER'])>len(poly_list[path_idx])*falloff_factor:
                        break
            path['EXTRUDE'] = path['OUTER']

            # Process each stroke
            for j,co_list in enumerate(poly_list):
                if self.path_type != 'VEC' and j == path_idx:
                    continue
                if len(co_list) < 3:    # Clipper cannot process them
                    continue
                new_material_index = None
                style = str(self.style)
                if self.invert_holdout and is_stroke_hole(stroke_list[j], current_gp_obj):
                    # When generating inner shadow, ignore holdout. Otherwise, invert the shadow direction
                    if style == 'INNER':
                        continue
                    else:
                        style = 'INNER'
                        holdout_material = current_gp_obj.material_slots[stroke_list[j].material_index].material
                        if 'original_material_index' in holdout_material:
                            new_material_index = holdout_material['original_material_index']
                poly_results = pyclipper.MinkowskiSum(co_list, path[style], path_is_closed)

                # Trim the result using Boolean operations
                clipper.Clear()
                op = pyclipper.CT_UNION if style == 'EXTRUDE' else pyclipper.CT_DIFFERENCE
                role0 = pyclipper.PT_CLIP if style != 'INNER' else pyclipper.PT_SUBJECT
                role1 = pyclipper.PT_SUBJECT if style != 'INNER' else pyclipper.PT_CLIP
                for result in poly_results:
                    clipper.AddPath(result, role1, True)
                clipper.AddPath(co_list, role0, True)
                poly_results = clipper.Execute(op, pyclipper.PFT_NONZERO, pyclipper.PFT_NONZERO)

                for result in poly_results:
                    if pyclipper.Area(result) < MIN_AREA:
                        continue
                    new_stroke, new_index, new_layer_index = generate_stroke_from_2d(result, inv_mat, [poly_list[j]], [depth_list[j]],
                                                                                        [stroke_info[j]], current_gp_obj, scale_factor,    
                                                                                        rearrange = True, 
                                                                                        arrange_offset = (style!='INNER'))
                    generated_strokes.append(new_stroke)
                    new_stroke.use_cyclic = True
                    if new_material_index != None:
                        new_stroke.material_index = new_material_index
                        
                    # Update the stroke index
                    for info in stroke_info:
                        if new_index <= info[2] and new_layer_index == info[1]:
                            info[2] += 1      

                if not self.keep_original:
                    stroke_info[j][3].nijigp_strokes.remove(stroke_list[j])            

        # Post-processing: change colors
        for stroke in generated_strokes:
            stroke.select = True
        bpy.ops.gpencil.nijigp_color_tint(tint_color=self.tint_color,
                                          tint_color_factor=self.tint_color_factor,
                                          tint_mode=self.tint_mode,
                                          blend_mode=self.blend_mode)
        refresh_strokes(current_gp_obj, list(frames_to_process.keys()))

        return {'FINISHED'}
    