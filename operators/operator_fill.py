import bpy
from .common import *
from ..utils import *

class SmartFillOperator(bpy.types.Operator):
    """Generate fill shapes given a line art layer and a hint layer"""
    bl_idname = "gpencil.nijigp_smart_fill"
    bl_label = "Smart Fill"
    bl_category = 'View'
    bl_options = {'REGISTER', 'UNDO'}

    line_layer: bpy.props.StringProperty(
        name='Line Art Layer',
        description='',
        default='',
        search=lambda self, context, edit_text: [layer.info for layer in context.object.data.layers]
    )
    hint_layer: bpy.props.StringProperty(
        name='Hint Layer',
        description='',
        default='',
        search=lambda self, context, edit_text: [layer.info for layer in context.object.data.layers]
    )
    fill_layer: bpy.props.StringProperty(
        name='Fill Layer',
        description='',
        default='',
        search=lambda self, context, edit_text: [layer.info for layer in context.object.data.layers]
    )
    precision: bpy.props.FloatProperty(
        name='Precision',
        default=0.01, min=0.001, max=1,
        description='Treat points in proximity as one to speed up'
    )
    fill_holes: bpy.props.BoolProperty(
        name='Fill Holes',
        default=True,
        description='Fill holes as much as possible'
    )
    clear_hint_layer: bpy.props.BoolProperty(
        name='Clear Hints',
        default=False,
        description=''
    )
    clear_fill_layer: bpy.props.BoolProperty(
        name='Clear Previous Fills',
        default=False,
        description=''
    )
    color_mode: bpy.props.EnumProperty(            
        name='Color Mode',
        items=[ ('VERTEX', 'Vertex Color', ''),
                ('MATERIAL', 'New Materials', '')],
        default='MATERIAL',
        description='Whether using an existing material with vertex colors, or creating new materials'
    )
    output_material: bpy.props.StringProperty(
        name='Output Material',
        description='Draw the new strokes using this material. If empty, use the active material',
        default='',
        search=lambda self, context, edit_text: [material.name for material in context.object.data.materials if material]
    )

    def draw(self, context):
        layout = self.layout
        layout.label(text = "Input/Output Layers:")
        box1 = layout.box()
        row = box1.row()
        row.label(text = "Line Art Layer:")
        row.prop(self, "line_layer", icon='OUTLINER_DATA_GP_LAYER', text='')
        row = box1.row()
        row.label(text = "Hint Layer:")
        row.prop(self, "hint_layer", icon='OUTLINER_DATA_GP_LAYER', text='')
        row = box1.row()
        row.label(text = "Fill Layer:")
        row.prop(self, "fill_layer", icon='OUTLINER_DATA_GP_LAYER', text='')

        layout.label(text = "Geometry Options:")
        box2 = layout.box()
        box2.prop(self, "precision")
        box2.prop(self, "fill_holes")

        layout.label(text = "Output Options:")
        box3 = layout.box()
        box3.prop(self, "clear_hint_layer")
        box3.prop(self, "clear_fill_layer")
        row = box3.row()
        row.label(text='Color Mode:')
        row.prop(self, "color_mode", text='')
        if self.color_mode == 'VERTEX':
            box3.prop(self, "output_material", text='Material', icon='MATERIAL')

    def execute(self, context):
        gp_obj = context.object
        current_mode = context.mode
        try:
            from ..solvers.graph import SmartFillSolver
        except:
            self.report({"ERROR"}, "Please install Scikit-Image in the Preferences panel.")
            return {'FINISHED'}
        
        # Get and validate layers
        if (len(self.line_layer) < 1 or
            len(self.hint_layer) < 1 or
            len(self.fill_layer) < 1):
            return {'FINISHED'}
        line_layer = gp_obj.data.layers[self.line_layer]
        hint_layer = gp_obj.data.layers[self.hint_layer]
        fill_layer = gp_obj.data.layers[self.fill_layer]
        if fill_layer.lock:
            self.report({"WARNING"}, "The output layer is locked.")
            return {'FINISHED'}
        if (self.line_layer == self.hint_layer or
            self.line_layer == self.fill_layer or
            self.hint_layer == self.fill_layer):
            self.report({"INFO"}, "Please select 3 different layers.")
            return {'FINISHED'}

        bpy.ops.object.mode_set(mode='EDIT_GPENCIL')
        bpy.ops.gpencil.select_all(action='DESELECT')

        def fill_single_frame(line_frame, hint_frame, fill_frame):
            resolution = self.precision
            if self.clear_fill_layer:
                for stroke in list(fill_frame.strokes):
                    fill_frame.strokes.remove(stroke)
                    
            # Get points and bound box of line frame
            margin_sizes = (0.1, 0.3, 0.5)
            corners = [None, None, None, None]
            stroke_list = []
            for stroke in line_frame.strokes:
                stroke_list.append(stroke)
                for co in (stroke.bound_box_min, stroke.bound_box_max):
                    u, v = vec3_to_vec2(co)
                    corners[0] = u if (not corners[0] or u<corners[0]) else corners[0]
                    corners[1] = v if (not corners[1] or v<corners[1]) else corners[1]
                    corners[2] = u if (not corners[2] or u>corners[2]) else corners[2]
                    corners[3] = v if (not corners[3] or v>corners[3]) else corners[3]
            poly_list, scale_factor = stroke_to_poly(stroke_list, True)
            corners = [co * scale_factor for co in corners]
            bound_W, bound_H = corners[2]-corners[0], corners[3]-corners[1]
            
            # Build triangles from points
            co_idx = {}
            tr_input = dict(vertices = [], segments = [])
            for i,co_list in enumerate(poly_list):
                for j,co in enumerate(co_list):
                    key = (int(co[0]*resolution), int(co[1]*resolution))
                    if key not in co_idx:
                        co_idx[key] = len(co_idx)
                        tr_input['vertices'].append(tuple(co))
                    if j>0:
                        key0 = (int(co_list[j-1][0]*resolution), int(co_list[j-1][1]*resolution))
                        tr_input['segments'].append( (co_idx[key], co_idx[key0]) )

            # Add multiple levels of bound boxes
            for ratio in margin_sizes:  
                tr_input['vertices'] += [(corners[0] - ratio * bound_W, corners[1] - ratio * bound_H),
                                        (corners[0] - ratio * bound_W, corners[3] + ratio * bound_H),
                                        (corners[2] + ratio * bound_W, corners[1] - ratio * bound_H),
                                        (corners[2] + ratio * bound_W, corners[3] + ratio * bound_H)]
            tr_output = {}
            tr_output['vertices'], tr_output['segments'], tr_output['triangles'], _,tr_output['orig_edges'],_ = geometry.delaunay_2d_cdt(tr_input['vertices'], tr_input['segments'], [], 0, 1e-9)

            # Build graph from triangles
            solver = SmartFillSolver()
            solver.build_graph(tr_output)
            
            # Extract colors from hint strokes to label the triangle node graph
            # Label 0 is reserved for transparent regions
            label_colors, label_color_map = [None], {}
            for stroke in hint_frame.strokes:
                hint_points_co, hint_points_label = [], []
                use_line_color = is_stroke_line(stroke, gp_obj)
                for point in stroke.points:
                        if use_line_color:
                            color = (point.vertex_color if point.vertex_color[3] > 0 else
                                    gp_obj.data.materials[stroke.material_index].grease_pencil.color)
                        else:
                            color = (stroke.vertex_color_fill if stroke.vertex_color_fill[3] > 0 else
                                    gp_obj.data.materials[stroke.material_index].grease_pencil.fill_color)
                        c_key = rgb_to_hex_code(color)
                        if c_key not in label_color_map:
                            label_color_map[c_key] = len(label_colors)
                            label_colors.append(color)
                        hint_points_co.append(vec3_to_vec2(point.co) * scale_factor)
                        hint_points_label.append(label_color_map[c_key])
                solver.set_labels_from_points(hint_points_co, hint_points_label)
            solver.propagate_labels()
            if self.fill_holes:
                solver.complete_labels()
            
            # Find or generate materials for each label (color)
            output_material_idx = [None]
            material_name = self.output_material
            if len(material_name)<1:
                material_name = gp_obj.active_material.name
            for color in label_colors:
                if not color:
                    continue
                if self.color_mode == 'MATERIAL':
                    material_name = 'GP_Fill' + rgb_to_hex_code(color)
                for i,material_slot in enumerate(gp_obj.material_slots):
                    # Case 1: Material added to active object
                    if material_slot.material and material_slot.material.name == material_name:
                        output_material_idx.append(i)
                        break
                else:
                    # Case 2: Material not created
                    if material_name not in bpy.data.materials:
                        mat = bpy.data.materials.new(material_name)
                        bpy.data.materials.create_gpencil_data(mat)
                        mat.grease_pencil.show_fill = True
                        mat.grease_pencil.show_stroke = False
                        mat.grease_pencil.fill_color = [color[0],color[1],color[2],1]
                    # Case 3: Material created but not added
                    gp_obj.data.materials.append(bpy.data.materials[material_name])
                    output_material_idx.append(len(gp_obj.material_slots)-1)

            # Generate new strokes from contours of the filled regions
            contours_co, contours_label = solver.get_contours()
            for i, contours in enumerate(contours_co):
                label = contours_label[i]
                if label < 1:
                    continue
                for c in contours:
                    new_stroke: bpy.types.GPencilStroke = fill_frame.strokes.new()
                    new_stroke.points.add(len(c))
                    new_stroke.use_cyclic = True
                    new_stroke.material_index = output_material_idx[label]
                    if self.color_mode == 'VERTEX':
                        new_stroke.vertex_color_fill = (label_colors[label][0],
                                                        label_colors[label][1],
                                                        label_colors[label][2],
                                                        1)
                    for i,co in enumerate(c):
                        new_stroke.points[i].co = vec2_to_vec3(co, 0, scale_factor)
                    new_stroke.select = True  

            if self.clear_hint_layer:
                for stroke in list(hint_frame.strokes):
                    hint_frame.strokes.remove(stroke)

        # Get the frames from each layer to process
        processed_frame_numbers = []
        if not gp_obj.data.use_multiedit:
            if fill_layer.active_frame:
                fill_single_frame(line_layer.active_frame,
                                hint_layer.active_frame,
                                fill_layer.active_frame)
            else:
                fill_frame = fill_layer.frames.new(line_layer.active_frame.frame_number)
                fill_single_frame(line_layer.active_frame,
                                hint_layer.active_frame,
                                fill_frame)
            processed_frame_numbers.append(fill_layer.active_frame.frame_number)
        else:
            # Process each selected line art frame
            for line_frame in line_layer.frames:
                if line_frame.select:
                    current_frame = line_frame.frame_number

                # Find the hint frame
                hint_frame = None
                for f in hint_layer.frames:
                    if f.frame_number > current_frame:
                        break
                    hint_frame = f
                if not hint_frame:
                    self.report({"WARNING"}, "Please add a keyframe in the hint layer.")
                    return {'FINISHED'}
                
                # Find or create the fill frame
                fill_frame = None
                for f in fill_layer.frames:
                    if f.frame_number == current_frame:
                        fill_frame = f
                        break
                if not fill_frame:
                    fill_frame = fill_layer.frames.new(current_frame)

                fill_single_frame(line_frame, hint_frame, fill_frame)
                processed_frame_numbers.append(line_frame.frame_number)

        refresh_strokes(gp_obj, processed_frame_numbers)
        bpy.ops.gpencil.nijigp_hole_processing(rearrange=True, apply_holdout=False)
        bpy.ops.object.mode_set(mode=current_mode)
        return {'FINISHED'}
