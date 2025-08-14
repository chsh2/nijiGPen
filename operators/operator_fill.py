import bpy
import numpy as np
from .common import *
from ..utils import *
from ..api_router import *

def lineart_triangulation(stroke_list, t_mat, poly_list, scale_factor, resolution):
    """Using Blender native Delaunay method on line art strokes"""
    corners = get_2d_bound_box(stroke_list, t_mat)
    corners = [co * scale_factor for co in corners]
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
            if j == len(co_list) - 1 and stroke_list[i].use_cyclic:
                key0 = (int(co_list[0][0]*resolution), int(co_list[0][1]*resolution))
                tr_input['segments'].append( (co_idx[key], co_idx[key0]) )                        
    # Add several margins to the bound boxes
    margin_sizes = (0.1, 0.3, 0.5)
    for ratio in margin_sizes:  
        tr_input['vertices'] += pad_2d_box(corners, ratio)
    tr_output = {}
    tr_output['vertices'], tr_output['segments'], tr_output['triangles'], _,tr_output['orig_edges'],_ = geometry.delaunay_2d_cdt(tr_input['vertices'], tr_input['segments'], [], 0, 1e-9)
    return tr_output

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
        search=multilayer_search_func
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
    use_boundary_strokes: bpy.props.BoolProperty(
        name='Boundary Strokes as Hints',
        default=False,
        description='Use boundary strokes in the fill layer as hints'
    )
    precision: bpy.props.FloatProperty(
        name='Precision',
        default=0.05, min=0.001, max=1,
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
    material_mode: bpy.props.EnumProperty(            
        name='Material Mode',
        items=[ ('NEW', 'New Materials', ''),
               ('SELECT', 'Select a Material', ''),
               ('HINT', 'From Hints', ''),],
        default='NEW',
        description='Whether using existing materials or creating new ones based on vertex colors'
    )
    output_material: bpy.props.StringProperty(
        name='Output Material',
        description='Draw the new strokes using this material. If empty, use the active material',
        default='',
        search=lambda self, context, edit_text: [material.name for material in context.object.data.materials if material]
    )

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self, width=300)

    def draw(self, context):
        layout = self.layout
        layout.label(text = "Input/Output Layers:")
        box1 = layout.box()
        row = box1.row()
        row.label(text = "Line Art Layer:")
        row.prop(self, "line_layer", icon='OUTLINER_DATA_GP_LAYER', text='')
        if not self.use_boundary_strokes:
            row = box1.row()
            row.label(text = "Hint Layer:")
            row.prop(self, "hint_layer", icon='OUTLINER_DATA_GP_LAYER', text='')
        row = box1.row()
        row.label(text = "Fill Layer:")
        row.prop(self, "fill_layer", icon='OUTLINER_DATA_GP_LAYER', text='')
        box1.prop(self, "use_boundary_strokes")

        layout.label(text = "Geometry Options:")
        box2 = layout.box()
        box2.prop(self, "precision")
        box2.prop(self, "fill_holes")

        layout.label(text = "Output Options:")
        box3 = layout.box()
        row = box3.row()
        row.prop(self, "clear_hint_layer")
        row.prop(self, "clear_fill_layer")
        row = box3.row()
        row.label(text='Material Mode:')
        row.prop(self, "material_mode", text='')
        if self.material_mode == 'SELECT':
            box3.prop(self, "output_material", text='Material', icon='MATERIAL')

    def execute(self, context):
        gp_obj = context.object
        current_mode = gp_obj.mode
        try:
            from ..solvers.graph import SmartFillSolver
        except:
            self.report({"ERROR"}, "Please install Scikit-Image in the Preferences panel.")
            return {'FINISHED'}
        
        # Get and validate input/output layers
        if (len(self.line_layer) < 1
            or (len(self.hint_layer) < 1 and not self.use_boundary_strokes)
            or len(self.fill_layer) < 1):
            return {'FINISHED'}
        line_layers, _ = multilayer_search_decode(self.line_layer)
        hint_layer = (gp_obj.data.layers[self.fill_layer] if self.use_boundary_strokes else
                      gp_obj.data.layers[self.hint_layer])
        fill_layer = gp_obj.data.layers[self.fill_layer]
        if fill_layer.lock:
            self.report({"WARNING"}, "The output layer is locked.")
            return {'FINISHED'}
        if self.hint_layer in [layer.info for layer in line_layers]:
            self.report({"WARNING"}, "Hint layer cannot be any of the line layers.")
            return {'FINISHED'}
        if self.fill_layer == self.hint_layer and not self.use_boundary_strokes:
            self.clear_fill_layer = False

        bpy.ops.object.mode_set(mode=get_obj_mode_str('EDIT'))
        op_deselect()

        def fill_single_frame(line_frames, hint_frame, fill_frame):
            if sum([len(line_frame.nijigp_strokes) for line_frame in line_frames]) < 1:
                return
            # Get points of line frame
            stroke_list = []
            for line_frame in line_frames:
                stroke_list += [stroke for stroke in line_frame.nijigp_strokes]
            t_mat, inv_mat = get_transformation_mat(mode=context.scene.nijigp_working_plane,
                                                    gp_obj=gp_obj, strokes=stroke_list, operator=self, 
                                                    requires_layer=False)
            poly_list, depth_list, scale_factor = get_2d_co_from_strokes(stroke_list, t_mat, scale=True)
            depth_lookup_tree = DepthLookupTree(poly_list, depth_list)
            
            # Build graph from triangles converted from the line art
            tr_output = lineart_triangulation(stroke_list, t_mat, poly_list, scale_factor, self.precision)
            solver = SmartFillSolver()
            solver.build_graph(tr_output)
            
            # Extract colors/materials from hint strokes to label the triangle node graph
            # Label 0 is reserved for transparent regions
            labels_info, label_map = [(None, None, False)], {}
            for stroke in reversed(hint_frame.nijigp_strokes):
                if self.use_boundary_strokes and not stroke.is_nofill_stroke:
                    continue
                hint_points_co, hint_points_label = [], []
                use_line_color = is_stroke_line(stroke, gp_obj)
                for point in stroke.points:
                        if use_line_color:
                            color = (point.vertex_color if point.vertex_color[3] > 0 else
                                    gp_obj.data.materials[stroke.material_index].grease_pencil.color)
                            use_vertex_color = (point.vertex_color[3] > 0)
                        else:
                            color = (stroke.vertex_color_fill if stroke.vertex_color_fill[3] > 0 else
                                    gp_obj.data.materials[stroke.material_index].grease_pencil.fill_color)
                            use_vertex_color = (stroke.vertex_color_fill[3] > 0)
                        # Use both color and material index to define a label
                        material_idx = stroke.material_index if self.material_mode == 'HINT' else -1
                        c_key = (rgb_to_hex_code(color), material_idx, use_vertex_color)
                        if c_key not in label_map:
                            label_map[c_key] = len(labels_info)
                            labels_info.append([color, material_idx, use_vertex_color])
                        hint_points_co.append((np.array(t_mat @ point.co) * scale_factor)[:2])
                        hint_points_label.append(label_map[c_key])
                solver.set_labels_from_points(hint_points_co, hint_points_label)
            solver.propagate_labels()
            if self.fill_holes:
                solver.complete_labels()
            
            # Find or generate materials for each label (color)
            material_name = self.output_material
            if len(material_name)<1:
                material_name = gp_obj.active_material.name
            for item in labels_info:
                color = item[0]
                if not color or item[1] > -1:   # Material already known
                    continue
                if self.material_mode == 'NEW':
                    material_name = 'GP_Fill' + rgb_to_hex_code(color)
                for i,material_slot in enumerate(gp_obj.material_slots):
                    # Case 1: Material added to active object
                    if material_slot.material and material_slot.material.name == material_name:
                        item[1] = i
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
                    item[1] = len(gp_obj.material_slots)-1

            if self.clear_fill_layer:
                for stroke in list(fill_frame.nijigp_strokes):
                    if not stroke.is_nofill_stroke:
                        fill_frame.nijigp_strokes.remove(stroke)

            # Generate new strokes from contours of the filled regions
            contours_co, contours_label = solver.get_contours()
            generated_strokes = set()
            for i, contours in enumerate(contours_co):
                label = contours_label[i]
                if label < 1:
                    continue
                for c in contours:
                    new_stroke = fill_frame.nijigp_strokes.new()
                    new_stroke.points.add(len(c))
                    new_stroke.use_cyclic = True
                    new_stroke.material_index = labels_info[label][1]
                    if (self.material_mode == 'SELECT' or
                        (self.material_mode == 'HINT' and labels_info[label][2]) ):
                        color = labels_info[label][0]
                        new_stroke.vertex_color_fill = (color[0], color[1], color[2], 1)
                    for i,co in enumerate(c):
                        new_stroke.points[i].co = restore_3d_co(co, depth_lookup_tree.get_depth(co), inv_mat, scale_factor)
                    new_stroke.select = True
                    generated_strokes.add(new_stroke)

            if self.clear_hint_layer:
                for stroke in list(hint_frame.nijigp_strokes):
                    if not self.use_boundary_strokes or stroke.is_nofill_stroke:
                        if stroke not in generated_strokes:
                            hint_frame.nijigp_strokes.remove(stroke)

        # Get the frames from each layer to process
        processed_frame_numbers = []
        if not get_multiedit(gp_obj):
            if fill_layer.active_frame:
                fill_single_frame([line_layer.active_frame for line_layer in line_layers],
                                hint_layer.active_frame,
                                fill_layer.active_frame)
                processed_frame_numbers.append(fill_layer.active_frame.frame_number)
            else:
                fill_frame = fill_layer.frames.new(bpy.context.scene.frame_current)
                fill_single_frame([line_layer.active_frame for line_layer in line_layers],
                                hint_layer.active_frame,
                                fill_frame)
                processed_frame_numbers.append(bpy.context.scene.frame_current)
        else:
            # If any line layer has a keyframe at a certain frame number, that frame should be processed
            for line_layer in list(line_layers) + [hint_layer]:
                for line_frame in line_layer.frames:
                    if line_frame.select:
                        processed_frame_numbers.append(line_frame.frame_number)
            processed_frame_numbers = list(set(processed_frame_numbers))

            for frame_number in processed_frame_numbers:
                line_frames = []
                for line_layer in line_layers:
                    line_frame = get_layer_latest_frame(line_layer, frame_number)
                    if line_frame:
                        line_frames.append(line_frame)

                    hint_frame = get_layer_latest_frame(hint_layer, frame_number)
                    if not hint_frame:
                        continue

                    fill_frame = get_layer_latest_frame(fill_layer, frame_number)
                    if not fill_frame or fill_frame.frame_number != frame_number:
                        fill_frame = fill_layer.frames.new(frame_number)
                    fill_single_frame(line_frames, hint_frame, fill_frame)

        refresh_strokes(gp_obj, processed_frame_numbers)
        bpy.ops.gpencil.nijigp_hole_processing(rearrange=True, apply_holdout=False)
        bpy.ops.object.mode_set(mode=current_mode)
        return {'FINISHED'}

class HatchFillOperator(bpy.types.Operator, ColorTintConfig, NoiseConfig):
    """Generate hatch strokes inside selected polygons"""
    bl_idname = "gpencil.nijigp_hatch_fill"
    bl_label = "Hatch Fill"
    bl_category = 'View'
    bl_options = {'REGISTER', 'UNDO'}
    
    gap: bpy.props.FloatProperty(
            name='Line Gap',
            default=0.1, soft_min=0.05, soft_max=0.5, min=0.01, unit='LENGTH',
            description='Length between two adjacent generated points'
    )
    angle: bpy.props.FloatProperty(
            name='Angle',
            default=0.25*math.pi, min=-2*math.pi, max=2*math.pi,
            unit='ROTATION',
            description='Rotation angle of the hatch pattern'
    )
    line_width: bpy.props.IntProperty(
            name='Line Width',
            description='The line width of the newly generated stroke',
            default=10, min=1, soft_max=200, subtype='PIXEL'
    ) 
    strength: bpy.props.FloatProperty(
            name='Strength',
            default=1, min=0, max=1
    ) 
    style_line: bpy.props.EnumProperty(
            name='Line Style',
            items=[ ('PARA', 'Parallel Lines', ''),
                    ('DOODLE', 'Doodles', ''),
                    ('CONVEX', 'Single-Line Doodle', ''),
                    ('RIDGE', 'Centerline', '')],
            default='DOODLE',
            description='The way of connecting points to one or multiple lines'
    )
    style_doodle: bpy.props.EnumProperty(
            name='Doodle',
            items=[ ('Z', 'Z-Shape', ''),
                    ('S', 'S-Shape', '')],
            default='Z',
            description='The way of connecting parallel lines to a doodle'
    )
    style_tile: bpy.props.EnumProperty(
            name='Tile',
            items=[ ('RECT', 'Squares', ''),
                    ('HEX', 'Hexagons', '')],
            default='RECT',
            description='The way of points in the grid are aligned'
    )
    keep_original: bpy.props.BoolProperty(
            name='Keep Original',
            default=True,
            description='Do not delete the original stroke'
    )
    ignore_mode: bpy.props.EnumProperty(
            name='Ignore',
            items=[('NONE', 'None', ''),
                    ('LINE', 'All Lines', ''),
                    ('OPEN', 'All Open Lines', '')],
            default='NONE',
            description='Skip strokes without fill'
    )
    output_material: bpy.props.StringProperty(
        name='Output Material',
        description='Draw the new strokes using this material. If empty, use the active material',
        default='',
        search=lambda self, context, edit_text: [material.name for material in context.object.data.materials if material]
    )
    vertex_color_mode: bpy.props.EnumProperty(
            name='Vertex Color',
            items=[ ('NONE', 'No Vertex Color', ''),
                    ('LINE', 'Use Line Color', ''),
                    ('FILL', 'Use Fill Color', '')],
            default='NONE',
            description='Whether to use the vertex color from the input strokes'
    )
    random_angle: bpy.props.FloatProperty(
            name='Angle',
            default=0, min=0, max=2*math.pi, unit='ROTATION',
            description='Additional random rotation angle'
    )
    random_uv: bpy.props.FloatProperty(
            name='UV', default=0, min=0, max=1,
    )
    random_pos: bpy.props.FloatProperty(
            name='Position', default=0, min=0, max=1,
    )
    random_radius: bpy.props.FloatProperty(
            name='Radius', default=0, min=0, max=1,
    )

    def draw(self, context):
        layout = self.layout
        layout.label(text = "Input Options:")
        box1 = layout.box()
        box1.prop(self, "ignore_mode")
        layout.label(text = "Geometry Options:")
        box2 = layout.box()
        box2.prop(self, "angle")
        box2.prop(self, "style_line")
        box2.prop(self, "gap")
        row = box2.row()
        row.prop(self, "style_doodle")
        if self.style_line in ['RIDGE', 'PARA']:
            row.enabled = False
        row = box2.row()
        row.prop(self, "style_tile")
        if self.style_line == 'RIDGE':
            row.enabled = False
        row = box2.row()
        row.prop(self, "line_width")
        row.prop(self, "strength")
        box2.label(text="Noise:")
        row = box2.row()
        row.prop(self, "random_scale")
        row.prop(self, "random_seed")
        row = box2.row()
        row.prop(self, "random_angle")
        row.prop(self, "random_pos")
        row = box2.row()
        row.prop(self, "random_uv")
        row.prop(self, "random_radius")
        layout.label(text = "Output Options:")
        box3 = layout.box()
        box3.prop(self, "output_material", icon='MATERIAL')
        box3.prop(self, "vertex_color_mode")
        if self.vertex_color_mode != 'NONE':
            box4 = box3.box()
            box4.prop(self, "tint_color")
            box4.prop(self, "tint_color_factor")
            box4.prop(self, "blend_mode") 
            box4.prop(self, "random_factor", text="Noise")   
        box3.prop(self, "keep_original")    

    def execute(self, context):
        import random
        try:
            import pyclipper
            if self.style_line == 'RIDGE':
                from scipy.spatial import Voronoi
                from ..solvers.graph import MstSolver
        except ImportError:
            self.report({"ERROR"}, "Please install dependencies in the Preferences panel.")
            return {'FINISHED'}
        
        gp_obj = context.object
        material = gp_obj.active_material
        if self.output_material in bpy.data.materials:
            material = bpy.data.materials[self.output_material]
        material_idx = gp_obj.material_slots.find(material.name)
        frames_to_process = get_input_frames(gp_obj, get_multiedit(gp_obj))
        generated_strokes = []
        noise.seed_set(self.random_seed)

        def across_boundary(p1, p2, poly, scale_factor=1):
            for i,point in enumerate(poly):
                if geometry.intersect_line_line_2d(p1,p2,
                                                   (poly[i][0]/scale_factor, poly[i][1]/scale_factor),
                                                   (poly[i-1][0]/scale_factor, poly[i-1][1]/scale_factor) ):
                    return True
            return False

        for frame in frames_to_process:
            stroke_list = get_input_strokes(gp_obj, frame)
            stroke_list = [s for s in stroke_list if
                            ((self.ignore_mode == 'LINE' and not is_stroke_line(s, gp_obj)) or
                                (self.ignore_mode == 'OPEN' and not (is_stroke_line(s, gp_obj) and not s.use_cyclic)) or
                                (self.ignore_mode == 'NONE')
                            )]
            t_mat, inv_mat = get_transformation_mat(mode=context.scene.nijigp_working_plane,
                                                    gp_obj=gp_obj, strokes=stroke_list, operator=self)
            # Process each stroke independently
            for stroke in stroke_list:
                delta_angle = (noise.random() - 0.5) * self.random_angle
                rot_mat = Euler((0,0, self.angle + delta_angle)).to_matrix()
                poly_list, depth_list, scale_factor = get_2d_co_from_strokes([stroke], rot_mat @ t_mat, scale=True)
                depth_lookup_tree = DepthLookupTree(poly_list, depth_list)
                hatch_polys = []

                # Use Voronoi diagram to generate centerlines
                if self.style_line == 'RIDGE':
                    vor = Voronoi(poly_list[0])
                    solver = MstSolver()
                    solver.mst_from_voronoi(vor, poly_list[0])
                    _, path = solver.get_longest_path()
                    path = [Vector(vor.vertices[i])/scale_factor for i in path]
                    hatch_polys.append(path)
                # For all other styles, generate hatch patterns from a grid
                else:
                    # Create a grid according to the bounding box
                    corners = get_2d_bound_box([stroke], rot_mat @ t_mat)
                    corners = pad_2d_box(corners, 0.05, return_bounds=True)
                    grid_points = []
                    grid_points_inside = []
                    v0 = 1./np.sqrt(3)*self.gap if self.style_tile=='HEX' else 0
                    v_scale = 2./np.sqrt(3) if self.style_tile=='HEX' else 1
                    odd_row = 0
                    for u in np.arange(corners[0], corners[2], self.gap):
                        grid_points.append([])
                        grid_points_inside.append([])
                        odd_row = 1 - odd_row
                        for v in np.arange(corners[1] - v0 * odd_row, corners[3] + v0 * (1-odd_row), self.gap * v_scale):
                            co_2d = Vector((u,v))
                            grid_points[-1].append(co_2d)
                            grid_points_inside[-1].append(pyclipper.PointInPolygon(co_2d * scale_factor, poly_list[0])==1)
                    if len(grid_points) < 1:
                        continue
                    grid_U = len(grid_points)
                    grid_V = len(grid_points[0])

                    # Connecting grid points to a line pattern
                    to_reverse = False
                    if self.style_line == 'CONVEX':
                        hatch_polys.append([])
                        for u in range(grid_U):
                            seq = reversed(range(grid_V)) if to_reverse else range(grid_V)
                            to_reverse = not to_reverse if self.style_doodle == 'S' else False
                            for v in seq:
                                if grid_points_inside[u][v]:
                                    hatch_polys[-1].append(grid_points[u][v])
                    else:             
                        # For each row of the grid, get all continuous intervals
                        row_segments = []
                        for u in range(grid_U):
                            row_segments.append({})
                            start = None
                            for v in range(grid_V):
                                if grid_points_inside[u][v] and start == None:
                                    start = v
                                elif (not grid_points_inside[u][v]) and start != None:
                                    row_segments[-1][start] = (start, v)
                                    start = None
                            if start != None:
                                row_segments[-1][start] = (start, grid_V)
                        # Create a new stroke for each previously unconnected segment
                        for u in range(grid_U):
                            for start,end in row_segments[u].values():
                                hatch_polys.append([])
                                current_seg = [u, start, end]
                                while True:
                                    # Doodle style: Greedy connects to a point in the next row
                                    seq = range(current_seg[2]-1, current_seg[1]-1, -1) if to_reverse else range(current_seg[1], current_seg[2])
                                    for v in seq:
                                        hatch_polys[-1].append(grid_points[current_seg[0]][v])
                                    to_reverse = not to_reverse if self.style_doodle == 'S' else False
                                    # Parallel line style: Never connect segments
                                    if self.style_line == 'PARA':
                                        break
                                    if current_seg[0] >= grid_U - 1:
                                        break
                                    for next_start,next_end in row_segments[current_seg[0]+1].values():
                                        pair = (grid_points[current_seg[0]][current_seg[2]-1], grid_points[current_seg[0]+1][next_start]) if self.style_doodle == 'Z' else \
                                            (grid_points[current_seg[0]][current_seg[2]-1], grid_points[current_seg[0]+1][next_end-1]) if to_reverse else \
                                            (grid_points[current_seg[0]][current_seg[1]], grid_points[current_seg[0]+1][next_start])
                                        if not across_boundary(pair[0], pair[1], poly_list[0], scale_factor):
                                            current_seg = [current_seg[0]+1, next_start, next_end]
                                            row_segments[current_seg[0]].pop(next_start)
                                            break
                                    else:
                                        break
                # Generate output strokes
                for co_list in hatch_polys:
                    new_stroke = frame.nijigp_strokes.new()
                    new_stroke.material_index = material_idx
                    new_stroke.points.add(len(co_list))
                    new_stroke.line_width = self.line_width
                    for i,point in enumerate(new_stroke.points):
                        depth = depth_lookup_tree.get_depth(co_list[i]*scale_factor)
                        _, orig_idx, dist = depth_lookup_tree.get_info(co_list[i]*scale_factor)
                        # Set point attributes
                        if self.vertex_color_mode == 'FILL':
                            point.vertex_color = get_mixed_color(gp_obj, stroke, to_linear=True)
                        elif self.vertex_color_mode == 'LINE':
                            point.vertex_color = get_mixed_color(gp_obj, stroke, orig_idx, to_linear=True)
                        co_noise = noise.noise_vector(co_list[i].to_3d() * self.random_scale)
                        point.strength = self.strength
                        point.co = restore_3d_co(co_list[i] + self.gap * self.random_pos * co_noise.xy , depth, inv_mat @ rot_mat.transposed(), 1)  
                        point.uv_rotation = math.pi * self.random_uv * co_noise.z
                        set_point_radius(point, 1 + self.random_radius * noise.noise(co_noise), self.line_width)
                    generated_strokes.append(new_stroke)   
                if not self.keep_original:
                    frame.nijigp_strokes.remove(stroke)
        # Post-processing
        op_deselect()
        for stroke in generated_strokes:
            stroke.select = True
        if self.vertex_color_mode != 'NONE':
            bpy.ops.gpencil.nijigp_color_tint(tint_color=self.tint_color,
                                            tint_color_factor=self.tint_color_factor,
                                            tint_mode='LINE', blend_mode=self.blend_mode,
                                            random_seed=self.random_seed, random_scale=self.random_scale, random_factor=self.random_factor)
        refresh_strokes(gp_obj, [f.frame_number for f in frames_to_process])                      
        
        return {'FINISHED'}