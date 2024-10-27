import bpy
import numpy as np
from .common import *
from ..utils import *
from ..api_router import *

def generate_shading_stroke(co_list, inv_mat, scale_factor, gp_obj, ref_stroke_info, ref_kdtree: DepthLookupTree):
    """
    Generate a new stroke representing a shade/light area arranged above a given stroke.
    Some generated points belong to the existing contour, and the others form a new terminator line, which should be marked.
    Format of ref_stroke_info: [stroke, layer_index, stroke_index, frame]
    """
    ref_stroke, layer_index, stroke_index = ref_stroke_info[0], ref_stroke_info[1], ref_stroke_info[2]
    frame = gp_obj.data.layers[layer_index].active_frame
    if len(ref_stroke_info) > 3:
        frame = ref_stroke_info[3]
    # Create a stroke
    new_stroke = frame.nijigp_strokes.new()
    copy_stroke_attributes(new_stroke, [ref_stroke],
                           copy_hardness=True, copy_linewidth=True,
                           copy_cap=True, copy_cyclic=False,
                           copy_uv=True, copy_material=True, copy_color=True)
    # Create stroke points
    terminator_points = []
    N = len(co_list)
    new_stroke.points.add(N)
    for i in range(N):
        co_2d = co_list[i]
        _, ref_i, dist = ref_kdtree.get_info(co_2d)
        new_stroke.points[i].co = restore_3d_co(co_2d, ref_kdtree.get_depth(co_2d), inv_mat, scale_factor)
        if dist > 2:
            terminator_points.append((new_stroke, i))
        new_stroke.points[i].pressure = ref_stroke.points[ref_i].pressure
        new_stroke.points[i].strength = ref_stroke.points[ref_i].strength
        new_stroke.points[i].uv_factor = ref_stroke.points[ref_i].uv_factor
        new_stroke.points[i].uv_fill = ref_stroke.points[ref_i].uv_fill
        new_stroke.points[i].uv_rotation = ref_stroke.points[ref_i].uv_rotation
        new_stroke.points[i].vertex_color = ref_stroke.points[ref_i].vertex_color
    # Rearrange the new stroke
    current_index = len(frame.nijigp_strokes) - 1
    new_index = stroke_index + 1
    op_deselect()
    new_stroke.select = True
    for i in range(current_index - new_index):
        op_arrange_stroke(direction='DOWN')
        
    return new_stroke, new_index, layer_index, terminator_points
        
class ShadeSelectedOperator(bpy.types.Operator):
    """This operator uses a light in the 3D scenario to calculate shadows and rim lights and generate corresponding strokes"""
    bl_idname = "gpencil.nijigp_shade_selected"
    bl_label = "Shade Selected"
    bl_category = 'View'
    bl_options = {'REGISTER', 'UNDO'}

    is_initialized: bpy.props.BoolProperty(default=False)
    ignore_mode: bpy.props.EnumProperty(
            name='Ignore',
            items=[('NONE', 'None', ''),
                    ('LINE', 'All Lines', ''),
                    ('OPEN', 'All Open Lines', '')],
            default='NONE',
            description='Skip strokes without fill'
    )
    fill_always_closed: bpy.props.BoolProperty(
            name='Treat Fills as Closed Paths',
            default=True,
            description='If an input stroke has only fill material, always treat it as a closed path'
    )
    smooth_repeat: bpy.props.IntProperty(
            name='Smooth Level',
            default=5, min=0, soft_max=10,
            description='Perform smoothing to the generated terminator lines'
    )
    
    # Light source configurations
    light_type: bpy.props.EnumProperty(
            name='Light Type',
            items=[('VEC', 'Constant Vector', 'Type a vector as the light direction'),
                    ('OBJ', 'Reference Object', 'Select any object and regard it as the light source'),
                    ('LIGHT', 'Light Object', 'Select a light in the scene to use its attributes. Support Point and Sun lights only')],
            default='LIGHT'
    )
    light_vector: bpy.props.FloatVectorProperty(
            name='Vector',
            default=(1.0, -1.0, 0.5), size=3, soft_min=-5, soft_max=5,
            description='A vector representing the light direction'
    )
    ref_obj: bpy.props.StringProperty(
        name='Reference Object',
        description='The location of this object will be regarded as the light source',
        default='',
        search=lambda self, context, edit_text: [object.name for object in context.scene.objects]
    )
    ref_light: bpy.props.StringProperty(
        name='Light Object',
        description='',
        default='',
        search=lambda self, context, edit_text: [object.name for object in context.scene.objects if object.type=='LIGHT']
    )
    light_energy: bpy.props.FloatProperty(
        name='Power',
        default=10, min=0.1, max=1000,
        description='Strength of the input light'
    )
    
    # Rim light output configurations
    rim_enabled: bpy.props.BoolProperty(name='Generate Rim Light', default=False)    
    rim_tint_config: bpy.props.PointerProperty(type=ColorTintPropertyGroup)
    rim_width: bpy.props.FloatProperty(
            name='Width', description='Width of generated rim light',
            default=0.03, min=0, unit='LENGTH',
    )
    rim_material: bpy.props.StringProperty(
        name='Material',
        description='Set a material to all generated strokes. Use the material of the input stroke if empty',
        default='', search=lambda self, context, edit_text: [material.name for material in context.object.data.materials if material]
    )
    
    # shadow output configurations
    shadow_enabled: bpy.props.BoolProperty(name='Generate Shadow', default=False)  
    shadow_double_level: bpy.props.BoolProperty(name='Double', default=False) 
    shadow_tint_config: bpy.props.PointerProperty(type=ColorTintPropertyGroup)
    shadow_resolution: bpy.props.IntProperty(
            name='Resolution',
            default=20, min=5, soft_max=50,
            description='Determine how precisely the shading will be calculated'
    )
    shadow_threshold: bpy.props.FloatProperty(
        name='Threshold',
        default=0, soft_min=-5, soft_max=5,
        description='Determine the terminator line between shadow and lighter parts based on the light intensity hitting the surface'
    )
    shadow_second_threshold: bpy.props.FloatProperty(
        name='',
        default=0.5, min=0, soft_max=5,
        description='Determine an additional terminator line to generate two levels of shadow'
    )
    shadow_scale: bpy.props.FloatProperty(
            name='Vertical Scale',
            default=1, soft_max=5, soft_min=-5,
            description='Scale the vertical component of generated normal vectors. Negative values result in concave shapes'
    )
    shadow_min_area: bpy.props.FloatProperty(
            name='Min Area',
            default=1, min=0, max=100,
            subtype='PERCENTAGE',
            description='Ignore shadows with too small an area'
    )
    shadow_material: bpy.props.StringProperty(
        name='Material',
        description='Set a material to all generated strokes. Use the material of the input stroke if empty',
        default='', search=lambda self, context, edit_text: [material.name for material in context.object.data.materials if material]
    )

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self, width=500)
        
    def draw(self, context):
        if not self.is_initialized:
            self.rim_tint_config.tint_color = (1, 1, 1, 1)
            self.rim_tint_config.tint_color_factor = 0.2
            self.shadow_tint_config.tint_color_factor = 0.2
            self.is_initialized = True

        layout = self.layout
        layout.prop(self, "ignore_mode")
        layout.prop(self, "fill_always_closed")
        
        layout.label(text = "Input Light:")
        box1 = layout.box()
        box1.prop(self, 'light_type')
        if self.light_type == 'VEC':
            row = box1.row(align=True)
            row.prop(self, 'light_vector')
            box1.prop(self, 'light_energy')
        elif self.light_type == 'OBJ':
            box1.prop(self, 'ref_obj', icon='OBJECT_DATA')
            box1.prop(self, 'light_energy')
        elif self.light_type == 'LIGHT':
            box1.prop(self, 'ref_light', icon='LIGHT_DATA')
            
        layout.prop(self, "rim_enabled")
        if self.rim_enabled:
            box2 = layout.box()
            box2.prop(self, 'rim_width')
            box2.prop(self, 'rim_material', icon='MATERIAL')
            row = box2.row(align=True)
            row.prop(self.rim_tint_config, "tint_color")
            row.prop(self.rim_tint_config, "tint_color_factor", slider=True)
            row.prop(self.rim_tint_config, "blend_mode", text='')     
            
        row = layout.row()
        row.prop(self, "shadow_enabled")
        if self.shadow_enabled:
            row.prop(self, "shadow_double_level")
        if self.shadow_enabled:
            box3 = layout.box()
            row = box3.row(align=True)
            row.prop(self, 'shadow_threshold')
            if self.shadow_double_level:
                row.prop(self, 'shadow_second_threshold')
            box3.prop(self, 'shadow_scale')
            row = box3.row(align=True)
            row.prop(self, 'shadow_resolution')
            row.prop(self, 'shadow_min_area')
            box3.prop(self, 'shadow_material', icon='MATERIAL')
            row = box3.row(align=True)
            row.prop(self.shadow_tint_config, "tint_color")
            row.prop(self.shadow_tint_config, "tint_color_factor", slider=True)
            row.prop(self.shadow_tint_config, "blend_mode", text='')   
                    
        layout.label(text = "Post-Processing Options:")  
        box4 = layout.box()
        box4.prop(self, 'smooth_repeat')
        
    def execute(self, context):   
        MIN_AREA = 4    # Eliminate too small shapes which may be errors of Boolean operations
        current_gp_obj = context.object
        # Import and configure Clipper
        try:
            import pyclipper
            from skimage import morphology, measure
        except ImportError:
            self.report({"ERROR"}, "Please install PyClipper and Scikit-Image in the Preferences panel.")
            return {'FINISHED'}
        clipper = pyclipper.Pyclipper()
        clipper.PreserveCollinear = True
        
        # Handle input parameters
        rim_material_idx = current_gp_obj.material_slots.find(self.rim_material)
        shadow_material_idx = current_gp_obj.material_slots.find(self.shadow_material)
        # Get the light source
        light_obj_name = self.ref_obj if self.light_type == 'OBJ' else \
                         self.ref_light if self.light_type == 'LIGHT' else None
        light_obj: bpy.types.Object = None
        if light_obj_name != None:
            if light_obj_name not in context.scene.objects:
                return {'FINISHED'}
            else:
                light_obj = bpy.data.objects[light_obj_name]
        if self.light_type == 'LIGHT' and light_obj.type != 'LIGHT':
            return {'FINISHED'}
        is_point_light = (self.light_type == 'OBJ') or (self.light_type == 'LIGHT' and light_obj.data.type == 'POINT')

        # Get a list of layers / frames to process
        frames_to_process = get_input_frames(current_gp_obj,
                                             multiframe = get_multiedit(current_gp_obj),
                                             return_map = True)

        select_map = save_stroke_selection(current_gp_obj)
        generated_shadow_strokes_multilevel = [[], []] if self.shadow_double_level else [[]]
        generated_rim_strokes = []
        
        current_frame = context.scene.frame_current
        for frame_number, layer_frame_map in frames_to_process.items():
            context.scene.frame_set(frame_number)
            load_stroke_selection(current_gp_obj, select_map)
            stroke_info = []
            stroke_list = []
            rim_terminator_points = []
            shadow_terminator_points = []
        
            # Convert selected strokes to 2D polygon point lists
            for i,item in layer_frame_map.items():
                frame = item[0]
                layer = current_gp_obj.data.layers[i]
                if is_frame_valid(frame):
                    for j,stroke in enumerate(frame.nijigp_strokes):
                        if ((self.ignore_mode == 'LINE' and is_stroke_line(stroke, current_gp_obj)) or
                            (self.ignore_mode == 'OPEN' and is_stroke_line(stroke, current_gp_obj) and not stroke.use_cyclic)):
                            continue
                        if stroke.select and not is_stroke_protected(stroke, current_gp_obj):
                            stroke_info.append([stroke, i, j, frame])
                            stroke_list.append(stroke)
                            
            t_mat, inv_mat = get_transformation_mat(mode=context.scene.nijigp_working_plane,
                                                    gp_obj=current_gp_obj, 
                                                    strokes=stroke_list, operator=self)
            poly_list, depth_list, scale_factor = get_2d_co_from_strokes(stroke_list, t_mat,
                                                                         scale=True,
                                                                         correct_orientation = True)
            
            # Process each stroke
            for i,co_list in enumerate(poly_list):
                src_stroke = stroke_list[i]
                if len(co_list) < 3:
                    continue
                # Get the light vector: direction and strength
                # Currently, the calculation is largely simplified with several assumptions:
                #  - Accept only one light source which has either point or sun (parallel) type.
                #  - Approximately regard the light vector to each point of the same stroke to be identical.
                light_vec_world = Vector()
                if is_point_light:
                    stroke_center_world = current_gp_obj.matrix_world @ get_stroke_center(src_stroke)
                    light_vec_world = stroke_center_world - light_obj.matrix_world.translation 
                else:
                    if self.light_type == 'VEC':
                        light_vec_world = - Vector(self.light_vector)
                    else:
                        light_vec_world = Vector((0,0,-1))
                        light_vec_world.rotate(light_obj.matrix_world.to_euler())
                light_vec_world.normalize()
                if self.light_type != 'LIGHT':
                    light_energy = self.light_energy
                else:
                    light_energy = light_obj.data.energy
                if is_point_light:
                    light_energy /= (light_vec_world.length ** 2)
                
                # Rotate the transform matrix to the light direction
                light_angle = Vector((0,1)).angle_signed((t_mat @ (-light_vec_world)).xy)
                new_t_mat = Matrix.Rotation(light_angle, 3, 'Z') @ t_mat
                new_co_list = [Matrix.Rotation(light_angle, 2) @ Vector(co) for co in co_list]
                light_vec_local = new_t_mat @ light_vec_world
                ref_kdtree = DepthLookupTree([new_co_list], [depth_list[i]])

                # Rim light: use Boolean operations to achieve
                if self.rim_enabled:
                    clipper.Clear()
                    rim_trans_vec = xy0(light_vec_local).normalized() * scale_factor * self.rim_width
                    rim_trans_poly = [[co[0]+rim_trans_vec[0], co[1]+rim_trans_vec[1]] for co in new_co_list]
                    try:
                        # Steps of processing a closed path: translate -> substract
                        if src_stroke.use_cyclic or (is_stroke_fill(src_stroke, current_gp_obj) and self.fill_always_closed):
                            op = pyclipper.CT_DIFFERENCE
                            clipper.AddPath(new_co_list, pyclipper.PT_SUBJECT, True)
                            clipper.AddPath(rim_trans_poly, pyclipper.PT_CLIP, True)
                        # Steps of processing an open path: translate -> invert -> join -> intersect
                        else:
                            op = pyclipper.CT_INTERSECTION
                            clipper.AddPath(new_co_list, pyclipper.PT_SUBJECT, True)
                            clipper.AddPath(new_co_list + rim_trans_poly[::-1], pyclipper.PT_CLIP, True)
                        rim_results = clipper.Execute(op, pyclipper.PFT_NONZERO, pyclipper.PFT_NONZERO)
                    except:
                        rim_results = []

                    # Turn rim paths into strokes
                    for result in rim_results:
                        if pyclipper.Area(result) < MIN_AREA:
                            continue
                        new_stroke, new_index, new_layer_index, new_terminator_points = \
                                generate_shading_stroke(result, new_t_mat.transposed(), scale_factor, current_gp_obj,
                                                        stroke_info[i], ref_kdtree)
                        generated_rim_strokes.append(new_stroke)
                        rim_terminator_points += new_terminator_points
                        if rim_material_idx >= 0:
                            new_stroke.material_index = rim_material_idx
                        # Update the stroke index
                        for info in stroke_info:
                            if new_index <= info[2] and new_layer_index == info[1]:
                                info[2] += 1 

                # Shadow: use normal interpolation method to achieve
                if self.shadow_enabled:
                    shadow_thresholds = [self.shadow_threshold]
                    if self.shadow_double_level:
                        shadow_thresholds.append(self.shadow_threshold + self.shadow_second_threshold)
                    # Support either 1 or 2 levels of shadow
                    for shadow_level, current_shadow_threshold in enumerate(shadow_thresholds):
                        # Get contour point normals
                        contour_co_array = []
                        contour_normal_array = []
                        contour_normal_map = {}
                        for j,co in enumerate(new_co_list):
                            _co = new_co_list[j-1]
                            norm = Vector([co[1]-_co[1], -co[0]+_co[0], 0]).normalized()
                            contour_co_array.append(co)
                            contour_normal_array.append(norm)
                            contour_normal_map[(int(co[0]),int(co[1]))] = norm
                        contour_co_array = np.array(contour_co_array)
                        contour_normal_array = np.array(contour_normal_array)
                        
                        # Generate a grid inside the shape to sample lighting
                        corners = get_2d_bound_box([src_stroke], new_t_mat)
                        corners = [co * scale_factor for co in pad_2d_box(corners, 0.05, return_bounds=True)]
                        U = V = int(self.shadow_resolution)
                        def get_grid_co(u: float, v: float):
                            """Should allow float/unbounded inputs"""
                            return (corners[0] + (corners[2] - corners[0]) * (u - 1) / (U - 1),
                                    corners[1] + (corners[3] - corners[1]) * (v - 1) / (V - 1))
                            
                        # Check which grid points are inside the shape
                        grid_is_inside = np.zeros((V + 2, U + 2), dtype='int') # Pad the grid once more
                        for v in range(1, V+1):
                            for u in range(1, U+1):
                                grid_is_inside[v][u] = (pyclipper.PointInPolygon(get_grid_co(u, v), new_co_list) == 1)
                        # Dilate the result, so the grid points now enclose the original shape
                        grid_is_inside = morphology.binary_dilation(grid_is_inside)    
                        
                        # Calculate the light information for each grid point  
                        grid_is_shadow = np.zeros((V + 2, U + 2), dtype='int')
                        for v in range(V+2):
                            for u in range(U+2):
                                if not grid_is_inside[v][u]:
                                    continue
                                # Get the normal by interpolation
                                co_2d = get_grid_co(u, v)
                                if (int(co_2d[0]), int(co_2d[1])) in contour_normal_map:
                                    norm = contour_normal_map[(int(co_2d[0]), int(co_2d[1]))].to_3d()
                                    norm.z = np.sqrt(1 - norm.x ** 2 - norm.y ** 2)
                                else:
                                    dist_sq = (contour_co_array[:,0]-co_2d[0])**2 + (contour_co_array[:,1]-co_2d[1])**2
                                    weights = 1.0 / dist_sq
                                    weights /= np.sum(weights)
                                    norm_u = np.dot(contour_normal_array[:,0], weights)
                                    norm_v = np.dot(contour_normal_array[:,1], weights)
                                    norm = Vector((norm_u, norm_v, np.sqrt(1 - norm_u ** 2 - norm_v ** 2)))
                                norm = Vector((norm.x * self.shadow_scale, norm.y * self.shadow_scale, norm.z)).normalized()
                                grid_is_shadow[v][u] = (norm.dot(- light_vec_local) * light_energy < current_shadow_threshold)
                        
                        # Get the contour of shadow areas to generate new strokes
                        shadow_paths = []
                        res = measure.find_contours( grid_is_shadow, 0.5, positive_orientation='high')
                        for path in res:
                            if pyclipper.Area(path) > max(1, self.shadow_min_area * 0.01 * (self.shadow_resolution ** 2)):
                                shadow_paths.append([get_grid_co(co[1], co[0]) for co in path])
                        # Perform intersect with the original shape to get the final results
                        op = pyclipper.CT_INTERSECTION
                        for poly in shadow_paths:
                            clipper.Clear()
                            clipper.AddPath(new_co_list, pyclipper.PT_SUBJECT, True)
                            clipper.AddPath(poly, pyclipper.PT_CLIP, True)
                            shadow_results = clipper.Execute(op, pyclipper.PFT_NONZERO, pyclipper.PFT_NONZERO)
                            
                            for result in shadow_results:
                                if pyclipper.Area(result) < MIN_AREA:
                                    continue
                                new_stroke, new_index, new_layer_index, new_terminator_points = \
                                    generate_shading_stroke(result, new_t_mat.transposed(), scale_factor, current_gp_obj,
                                                                    stroke_info[i], ref_kdtree)
                                generated_shadow_strokes_multilevel[shadow_level].append(new_stroke)
                                shadow_terminator_points += new_terminator_points
                                if shadow_material_idx >= 0:
                                    new_stroke.material_index = shadow_material_idx
                                # Update the stroke index
                                for info in stroke_info:
                                    if new_index <= info[2] and new_layer_index == info[1]:
                                        info[2] += 1            
                                                                        
            # Single-frame post-processing
            context.scene.tool_settings.gpencil_selectmode_edit = 'POINT'
            op_deselect()
            for stroke,index in rim_terminator_points + shadow_terminator_points:
                stroke.points[index].select = True
            op_stroke_smooth(repeat=self.smooth_repeat)
            context.scene.tool_settings.gpencil_selectmode_edit = 'STROKE'
                                
        # Overall post-processing
        bpy.context.scene.frame_set(current_frame)
        # Rim strokes
        op_deselect()
        for stroke in generated_rim_strokes:
            stroke.select = True
            stroke.use_cyclic = True
        bpy.ops.gpencil.nijigp_color_tint(tint_color=self.rim_tint_config.tint_color,
                                        tint_color_factor=self.rim_tint_config.tint_color_factor,
                                        tint_mode='FILL', blend_mode=self.rim_tint_config.blend_mode)
        # Shadow strokes
        num_levels = len(generated_shadow_strokes_multilevel)
        for shadow_level, generated_shadow_strokes in enumerate(generated_shadow_strokes_multilevel):
            op_deselect()
            for stroke in generated_shadow_strokes:
                stroke.select = True
                stroke.use_cyclic = True
            bpy.ops.gpencil.nijigp_color_tint(tint_color=self.shadow_tint_config.tint_color,
                                            tint_color_factor=self.shadow_tint_config.tint_color_factor * (num_levels - shadow_level) / num_levels,
                                            tint_mode='FILL', blend_mode=self.shadow_tint_config.blend_mode)
        # All strokes
        op_deselect()
        for stroke in generated_rim_strokes + sum(generated_shadow_strokes_multilevel, []):
            stroke.select = True        
        refresh_strokes(current_gp_obj, list(frames_to_process.keys()))
                
        return {'FINISHED'}