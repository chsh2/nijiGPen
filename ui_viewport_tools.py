import bpy, blf, gpu
import math
from bpy_extras import view3d_utils
from gpu_extras.batch import batch_for_shader
from mathutils import *
from .utils import *
from .resources import *
from .api_router import *
from .operators.common import ColorTintConfig, refresh_strokes, smooth_stroke_attributes
from .operators.operator_fill import lineart_triangulation

def draw_button(x, y, width, height, color, text):
    """Draw a rectangle button in the screen using the GPU module"""
    # Draw rectangle background
    positions = (
        (x,  y), (x,  y+height),
        (x+width, y+height), (x+width, y))
    indices = ((0, 1, 2), (2, 3, 0))
    if bpy.app.version >= (3, 4, 0):
        shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    else:
        shader = gpu.shader.from_builtin('2D_UNIFORM_COLOR')
        shader.bind()
    batch = batch_for_shader(shader, 'TRIS', {"pos": positions}, indices=indices)
    shader.uniform_float("color", color)
    batch.draw(shader)
    
    # Draw text
    font_id = 0
    blf.color(font_id, 1, 1, 1, 1)
    blf_set_size(blf, font_id, height * 0.8)
    blf.position(font_id, x + height * 0.1, y + height * 0.1, 0)
    blf.draw(font_id, text)

def is_button_clicked(event, button_position, button_size):
    if event.type != 'LEFTMOUSE':
        return False
    if event.mouse_region_x < button_position[0] or event.mouse_region_x > button_position[0] + button_size[0]:
        return False
    if event.mouse_region_y < button_position[1] or event.mouse_region_y > button_position[1] + button_size[1]:
        return False
    return True

class BooleanModalOperator(bpy.types.Operator):
    """Use mouse or pen to draw shapes for Boolean operations"""
    bl_idname = "gpencil.nijigp_boolean_modal"
    bl_label = "Boolean (Modal)"
    bl_options = {'REGISTER', 'UNDO'}

    use_pressure: bpy.props.BoolProperty(
        name='Use Pressure',
        default=True,
    )
    caps_type: bpy.props.EnumProperty(
        name='Caps Type',
        items=[('ROUND', 'Round', ''),
                ('FLAT', 'Flat', ''),
                ('TAPER', 'Taper', ''),
                ('LASSO', 'Lasso', '')],
        default='ROUND'
    )
    operation_type: bpy.props.EnumProperty(
        name='Operation',
        items=[('DIFFERENCE', 'Erase', ''),
                ('UNION', 'Append', '')],
        default='DIFFERENCE'
    )
    smooth_level: bpy.props.IntProperty(
            name='Smooth Level',
            default=0, min=0, soft_max=10,
            description='Perform smoothing to reduce the alias'
    )

    def boolean_eraser_setup(self, context):
        # Create a temporary material for preview only
        mat = bpy.data.materials.new('nijigp_Boolean_Eraser_Preview')
        bpy.data.materials.create_gpencil_data(mat)
        mat.grease_pencil.color = (0.2, 0.2, 0.2, 0.5)
        context.active_object.data.materials.append(mat)
        if self.caps_type == 'LASSO':
            mat.grease_pencil.show_stroke = False
            mat.grease_pencil.show_fill = True
            mat.grease_pencil.fill_color = (0.2, 0.2, 0.2, 0.5)
        
        frame = context.active_object.data.layers.active.active_frame
        stroke = frame.nijigp_strokes.new()
        stroke.line_width = context.scene.tool_settings.gpencil_paint.brush.size
        stroke.material_index = len(context.active_object.data.materials) - 1
        stroke.start_cap_mode = 'FLAT' if self.caps_type == 'FLAT' else 'ROUND'
        stroke.end_cap_mode = 'FLAT' if self.caps_type == 'FLAT' else 'ROUND'
        self._stroke = stroke

    def boolean_eraser_update(self, context, event):
        origin = context.object.matrix_world.translation
        if context.scene.tool_settings.gpencil_stroke_placement_view3d == 'CURSOR':
            origin = context.scene.cursor.location
        self._raw_pressure.append(event.pressure if self.use_pressure else 1)
        self._stroke.points.add(1)
        self._stroke.points[-1].co = self._t_world @ view3d_utils.region_2d_to_location_3d(context.region,
                                    context.space_data.region_3d,
                                    (event.mouse_region_x, event.mouse_region_y), origin)
        set_point_radius(self._stroke.points[-1], self._raw_pressure[-1])
        # For taper mode, reshape the whole stroke 
        if self.caps_type == 'TAPER' and len(self._stroke.points) > 1:
            seg_length = (self._stroke.points[-1].co - self._stroke.points[-2].co).length
            self._segments_length.append(seg_length)
            self._total_length += seg_length
            factor = 0
            for i,point in enumerate(self._stroke.points):
                factor += self._segments_length[i]
                adjusted_pressure = self._raw_pressure[i] * (factor/self._total_length) * (1-factor/self._total_length) * 4
                set_point_radius(point, max(adjusted_pressure, 1e-3))

    def boolean_eraser_finalize(self, context):
        # Execute Draw mode Boolean operator
        if self.smooth_level > 0:
            smooth_stroke_attributes(self._stroke, self.smooth_level, {'co':3, 'pressure':1})
        self._stroke.material_index = context.active_object.active_material_index
        refresh_strokes(context.active_object, [context.scene.frame_current])
        if len(self._stroke.points) > 0:
            bpy.ops.gpencil.nijigp_bool_last(
                operation_type = self.operation_type,
                clip_mode = 'LINE' if self.caps_type != 'LASSO' else 'FILL',
            )
        # If the newly drawn stroke still exist, remove it
        frame = context.active_object.data.layers.active.active_frame
        if self._stroke == frame.nijigp_strokes[-1]:
            frame.nijigp_strokes.remove(self._stroke)
        # Purge the temporary preview material
        mat = bpy.data.materials['nijigp_Boolean_Eraser_Preview']
        context.active_object.data.materials.pop()
        bpy.data.materials.remove(mat)

    def modal(self, context, event): 
        if event.type == 'MOUSEMOVE' and event.pressure > 0:
            if event.mouse_x != self._last_x or event.mouse_y != self._last_y:
                self._last_x, self._last_y = event.mouse_x, event.mouse_y
                self.boolean_eraser_update(context, event)
                return {'RUNNING_MODAL'}
        if event.type in {'LEFTMOUSE'} and event.value == 'RELEASE':
            self.boolean_eraser_finalize(context)
            context.scene.tool_settings.use_gpencil_draw_onback = self._onback
            context.object.show_in_front = self._show_in_front
            return {'FINISHED'}
        return {'RUNNING_MODAL'}

    def invoke(self, context, event):
        try:
            import pyclipper
        except:
            self.report({"ERROR"}, "Please install PyClipper in the Preferences panel.")
            return {'FINISHED'}
        if not context.object.data.layers.active:
            self.report({"INFO"}, "Please select a layer.")
            return {'FINISHED'}
        if is_layer_protected(context.object.data.layers.active):
            self.report({"ERROR"}, "Active layer is locked or hidden.")
            return {'FINISHED'}
                
        self._onback = context.scene.tool_settings.use_gpencil_draw_onback
        self._show_in_front = context.object.show_in_front
        self._stroke = None
        self._raw_pressure = []
        self._segments_length = [0]
        self._total_length = 0
        self._last_x, self._last_y = -1, -1
        self._t_world = context.object.data.layers.active.matrix_layer.inverted_safe() @ context.object.matrix_world.inverted_safe()
        context.scene.tool_settings.use_gpencil_draw_onback = False
        context.object.show_in_front = True
        self.boolean_eraser_setup(context)
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}  

class SmartFillModalOperator(bpy.types.Operator):
    """Filling areas by giving hints with mouse clicks"""
    bl_idname = "gpencil.nijigp_smart_fill_modal"
    bl_label = "Smart Fill (Modal)"
    bl_options = {'REGISTER', 'UNDO'}

    line_layer: bpy.props.StringProperty(
        name='Line Art Layer',
        description='',
        default='',
        search=multilayer_search_func
    )
    use_all_visible: bpy.props.BoolProperty(
        name='Use All Visible Layers',
        default=False,
        description='Every visible layer is considered line art layer. Please note that processing may be slow if there are too many lines.'
    )
    precision: bpy.props.FloatProperty(
        name='Precision',
        default=0.05, min=0.001, max=1,
        description='Treat points in proximity as one to speed up'
    )
    mode: bpy.props.EnumProperty(
        name='Mode',
        items=[('INTERACTIVE', 'Interactive', 'The user can give multiple hints to adjust the fill until confirmed/canceled.'),
                ('SINGLE', 'Single Click', 'Filling will finish instantly without further adjustments.')],
        default='INTERACTIVE'
    )
    incremental: bpy.props.BoolProperty(
        name='Incremental',
        default=True,
        description='If disabled, ignore all previously generated fills when painting a new one'
    )
    _hint_text_position = [150, 100 + get_viewport_bottom_offset() * 1.25]
    _confirm_button = [150, 50 + get_viewport_bottom_offset() * 1.25]
    _cancel_button = [350, 50 + get_viewport_bottom_offset() * 1.25]
    _button_size = [150, 40]
    
    def smart_fill_setup(self):
        # Get line art strokes. Skip the whole operation if cannot find a proper frame
        gp_obj = bpy.context.object
        if self.use_all_visible:
            line_art_layers = [layer for layer in gp_obj.data.layers if not layer_hidden(layer)]
        else:
            line_art_layers, _ = multilayer_search_decode(self.line_layer)
        if len(line_art_layers) < 1:
            line_art_layers = [gp_obj.data.layers.active]
        # Find the frame shown in the viewport
        for line_art_layer in line_art_layers:
            if layer_hidden(line_art_layer):
                continue
            line_art_frame = get_layer_latest_frame(line_art_layer, bpy.context.scene.frame_current)
            if line_art_frame:
                self.line_art_frames.append(line_art_frame)
        if len(self.line_art_frames) < 1:
            return 1
        if sum([len(f.nijigp_strokes) for f in self.line_art_frames]) < 1:
            return 1
        
        # Get or create an output frame from active layer
        fill_layer = gp_obj.data.layers.active
        if not bpy.context.scene.tool_settings.use_keyframe_insert_auto:
            self.output_frame = fill_layer.active_frame
        else:
            for f in fill_layer.frames:
                if f.frame_number == bpy.context.scene.frame_current:
                    self.output_frame = f
                    break
        if not self.output_frame:
            self.output_frame = fill_layer.frames.new(bpy.context.scene.frame_current)

        # Process line art as solver input
        stroke_list = []
        for f in self.line_art_frames:
            stroke_list += [stroke for stroke in f.nijigp_strokes]
        if self.incremental and self.output_frame not in self.line_art_frames:
            stroke_list += [stroke for stroke in self.output_frame.nijigp_strokes]
        self.t_mat, _ = get_transformation_mat(mode='VIEW',
                                                gp_obj=gp_obj, strokes=stroke_list, operator=self)
        poly_list, depth_list, self.scale_factor = get_2d_co_from_strokes(stroke_list, self.t_mat, scale=True)
        self.depth_lookup_tree = DepthLookupTree(poly_list, depth_list)
        tr_output = lineart_triangulation(stroke_list, self.t_mat, poly_list, self.scale_factor, self.precision)
        self.solver.build_graph(tr_output)
        self.solver_init_state = self.solver.labels.copy()
        return 0

    def smart_fill_update(self, mouse_region_co, label):
        """Clear pervious results and recalculate with the newly added hint point"""
        if not self.output_frame:
            return 1
        # Solve the graph problem
        origin = bpy.context.object.matrix_world.translation
        if bpy.context.scene.tool_settings.gpencil_stroke_placement_view3d == 'CURSOR':
            origin = bpy.context.scene.cursor.location
        mouse_global_co = self._t_world @ view3d_utils.region_2d_to_location_3d(bpy.context.region,
                                              bpy.context.space_data.region_3d,
                                              mouse_region_co, origin)
        self.hint_points_co.append((self.t_mat @ mouse_global_co) * self.scale_factor)
        self.hint_points_label.append(label)
        self.smart_fill_clear()
        self.solver.labels = self.solver_init_state.copy()
        # Handle exceptions to prevent the modal from being interrupted
        try:
            for co, label in zip(reversed(self.hint_points_co), reversed(self.hint_points_label)):
                self.solver.set_labels_from_points([co], [label])
            self.solver.propagate_labels()
            self.solver.complete_labels()
        except:
            return 1
        
        # Generate new strokes
        contours_co, contours_label = self.solver.get_contours()
        inv_mat = self.t_mat.inverted_safe()
        gp_settings = bpy.context.scene.tool_settings.gpencil_paint
        for i, contours in enumerate(contours_co):
            label = contours_label[i]
            if label < 1:
                continue
            for c in contours:
                new_stroke = self.output_frame.nijigp_strokes.new()
                new_stroke.line_width = gp_settings.brush.size
                new_stroke.use_cyclic = True
                new_stroke.material_index = bpy.context.object.active_material_index
                if gp_settings.color_mode == 'VERTEXCOLOR':
                    new_stroke.vertex_color_fill = [srgb_to_linear(color) for color in gp_settings.brush.color] + [1]
                new_stroke.points.add(len(c))
                for i,co in enumerate(c):
                    new_stroke.points[i].co = restore_3d_co(co, self.depth_lookup_tree.get_depth(co), inv_mat, self.scale_factor)
                    set_point_radius(new_stroke.points[i], 1)
                    if gp_settings.color_mode == 'VERTEXCOLOR':
                        new_stroke.points[i].vertex_color = [srgb_to_linear(color) for color in gp_settings.brush.color] + [1]
                self.generated_strokes.append(new_stroke)
        refresh_strokes(bpy.context.object)

    def smart_fill_clear(self):
        if not self.output_frame:
            return 1
        for stroke in self.generated_strokes:
            self.output_frame.nijigp_strokes.remove(stroke)
        self.generated_strokes = []

    def smart_fill_finalize(self):
        """Post-processing."""
        bpy.ops.object.mode_set(mode=get_obj_mode_str('EDIT'))
        op_deselect()
        for stroke in self.generated_strokes:
            stroke.select = True
        if bpy.context.scene.tool_settings.use_gpencil_draw_onback:
            op_arrange_stroke(direction='BOTTOM')
        bpy.ops.object.mode_set(mode=get_obj_mode_str('PAINT'))
    
    def draw_callback_px(self, op, context):
        preferences = context.preferences.addons[__package__].preferences     
        hint_texts = ["[Interactive Fill Mode]",
                      "Include Area - <Left Click>",
                      "Exclude Area - <Right Click>",
                      f"Confirm - <Enter> / <{preferences.tool_shortcut_confirm.title()}>",
                      f"Cancel - <ESC> / <{preferences.tool_shortcut_cancel.title()}>",
                      "------"]
        if self._perspective != 'PERSP':
            hint_texts.insert(3, f"Pan View - <{preferences.tool_shortcut_pan.title()}>")
        font_id = 0
        font_size = 18
        line_height = 22

        # Draw hint texts in a box on the screen
        draw_button(self._hint_text_position[0], self._hint_text_position[1], 350, line_height * len(hint_texts), (.95, .95, .95, 1), '')
        blf.color(font_id, 0.1, 0.1, 0.1, 1)
        blf_set_size(blf, font_id, font_size)
        for i,text in enumerate(reversed(hint_texts)):
            blf.position(font_id, self._hint_text_position[0], self._hint_text_position[1] + line_height * i, 0)
            blf.draw(font_id, text)
            
        # Draw hint points according to user clicks
        for symbol in ['+', '-']:
            for loc in self._screen_hint_points[symbol]:
                co = view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.space_data.region_3d, loc)
                blf.position(font_id, 
                    co[0] - font_size * .25,
                    co[1] - font_size * .25,
                    0)
                blf.draw(font_id, symbol)
        
        # Draw interactive buttons
        draw_button(self._confirm_button[0], self._confirm_button[1], self._button_size[0], self._button_size[1], 
                    (.24, .73, .58, 1), 'Confirm')
        draw_button(self._cancel_button[0], self._cancel_button[1], self._button_size[0], self._button_size[1],
                    (.75, .22, .34, 1), 'Cancel')
    
    def modal(self, context, event):
        preferences = context.preferences.addons[__package__].preferences
        context.area.tag_redraw()

        # Process view panning
        if self._perspective != 'PERSP':
            if event.type == preferences.tool_shortcut_pan and event.value == 'PRESS':
                self._panning_ongoing = True
                self._last_x, self._last_y = event.mouse_region_x, event.mouse_region_y
            if event.type == preferences.tool_shortcut_pan and event.value == 'RELEASE':
                self._panning_ongoing = False
            if self._panning_ongoing and event.type == 'MOUSEMOVE':
                delta_x = event.mouse_region_x - self._last_x
                delta_y = event.mouse_region_y - self._last_y
                if self._perspective == 'CAMERA':
                    context.space_data.region_3d.view_camera_offset[0] -= delta_x / context.region.width
                    context.space_data.region_3d.view_camera_offset[1] -= delta_y / context.region.height
                else:
                    context.space_data.region_3d.view_location -= (context.space_data.region_3d.view_rotation @ Vector((delta_x, delta_y, 0))) * (context.space_data.region_3d.view_distance / context.region.width)
                self._last_x, self._last_y = event.mouse_region_x, event.mouse_region_y

        # Process events that terminate the modal
        if event.type in {'RET', 'NUMPAD_ENTER', preferences.tool_shortcut_confirm} or is_button_clicked(event, self._confirm_button, self._button_size):
            bpy.types.SpaceView3D.draw_handler_remove(self._handle, 'WINDOW')
            self.smart_fill_finalize()
            context.object.show_in_front = self._show_in_front
            return {'FINISHED'}
        if event.type in {'ESC', preferences.tool_shortcut_cancel} or is_button_clicked(event, self._cancel_button, self._button_size):
            bpy.types.SpaceView3D.draw_handler_remove(self._handle, 'WINDOW')
            self.smart_fill_clear()
            context.object.show_in_front = self._show_in_front
            return {'CANCELLED'}

        # Process user hints
        ret = None
        hint_point_loc = view3d_utils.region_2d_to_location_3d(
            bpy.context.region, bpy.context.space_data.region_3d, 
            (event.mouse_region_x, event.mouse_region_y), (0,0,0)
            )
        if event.type == 'LEFTMOUSE' and event.value == 'RELEASE':
            self._screen_hint_points['+'].append(hint_point_loc)
            ret = self.smart_fill_update((event.mouse_region_x, event.mouse_region_y), 1)
            if self.mode == 'SINGLE':
                bpy.types.SpaceView3D.draw_handler_remove(self._handle, 'WINDOW')
                self.smart_fill_finalize()
                context.object.show_in_front = self._show_in_front
                return {'FINISHED'}
        if event.type == 'RIGHTMOUSE' and event.value == 'RELEASE':
            self._screen_hint_points['-'].append(hint_point_loc)
            ret = self.smart_fill_update((event.mouse_region_x, event.mouse_region_y), 0)

        # Process error cases
        if ret and ret > 0:
            self.report({"ERROR"}, "Cannot calculate the fill area. Please select a proper line art layer and ensure Python packages are installed.")
            bpy.types.SpaceView3D.draw_handler_remove(self._handle, 'WINDOW')
            context.object.show_in_front = self._show_in_front
            return {'FINISHED'}
        return {'RUNNING_MODAL'}
    
    def invoke(self, context, event):
        try:
            from .solvers.graph import SmartFillSolver
        except:
            self.report({"ERROR"}, "Please install Scikit-Image in the Preferences panel.")
            return {'FINISHED'}
        if is_layer_protected(context.object.data.layers.active):
            self.report({"ERROR"}, "Active layer is locked or hidden.")
            return {'FINISHED'}

        self._t_world = context.object.data.layers.active.matrix_layer.inverted_safe() @ context.object.matrix_world.inverted_safe()
        self._show_in_front = context.object.show_in_front
        context.object.show_in_front = True

        # Setup the solver for calculations
        self.t_mat = Matrix()
        self.scale_factor = 1
        self.depth_lookup_tree = None
        self.hint_points_co = []
        self.hint_points_label = []
        self.output_frame = None
        self.line_art_frames = []
        self.generated_strokes = []
        self.solver = SmartFillSolver()
        self.solver_init_state = None
        self.smart_fill_setup()
        
        # Setup the handler for screen hint messages
        self._screen_hint_points = {'+':[], '-':[]}
        self._perspective = context.space_data.region_3d.view_perspective
        self._panning_ongoing = False
        args = (self, context)
        self._handle = bpy.types.SpaceView3D.draw_handler_add(
            self.draw_callback_px, args, 'WINDOW', 'POST_PIXEL'
        )
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}    

class SweepModalOperator(bpy.types.Operator, ColorTintConfig):
    """Executing sweep with the path vector determined by mouse position"""
    bl_idname = "gpencil.nijigp_sweep_selected_modal"
    bl_label = "Sweep Selected (Modal)"
    bl_options = {'REGISTER', 'UNDO'}

    starting_mouse_pos: bpy.props.IntVectorProperty(size=2)
    multiframe_falloff: bpy.props.FloatProperty(
            name='Multiframe Falloff',
            default=0, min=0, max=1,
            description='The ratio of offset length falloff per frame in the multiframe editing mode',
    )
    style: bpy.props.EnumProperty(
            name='Style',
            items=[('EXTRUDE', 'Extrude', ''),
                    ('OUTER', 'Outer Shadow', ''),
                    ('INNER', 'Inner Shadow', '')],
            default='EXTRUDE'
    )
    moused_moved: bool

    def modal(self, context, event):
        if event.type == 'MOUSEMOVE':
            self.mouse_moved = True
            delta = [2 * (event.mouse_x - self.starting_mouse_pos[0]) / LINE_WIDTH_FACTOR,
                     2 * (event.mouse_y - self.starting_mouse_pos[1]) / LINE_WIDTH_FACTOR]
            bpy.ops.ed.undo()
            bpy.ops.ed.undo_push()
            context.area.header_text_set('Path Vector: {:f} m, {:f} m'.format(delta[0], delta[1]))
            bpy.ops.gpencil.nijigp_sweep_selected(multiframe_falloff=self.multiframe_falloff,
                                                  path_type='VEC', path_vector=delta, style=self.style,
                                                  tint_color=self.tint_color,
                                                  tint_color_factor=self.tint_color_factor,
                                                  tint_mode=self.tint_mode,
                                                  blend_mode=self.blend_mode)

        elif event.type == 'LEFTMOUSE':
            # Fall back to the select operation if mouse is never moved
            if not self.mouse_moved:
                op_select(location=(event.mouse_region_x, event.mouse_region_y),
                            extend=(self.style=='OUTER'))
            context.area.header_text_set(None)
            return {'FINISHED'}
        
        elif event.type in {'RIGHTMOUSE', 'ESC'}:
            context.area.header_text_set(None)
            bpy.ops.ed.undo()
            return {'CANCELLED'}

        return {'RUNNING_MODAL'}

    def invoke(self, context, event):
        self.mouse_moved = False
        if context.object:
            self.starting_mouse_pos = [event.mouse_x, event.mouse_y]
            context.window_manager.modal_handler_add(self)
            bpy.ops.ed.undo_push()
            return {'RUNNING_MODAL'}
        else:
            self.report({'WARNING'}, "No active object, could not finish.")
            return {'CANCELLED'}

class OffsetModalOperator(bpy.types.Operator):
    """Executing offset with the value determined by mouse position"""
    bl_idname = "gpencil.nijigp_offset_selected_modal"
    bl_label = "Offset Selected (Modal)"
    bl_options = {'REGISTER', 'UNDO'}

    starting_mouse_x: bpy.props.IntProperty()
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
    moused_moved: bool

    def modal(self, context, event):
        if event.type == 'MOUSEMOVE':
            self.mouse_moved = True
            delta = (event.mouse_x - self.starting_mouse_x) / LINE_WIDTH_FACTOR
            bpy.ops.ed.undo()
            bpy.ops.ed.undo_push()
            context.area.header_text_set('Offset: {:f} m'.format(delta))
            bpy.ops.gpencil.nijigp_offset_selected(offset_amount=delta,
                                                   corner_shape=self.corner_shape,
                                                   end_type_mode=self.end_type_mode,
                                                   multiframe_falloff=self.multiframe_falloff)   

        elif event.type == 'LEFTMOUSE':
            # Fall back to the select operation if mouse is never moved
            if not self.mouse_moved:
                op_select(location=(event.mouse_region_x, event.mouse_region_y),
                            extend=(self.end_type_mode=='ET_OPEN'))
            context.area.header_text_set(None)
            return {'FINISHED'}
        
        elif event.type in {'RIGHTMOUSE', 'ESC'}:
            context.area.header_text_set(None)
            bpy.ops.ed.undo()
            return {'CANCELLED'}

        return {'RUNNING_MODAL'}

    def invoke(self, context, event):
        self.mouse_moved = False
        if context.object:
            self.starting_mouse_x = event.mouse_x
            context.window_manager.modal_handler_add(self)
            bpy.ops.ed.undo_push()
            #bpy.ops.gpencil.nijigp_offset_selected(offset_amount=0) 
            return {'RUNNING_MODAL'}
        else:
            self.report({'WARNING'}, "No active object, could not finish.")
            return {'CANCELLED'}

nijigp_cached_view_rotation = {}
class RollViewModalOperator(bpy.types.Operator):
    """Drag to roll the view, or click to undo all previous rolling"""
    bl_idname = "gpencil.nijigp_roll_view_modal"
    bl_label = "Roll the View (Modal)"
    bl_options = {'REGISTER'}

    mouse_x: bpy.props.IntProperty()
    starting_mouse_x: bpy.props.IntProperty()
    moving_rate: bpy.props.FloatProperty(default=0.01)
    moving_threshold: bpy.props.IntProperty(default=5)

    def modal(self, context, event):
        if event.type == 'MOUSEMOVE':
            delta = (event.mouse_x - self.mouse_x) * self.moving_rate
            accumulated_delta = (event.mouse_x - self.starting_mouse_x) * self.moving_rate
            self.mouse_x = event.mouse_x
            rotation: Euler = Quaternion(self.region.view_rotation).to_euler()
            rotation.rotate_axis('Z', delta)
            self.region.view_rotation = rotation.to_quaternion()
            context.area.header_text_set('Angle: {:f} degrees'.format(math.degrees(accumulated_delta)))

        # Left click: reset the view to the state before rolling
        elif event.type == 'LEFTMOUSE':
            if abs(event.mouse_x - self.starting_mouse_x) < self.moving_threshold:
                for i,area in enumerate(context.screen.areas):
                    if area.type == 'VIEW_3D':
                        for j,space in enumerate(area.spaces):
                            key=str([i,j])
                            if key in nijigp_cached_view_rotation:
                                space.region_3d.view_rotation = nijigp_cached_view_rotation[key]
                                nijigp_cached_view_rotation[key] = None
            context.area.header_text_set(None)
            return {'FINISHED'}
        
        elif event.type in {'RIGHTMOUSE', 'ESC'}:
            self.region.view_rotation = self.original_rotation
            context.area.header_text_set(None)
            return {'CANCELLED'}

        return {'RUNNING_MODAL'}

    def invoke(self, context, event):
        self.region: bpy.types.RegionView3D = None
        if context.region_data:
            self.region = context.region_data

            if self.region.view_perspective == 'CAMERA':
                self.region.view_perspective = 'ORTHO'
            self.mouse_x = event.mouse_x
            self.starting_mouse_x = event.mouse_x
            context.window_manager.modal_handler_add(self)

            # Save the rotation state before rolling for possible reset
            self.original_rotation = Quaternion(self.region.view_rotation)
            for i,area in enumerate(context.screen.areas):
                if area.type == 'VIEW_3D':
                    for j,space in enumerate(area.spaces):
                        key=str([i,j])
                        if key not in nijigp_cached_view_rotation or nijigp_cached_view_rotation[key] is None:
                            nijigp_cached_view_rotation[key] = Quaternion(space.region_3d.view_rotation)
            return {'RUNNING_MODAL'}
        else:
            return {'CANCELLED'}

class ArrangeModalOperator(bpy.types.Operator):
    """
    Drag left/right to arrange strokes downward/upward
    """
    bl_idname = "gpencil.nijigp_arrange_modal"
    bl_label = "Arrange Strokes (Modal)"
    bl_options = {'REGISTER', 'UNDO'}

    max_display: bpy.props.IntProperty(default=300)
    font_id: bpy.props.IntProperty(default=0)
    starting_mouse_x: bpy.props.IntProperty()
    arrange_offset: bpy.props.IntProperty(default=0)
    moving_rate: bpy.props.FloatProperty(default=0.01)

    def draw_callback_px(self, op, context):
        """
        Show the order of the active layer&frame by number
        """
        if context.object and obj_is_gp(context.object):
            layers = context.object.data.layers
            if layers.active and not layers.active.hide and layers.active.active_frame:
                strokes = layers.active.active_frame.nijigp_strokes
                if len(strokes) < self.max_display:
                    for i,stroke in enumerate(strokes):
                        if len(stroke.points) > 0:
                            # Render the number text on each stroke
                            # Mark the starting point for open stroke and center for closed/fill stroke
                            if is_stroke_line(stroke, context.object) and not stroke.use_cyclic:
                                view_co = view3d_utils.location_3d_to_region_2d(context.region, 
                                                                                context.space_data.region_3d, 
                                                                                stroke.points[0].co)
                            else:
                                view_co = view3d_utils.location_3d_to_region_2d(context.region, 
                                                                                context.space_data.region_3d, 
                                                                                (stroke.points[0].co + stroke.points[len(stroke.points)//3].co + stroke.points[len(stroke.points)*2//3].co) / 3)                                
                            if stroke.select:
                                blf.color(self.font_id, 1, 0.5, 0.5, 1)
                            else:
                                blf.color(self.font_id, 0, 0, 0, 1)
                            blf.enable(self.font_id, blf.SHADOW)
                            blf.shadow(self.font_id, 3, 0.8, 0.8, 0.8, 0.95)
                            blf_set_size(blf, self.font_id, 36 if stroke.select else 24)
                            blf.position(self.font_id, view_co[0], view_co[1], 0)
                            blf.draw(self.font_id, str(i))

    def modal(self, context, event):
        context.area.tag_redraw()

        if event.type == 'MOUSEMOVE':
            delta = round((event.mouse_x - self.starting_mouse_x) * self.moving_rate)
            if delta > self.arrange_offset:
                for i in range(delta-self.arrange_offset):
                    op_arrange_stroke('UP')
            elif delta < self.arrange_offset:
                for i in range(self.arrange_offset-delta):
                    op_arrange_stroke('DOWN')
            self.arrange_offset = delta 
            context.area.header_text_set('Arrangement Offset: {:d}'.format(self.arrange_offset))
        elif event.type == 'LEFTMOUSE':
            context.area.header_text_set(None)
            bpy.types.SpaceView3D.draw_handler_remove(self._handle, 'WINDOW')
            return {'FINISHED'}            
        elif event.type in {'RIGHTMOUSE', 'ESC'}:
            context.area.header_text_set(None)
            bpy.types.SpaceView3D.draw_handler_remove(self._handle, 'WINDOW')
            return {'CANCELLED'}
        return {'RUNNING_MODAL'}

    def invoke(self, context, event):
        self.starting_mouse_x = event.mouse_x
        self.arrange_offset = 0

        args = (self, context)
        self._handle = bpy.types.SpaceView3D.draw_handler_add(
            self.draw_callback_px, args, 'WINDOW', 'POST_PIXEL'
        )
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

class BooleanEraserTool(bpy.types.WorkSpaceTool):
    bl_space_type = 'VIEW_3D'
    bl_context_mode = get_ctx_mode_str('PAINT')
    bl_idname = "nijigp.boolean_eraser_tool"
    bl_label = "Boolean Eraser"
    bl_description = (
        "Use mouse or pen drawing to perform Boolean operations. Hold CTRL to append, otherwise erase"
    )
    bl_icon = get_workspace_tool_icon('ops.nijigp.boolean_eraser_tool')
    bl_cursor = 'ERASER'
    bl_widget = None
    bl_keymap = (
        ("gpencil.nijigp_boolean_modal", {"type": 'LEFTMOUSE', "value": 'PRESS'},
         {"properties": []}),
        ("gpencil.nijigp_boolean_modal", {"type": 'LEFTMOUSE', "value": 'PRESS', "ctrl": True},
         {"properties": [("operation_type", 'UNION')]}),
    )
    def draw_settings(context, layout, tool):
        props = tool.operator_properties("gpencil.nijigp_boolean_modal")
        gp_settings = context.scene.tool_settings.gpencil_paint
        active_material_name = context.object.active_material.name if context.object.active_material else ""
        
        # Brush setting panel imitating the native style
        row = layout.row(align=True)
        row.popover(panel=get_panel_str('TOPBAR_PT', 'materials'), text=active_material_name)
        row = layout.row(align=True)
        row.prop(gp_settings.brush, "size", text="Radius")
        row.prop(props, "use_pressure", text="", icon='STYLUS_PRESSURE')
        layout.prop(props, "smooth_level")
        layout.prop(props, "caps_type")
        layout.prop(props, "operation_type", expand=True)
        
        layout.label(text="Affected Strokes:")
        layout.prop(context.scene, "nijigp_draw_bool_material_constraint", text = "")
        layout.prop(context.scene, "nijigp_draw_bool_fill_constraint", text = "")
        layout.prop(context.scene, "nijigp_draw_bool_selection_constraint", text = "Selected Strokes Only", icon = "GP_SELECT_STROKES")     

class SmartFillTool(bpy.types.WorkSpaceTool):
    bl_space_type = 'VIEW_3D'
    bl_context_mode = get_ctx_mode_str('PAINT')
    bl_idname = "nijigp.smart_fill_tool"
    bl_label = "Interactive Smart Fill"
    bl_description = (
        "Filling areas by giving hints with mouse clicks"
    )
    bl_icon = get_workspace_tool_icon('ops.nijigp.smart_fill_tool')
    bl_widget = None
    bl_keymap = (
        ("gpencil.nijigp_smart_fill_modal", {"type": 'LEFTMOUSE', "value": 'PRESS'},
         {"properties": []}),
    )
    def draw_settings(context, layout, tool):
        props = tool.operator_properties("gpencil.nijigp_smart_fill_modal")
        gp_settings = context.scene.tool_settings.gpencil_paint
        active_material_name = context.object.active_material.name if context.object.active_material else ""
        
        # Color/material setting panel imitating the native style
        row = layout.row(align=True)
        row.popover(panel=get_panel_str('TOPBAR_PT', 'materials'), text=active_material_name)
        row = layout.row(align=True)
        row.separator(factor=1.0)
        sub_row = row.row(align=True)
        sub_row.prop_enum(gp_settings, "color_mode", 'MATERIAL', text="", icon='MATERIAL')
        sub_row.prop_enum(gp_settings, "color_mode", 'VERTEXCOLOR', text="", icon='VPAINT_HLT')
        sub_row = row.row(align=True)
        sub_row.enabled = gp_settings.color_mode == 'VERTEXCOLOR'
        sub_row.prop_with_popover(gp_settings.brush, "color", text="", panel=get_panel_str('TOPBAR_PT', 'vertexcolor'))
        
        # Other attributes
        row = layout.row(align=True)
        row.prop(props, "line_layer")
        if props.use_all_visible:
            row.enabled = False
        row = layout.row(align=True)
        row.prop(props, "use_all_visible")
        row = layout.row(align=True)
        row.prop(props, "precision")
        row = layout.row(align=True)
        row.prop(props, "mode")
        row = layout.row(align=True)
        row.prop(props, "incremental")
        if props.use_all_visible:
            row.enabled = False

class OffsetTool(bpy.types.WorkSpaceTool):
    bl_space_type = 'VIEW_3D'
    bl_context_mode = get_ctx_mode_str('EDIT')

    bl_idname = "nijigp.offset_tool"
    bl_label = "2D Offset"
    bl_description = (
        "Offset/inset the shape of selected strokes with mouse dragging. "
        "Holding CTRL for corner mode, or holding SHIFT for line mode"
    )
    bl_icon = get_workspace_tool_icon('ops.nijigp.offset_tool')
    bl_widget = None
    bl_keymap = (
        ("gpencil.nijigp_offset_selected_modal", {"type": 'LEFTMOUSE', "value": 'PRESS'},
         {"properties": []}),
        ("gpencil.nijigp_offset_selected_modal", {"type": 'LEFTMOUSE', "value": 'PRESS', "ctrl": True},
         {"properties": [("end_type_mode", 'ET_CLOSE_CORNER')]}),
        ("gpencil.nijigp_offset_selected_modal", {"type": 'LEFTMOUSE', "value": 'PRESS', "shift": True},
         {"properties": [("end_type_mode", 'ET_OPEN')]}),
    )

    def draw_settings(context, layout, tool):
        props = tool.operator_properties("gpencil.nijigp_offset_selected_modal")
        layout.prop(props, "corner_shape")
        layout.prop(props, "multiframe_falloff")

class SweepTool(bpy.types.WorkSpaceTool):
    bl_space_type = 'VIEW_3D'
    bl_context_mode = get_ctx_mode_str('EDIT')

    bl_idname = "nijigp.sweep_tool"
    bl_label = "2D Sweep"
    bl_description = (
        "Sweep the shape of selected strokes in the direction of mouse dragging. "
        "Holding CTRL for inner shadow mode, or holding SHIFT for outer shadow mode"
    )
    bl_icon = get_workspace_tool_icon('ops.nijigp.sweep_tool')
    bl_widget = None
    bl_keymap = (
        ("gpencil.nijigp_sweep_selected_modal", {"type": 'LEFTMOUSE', "value": 'PRESS'},
         {"properties": []}),
        ("gpencil.nijigp_sweep_selected_modal", {"type": 'LEFTMOUSE', "value": 'PRESS', "ctrl": True},
         {"properties": [("style", 'INNER')]}),
        ("gpencil.nijigp_sweep_selected_modal", {"type": 'LEFTMOUSE', "value": 'PRESS', "shift": True},
         {"properties": [("style", 'OUTER')]}),
    )

    def draw_settings(context, layout, tool):
        props = tool.operator_properties("gpencil.nijigp_sweep_selected_modal")
        layout.prop(props, "multiframe_falloff")
        layout.prop(props, "tint_color")
        layout.prop(props, "tint_color_factor")
        layout.prop(props, "tint_mode")
        layout.prop(props, "blend_mode")

class ViewportShortcuts(bpy.types.GizmoGroup):
    bl_idname = "nijigp_viewport_shortcuts"
    bl_label = "NijiGPen Viewport Shortcuts"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'WINDOW'
    bl_options = {'PERSISTENT', 'SCALE'}

    @classmethod
    def poll(cls, context):
        return (context.mode==get_ctx_mode_str('EDIT') or context.mode==get_ctx_mode_str('PAINT') or context.mode==get_ctx_mode_str('SCULPT'))

    def draw_prepare(self, context):
        preferences = context.preferences.addons[__package__].preferences
        if not preferences.shortcut_button_enabled or not hasattr(self, "gizmo_list"):
            return
        region = context.region
        spacing = preferences.shortcut_button_spacing
        total_width = len(self.gizmo_list) * self.button_size * 4 * context.preferences.view.ui_scale
        for i, gizmo in enumerate(self.gizmo_list):
            if gizmo:
                if preferences.shortcut_button_style == 'BOTTOM':
                    gizmo.matrix_basis[0][3] = (region.width/2 + total_width/2
                                                - self.button_size*spacing*(i+1) * context.preferences.view.ui_scale
                                                + preferences.shortcut_button_location[0])
                    gizmo.matrix_basis[1][3] = self.button_size*2 + preferences.shortcut_button_location[1] + get_viewport_bottom_offset()
                elif preferences.shortcut_button_style == 'TOP':
                    gizmo.matrix_basis[0][3] = (region.width/2 + total_width/2
                                                - self.button_size*spacing*(i+1) * context.preferences.view.ui_scale
                                                + preferences.shortcut_button_location[0])
                    gizmo.matrix_basis[1][3] = region.height - self.button_size*4 - preferences.shortcut_button_location[1]
                elif preferences.shortcut_button_style == 'RIGHT':
                    gizmo.matrix_basis[0][3] = region.width - self.button_size*3 - preferences.shortcut_button_location[0]
                    gizmo.matrix_basis[1][3] = (self.button_size*spacing*(i+1) * context.preferences.view.ui_scale
                                                + preferences.shortcut_button_location[1])   
                else:             
                    gizmo.matrix_basis[0][3] = self.button_size*6 + preferences.shortcut_button_location[0]
                    gizmo.matrix_basis[1][3] = (self.button_size*spacing*(i+1) * context.preferences.view.ui_scale
                                                + preferences.shortcut_button_location[1])                    
    def setup(self, context):
        preferences = context.preferences.addons[__package__].preferences
        if not preferences.shortcut_button_enabled:
            return
        self.gizmo_list = []
        self.button_profile_list = [{'op': 'ed.redo', 'icon': 'LOOP_FORWARDS'},
                               {'op': 'ed.undo', 'icon': 'LOOP_BACK'},
                               None,
                               {'op': 'gpencil.nijigp_roll_view_modal', 'icon': 'FILE_REFRESH'},
                               None,
                               {'op': get_ops_str('gpencil.stroke_arrange'), 'icon': 'TRIA_UP_BAR', 'direction': 'TOP'},
                               {'op': 'gpencil.nijigp_arrange_modal' ,'icon': 'MOD_DISPLACE'},
                               {'op': get_ops_str('gpencil.stroke_arrange'), 'icon': 'TRIA_DOWN_BAR', 'direction': 'BOTTOM'}
                               ]
        self.button_size = preferences.shortcut_button_size 
        for profile in self.button_profile_list:
            if profile:
                button = self.gizmos.new("GIZMO_GT_button_2d")
                button.draw_options = {'BACKDROP', 'OUTLINE'}
                button.scale_basis = self.button_size
                button.icon = profile['icon']
                op = button.target_set_operator(profile['op'])
                if 'direction' in profile:
                    op.direction = profile['direction']

                button.color = 0.3, 0.3, 0.3
                button.alpha = 0.5
                button.color_highlight = 0.6, 0.6, 0.6
                button.alpha_highlight = 0.5
                self.gizmo_list.append(button)
            # Add a placeholder as a separator in the viewport
            else:
                self.gizmo_list.append(None)

class RefreshGizmoOperator(bpy.types.Operator):
    bl_idname = "gpencil.nijigp_refresh_gizmo"
    bl_label = "Refresh Viewport Shortcuts"

    def execute(self, context):
        bpy.utils.unregister_class(ViewportShortcuts)
        bpy.utils.register_class(ViewportShortcuts)
        return {'FINISHED'}

def register_viewport_tools():
    # Edit mode tools
    bpy.utils.register_tool(OffsetTool, after={"builtin.transform_fill", "builtin.texture_gradient"}, separator=True, group=True)
    bpy.utils.register_tool(SweepTool, after={OffsetTool.bl_idname})
    # Draw mode tools
    bpy.utils.register_tool(SmartFillTool, after={"builtin.circle"}, separator=True, group=True)
    bpy.utils.register_tool(BooleanEraserTool, after={SmartFillTool.bl_idname})

def unregister_viewport_tools():
    bpy.utils.unregister_tool(OffsetTool)
    bpy.utils.unregister_tool(SweepTool)
    bpy.utils.unregister_tool(SmartFillTool)
    bpy.utils.unregister_tool(BooleanEraserTool)