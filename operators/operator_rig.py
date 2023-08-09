import bpy

class BakeRiggingOperator(bpy.types.Operator):
    """Apply deformation caused by armature modifiers on each frame. Create new keyframes when necessary"""
    bl_idname = "gpencil.nijigp_bake_rigging_animation"
    bl_label = "Bake Rigging to Keyframes"
    bl_options = {'REGISTER', 'UNDO'}

    start_frame: bpy.props.IntProperty(
        name='Start Frame',
        default=0, min=1
        )
    end_frame: bpy.props.IntProperty(
        name='End Frame',
        default=0, min=1
        )
    frame_step: bpy.props.IntProperty(
        name='Frame Step',
        default=1, min=1
        )
    target_mode: bpy.props.EnumProperty(            
        name='Target Layers',
        items=[ ('LAYER', 'Active Layer', ''),
               ('PASS', 'Active Layer Pass Index', ''),
               ('ALL', 'All Layers', ''),],
        default='ALL',
    )
    clear_parents: bpy.props.BoolProperty(
        name='Clear Parents',
        default=False,
        description='Clear the parents of the active Grease Pencil object'
    )

    def execute(self, context):
        gp_obj = context.object
        bpy.ops.object.mode_set(mode='OBJECT')
        
        def switch_to(obj):
            bpy.ops.object.select_all(action='DESELECT')
            obj.select_set(True)
            bpy.context.view_layer.objects.active = obj
        switch_to(gp_obj)
        
        # Get output layers
        target_layers = []
        active_pass_index = gp_obj.data.layers.active.pass_index
        for i,layer in enumerate(gp_obj.data.layers):
            if (layer == gp_obj.data.layers.active or
                self.target_mode == 'ALL' or 
                (self.target_mode == 'PASS' and layer.pass_index == active_pass_index)):
                    target_layers.append(i)
                    
        # Get frame information
        target_frame_numbers = range(self.start_frame, self.end_frame + 1, self.frame_step)
        is_target_frame_number = lambda f: (f>=self.start_frame and f<=self.end_frame and 
                                            (f - self.start_frame) % self.frame_step == 0)
        layer_get_keyframe = {}         # 2D map, for original object
        layer_get_ref_keyframe_idx = {} # 2D map, for duplicated objects
        for i in target_layers:
            layer = gp_obj.data.layers[i]
            layer_get_keyframe[i] = {}
            for frame in layer.frames:
                layer_get_keyframe[i][frame.frame_number] = frame
                
        # Duplicate the active object for reference
        # TODO: This method has a very low efficiency. Is there a better way?
        bpy.ops.object.duplicate()
        ref_gp_obj = bpy.context.object
        for i in target_layers:
            layer_get_ref_keyframe_idx[i] = {}
            layer = ref_gp_obj.data.layers[i]
            for j,frame in enumerate(layer.frames):
                layer_get_ref_keyframe_idx[i][frame.frame_number] = j
            last_frame_idx = None
            for frame_number in range(1, self.end_frame + 1):
                if frame_number in layer_get_ref_keyframe_idx[i]:
                    last_frame_idx = layer_get_ref_keyframe_idx[i][frame_number]
                elif last_frame_idx != None:
                    layer_get_ref_keyframe_idx[i][frame_number] = last_frame_idx
        switch_to(gp_obj)

        # Create missing keyframes on the original object
        for i in target_layers:
            layer = gp_obj.data.layers[i]
            last_frame = None
            for frame_number in range(1, self.end_frame + 1):
                if frame_number in layer_get_keyframe[i]:
                    last_frame = layer_get_keyframe[i][frame_number]
                elif last_frame != None and is_target_frame_number(frame_number):
                    layer_get_keyframe[i][frame_number] = layer.frames.copy(last_frame)
                    layer_get_keyframe[i][frame_number].frame_number = frame_number

        # Get all armature modifiers
        target_modifiers = []
        for mod in gp_obj.grease_pencil_modifiers:
            if mod.type == 'GP_ARMATURE' and mod.object:
                target_modifiers.append(mod.name)
                
        # Process by frame number
        switch_to(ref_gp_obj)
        for frame_number in target_frame_numbers:
            context.scene.frame_current = frame_number
            
            # Duplicate the object to apply all armature modifiers, and record coordinate changes
            bpy.ops.object.duplicate()
            dup_gp_obj = bpy.context.object
            new_coordinates = {}    # 4D list: layer->stroke->point->xyz
            for mod_name in target_modifiers:
                bpy.ops.object.gpencil_modifier_apply(modifier=mod_name)
            for i in target_layers:
                layer = dup_gp_obj.data.layers[i]
                new_coordinates[i] = []
                if frame_number not in layer_get_ref_keyframe_idx[i]:
                    continue
                frame = layer.frames[layer_get_ref_keyframe_idx[i][frame_number]]
                for stroke in frame.strokes:
                    new_coordinates[i].append([])
                    for point in stroke.points:
                        new_coordinates[i][-1].append(tuple(point.co))
            # Purge the duplicated object
            bpy.ops.object.delete(use_global=True)
            switch_to(ref_gp_obj)
            
            # Apply recorded coordinate changes
            for i in target_layers:
                layer = gp_obj.data.layers[i]
                if frame_number not in layer_get_keyframe[i]:
                    continue
                frame = layer_get_keyframe[i][frame_number]
                for j,stroke in enumerate(frame.strokes):
                    for k,point in enumerate(stroke.points):
                        point.co = new_coordinates[i][j][k]
        # Cleanup
        bpy.ops.object.delete(use_global=True)
        switch_to(gp_obj)
        for mod_name in target_modifiers:
            bpy.ops.object.gpencil_modifier_remove(modifier=mod_name)
        if self.clear_parents:
            bpy.ops.object.parent_clear(type='CLEAR')
        bpy.ops.object.mode_set(mode='EDIT_GPENCIL')
        return {'FINISHED'}

    def invoke(self, context, event):
        self.start_frame = bpy.context.scene.frame_start
        self.end_frame = bpy.context.scene.frame_end
        return context.window_manager.invoke_props_dialog(self)

    def draw(self, context):
        row = self.layout
        row.prop(self, "start_frame")
        row.prop(self, "end_frame")
        row.prop(self, "frame_step")
        row.prop(self, "target_mode")
        row.prop(self, "clear_parents")