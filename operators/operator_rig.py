import bpy
from mathutils import *
from .common import *
from ..utils import *

def switch_to(objs):
    """Rig operations need frequent object re-selections"""
    bpy.ops.object.select_all(action='DESELECT')
    for obj in objs:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]

def get_output_layers(gp_obj, target_mode:str):
    """Common way of selecting output layers for rig operators"""
    target_layers = []
    active_pass_index = gp_obj.data.layers.active.pass_index
    for i,layer in enumerate(gp_obj.data.layers):
        if (layer == gp_obj.data.layers.active or
            target_mode == 'ALL' or 
            (target_mode == 'PASS' and layer.pass_index == active_pass_index)):
                target_layers.append(i)
    return target_layers

class TransferWeightOperator(bpy.types.Operator):
    """Transfer bone weights from meshes to Grease Pencil strokes by looking for the nearest vertex"""
    bl_idname = "gpencil.nijigp_rig_by_transfer_weights"
    bl_label = "Bone Weights from Meshes"
    bl_options = {'REGISTER', 'UNDO'}

    source_arm: bpy.props.StringProperty(
        name='Source Armature',
        description='The armature that meshes and Grease Pencil strokes are or will be attached to',
        default='',
        search=lambda self, context, edit_text: [obj.name for obj in bpy.context.scene.objects if obj.type=='ARMATURE']
    ) 
    source_type: bpy.props.EnumProperty(            
        name='Source Meshes',
        items=[ ('THIS', 'Generated Meshes', ''),
               ('OBJ', 'An Object', ''),
               ('COLL', 'A Collection', ''),],
        default='THIS',
    )
    source_obj: bpy.props.StringProperty(
        name='Object',
        default='',
        search=lambda self, context, edit_text: [obj.name for obj in bpy.context.scene.objects if obj.type=='MESH']
    ) 
    source_coll: bpy.props.StringProperty(
        name='Collection',
        default='',
        search=lambda self, context, edit_text: [coll.name_full for coll in bpy.context.scene.collection.children_recursive]
    ) 
    mapping_type: bpy.props.EnumProperty(            
        name='Mapping',
        items=[ ('2D', '2D Nearest Vertex', ''),
               ('3D', '3D Nearest Vertex', ''),],
        default='2D',
    )
    target_mode: bpy.props.EnumProperty(            
        name='Target Layers',
        items=[ ('LAYER', 'Active Layer', ''),
               ('PASS', 'Active Layer Pass Index', ''),
               ('ALL', 'All Layers', ''),],
        default='ALL',
    )
    weights_exist: bpy.props.BoolProperty(            
        name='Use Existing Weights',
        default=False,
        description='If not checked, generate automatic weights for the meshes first'
    )
    adjust_parenting: bpy.props.BoolProperty(            
        name='Adjust Parenting',
        default=True,
        description='Set necessary parent relationships after transferring the weights'
    )

    def execute(self, context):
        bpy.ops.object.mode_set(mode='OBJECT')
        gp_obj: bpy.types.Object = context.object
        gp_obj_inv_mat = gp_obj.matrix_world.inverted_safe()
        if self.source_arm not in bpy.context.scene.objects:
            return {'FINISHED'}
        arm = bpy.context.scene.objects[self.source_arm]
        if self.source_type == 'OBJ' and self.source_obj not in bpy.context.scene.objects:
            return {'FINISHED'}
        if self.source_type == 'COLL' and self.source_coll not in bpy.data.collections:
            return {'FINISHED'}   
        
        t_mat, _ = get_transformation_mat(mode=context.scene.nijigp_working_plane,
                                                gp_obj=gp_obj, operator=self)
        bone_name_set = set()
        for bone in arm.data.bones:
            bone_name_set.add(bone.name)
            
        # Get all source objects
        if self.source_type == 'THIS':
            src_objs = get_generated_meshes(gp_obj)
        elif self.source_type == 'OBJ':
            src_objs = [bpy.context.scene.objects[self.source_obj]]
        elif self.source_type == 'COLL':
            source_coll = bpy.data.collections[self.source_coll]
            src_objs = [obj for obj in source_coll.objects if obj.type == 'MESH']

        if not self.weights_exist:
            switch_to([arm] + src_objs)
            bpy.ops.object.parent_set(type='ARMATURE_AUTO')
        switch_to([gp_obj])

        # Set up KDTree for vertex lookup
        num_verts = sum([len(obj.data.vertices) for obj in src_objs])
        kdt = kdtree.KDTree(num_verts)
        kdt_vert_lookup = []
        i = 0
        for obj in src_objs:
            trans = gp_obj_inv_mat @ obj.matrix_world
            if self.mapping_type == '2D':
                trans = t_mat.to_4x4() @ trans
            for v in obj.data.vertices:
                key = xy0(trans @ v.co) if self.mapping_type == '2D' else trans @ v.co
                kdt.insert(key, i)
                i += 1
                kdt_vert_lookup.append((obj,v))
        kdt.balance()
        
        # Get indices of bone groups of the Grease Pencil object. 
        gp_group_map = {}
        for group in gp_obj.vertex_groups:
            if group.name in bone_name_set:
                gp_group_map[group.name] = group.index
        # Create new groups when necessary
        for name in bone_name_set:
            if name not in gp_group_map:
                group = gp_obj.vertex_groups.new(name=name)
                gp_group_map[name] = group.index
                
        # Get target layers and frames
        target_layers = get_output_layers(gp_obj, self.target_mode)
        target_layers = [gp_obj.data.layers[i] for i in target_layers]
        frames_to_process = get_input_frames(gp_obj,
                                             multiframe = gp_obj.data.use_multiedit,
                                             layers = target_layers)
        # Transfer weights
        for frame in frames_to_process:
            for stroke in frame.strokes:
                for i,p in enumerate(stroke.points):
                    key = xy0(t_mat @ p.co) if self.mapping_type == '2D' else p.co
                    _,idx,_ = kdt.find(key)
                    obj, v = kdt_vert_lookup[idx]
                    for group in v.groups:
                        name = obj.vertex_groups[group.group].name
                        if name in gp_group_map:
                            stroke.points.weight_set(vertex_group_index=gp_group_map[name], 
                                                     point_index=i, 
                                                     weight=group.weight)
        
        # Add modifiers
        mod = gp_obj.grease_pencil_modifiers.new(name='nijigp_FromMesh', type='GP_ARMATURE')
        mod.object = arm
        mod.use_vertex_groups = True

        if self.adjust_parenting:
            if self.source_type == 'THIS':
                switch_to([gp_obj]+src_objs)
                bpy.ops.object.parent_set(type='OBJECT')
            switch_to([arm, gp_obj])
            bpy.ops.object.parent_set(type='OBJECT')
            switch_to([gp_obj])

        bpy.ops.object.mode_set(mode='WEIGHT_GPENCIL')
        return {'FINISHED'}

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self, width=500)

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "source_arm", icon='ARMATURE_DATA')
        layout.prop(self, "source_type")
        if self.source_type == 'OBJ':
            layout.prop(self, "source_obj", icon='OBJECT_DATA')
        elif self.source_type == 'COLL':
            layout.prop(self, "source_coll", icon='OUTLINER_COLLECTION')
        layout.prop(self, "weights_exist")
        layout.prop(self, "mapping_type")
        layout.prop(self, "target_mode")
        layout.prop(self, "adjust_parenting")

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
        switch_to([gp_obj])
        
        # Get output layers
        target_layers = get_output_layers(gp_obj, self.target_mode)
                    
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
        switch_to([gp_obj])

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
        switch_to([ref_gp_obj])
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
            switch_to([ref_gp_obj])
            
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
        switch_to([gp_obj])
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
        layout = self.layout
        layout.prop(self, "start_frame")
        layout.prop(self, "end_frame")
        layout.prop(self, "frame_step")
        layout.prop(self, "target_mode")
        layout.prop(self, "clear_parents")