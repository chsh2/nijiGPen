import bpy
import math
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

class VertexGroupClearOperator(bpy.types.Operator):
    """Clear all vertex groups and weights data"""
    bl_idname = "gpencil.nijigp_vertex_group_clear"
    bl_label = "Clear All Groups"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        num_groups = len(context.object.vertex_groups)
        context.object.vertex_groups.active_index = 0
        for _ in range(num_groups):
            # The option all=True removes only groups but not weights. Therefore, this new operator is needed
            bpy.ops.object.vertex_group_remove(all=False)
        return {'FINISHED'}

class PinRigOperator(bpy.types.Operator):
    """Using hint strokes to generate a pin-style armature and corresponding weights"""
    bl_idname = "gpencil.nijigp_rig_by_pin_hints"
    bl_label = "Generate Pins From Hints"
    bl_options = {'REGISTER', 'UNDO'}

    hint_layer: bpy.props.StringProperty(
        name='Hint Layer',
        description='Each stroke in this layer will be converted to a bone/pin',
        default='',
        search=lambda self, context, edit_text: [layer.info for layer in context.object.data.layers]
    )    
    target_mode: bpy.props.EnumProperty(            
        name='Target Layers',
        items=[ ('LAYER', 'Active Layer', ''),
               ('PASS', 'Active Layer Pass Index', ''),
               ('ALL', 'All Layers', ''),],
        default='ALL',
    ) 
    falloff_multiplier: bpy.props.FloatProperty(
        name='Falloff Distance',
        description='Weights of points decrease when they get farther from the bone. Larger value may lead to smoother rigging',
        default=1, min=0.1, max=5,
    ) 
    bone_style: bpy.props.EnumProperty(            
        name='Bone Style',
        items=[ ('HALF', 'Half Size', ''),
               ('FULL', 'Full Size', ''),],
        default='HALF',
    ) 
    bone_scale: bpy.props.FloatProperty(
        name='Bone Scale',
        default=1, min=0.1, max=5,
    )    
    bone_rotate: bpy.props.FloatProperty(
        name='Bone Rotation',
        default=0, min=-math.pi, max=math.pi, subtype='ANGLE'
    )  
    bone_set_parent: bpy.props.BoolProperty(
        name='Set Bone Parent',
        default=False,
        description='Determine the parent relation between bones according to the drawing sequence and locations of hint strokes'
    )  
    
    def execute(self, context):
        gp_obj: bpy.types.Object = context.object
        try:
            from ..solvers.graph import TriangleMst
        except:
            self.report({"ERROR"}, "Please install Scipy in the Preferences panel.")
        if len(self.hint_layer) < 1 or self.hint_layer not in gp_obj.data.layers:
            return {'FINISHED'}
        
        # Get frame and transform from the hint layer
        hint_frame = gp_obj.data.layers[self.hint_layer].active_frame
        num_pins = len(hint_frame.strokes)
        t_mat, _ = get_transformation_mat(mode=context.scene.nijigp_working_plane,
                                                gp_obj=gp_obj, strokes=hint_frame.strokes, operator=self)
        # Generate an armature
        arm_data = bpy.data.armatures.new(gp_obj.name_full + '_pins')
        arm_obj = bpy.data.objects.new(arm_data.name, arm_data)
        bpy.context.collection.objects.link(arm_obj)
        bpy.context.view_layer.objects.active = arm_obj
        bpy.ops.object.mode_set(mode='EDIT')
        
        # Generate each bone from hint strokes
        bones = []
        bone_centers = []
        bone_radiuses =[]
        for i in range(num_pins):
            hint_stroke = hint_frame.strokes[i]
            tail_pos:Vector = hint_stroke.points[0].co
            head_pos:Vector = hint_stroke.points[-1].co
            pin_radius = (tail_pos - head_pos).length / 2.0
            pin_direction = (tail_pos - head_pos).normalized()
            pin_center = (tail_pos + head_pos) / 2
            bone_centers.append(pin_center)
            bone_radiuses.append(pin_radius)

            bone = arm_data.edit_bones.new(name="Pin_"+str(i))
            bones.append(bone)
            rotation = Euler(t_mat @ Vector((0,0,self.bone_rotate)))
            pin_direction.rotate(rotation)
            pin_direction *= self.bone_scale
            arm_data.edit_bones[-1].head = pin_center - pin_direction * pin_radius * int(self.bone_style=='FULL')
            arm_data.edit_bones[-1].tail = pin_center + pin_direction * pin_radius 
                   
            if self.bone_set_parent:
                bone.use_connect = False
                min_dist = None
                for j in range(i):
                    if not min_dist or (pin_center - bone_centers[j]).length < min_dist:
                        min_dist = (pin_center - bone_centers[j]).length
                        bone.parent = bones[j]
        # Set vertex groups
        gp_group_indices = []
        for i in range(num_pins):
            for group in gp_obj.vertex_groups:
                if group.name == bones[i].name:
                    gp_group_indices.append(group.index)
                    break
            else:
                group = gp_obj.vertex_groups.new(name=bones[i].name)
                gp_group_indices.append(group.index)
        
        def set_weights(strokes):
            point_info = []
            co_list = []
            point_weight_groups = []
            
            # Build data structures for geometry lookup: KDTree, MST
            for stroke in strokes:
                for j,point in enumerate(stroke.points):
                    point_info.append((stroke,point,j))
                    co_list.append((t_mat @ point.co).xy)
                    point_weight_groups.append([])    
            kdt = kdtree.KDTree(len(point_info))
            for i,info in enumerate(point_info):
                kdt.insert(info[1].co, i)
            kdt.balance()

            tr_output = {}
            tr_output['vertices'], _, tr_output['triangles'], tr_output['orig_verts'],_,_ = geometry.delaunay_2d_cdt(co_list, [], [], 0, 1e-9)
            updated_idx_map = {}    # Delaunay method may delete or create points
            for new_idx,orig_verts in enumerate(tr_output['orig_verts']):
                for old_idx in orig_verts:
                    updated_idx_map[old_idx] = new_idx
            mst_builder = TriangleMst()
            mst_builder.build_mst(tr_output)
            
            # Process bones one by one
            for i in range(num_pins):
                # First, get all points inside this bone's radius
                search_dist = self.falloff_multiplier * bone_radiuses[i]
                search_queue = []
                search_queue_pointer = 0
                dist_map = {}
                res = kdt.find_range(co=bone_centers[i], radius=bone_radiuses[i])
                for _,p_idx,_ in res:
                    search_queue.append(updated_idx_map[p_idx])
                    dist_map[updated_idx_map[p_idx]] = 0
                # Then, propagate the weight along the spanning tree
                while search_queue_pointer < len(search_queue):
                    p_idx = search_queue[search_queue_pointer]
                    search_queue_pointer += 1
                    for next_idx in mst_builder.mst.getcol(p_idx).nonzero()[0]:
                        dist = mst_builder.mst[next_idx, p_idx] + dist_map[p_idx]
                        if next_idx not in dist_map and dist < search_dist:
                            dist_map[next_idx] = dist
                            search_queue.append(next_idx)
                    for next_idx in mst_builder.mst.getrow(p_idx).nonzero()[1]:
                        dist = mst_builder.mst[p_idx, next_idx] + dist_map[p_idx]
                        if next_idx not in dist_map and dist < search_dist:
                            dist_map[next_idx] = dist
                            search_queue.append(next_idx)  
                # Set weights according to the propagated distance
                for p_idx in dist_map:
                    weight = smoothstep(1 - dist_map[p_idx] / search_dist )
                    if weight > 1e-3:
                        for orig_idx in tr_output['orig_verts'][p_idx]:
                            point_info[orig_idx][0].points.weight_set(vertex_group_index=gp_group_indices[i], 
                                                            point_index=point_info[orig_idx][2], 
                                                            weight=weight)
                            point_weight_groups[orig_idx].append(gp_group_indices[i])

            # Lastly, process points without any weight assigned by copying from the point nearby
            for p_idx,groups in enumerate(point_weight_groups):
                if len(groups) == 0:
                    _, src_idx, _ = kdt.find(point_info[p_idx][1].co, filter=lambda i: len(point_weight_groups[i])>0)
                    if src_idx:
                        for group_idx in point_weight_groups[src_idx]:
                            weight = point_info[src_idx][0].points.weight_get(vertex_group_index=group_idx, 
                                                                            point_index=point_info[src_idx][2])
                            point_info[p_idx][0].points.weight_set(vertex_group_index=group_idx, 
                                                            point_index=point_info[p_idx][2], 
                                                            weight=weight)
        # Get target layers and frames
        bpy.ops.object.mode_set(mode='OBJECT')
        target_layers = get_output_layers(gp_obj, self.target_mode)
        target_layers = [gp_obj.data.layers[i] for i in target_layers]
        frames_to_process = get_input_frames(gp_obj,
                                             multiframe = gp_obj.data.use_multiedit,
                                             layers = target_layers,
                                             return_map = True)
        # Set weights for each frame number
        for frame_number, layer_frame_map in frames_to_process.items():
            strokes_to_process = []
            for i,item in layer_frame_map.items():
                frame = item[0]
                strokes_to_process += list(frame.strokes)
            set_weights(strokes_to_process)
        
        # Add modifiers
        mod = gp_obj.grease_pencil_modifiers.new(name='nijigp_Pins', type='GP_ARMATURE')
        mod.object = arm_obj
        mod.use_vertex_groups = True
        switch_to([arm_obj, gp_obj])
        bpy.ops.object.parent_set(type='OBJECT')
        switch_to([gp_obj])
        bpy.ops.object.mode_set(mode='WEIGHT_GPENCIL')
        return {'FINISHED'}

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self, width=300)

    def draw(self, context):
        layout = self.layout
        layout.label(text='Input Setting:')
        box1 = layout.box()
        box1.prop(self, "hint_layer", icon='OUTLINER_DATA_GP_LAYER')
        box1.prop(self, "target_mode")
        layout.label(text='Bone Setting:')
        box2 = layout.box()
        box2.prop(self, "falloff_multiplier")
        box2.prop(self, "bone_style")
        box2.prop(self, "bone_scale")
        box2.prop(self, "bone_rotate")
        box2.prop(self, "bone_set_parent")

class TransferWeightOperator(bpy.types.Operator):
    """Transfer bone weights from meshes to Grease Pencil strokes by looking for the nearest vertex"""
    bl_idname = "gpencil.nijigp_rig_by_transfer_weights"
    bl_label = "Transfer Bone Weights from Meshes"
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