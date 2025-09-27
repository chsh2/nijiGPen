import bpy
import math
from mathutils import *
from .common import *
from ..utils import *
from ..api_router import *

def switch_to(objs):
    """Rig operations need frequent object re-selections"""
    bpy.ops.object.select_all(action='DESELECT')
    for obj in objs:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]

def get_output_layers(gp_obj, target_mode:str):
    """Common way of selecting output layers for rig operators. Return indices"""
    target_layers = []
    if target_mode == 'ALL':
        target_layers = [i for i,layer in enumerate(gp_obj.data.layers)]
    elif target_mode == 'PASS':
        if gp_obj.data.layers.active:
            active_pass_index = gp_obj.data.layers.active.pass_index
            target_layers = [i for i,layer in enumerate(gp_obj.data.layers) if layer.pass_index == active_pass_index]
    # May either be an active layer or an active layer group (for GPv3 only)
    elif target_mode == 'ACTIVE':
        if is_gpv3() and gp_obj.data.layer_groups.active:
            active_group = gp_obj.data.layer_groups.active
            target_layers = [i for i,layer in enumerate(gp_obj.data.layers) if is_layer_inside_group(layer, active_group)]
        else:
            target_layers = [i for i,layer in enumerate(gp_obj.data.layers) if layer == gp_obj.data.layers.active]
    return target_layers

class VertexGroupClearOperator(bpy.types.Operator):
    """Clear all vertex groups and weights data"""
    bl_idname = "gpencil.nijigp_vertex_group_clear"
    bl_label = "Clear All Groups"
    bl_options = {'REGISTER', 'UNDO'}

    remove_modifiers: bpy.props.BoolProperty(
        name='Remove Related Modifiers',
        default=False,
        description='Remove all armature modifiers and modifiers taking vertex groups as input'
    )  

    def draw(self, context):
        self.layout.prop(self, "remove_modifiers")

    def execute(self, context):
        # The option all=True removes only groups but not weights. Therefore, use a loop here.
        num_groups = len(context.object.vertex_groups)
        context.object.vertex_groups.active_index = 0
        for _ in range(num_groups):
            bpy.ops.object.vertex_group_remove(all=False)
        # Check each modifier
        if self.remove_modifiers:
            mods_to_remove = []
            for mod in get_gp_modifiers(context.object):
                if mod.type == get_modifier_str('ARMATURE'):
                    mods_to_remove.append(mod)
                elif hasattr(mod, 'vertex_group') and len(mod.vertex_group) > 0:
                    mods_to_remove.append(mod)
            for mod in mods_to_remove:
                get_gp_modifiers(context.object).remove(mod)

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
    hint_shape: bpy.props.EnumProperty(            
        name='Hint Shape',
        items=[ ('STICK', 'Stick', 'Draw a straight line indicating the bone shape'),
               ('LASSO', 'Lasso', 'Draw a circle to select points where weights should be assigned'),],
        default='STICK',
    )   
    target_mode: bpy.props.EnumProperty(            
        name='Target Layers',
        items=[ ('ACTIVE', 'Active Layer/Group', ''),
               ('PASS', 'Active Layer Pass Index', ''),
               ('ALL', 'All Layers', ''),],
        default='ALL',
    ) 
    rig_hints: bpy.props.BoolProperty(
        name='Rig Hints',
        default=False,
        description='Hint strokes are not rigged by default'
    )  
    rig_all: bpy.props.BoolProperty(
        name='Ensure Non-Zero Weights for All Points',
        default=True,
        description='Make sure that every target stroke is assigned at least one weight, even if it is far from any hint'
    ) 
    exclusive_binding: bpy.props.BoolProperty(
        name='Bind Single Bone to Stroke',
        default=False,
        description='Assign at most one weight group to all points of a specific stroke to avoid distortion'
    ) 
    falloff_multiplier: bpy.props.FloatProperty(
        name='Falloff Distance',
        description='Weights of points decrease when they get farther from the bone. Larger value may lead to smoother rigging',
        default=0.1, min=0.01, max=5,
    ) 
    bone_prefix: bpy.props.StringProperty(            
        name='Name Prefix',
        default='Pin',
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

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self, width=300)

    def draw(self, context):
        layout = self.layout
        layout.label(text='Input Setting:')
        box1 = layout.box()
        box1.prop(self, "hint_layer", icon='OUTLINER_DATA_GP_LAYER')
        box1.prop(self, "hint_shape")
        box1.prop(self, "target_mode")
        box1.prop(self, 'rig_hints')
        layout.label(text='Weight Calculation:')
        box2 = layout.box()
        box2.prop(self, 'rig_all')
        box2.prop(self, "exclusive_binding")
        box2.prop(self, "falloff_multiplier")
        layout.label(text='Bone Setting:')
        box3 = layout.box()
        box3.prop(self, "bone_prefix")
        box3.prop(self, "bone_style")
        box3.prop(self, "bone_scale")
        box3.prop(self, "bone_rotate")
        box3.prop(self, "bone_set_parent")  
          
    def execute(self, context):
        gp_obj: bpy.types.Object = context.object
        try:
            from ..solvers.graph import MstSolver
        except:
            self.report({"ERROR"}, "Please install Scipy in the Preferences panel.")
        if self.hint_shape == 'LASSO':
            try:
                import pyclipper
            except:
                self.report({"WARNING"}, "The lasso mode is available only when PyClipper is installed.")
                self.hint_shape = 'STICK'
        if len(self.hint_layer) < 1 or self.hint_layer not in gp_obj.data.layers:
            return {'FINISHED'}
        
        # Get frame and transform from the hint layer
        hint_frame = gp_obj.data.layers[self.hint_layer].active_frame
        num_pins = len(hint_frame.nijigp_strokes)
        t_mat, _ = get_transformation_mat(mode=context.scene.nijigp_working_plane,
                                                gp_obj=gp_obj, strokes=hint_frame.nijigp_strokes, operator=self,
                                                requires_layer=False)
        # Generate an armature
        arm_data = bpy.data.armatures.new(gp_obj.name + '_pins')
        arm_obj = bpy.data.objects.new(arm_data.name, arm_data)
        bpy.context.collection.objects.link(arm_obj)
        bpy.context.view_layer.objects.active = arm_obj
        bpy.ops.object.mode_set(mode='EDIT')
        
        # Generate each bone from hint strokes
        bones = []
        bone_centers = []
        bone_radiuses =[]
        for i in range(num_pins):
            hint_stroke = hint_frame.nijigp_strokes[i]
            if self.hint_shape == 'STICK':
                tail_pos:Vector = hint_stroke.points[0].co
                head_pos:Vector = hint_stroke.points[-1].co
                pin_radius = (tail_pos - head_pos).length / 2.0
                pin_direction = (tail_pos - head_pos).normalized()
                pin_center = (tail_pos + head_pos) / 2
            elif self.hint_shape == 'LASSO':
                tail_pos = hint_stroke.points[0].co
                pin_center = Vector((0,0,0))
                for point in hint_stroke.points:
                    pin_center += point.co
                pin_center /= len(hint_stroke.points)
                pin_radius = (tail_pos - pin_center).length
                pin_direction = (tail_pos - pin_center).normalized()
            bone_centers.append(pin_center)
            bone_radiuses.append(pin_radius)

            bone = arm_data.edit_bones.new(name=f"{self.bone_prefix}_{i}")
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
                    if not min_dist or (arm_data.edit_bones[i].head - arm_data.edit_bones[j].tail).length < min_dist:
                        min_dist = (arm_data.edit_bones[i].head - arm_data.edit_bones[j].tail).length
                        bone.parent = bones[j]
        if self.hint_shape == 'LASSO':
            lasso_polys, _, scale_factor = get_2d_co_from_strokes(hint_frame.nijigp_strokes, t_mat, scale=True)

        # Set vertex groups
        gp_group_indices = []
        for i in range(num_pins):
            for group in gp_obj.vertex_groups:
                if group.name == arm_data.edit_bones[i].name:
                    gp_group_indices.append(group.index)
                    break
            else:
                group = gp_obj.vertex_groups.new(name=bones[i].name)
                gp_group_indices.append(group.index)
        
        def set_weights(strokes):
            point_info = []
            co_list = []
            point_weight_groups = []
            stroke_weight_scores = []
            
            # Build data structures for geometry lookup: KDTree, MST
            for i,stroke in enumerate(strokes):
                for j,point in enumerate(stroke.points):
                    point_info.append((stroke,point,j,i))
                    co_list.append((t_mat @ point.co).xy)
                    point_weight_groups.append({})  
                    stroke_weight_scores.append({})  
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
            mst_builder = MstSolver()
            mst_builder.mst_from_triangles(tr_output)
            
            # Process bones one by one
            for i in range(num_pins):
                # First, get all points inside this hint's range
                search_dist = self.falloff_multiplier * bone_radiuses[i]
                search_queue = []
                search_queue_pointer = 0
                dist_map = {}
                # Use the radius to determine the range 
                if self.hint_shape == 'STICK':
                    res = kdt.find_range(co=bone_centers[i], radius=bone_radiuses[i])
                    for _,p_idx,_ in res:
                        search_queue.append(updated_idx_map[p_idx])
                        dist_map[updated_idx_map[p_idx]] = 0
                # Or check each point precisely
                elif self.hint_shape == 'LASSO':
                    for p_idx,info in enumerate(point_info):
                        if pyclipper.PointInPolygon(co_list[p_idx] * scale_factor, lasso_polys[i]) == 1:
                            search_queue.append(updated_idx_map[p_idx])
                            dist_map[updated_idx_map[p_idx]] = 0
                # Then, propagate the weight along the spanning tree until exceeding the falloff distance
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
                            point_weight_groups[orig_idx][gp_group_indices[i]] = weight
                            # Count the weight of this bone at the stroke level
                            stroke_idx = point_info[orig_idx][3]
                            if i not in stroke_weight_scores[stroke_idx]:
                                stroke_weight_scores[stroke_idx][i] = 0
                            stroke_weight_scores[stroke_idx][i] += weight
            # Process points without any weight assigned by copying from the point nearby
            new_point_weight_groups = {}
            for p_idx,groups in enumerate(point_weight_groups):
                if len(groups) == 0:
                    tmp_group = {}
                    if self.rig_all:
                        _, src_idx, _ = kdt.find(point_info[p_idx][1].co, filter=lambda i: len(point_weight_groups[i])>0)
                    else:   # Limit the search range inside the same stroke
                        _, src_idx, _ = kdt.find(point_info[p_idx][1].co, 
                                                 filter=lambda i: len(point_weight_groups[i])>0 and point_info[p_idx][3]==point_info[i][3])
                    if src_idx:
                        for group_idx in point_weight_groups[src_idx]:
                            weight = point_weight_groups[src_idx][group_idx]
                            tmp_group[group_idx] = weight

                            stroke_idx = point_info[p_idx][3]
                            if group_idx not in stroke_weight_scores[stroke_idx]:
                                stroke_weight_scores[stroke_idx][group_idx] = 0
                            stroke_weight_scores[stroke_idx][group_idx] += weight

                        new_point_weight_groups[p_idx] = tmp_group
            for p_idx in new_point_weight_groups:
                point_weight_groups[p_idx] = new_point_weight_groups[p_idx]
                            
            # Find the most important bone for each stroke and remove all other weights
            if self.exclusive_binding:
                stroke_weight_group = []
                for i,stroke in enumerate(strokes):
                    if len(stroke_weight_scores[i]) > 0:
                        stroke_weight_group.append(max(stroke_weight_scores[i], key=stroke_weight_scores[i].get))
                    else:
                        stroke_weight_group.append(None)
            # Set the final weights in Blender
            for p_idx,info in enumerate(point_info):
                stroke, stroke_idx = info[0], info[3]
                if self.exclusive_binding and stroke_weight_group[stroke_idx] != None:
                    point_weight_groups[p_idx] = {stroke_weight_group[stroke_idx]: 1.0}
                for i in point_weight_groups[p_idx]:
                    stroke.points.weight_set(vertex_group_index=i, 
                                                    point_index=info[2], 
                                                    weight=point_weight_groups[p_idx][i])
        
        # Get target layers and frames
        target_layers = get_output_layers(gp_obj, self.target_mode)
        target_layers = [gp_obj.data.layers[i] for i in target_layers]
        if not self.rig_hints and gp_obj.data.layers[self.hint_layer] in target_layers:
            target_layers.remove(gp_obj.data.layers[self.hint_layer])
        frames_to_process = get_input_frames(gp_obj,
                                             multiframe = get_multiedit(gp_obj),
                                             layers = target_layers,
                                             return_map = True)
        # Lock non-target layers to protect them
        layers_lock_status = [layer.lock for layer in gp_obj.data.layers]
        for i,layer in enumerate(gp_obj.data.layers):
            if layer not in target_layers:
                layer.lock = True

        # Set armature relationship
        bpy.ops.object.mode_set(mode='OBJECT')
        switch_to([arm_obj, gp_obj])
        bpy.ops.object.parent_set(type='ARMATURE')
        # For GPv3, weights must be initialized by a native operator first
        if is_gpv3():
            for frame_number in frames_to_process:
                context.scene.frame_set(frame_number)
                bpy.ops.object.parent_set(type='ARMATURE_AUTO')
        switch_to([gp_obj])
        
        # Set weights for each frame number
        weight_helper = GPv3WeightHelper(gp_obj)
        current_frame_number = context.scene.frame_current
        for frame_number, layer_frame_map in frames_to_process.items():
            context.scene.frame_set(frame_number)
            strokes_to_process = []
            for i,item in layer_frame_map.items():
                frame = item[0]
                strokes_to_process += list(frame.nijigp_strokes)
            weight_helper.setup()
            set_weights(strokes_to_process)
            weight_helper.commit()
        context.scene.frame_set(current_frame_number)
        
        # Recover the state
        for i,layer in enumerate(gp_obj.data.layers):
            layer.lock = layers_lock_status[i]
        bpy.ops.object.mode_set(mode=get_obj_mode_str('WEIGHT'))
        return {'FINISHED'}

class MeshFromArmOperator(bpy.types.Operator):
    """Use smart fill to generate a temporary mesh for later rigging operations"""
    bl_idname = "gpencil.nijigp_mesh_from_bones"
    bl_label = "Generate Meshes From Armature"
    bl_options = {'REGISTER', 'UNDO'}

    source_arm: bpy.props.StringProperty(
        name='Armature',
        description='The armature that meshes and Grease Pencil strokes are or will be attached to',
        default='',
        search=lambda self, context, edit_text: [obj.name for obj in bpy.context.scene.objects if obj.type=='ARMATURE']
    ) 
    resolution: bpy.props.IntProperty(
        name='Resolution',
        description='The resolution to generate hint strokes from bones and generate triangle faces',
        default=10, min=3, max=20,
    ) 

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "source_arm", icon='ARMATURE_DATA')
        layout.prop(self, "resolution")

    def execute(self, context):
        if self.source_arm not in bpy.context.scene.objects:
            return {'FINISHED'}
        gp_obj: bpy.types.Object = context.object
        current_mode = gp_obj.mode
        gp_obj_inv_mat = gp_obj.matrix_world.inverted_safe()
        arm = bpy.context.scene.objects[self.source_arm]    

        # Generate temporary layers for fills and hints
        fill_layer = gp_obj.data.layers.new("tmp_fill", set_active=False)
        fill_frame = new_active_frame(fill_layer.frames, context.scene.frame_current)
        hint_layer = gp_obj.data.layers.new("tmp_hint", set_active=False)
        hint_frame = new_active_frame(hint_layer.frames, context.scene.frame_current)

        # Convert bones to hint strokes
        arm_trans = gp_obj_inv_mat @ arm.matrix_world
        for bone in arm.data.bones:
            head:Vector = arm_trans @ bone.head_local
            tail:Vector = arm_trans @ bone.tail_local
            stroke = hint_frame.nijigp_strokes.new()
            stroke.points.add(self.resolution)
            for i in range(self.resolution):
                factor = i / (self.resolution - 1.0)
                stroke.points[i].co = head * (1 - factor) + tail * factor
        bpy.ops.object.mode_set(mode=get_obj_mode_str('EDIT'))
        op_select_all()
        if not is_gpv3():
            bpy.ops.gpencil.recalc_geometry()

        # Generate fills and meshes using other operators
        multiframe = get_multiedit(gp_obj)
        set_multiedit(gp_obj, False)
        if is_gpv3() and gp_obj.data.layer_groups.active:
            line_layer_name = gp_obj.data.layer_groups.active.name
        else:
            line_layer_name = gp_obj.data.layers.active.info
        bpy.ops.gpencil.nijigp_smart_fill(line_layer = line_layer_name,
                                          hint_layer = hint_layer.info,
                                          fill_layer = fill_layer.info,
                                          precision = 0.05,
                                          material_mode = 'HINT')
        op_deselect()
        for stroke in fill_frame.nijigp_strokes:
            stroke.select = True
        bpy.ops.gpencil.nijigp_mesh_generation_normal(use_native_triangulation = True,
                                                      resolution = self.resolution * 2)
        # Cleanup temporary layers
        set_multiedit(gp_obj, multiframe)
        gp_obj.data.layers.remove(fill_layer)
        gp_obj.data.layers.remove(hint_layer)
        bpy.ops.object.mode_set(mode=current_mode)
        return {'FINISHED'}

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
        search=lambda self, context, edit_text: [coll.name for coll in bpy.context.scene.collection.children_recursive]
    ) 
    mapping_type: bpy.props.EnumProperty(            
        name='Mapping',
        items=[ ('2D', '2D Nearest Vertex', ''),
               ('3D', '3D Nearest Vertex', ''),],
        default='2D',
    )
    target_mode: bpy.props.EnumProperty(            
        name='Target Layers',
        items=[ ('ACTIVE', 'Active Layer/Group', ''),
               ('PASS', 'Active Layer Pass Index', ''),
               ('ALL', 'All Layers', ''),],
        default='ALL',
    )
    auto_mesh: bpy.props.BoolProperty(            
        name='Quick New Mesh Using Smart Fill',
        default=False,
        description='Use the bones as hint strokes to generate new meshes based on the active layer and frame'
    )
    weights_exist: bpy.props.BoolProperty(            
        name='Use Existing Weights',
        default=False,
        description='If not checked, generate automatic weights for the meshes first'
    )
    
    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self, width=500)

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "source_arm", icon='ARMATURE_DATA')
        layout.prop(self, "source_type")
        if self.source_type == 'THIS':
            layout.prop(self, 'auto_mesh')
        elif self.source_type == 'OBJ':
            layout.prop(self, "source_obj", icon='OBJECT_DATA')
        elif self.source_type == 'COLL':
            layout.prop(self, "source_coll", icon='OUTLINER_COLLECTION')
        layout.prop(self, "weights_exist")
        layout.prop(self, "mapping_type")
        layout.prop(self, "target_mode")

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
                                                gp_obj=gp_obj, operator=self,
                                                requires_layer=False)
        bone_name_set = set()
        for bone in arm.data.bones:
            bone_name_set.add(bone.name)

        # Remove conflicting modifiers
        mods_to_remove = []
        for mod in get_gp_modifiers(gp_obj):
            if mod.type == get_modifier_str('ARMATURE') and mod.object == arm:
                mods_to_remove.append(mod)
        for mod in mods_to_remove:
            get_gp_modifiers(gp_obj).remove(mod)        
                
        # Get all source objects
        if self.source_type == 'THIS':
            if self.auto_mesh:
                bpy.ops.gpencil.nijigp_mesh_from_bones(source_arm = self.source_arm)
            src_objs = get_generated_meshes(gp_obj)
        elif self.source_type == 'OBJ':
            src_objs = [bpy.context.scene.objects[self.source_obj]]
        elif self.source_type == 'COLL':
            source_coll = bpy.data.collections[self.source_coll]
            src_objs = [obj for obj in source_coll.objects if obj.type == 'MESH']
        if len(src_objs) < 1:
            return {'FINISHED'}

        if not self.weights_exist:
            switch_to([arm] + src_objs)
            bpy.ops.object.parent_set(type='ARMATURE_AUTO')

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
                                             multiframe = get_multiedit(gp_obj),
                                             layers = target_layers)
        # Lock non-target layers to protect them
        layers_lock_status = [layer.lock for layer in gp_obj.data.layers]
        for i,layer in enumerate(gp_obj.data.layers):
            if layer not in target_layers:
                layer.lock = True
                
        # Set armature relationship
        switch_to([arm, gp_obj])
        bpy.ops.object.parent_set(type='ARMATURE_AUTO' if is_gpv3() else 'ARMATURE')
        # For GPv3, weights must be initialized by a native operator first
        if is_gpv3():
            for frame_number in set([frame.frame_number for frame in frames_to_process]):
                context.scene.frame_set(frame_number)
                bpy.ops.object.parent_set(type='ARMATURE_AUTO')
        switch_to([gp_obj])
        
        # Transfer weights
        weight_helper = GPv3WeightHelper(gp_obj)
        current_frame_number = context.scene.frame_current
        for frame in frames_to_process:
            context.scene.frame_set(frame.frame_number)
            weight_helper.setup()
            for stroke in frame.nijigp_strokes:
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
            weight_helper.commit()
        context.scene.frame_set(current_frame_number)
        
        # Recover the state
        for i,layer in enumerate(gp_obj.data.layers):
            layer.lock = layers_lock_status[i]
        bpy.ops.object.mode_set(mode=get_obj_mode_str('WEIGHT'))
        return {'FINISHED'}

class LayersToGroupsOperator(bpy.types.Operator):
    """For each layer, generates a vertex group with the same name, and assign weights to all frames of that layer"""
    bl_idname = "gpencil.nijigp_layers_to_groups"
    bl_label = "Layers to Groups"
    bl_options = {'REGISTER', 'UNDO'}

    ignore_layer_groups: bpy.props.BoolProperty(
        name='Ignore Layer Groups',
        default=False,
        description='Do not generate vertex groups for layer groups'
    )

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "ignore_layer_groups")

    def execute(self, context):
        gp_obj = context.object

        # Get all vertex group names to generate
        layers = [layer for layer in gp_obj.data.layers if not is_layer_protected(layer)]
        groups = set()
        layer_group_map = {layer.info:[] for layer in layers}
        if is_gpv3() and not self.ignore_layer_groups:
            for layer in layers:
                p = layer
                while p.parent_group is not None:
                    groups.add(p.parent_group)
                    layer_group_map[layer.info].append(p.parent_group.name)
                    p = p.parent_group

        # Get all frames to process
        frames_by_number = {}
        for layer in layers:
            for frame in layer.frames:
                if frame.frame_number not in frames_by_number:
                    frames_by_number[frame.frame_number] = []
                frames_by_number[frame.frame_number].append((layer, frame))

        # Generate a temporary armature to assist assigning weights and avoid possible bugs in GPv3
        if is_gpv3():
            arm_data = bpy.data.armatures.new('nijigp_tmp_armature')
            arm_obj = bpy.data.objects.new(arm_data.name, arm_data)
            bpy.context.collection.objects.link(arm_obj)
            bpy.context.view_layer.objects.active = arm_obj
            bpy.ops.object.mode_set(mode='EDIT')
            for name in [l.info for l in layers]+[g.name for g in groups]:
                eb = arm_data.edit_bones.new(name=name)
                eb.head = Vector((0,0,0))
                eb.tail = Vector((0,.1,0))

            bpy.ops.object.mode_set(mode='OBJECT')
            switch_to([arm_obj, gp_obj])
            for frame_number in frames_by_number:
                context.scene.frame_set(frame_number)
                bpy.ops.object.parent_set(type='ARMATURE_AUTO')
            switch_to([gp_obj])
        # In GPv2, simply create vertex groups
        else:
            for name in [l.info for l in layers]:
                if name not in gp_obj.vertex_groups:
                    gp_obj.vertex_groups.new(name=name)

        # Set weights
        weight_helper = GPv3WeightHelper(gp_obj)
        current_frame_number = context.scene.frame_current
        for frame in frames_by_number:
            context.scene.frame_set(frame)
            weight_helper.setup()
            for layer, f in frames_by_number[frame]:
                group_indices = [gp_obj.vertex_groups[layer.info].index]
                if layer.info in layer_group_map:
                    for gname in layer_group_map[layer.info]:
                        group_indices.append(gp_obj.vertex_groups[gname].index)
                for stroke in get_input_strokes(gp_obj, f, select_all=True):
                    for i,p in enumerate(stroke.points):
                        for gi in group_indices:
                            stroke.points.weight_set(vertex_group_index=gi, point_index=i, weight=1.0)
            weight_helper.commit()
        context.scene.frame_set(current_frame_number)

        # Cleanup
        if is_gpv3():
            mods = get_gp_modifiers(gp_obj)
            for mod in mods:
                if mod.type == get_modifier_str('ARMATURE') and mod.object == arm_obj:
                    mods.remove(mod)
                    break
            bpy.ops.object.mode_set(mode='OBJECT')
            switch_to([arm_obj])
            bpy.ops.object.delete(use_global=True)
            switch_to([gp_obj])

        return {'FINISHED'}

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
        items=[ ('ACTIVE', 'Active Layer/Group', ''),
               ('PASS', 'Active Layer Pass Index', ''),
               ('ALL', 'All Layers', ''),],
        default='ALL',
    )
    clear_parents: bpy.props.BoolProperty(
        name='Clear Parents',
        default=False,
        description='Clear the parents of the active Grease Pencil object'
    )

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
            last_frame_number = None
            for frame_number in range(1, self.end_frame + 1):
                if frame_number in layer_get_keyframe[i]:
                    last_frame = layer_get_keyframe[i][frame_number]
                    last_frame_number = last_frame.frame_number
                elif last_frame != None and is_target_frame_number(frame_number):
                    layer_get_keyframe[i][frame_number] = copy_frame(layer.frames, last_frame, last_frame_number, frame_number)

        # Get all armature modifiers
        target_modifiers = []
        for mod in get_gp_modifiers(gp_obj):
            if mod.type == get_modifier_str('ARMATURE') and mod.object:
                target_modifiers.append(mod.name)
                
        # Process by frame number
        switch_to([ref_gp_obj])
        for frame_number in target_frame_numbers:
            context.scene.frame_set(frame_number)
            
            # Duplicate the object to apply all armature modifiers, and record coordinate changes
            bpy.ops.object.duplicate()
            dup_gp_obj = bpy.context.object
            new_coordinates = {}    # 4D list: layer->stroke->point->xyz
            for mod_name in target_modifiers:
                op_modifier_apply(mod_name)
            for i in target_layers:
                layer = dup_gp_obj.data.layers[i]
                new_coordinates[i] = []
                if frame_number not in layer_get_ref_keyframe_idx[i]:
                    continue
                frame = layer.frames[layer_get_ref_keyframe_idx[i][frame_number]]
                for stroke in frame.nijigp_strokes:
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
                frame = [frame for frame in layer.frames if frame.frame_number==frame_number][0]
                for j,stroke in enumerate(frame.nijigp_strokes):
                    for k,point in enumerate(stroke.points):
                        point.co = new_coordinates[i][j][k]
        # Cleanup
        bpy.ops.object.delete(use_global=True)
        switch_to([gp_obj])
        for mod_name in target_modifiers:
            op_modifier_remove(mod_name)
        if self.clear_parents:
            bpy.ops.object.parent_clear(type='CLEAR')
        bpy.ops.object.mode_set(mode=get_obj_mode_str('EDIT'))
        return {'FINISHED'}