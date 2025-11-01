import bpy
import random
from .resources import append_geometry_nodes
from mathutils import *

#region [Constants & Names]
ops_trans_map = {
    "gpencil.stroke_smooth": "grease_pencil.stroke_smooth",
    "gpencil.stroke_arrange": "grease_pencil.reorder"
}

def get_bl_context_str(mode: str):
    """Construct bl_context string required by panels"""
    if bpy.app.version >= (4, 3, 0):
        return 'grease_pencil_' + mode.lower()
    else:
        return 'greasepencil_' + mode.lower()

def get_ctx_mode_str(mode: str):
    if bpy.app.version >= (4, 3, 0):
        return mode.upper() + '_GREASE_PENCIL'
    else:
        return mode.upper() + '_GPENCIL'

def get_panel_str(prefix, suffix):
    if bpy.app.version >= (4, 3, 0):
        if suffix == 'vertexcolor':
            suffix = 'vertex_color'
        return f'{prefix.upper()}_grease_pencil_{suffix.lower()}'
    else:
        return f'{prefix.upper()}_gpencil_{suffix.lower()}'

def get_obj_mode_str(mode: str): 
    """Basically the same as context mode, except the EDIT mode"""
    if bpy.app.version >= (4, 3, 0) and mode == 'EDIT':
        return mode
    return get_ctx_mode_str(mode)
   
def get_ops_str(ops: str):
    """Translate native GPv2 operator names to GPv3 ones"""
    if bpy.app.version >= (4, 3, 0) and ops in ops_trans_map:
        return ops_trans_map[ops]
    else:
        return ops

def get_modifier_str(name: str):
    if bpy.app.version >= (4, 3, 0):
        return 'GREASE_PENCIL_' + name.upper()
    else:
        return 'GP_' + name.upper()   

def get_viewport_bottom_offset():
    """Blender 4.3 adds an asset shelf region at the viewport bottom, therefore gizmos need an additional offset"""
    return 100 if bpy.app.version >= (4, 3, 0) else 0

#endregion


#region [Wrapped APIs]
def blf_set_size(obj, fontid, size, resolution=72):
    # The third parameter became optional since Blender 3.4,
    # and was deprecated in Blender 4.0
    if bpy.app.version > (3, 6, 0):
        obj.size(fontid, size)
    else:
        obj.size(fontid, size, resolution)
    
def is_gpv3():
    return bpy.app.version >= (4, 3, 0)
    
def obj_is_gp(obj):
    if bpy.app.version >= (4, 3, 0):
        return obj.type == "GREASEPENCIL"
    else:
        return obj.type == "GPENCIL"

def get_gp_modifiers(obj):
    if bpy.app.version >= (4, 3, 0):
        return obj.modifiers
    else:
        return obj.grease_pencil_modifiers

def new_gp_brush(name):
    """Copy an existing one if GPv2, since files always have internal brushes; create a new one if GPv3 since it is easier"""
    if bpy.app.version >= (4, 3, 0):
        res = bpy.data.brushes.new(name, mode=get_ctx_mode_str('PAINT'))
        res.color = (0,0,0)
        res.gpencil_settings.vertex_color_factor = 1
        res.gpencil_settings.vertex_mode = 'BOTH'
        res.gpencil_settings.aspect = (1.0,1.0)
    else:
        src = [brush for brush in bpy.data.brushes if brush.use_paint_grease_pencil and brush.gpencil_tool=='DRAW'][0]
        res = src.copy()
        res.name = name
    return res

def layer_locked(layer):
    if bpy.app.version >= (4, 3, 0) and layer.parent_group:
        # Layer group is introduced in GPv3
        if layer.lock:
            return True
        g = layer.parent_group
        while g is not None:
            if g.lock:
                return True
            g = g.parent_group
        return False
    else:
        return layer.lock

def layer_hidden(layer):
    if bpy.app.version >= (4, 3, 0) and layer.parent_group:
        if layer.hide:
            return True
        g = layer.parent_group
        while g is not None:
            if g.hide:
                return True
            g = g.parent_group
        return False
    else:
        return layer.hide

def get_active_layer_index(obj):
    if bpy.app.version >= (4, 3, 0):
        for i,layer in enumerate(obj.data.layers):
            if obj.data.layers.active == layer:
                return i
        return -1
    else:
        return obj.data.layers.active_index
    
def set_active_layer_index(obj, index):
    if bpy.app.version >= (4, 3, 0):
        obj.data.layers.active == obj.data.layers[index]
    else:
        obj.data.layers.active_index = index

def get_multiedit(obj):
    if bpy.app.version >= (4, 3, 0):
        return bpy.context.scene.tool_settings.use_grease_pencil_multi_frame_editing
    else:
        return obj.data.use_multiedit

def set_multiedit(obj, enabled):
    if bpy.app.version >= (4, 3, 0):
        bpy.context.scene.tool_settings.use_grease_pencil_multi_frame_editing = enabled
    else:
        obj.data.use_multiedit = enabled

def get_layer_frame_by_number(layer, frame_number):
    if bpy.app.version >= (4, 3, 0):
        return layer.get_frame_at(frame_number)
    else:
        res = None
        for frame in layer.frames:
            if frame.frame_number <= frame_number and (res == None or res.frame_number <= frame.frame_number):
                res = frame
        return res

def is_frame_valid(frame):
    if bpy.app.version >= (4, 3, 0):
        return frame and hasattr(frame, "drawing") and frame.drawing
    else:
        return frame and hasattr(frame, "strokes")

def remove_frame(frames, frame):
    if bpy.app.version >= (4, 3, 0):
        frames.remove(frame.frame_number)
    else:
        frames.remove(frame)

def copy_frame(frames, src_frame, src_frame_number, new_frame_number):
    if bpy.app.version >= (4, 3, 0):
        dst_frame = frames.copy(src_frame_number, new_frame_number)
    else:
        dst_frame = frames.copy(src_frame)
        dst_frame.frame_number = new_frame_number
    return dst_frame

def new_active_frame(frames, frame_number):
    if bpy.app.version >= (4, 3, 0):
        return frames.new(frame_number)
    else:
        return frames.new(frame_number, active=True)

def set_point_radius(point, value, line_width = None):
    """GPv3 uses a single radius value that equals (line_width / 2000 * pressure) in GPv2"""
    if bpy.app.version >= (4, 3, 0):
        if not line_width:
            active_brush = bpy.context.tool_settings.gpencil_paint.brush
            if hasattr(active_brush, 'unprojected_radius'):
                point.pressure = active_brush.unprojected_radius * value if active_brush else 0.02 * value
            else:
                point.pressure = 0.5 * active_brush.unprojected_size * value if active_brush else 0.01 * value
        else:
            point.pressure = line_width / 2000.0 * value
    else:
        point.pressure = value

def get_point_radius(point, line_width = None):
    if bpy.app.version >= (4, 3, 0):
        return point.pressure
    else:
        if not line_width:
            return point.pressure * bpy.context.scene.tool_settings.gpencil_paint.brush.size / 2000.0
        else:
            return point.pressure * line_width / 2000.0

def op_modifier_apply(name):
    if bpy.app.version >= (4, 3, 0):
        bpy.ops.object.modifier_apply(modifier=name)
    else:
        bpy.ops.object.gpencil_modifier_apply(modifier=name)

def op_modifier_remove(name):
    if bpy.app.version >= (4, 3, 0):
        bpy.ops.object.modifier_remove(modifier=name)
    else:
        bpy.ops.object.gpencil_modifier_remove(modifier=name)

def op_layer_merge(mode):
    if bpy.app.version >= (4, 3, 0):
        bpy.ops.grease_pencil.layer_merge(mode=mode)
    else:
        bpy.ops.gpencil.layer_merge(mode=mode)
            
def op_arrange_stroke(direction):
    if bpy.app.version >= (4, 3, 0):
        bpy.ops.grease_pencil.reorder(direction=direction)
    else:
        bpy.ops.gpencil.stroke_arrange(direction=direction)

def op_join_strokes():
    if bpy.app.version >= (4, 3, 0):
        bpy.ops.grease_pencil.join_selection(type='JOIN')
    else:
        bpy.ops.gpencil.stroke_join()

def op_copy_strokes():
    if bpy.app.version >= (4, 3, 0):
        bpy.ops.grease_pencil.copy()
    else:
        bpy.ops.gpencil.copy()

def op_paste_strokes():
    if bpy.app.version >= (4, 3, 0):
        bpy.ops.grease_pencil.paste()
    else:
        bpy.ops.gpencil.paste()

def op_stroke_smooth(repeat):
    if repeat < 1:
        return
    if bpy.app.version >= (4, 3, 0):
        bpy.ops.grease_pencil.stroke_smooth(iterations=repeat, keep_shape=True)
    else:
        bpy.ops.gpencil.stroke_smooth(repeat=repeat)

def op_reproject():
    if bpy.app.version >= (4, 3, 0):
        bpy.ops.grease_pencil.reproject(type='VIEW')
    else:
        bpy.ops.gpencil.reproject(type='VIEW')

def op_select_all():
    if bpy.app.version >= (4, 3, 0):
        bpy.ops.grease_pencil.select_all(action='SELECT')
    else:
        bpy.ops.gpencil.select_all(action='SELECT')
                
def op_deselect():
    if bpy.app.version >= (4, 3, 0):
        bpy.ops.grease_pencil.select_all(action='DESELECT')
    else:
        bpy.ops.gpencil.select_all(action='DESELECT')
        
def op_select(location, extend):
    if bpy.app.version >= (4, 3, 0):
        bpy.ops.view3d.select(location=location, extend=extend)
    else:
        bpy.ops.gpencil.select(location=location, extend=extend)
        
def op_import_svg(filepath, directory, files, resolution, scale):
    if bpy.app.version >= (4, 3, 0):
        bpy.ops.wm.grease_pencil_import_svg("EXEC_DEFAULT", filepath=filepath, directory=directory, files=files, resolution=resolution, scale=scale)
    elif bpy.app.version > (3, 3, 0):
        bpy.ops.wm.gpencil_import_svg("EXEC_DEFAULT", filepath=filepath, directory=directory, files=files, resolution=resolution, scale=scale)
    else:
        bpy.ops.wm.gpencil_import_svg("EXEC_DEFAULT", filepath=filepath, resolution=resolution, scale=scale)
#endregion

#region [Point Wrapper Classes]
class LegacyPointRef:
    """
    A stroke point reference class that presents GPv3 point attributes in GPv2 style.
    Different from the stroke reference, this one aims at being lightweight and is not hashable nor persistent
    """
    def __init__(self, drawing, stroke_ref, stroke_index, point_index):
        self._slice = drawing.strokes[stroke_index].points[point_index]
        self._stroke = stroke_ref
        
    def __getattr__(self, name):
        if name == 'uv_fill':
            return 0
        elif name == 'uv_factor':
            return Vector((0, 0))
        elif name == 'co':
            return self._slice.position
        elif name == 'strength':
            return self._slice.opacity
        elif name == 'pressure':
            return self._slice.radius
        elif name == 'uv_rotation':
            return self._slice.rotation
        else:
            return getattr(self._slice, name)

    def __setattr__(self, name, value):
        writable = {'select', 'co', 'strength', 'pressure', 'uv_rotation', 'vertex_color'}
        if name not in writable:
            super().__setattr__(name, value)
            return
            
        if name == 'co':
            self._slice.position = value
            self._stroke._bound_box_max = None
            self._stroke._bound_box_min = None
        elif name == 'strength':
            self._slice.opacity = value
        elif name == 'pressure':
            self._slice.radius = value
        elif name == 'uv_rotation':
            self._slice.rotation = value
        else:
            setattr(self._slice, name, value)
        
class LegacyPointCollection:
    """
    Provide point creation and attribute assignment APIs in GPv2 style
    """
    def __init__(self, stroke):
        self._drawing = stroke._drawing
        self._stroke = stroke
        
    def __getitem__(self, key):
        if key >= len(self):
            raise IndexError()
        return LegacyPointRef(self._drawing, self._stroke, self._stroke._index, key)    
    
    def __len__(self):
        self._stroke.update_index()
        return len(self._drawing.strokes[self._stroke._index].points)
    
    def add(self, count):
        self._stroke.update_index()
        # Different from GPv2, a new stroke always has a point, which should not be added again
        real_count = count
        if self._drawing.attributes['.nijigp_new'].data[self._stroke._index].value:
            self._drawing.attributes['.nijigp_new'].data[self._stroke._index].value = False
            real_count = count - 1
        self._drawing.resize_strokes([len(self) + real_count], indices=(self._stroke._index,))
        # Initialize attributes
        for i in range(count):
            self[-1-i].strength = 1.0
            self[-1-i].vertex_color = (0, 0, 0, 0)
        
    def foreach_get(self, name, buffer):
        offset = self._stroke.get_offset()
        num_points = len(self._drawing.attributes['position'].data)
        if name == 'co':
            full_buffer = [0] * num_points * 3
            self._drawing.attributes['position'].data.foreach_get('vector', full_buffer)
            buffer[:len(self)*3] = full_buffer[offset*3:offset*3+len(self)*3]
        elif name == 'pressure':
            full_buffer = [0] * num_points
            self._drawing.attributes['radius'].data.foreach_get('value', full_buffer)
            buffer[:len(self)] = full_buffer[offset:offset+len(self)]
        elif name == 'strength':
            full_buffer = [0] * num_points
            self._drawing.attributes['opacity'].data.foreach_get('value', full_buffer)
            buffer[:len(self)] = full_buffer[offset:offset+len(self)]
        elif name == 'uv_rotation':
            full_buffer = [0] * num_points
            self._drawing.attributes['rotation'].data.foreach_get('value', full_buffer)
            buffer[:len(self)] = full_buffer[offset:offset+len(self)]
        elif name == 'vertex_color':
            full_buffer = [0] * num_points * 4
            self._drawing.attributes['vertex_color'].data.foreach_get('color', full_buffer)
            buffer[:len(self)*4] = full_buffer[offset*4:offset*4+len(self)*4]
            
    def foreach_set(self, name, buffer):
        offset = self._stroke.get_offset()
        num_points = len(self._drawing.attributes['position'].data)
        if name == 'co':
            full_buffer = [0] * num_points * 3
            self._drawing.attributes['position'].data.foreach_get('vector', full_buffer)
            full_buffer[offset*3:offset*3+len(self)*3] = buffer
            self._drawing.attributes['position'].data.foreach_set('vector', full_buffer)
            self._stroke._bound_box_max = None
            self._stroke._bound_box_min = None
        elif name == 'pressure':
            full_buffer = [0] * num_points
            self._drawing.attributes['radius'].data.foreach_get('value', full_buffer)
            full_buffer[offset:offset+len(self)] = buffer
            self._drawing.attributes['radius'].data.foreach_set('value', full_buffer)
        elif name == 'strength':
            full_buffer = [0] * num_points
            self._drawing.attributes['opacity'].data.foreach_get('value', full_buffer)
            full_buffer[offset:offset+len(self)] = buffer
            self._drawing.attributes['opacity'].data.foreach_set('value', full_buffer)
        elif name == 'uv_rotation':
            full_buffer = [0] * num_points
            self._drawing.attributes['rotation'].data.foreach_get('value', full_buffer)
            full_buffer[offset:offset+len(self)] = buffer
            self._drawing.attributes['rotation'].data.foreach_set('value', full_buffer)
        if name == 'vertex_color':
            full_buffer = [0] * num_points * 4
            self._drawing.attributes['vertex_color'].data.foreach_get('color', full_buffer)
            full_buffer[offset*4:offset*4+len(self)*4] = buffer
            self._drawing.attributes['vertex_color'].data.foreach_set('color', full_buffer)
    
    def weight_get(self, vertex_group_index, point_index):
        """Must set up a GPv3WeightHelper for the object first"""
        offset = self._stroke.get_offset()
        group_name = f'.nijigp_weight_proxy_{vertex_group_index}'
        if group_name not in self._drawing.attributes:
            return 0.0
        return self._drawing.attributes[group_name].data[offset + point_index].value
        
    def weight_set(self, vertex_group_index, point_index, weight):
        offset = self._stroke.get_offset()
        group_name = f'.nijigp_weight_proxy_{vertex_group_index}'
        if group_name not in self._drawing.attributes:
            return
        self._drawing.attributes[group_name].data[offset + point_index].value = weight

class GPv3WeightHelper:
    """
    APIs for setting vertex group weights have not been available until Blender 4.5;
    APIs for getting vertex group weights are still unavailable in GPv3.
    This class provides a workaround to update weights through Geometry Nodes.
    """
    def __init__(self, gp_obj):
        self.gp_obj: bpy.types.Object = gp_obj
        self.groups = []
        self.on = False
    
    def setup(self):
        """
        Use Geometry Nodes to mirror all group weights to a new attribute
        """
        if not is_gpv3():
            return
        modifiers = get_gp_modifiers(self.gp_obj)
        mod_name = 'nijigp_WeightProxy'
        current_mode = self.gp_obj.mode
        bpy.ops.object.mode_set(mode='OBJECT')
        
        for i,group in enumerate(self.gp_obj.vertex_groups):
            self.groups.append(group.name)
            # Create attributes for each frame in the object
            proxy = f'.nijigp_weight_proxy_{i}'
            for layer in self.gp_obj.data.layers:
                for frame in layer.frames:
                    frame.drawing.attributes.new(proxy, 'FLOAT', 'POINT')
            # Apply Geometry Nodes
            # TODO: Currently, applying this modifier to all frames may crash Blender. May revisit in the future
            mod = modifiers.new(name=mod_name, type='NODES')
            mod.node_group = append_geometry_nodes(bpy.context, 'NijiGP Weight Proxy')
            mod['Input_2'] = group.name
            mod['Input_3'] = proxy
            bpy.ops.object.modifier_move_to_index(modifier=mod.name, index=0)
            bpy.ops.object.modifier_apply("EXEC_DEFAULT", modifier=mod.name)
        bpy.ops.object.mode_set(mode=current_mode)    
        self.on = True

    def commit(self, abort=False):
        """
        Copy weights from the proxy attribute to the actual group, and clean up the proxy
        """
        if not is_gpv3() or not self.on:
            return
        modifiers = get_gp_modifiers(self.gp_obj)
        mod_name = 'nijigp_WeightProxy'
        current_mode = self.gp_obj.mode
        bpy.ops.object.mode_set(mode='OBJECT')
        
        for i,group in enumerate(self.groups):
            proxy = f'.nijigp_weight_proxy_{i}'
            # Apply Geometry Nodes to write values back
            if not abort:
                mod = modifiers.new(name=mod_name, type='NODES')
                mod.node_group = append_geometry_nodes(bpy.context, 'NijiGP Weight Proxy')
                mod['Input_2'] = proxy
                mod['Input_3'] = group
                bpy.ops.object.modifier_move_to_index(modifier=mod.name, index=0)
                bpy.ops.object.modifier_apply("EXEC_DEFAULT", modifier=mod.name)
            # Remove added attributes
            for layer in self.gp_obj.data.layers:
                for frame in layer.frames:
                    attr = frame.drawing.attributes[proxy]
                    frame.drawing.attributes.remove(attr)  
        bpy.ops.object.mode_set(mode=current_mode)
        self.groups = []
        self.on = False
    
    def __del__(self):
        self.commit(True)
        pass
        
#endregion

#region [Stroke Wrapper Classes]            
class LegacyStrokeRef:
    """
    A stroke reference class that provides features missing in GPv3 slice: 
        being comparable, hashable and capable of tracking stroke index changes
    """
    def __init__(self, drawing, identifier, initial_index):
        self._drawing = drawing
        self._hash = identifier
        self._index = initial_index
        self._bound_box_min = None
        self._bound_box_max = None

    def update_index(self):
        """Before any access, check if the stroke index has been changed. Index -1 means removal"""
        hashes = self._drawing.attributes['.nijigp_hash'].data
        if self._index < 0 or self._index >= len(hashes) or hashes[self._index] != self._hash:
            for i, attr in enumerate(hashes):
                if attr.value == self._hash:
                    self._index = i
                    return
        self._index = -1

    def get_offset(self):
        """Return the index of the stroke's first point"""
        self.update_index()
        return self._drawing.curve_offsets[self._index].value

    def get_slice(self):
        """Return the GPv3 reference object"""
        self.update_index()
        return self._drawing.strokes[self._index]        

    @property
    def points(self):
        return LegacyPointCollection(self)
    
    @property
    def bound_box_min(self):
        if self._bound_box_min == None:
            buffer = [0] * len(self.get_slice().points) * 3
            self.points.foreach_get('co', buffer)
            self._bound_box_min = Vector((min(buffer[::3]), min(buffer[1::3]), min(buffer[2::3])))
        return self._bound_box_min

    @property
    def bound_box_max(self):
        if self._bound_box_max == None:
            buffer = [0] * len(self.get_slice().points) * 3
            self.points.foreach_get('co', buffer)
            self._bound_box_max = Vector((max(buffer[::3]), max(buffer[1::3]), max(buffer[2::3])))
        return self._bound_box_max
    
    def __eq__(self, other):
        return (self._drawing == other._drawing) and (self._hash == other._hash)
    
    def __hash__(self):
        return self._hash

    def __bool__(self):
        self.update_index()
        return self._index >= 0

    def __getattr__(self, name):
        self.update_index()

        if name == 'hardness':
            return 1.0 - self._drawing.strokes[self._index].softness
        elif name == 'use_cyclic':
            return self._drawing.strokes[self._index].cyclic
        elif name == 'start_cap_mode':
            return 'ROUND' if self._drawing.strokes[self._index].start_cap == 0 else 'FLAT'
        elif name == 'end_cap_mode':
            return 'ROUND' if self._drawing.strokes[self._index].end_cap == 0 else 'FLAT'
        elif name == 'vertex_color_fill':
            return self._drawing.strokes[self._index].fill_color
        elif name == 'uv_scale':
            return self._drawing.attributes['uv_scale'].data[self._index].vector
        elif name == 'uv_rotation':
            return self._drawing.attributes['uv_rotation'].data[self._index].value
        elif name == 'uv_translation':
            return self._drawing.attributes['uv_translation'].data[self._index].vector
        
        # These properties are available from Blender 4.5
        elif name == 'is_nofill_stroke':
            if '.is_fill_guide' in self._drawing.attributes:
                return self._drawing.attributes['.is_fill_guide'].data[self._index].value
            return False
        elif name == 'aspect_ratio':
            if 'aspect_ratio' in self._drawing.attributes:
                return self._drawing.attributes['aspect_ratio'].data[self._index].value
            return 1.0
        
        # The following properties do not exist in GPv3. Return a placeholder value instead.
        elif name == 'line_width':
            return 1 if bpy.app.version < (5, 0, 0) else 0.5
        elif name == 'select_index':
            return self.select * (self._index + 1)
        
        else:
            return getattr(self._drawing.strokes[self._index], name)
    
    def __setattr__(self, name, value):
        writable = {'select', 'use_cyclic', 'is_nofill_stroke',
                    'material_index', 'vertex_color_fill', 'line_width', 'hardness',
                    'uv_rotation', 'uv_translation', 'uv_scale', 'start_cap_mode', 'end_cap_mode',
                    'fill_opacity', 'aspect_ratio'    # New attribute that does not exist in GPv2
                    }
        if name not in writable:
            super().__setattr__(name, value)
            return
            
        self.update_index()
        if name == 'hardness':
            self._drawing.strokes[self._index].softness = 1.0 - value
        elif name == 'use_cyclic':
            self._drawing.strokes[self._index].cyclic = value
        elif name == 'start_cap_mode':
            self._drawing.strokes[self._index].start_cap = 0 if value == 'ROUND' else 1
        elif name == 'end_cap_mode':
            self._drawing.strokes[self._index].end_cap = 0 if value == 'ROUND' else 1
        elif name == 'vertex_color_fill':
            self._drawing.strokes[self._index].fill_color = value
        elif name == 'uv_scale':
            self._drawing.attributes['uv_scale'].data[self._index].vector = value
        elif name == 'uv_rotation':
            self._drawing.attributes['uv_rotation'].data[self._index].value = value
        elif name == 'uv_translation':
            self._drawing.attributes['uv_translation'].data[self._index].vector = value
        elif name == 'aspect_ratio':
            if 'aspect_ratio' in self._drawing.attributes:
                self._drawing.attributes['aspect_ratio'].data[self._index].value = value
        elif name in {'line_width', 'is_nofill_stroke'}:
            return
        else:
            setattr(self._drawing.strokes[self._index], name, value)
        
class LegacyStrokeCollection:
    """
    Initialize several stroke/point attributes of a frame, including a stroke hash value as a persistent identifier.
    Also provides stroke creation/deletion APIs in GPv2 style
    """
    def __init__(self, frame):
        self._drawing = frame.drawing
        hash_attr = frame.drawing.attributes.new(".nijigp_hash", 'INT', 'CURVE') if '.nijigp_hash' not in frame.drawing.attributes else frame.drawing.attributes['.nijigp_hash']
        used = set()    # If strokes are duplicated, there might be identical hash values, which need reassignment
        for item in hash_attr.data:
            if item.value == 0 or item.value in used:
                item.value = random.randint(1, 2 ** 28)
            else:
                used.add(item.value)
                
        # Initialize some required attributes
        if '.nijigp_new' not in frame.drawing.attributes:
            frame.drawing.attributes.new(".nijigp_new", 'BOOLEAN', 'CURVE')
        if 'fill_opacity' not in frame.drawing.attributes:
            attr = frame.drawing.attributes.new("fill_opacity", 'FLOAT', 'CURVE')
            attr.data.foreach_set('value', [1.0] * len(attr.data))
        if 'uv_rotation' not in frame.drawing.attributes:
            attr = frame.drawing.attributes.new("uv_rotation", 'FLOAT', 'CURVE')
        if 'uv_scale' not in frame.drawing.attributes:
            attr = frame.drawing.attributes.new("uv_scale", 'FLOAT2', 'CURVE')
            attr.data.foreach_set('vector', [1.0] * len(attr.data) * 2)
        if 'uv_translation' not in frame.drawing.attributes:
            attr = frame.drawing.attributes.new("uv_translation", 'FLOAT2', 'CURVE')
            attr.data.foreach_set('vector', [1.0] * len(attr.data) * 2)
        if 'fill_color' not in frame.drawing.attributes:
            attr = frame.drawing.attributes.new("fill_color", 'FLOAT_COLOR', 'CURVE')
            attr.data.foreach_set('color', [0] * len(attr.data) * 4)
        if 'vertex_color' not in frame.drawing.attributes:
            attr = frame.drawing.attributes.new("vertex_color", 'FLOAT_COLOR', 'POINT')
            attr.data.foreach_set('color', [0] * len(attr.data) * 4)
                        
    def __getitem__(self, key):
        if key >= len(self):
            raise IndexError()
        hash_attr = self._drawing.attributes['.nijigp_hash']
        return LegacyStrokeRef(self._drawing, hash_attr.data[key].value, key)

    def __len__(self):
        return len(self._drawing.strokes)
       
    def new(self):
        # GPv3 stroke must have at least 1 point, while GPv2 has 0 when created
        self._drawing.add_strokes([1])
        key = len(self._drawing.strokes) - 1
        
        # Set initial attribute values
        self._drawing.attributes['.nijigp_hash'].data[key].value = random.randint(1, 2 ** 28)
        self._drawing.attributes['.nijigp_new'].data[key].value = True
        if 'u_scale' in self._drawing.attributes:
            self._drawing.attributes['u_scale'].data[key].value = 1.0
        if 'uv_scale' in self._drawing.attributes:
            self._drawing.attributes['uv_scale'].data[key].vector = (1.0, 1.0)
        if 'fill_opacity' in self._drawing.attributes:
            self._drawing.attributes['fill_opacity'].data[key].value = 1.0
        if 'aspect_ratio' in self._drawing.attributes:
            self._drawing.attributes['aspect_ratio'].data[key].value = 1.0
        return LegacyStrokeRef(self._drawing, self._drawing.attributes['.nijigp_hash'].data[key].value, key)

    def remove(self, stroke: LegacyStrokeRef):
        stroke.update_index()
        if stroke._index >= 0:
            self._drawing.remove_strokes(indices=(stroke._index,))
            stroke._index = -1

#endregion

def register_alternative_api_paths():
    """
    Create new APIs to make GPv3 compatible with GPv2
    """
    if bpy.app.version >= (4, 3, 0):
        bpy.types.GreasePencilLayer.active_frame = property(lambda self: self.current_frame())
        bpy.types.GreasePencilLayer.matrix_layer = property(lambda self: self.matrix_local)
        bpy.types.GreasePencilLayer.use_mask_layer = property(lambda self: self.use_masks)
        bpy.types.GreasePencilLayer.info = property(lambda self: self.name, lambda self, value: setattr(self, 'name', value))
        bpy.types.GreasePencilFrame.nijigp_strokes = property(lambda self: LegacyStrokeCollection(self))
    else:
        bpy.types.GPencilFrame.nijigp_strokes = property(lambda self: self.strokes)
    
def unregister_alternative_api_paths():
    if bpy.app.version >= (4, 3, 0):
        delattr(bpy.types.GreasePencilLayer, "active_frame")
        delattr(bpy.types.GreasePencilLayer, "matrix_layer")
        delattr(bpy.types.GreasePencilLayer, "use_mask_layer")
        delattr(bpy.types.GreasePencilLayer, "info")
        delattr(bpy.types.GreasePencilFrame, "nijigp_strokes")
    else:
        delattr(bpy.types.GPencilFrame, "nijigp_strokes")