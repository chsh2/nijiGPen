import bpy
import random
import numpy as np
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

def get_mode_str(mode: str):
    if bpy.app.version >= (4, 3, 0):
        return mode.upper() + '_GREASE_PENCIL'
    else:
        return mode.upper() + '_GPENCIL'
    
def get_ops_str(ops: str):
    """Translate native GPv2 operator names to GPv3 ones"""
    if bpy.app.version >= (4, 3, 0) and bpy.app.version < (4, 4, 0) and ops == 'gpencil.stroke_sample':
        return None
    if bpy.app.version >= (4, 3, 0) and ops in ops_trans_map:
        return ops_trans_map[ops]
    else:
        return ops
    
def get_viewport_bottom_offset():
    """Blender 4.3 adds an asset shelf region at the viewport bottom, therefore gizmos need an additional offset"""
    return 100 if bpy.app.version >= (4, 3, 0) else 0

#endregion


#region [Wrapped APIs]
def obj_is_gp(obj):
    if bpy.app.version >= (4, 3, 0):
        return obj.type == "GREASEPENCIL"
    else:
        return obj.type == "GPENCIL"

def layer_locked(layer):
    # TODO: Blender 4.3 API cannot process nested groups. Should look back when 4.4 comes out
    if bpy.app.version >= (4, 3, 0) and layer.parent_group:
        return layer.lock and layer.parent_group.lock
    else:
        return layer.lock

def layer_hidden(layer):
    if bpy.app.version >= (4, 3, 0) and layer.parent_group:
        return layer.hide and layer.parent_group.hide
    else:
        return layer.hide
    
def new_gp_brush(name):
    """Copy an existing one if GPv2; create a new one if GPv3"""
    if bpy.app.version >= (4, 3, 0):
        res = bpy.data.brushes.new(name, mode=get_mode_str('PAINT'))
        res.gpencil_settings.vertex_color_factor = 1
    else:
        src = [brush for brush in bpy.data.brushes if brush.use_paint_grease_pencil and brush.gpencil_tool=='DRAW'][0]
        res = src.copy()
        res.name = name
    return res

def set_absolute_pressure(point, value, line_width = None):
    """GPv3 uses a single radius value that equals (line_width / 2000 * pressure) in GPv2"""
    if bpy.app.version >= (4, 3, 0):
        if not line_width:
            point.pressure = bpy.context.tool_settings.gpencil_paint.brush.unprojected_radius * value
        else:
            point.pressure = line_width / 2000.0 * value
    else:
        point.pressure = value
        
def op_arrange_stroke(direction):
    if bpy.app.version >= (4, 3, 0):
        bpy.ops.grease_pencil.reorder(direction=direction)
    else:
        bpy.ops.gpencil.stroke_arrange(direction=direction)
        
def op_deselect():
    if bpy.app.version >= (4, 3, 0):
        bpy.ops.grease_pencil.select_all(action='DESELECT')
    else:
        bpy.ops.gpencil.select_all(action='DESELECT')
#endregion

#region [Wrapper classes]
class LegacyPointRef:
    """
    TODO
    """
    def __init__(self, drawing, stroke_index, point_index):
        self._slice = drawing.strokes[stroke_index].points[point_index]
        
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
    TODO
    """
    def __init__(self, stroke):
        self._drawing = stroke._drawing
        self._stroke = stroke
        
    def __getitem__(self, key):
        return LegacyPointRef(self._drawing, self._stroke._index, key)    
    
    def __len__(self):
        return len(self._drawing.strokes[self._stroke._index].points)
    # def add
    # def foreach_get
            
class LegacyStrokeRef:
    """
    TODO
    """
    def __init__(self, drawing, identifier, initial_index):
        self._drawing = drawing
        self._hash = identifier
        self._index = initial_index

    def update_index(self):
        hashes = self._drawing.attributes['.nijigp_hash'].data
        if self._index < 0 or self._index >= len(hashes) or hashes[self._index] != self._hash:
            # TODO: Improve search efficiency
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

    def __eq__(self, other):
        return (self._drawing == other._drawing) and (self._hash == other._hash)
    
    def __hash__(self):
        return self._hash

    def __bool__(self):
        self.update_index()
        return self._index >= 0

    def __getattr__(self, name):
        self.update_index()

        if name == 'points':
            return LegacyPointCollection(self)
        elif name == 'hardness':
            return 1.0 - self._drawing.strokes[self._index].softness
        elif name == 'use_cyclic':
            return self._drawing.strokes[self._index].cyclic
        elif name == 'start_cap_mode':
            return 'ROUND' if self._drawing.strokes[self._index].start_cap == 0 else 'FLAT'
        elif name == 'end_cap_mode':
            return 'ROUND' if self._drawing.strokes[self._index].end_cap == 0 else 'FLAT'
        elif name == 'vertex_color_fill':
            return self._drawing.strokes[self._index].fill_color
        
        # The following properties do not exist in GPv3. Return a placeholder value instead.
        elif name == 'line_width':
            return 1
        elif name == 'is_nofill_stroke':
            return False
        elif name == 'select_index':
            return self._index
        
        else:
            return getattr(self._drawing.strokes[self._index], name)
    
    def __setattr__(self, name, value):
        writable = {'select', 'use_cyclic', 'is_nofill_stroke',
                    'material_index', 'vertex_color_fill', 'line_width', 'hardness',
                    'uv_rotation', 'uv_translation', 'uv_scale', 'start_cap_mode', 'end_cap_mode',
                    'fill_opacity',    # New attribute that does not exist in GPv2
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
        else:
            setattr(self._drawing.strokes[self._index], name, value)
        
class LegacyStrokeCollection:
    """
    TODO
    """
    def __init__(self, drawing):
        self._drawing = drawing
        self._hash_attr = drawing.attributes.new(".nijigp_hash", 'INT', 'CURVE') if '.nijigp_hash' not in drawing.attributes else drawing.attributes['.nijigp_hash']
        used = set()    # If strokes are duplicated, there might be identical hash values, which need reassignment
        for attr in self._hash_attr.data:
            if attr.value == 0 or attr.value in used:
                attr.value = random.randint(1, 2 ** 28)
            else:
                used.add(attr.value)

    def __getitem__(self, key):
        return LegacyStrokeRef(self._drawing, self._hash_attr.data[key].value, key)

    def __len__(self):
        return len(self._drawing.strokes)
       
    #def new()
    #def remove()

#endregion

def register_alternative_api_paths():
    """
    Create new APIs to make GPv3 compatible with GPv2
    """
    if bpy.app.version >= (4, 3, 0):
        bpy.types.GreasePencilv3.use_multiedit = property(lambda self: bpy.context.scene.tool_settings.use_grease_pencil_multi_frame_editing,
                                                        lambda self, value: setattr(bpy.context.scene.tool_settings, 'use_grease_pencil_multi_frame_editing', value))
        bpy.types.GreasePencilLayer.active_frame = property(lambda self: self.current_frame())
        bpy.types.GreasePencilLayer.matrix_layer = property(lambda self: self.matrix_local)
        bpy.types.GreasePencilLayer.use_mask_layer = property(lambda self: self.use_masks)
        bpy.types.GreasePencilLayer.info = property(lambda self: self.name, lambda self, value: setattr(self, 'name', value))
        bpy.types.GreasePencilFrame.strokes = property(lambda self: LegacyStrokeCollection(self.drawing))
        
    
def unregister_alternative_api_paths():
    if bpy.app.version >= (4, 3, 0):
        delattr(bpy.types.GreasePencilv3, "use_multiedit")
        delattr(bpy.types.GreasePencilLayer, "active_frame")
        delattr(bpy.types.GreasePencilLayer, "matrix_layer")
        delattr(bpy.types.GreasePencilLayer, "use_mask_layer")
        delattr(bpy.types.GreasePencilLayer, "info")
        delattr(bpy.types.GreasePencilFrame, "strokes")