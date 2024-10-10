import bpy

#region [Constants & Names]
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
    ops_map = {
                "gpencil.stroke_smooth": "grease_pencil.stroke_smooth",
                "gpencil.stroke_arrange": "grease_pencil.reorder"
                
                }
    if bpy.app.version >= (4, 3, 0) and ops in ops_map:
        return ops_map[ops]
    else:
        return ops
    
def get_viewport_bottom_offset():
    """Blender 4.3 adds an asset shelf region at the viewport bottom, therefore gizmos need an additional offset"""
    return 100 if bpy.app.version >= (4, 3, 0) else 0

#endregion


#region [Wrapped APIs]
def multiedit_enabled(gp_obj):
    if bpy.app.version >= (4, 3, 0):
        return bpy.context.scene.tool_settings.use_grease_pencil_multi_frame_editing
    else:
        return gp_obj.data.use_multiedit
    
def set_multiedit(gp_obj, enabled):
    if bpy.app.version >= (4, 3, 0):
        bpy.context.scene.tool_settings.use_grease_pencil_multi_frame_editing = enabled
    else:
        gp_obj.data.use_multiedit = enabled
#endregion


def register_alternative_api_paths():
    """
    Make it possible to interact with GPv2 objects in the same way as GPv3
    """
    bpy.types.GPencilLayer.current_frame = lambda self: self.active_frame
    bpy.types.GPencilLayer.matrix_local = property(lambda self: self.matrix_layer)
    
    bpy.types.GPencilFrame.drawing = property(lambda self: self)
    
    bpy.types.GPencilStroke.cyclic = property(lambda self: self.use_cyclic, lambda self, new_value: setattr(self, 'use_cyclic', new_value))
    
    bpy.types.GPencilStrokePoint.position = property(lambda self: self.co, lambda self, new_value: setattr(self, 'co', new_value))
    bpy.types.GPencilStrokePoint.radius = property(lambda self: self.pressure, lambda self, new_value: setattr(self, 'pressure', new_value))
    bpy.types.GPencilStrokePoint.opacity = property(lambda self: self.strength, lambda self, new_value: setattr(self, 'strength', new_value))
    
def unregister_alternative_api_paths():
    delattr(bpy.types.GPencilLayer, "current_frame")
    delattr(bpy.types.GPencilLayer, "matrix_local")
    
    delattr(bpy.types.GPencilFrame, "drawing")
    
    delattr(bpy.types.GPencilStroke, "cyclic")
    
    delattr(bpy.types.GPencilStrokePoint, "position")
    delattr(bpy.types.GPencilStrokePoint, "radius")
    delattr(bpy.types.GPencilStrokePoint, "opacity")