import bpy

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