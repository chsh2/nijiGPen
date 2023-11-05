import bpy
import numpy as np
from .common import *
from ..utils import *

def mix_color(rgb1, rgb2, factor, op):
    """Mixing either RGB or HSV colors"""
    if op in {'HUE', 'SATURATION', 'BRIGHTNESS'}:
        return mix_hsv(rgb1, rgb2, factor, {op})
    else:
        res = [0, 0, 0]
        for i in range(3):
            res[i] = mix_rgb(rgb1[i], rgb2[i], factor, op)
        return res

class TintSelectedOperator(bpy.types.Operator, ColorTintConfig):
    """Mixing the vertex color of selected points with another given color"""
    bl_idname = "gpencil.nijigp_color_tint"
    bl_label = "Tint Selected"
    bl_options = {'REGISTER', 'UNDO'}

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "tint_color")
        layout.prop(self, "tint_color_factor")
        layout.prop(self, "tint_mode")
        layout.prop(self, "blend_mode")

    def execute(self, context):
        if self.tint_color_factor == 0:
            return {'FINISHED'}
        gp_obj = context.object
        frames_to_process = get_input_frames(gp_obj, gp_obj.data.use_multiedit)
        for frame in frames_to_process:
            for stroke in frame.strokes:
                # Process fill color
                if self.tint_mode != 'LINE' and stroke.select:
                    r, g, b, _ = get_mixed_color(gp_obj, stroke, to_linear=True)
                    r, g, b = mix_color([r,g,b], self.tint_color, self.tint_color_factor, self.blend_mode)
                    stroke.vertex_color_fill = [r, g, b, 1]
                # Process line color
                if self.tint_mode != 'FILL':
                    for i,point in enumerate(stroke.points):
                        if point.select:
                            r, g, b, _ = get_mixed_color(gp_obj, stroke, i, to_linear=True)
                            r, g, b = mix_color([r,g,b], self.tint_color, self.tint_color_factor, self.blend_mode)
                            point.vertex_color = [r, g, b, 1]                        
        return {'FINISHED'}