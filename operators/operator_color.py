import bpy
import numpy as np
from .common import *
from ..utils import *
from ..api_router import *

def mix_color(rgb1, rgb2, factor, op):
    """Mixing either RGB or HSV colors"""
    if op in {'HUE', 'SATURATION', 'BRIGHTNESS'}:
        return mix_hsv(rgb1, rgb2, factor, {op})
    else:
        res = [0, 0, 0]
        for i in range(3):
            res[i] = mix_rgb(rgb1[i], rgb2[i], factor, op)
        return res

def is_fill_tintable(operator, gp_obj, stroke):
    """Determine if a stroke's vertex_color_fill should be considered in an operator"""
    gp_mat = gp_obj.material_slots[stroke.material_index].material
    if not gp_mat:
        return False
    return (operator.tint_mode != 'LINE' and stroke.select 
            and gp_mat.grease_pencil.show_fill and not gp_mat.grease_pencil.use_fill_holdout)

def is_point_tintable(operator, gp_obj, stroke, point):
    """Determine if a stroke point's vertex_color should be considered in an operator"""
    gp_mat = gp_obj.material_slots[stroke.material_index].material
    if not gp_mat:
        return False
    return (operator.tint_mode != 'FILL' and point.select 
            and gp_mat.grease_pencil.show_stroke and not gp_mat.grease_pencil.use_stroke_holdout)
    
class TintSelectedOperator(bpy.types.Operator, ColorTintConfig, NoiseConfig):
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
        layout.label(text="Noise:")
        row = layout.row()
        row.prop(self, "random_scale")
        row.prop(self, "random_seed")
        row = layout.row()
        row.prop(self, "random_factor")

    def execute(self, context):
        if self.tint_color_factor == 0:
            return {'FINISHED'}
        gp_obj = context.object
        frames_to_process = get_input_frames(gp_obj, get_multiedit(gp_obj))
        noise.seed_set(self.random_seed)
        for frame in frames_to_process:
            for stroke in get_input_strokes(gp_obj, frame):
                if len(stroke.points) < 1:
                    continue
                # Process fill color
                if is_fill_tintable(self, gp_obj, stroke):
                    r, g, b, _ = get_mixed_color(gp_obj, stroke, to_linear=True)
                    factor = self.tint_color_factor + self.random_factor * noise.noise_vector(stroke.points[0].co * self.random_scale)[0]
                    r, g, b = mix_color([r,g,b], self.tint_color, np.clip(factor,0,1), self.blend_mode)
                    stroke.vertex_color_fill = [r, g, b, 1]
                # Process line color
                if self.tint_mode != 'FILL':
                    for i,point in enumerate(stroke.points):
                        if is_point_tintable(self, gp_obj, stroke, point):
                            r, g, b, _ = get_mixed_color(gp_obj, stroke, i, to_linear=True)
                            factor = self.tint_color_factor + self.random_factor * noise.noise_vector(point.co * self.random_scale)[0]
                            r, g, b = mix_color([r,g,b], self.tint_color, np.clip(factor,0,1), self.blend_mode)
                            point.vertex_color = [r, g, b, 1]                        
        return {'FINISHED'}
    
class RecolorSelectedOperator(bpy.types.Operator, ColorTintConfig, NoiseConfig):
    """Recolor selected strokes/points according to a palette"""
    bl_idname = "gpencil.nijigp_recolor"
    bl_label = "Recolor Selected"
    bl_options = {'REGISTER', 'UNDO'}

    mapping_criterion: bpy.props.EnumProperty(            
        name='Criterion',
        items=[ ('RGB', 'RGB Similarity', ''),
               ('HSV', 'HSV Similarity', ''),],
        default='RGB',
        description='How to map colors to the new palette'
    )
    normalize: bpy.props.FloatProperty(
        name='Normalize',
        default=0, min=0, max=1,
        description='Color selected strokes to have similar mean and variance in each color channel with the palette'
    )    
    use_h: bpy.props.BoolProperty(default=True)   
    use_s: bpy.props.BoolProperty(default=True) 
    use_v: bpy.props.BoolProperty(default=True)  
    keep_v: bpy.props.FloatProperty(
        default=0, min=0, max=1,
        description='Keep the brightness of the original color'
    )
    keep_s: bpy.props.FloatProperty(
        default=0, min=0, max=1,
        description='Keep the saturation of the original color'
    )
    palette_name: bpy.props.StringProperty(
        name='Palette',
        default='',
        search=lambda self, context, edit_text: [palette.name for palette in bpy.data.palettes if palette]
    )

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "palette_name", icon='COLOR')
        layout.prop(self, "mapping_criterion")
        box = layout.box()
        if self.mapping_criterion == 'HSV':
            row = box.row()
            row.prop(self, "use_h", text='Hue')
            row.prop(self, "use_s", text='Saturation')
            row.prop(self, "use_v", text='Brightness')
        box.prop(self, "normalize")
        layout.prop(self, "tint_mode")
        layout.label(text="Preserve: ")
        row = layout.row()
        row.prop(self, "keep_s", text="Saturation")
        row.prop(self, "keep_v", text="Brightness")
        layout.label(text="Randomization: ")
        row = layout.row()
        row.prop(self, "random_factor", text="Factor")
        row.prop(self, "random_seed", text="Seed")        

    def execute(self, context):
        from colorsys import rgb_to_hsv, hsv_to_rgb
        if self.palette_name not in bpy.data.palettes:
            return {'FINISHED'}
        if self.mapping_criterion == 'HSV' and not self.use_h and not self.use_s and not self.use_v:
            self.use_h, self.use_s, self.use_v = True, True, True
        if self.mapping_criterion == 'RGB':
            self.use_h, self.use_s, self.use_v = False, False, False

        gp_obj = context.object
        frames_to_process = get_input_frames(gp_obj, get_multiedit(gp_obj))
        noise.seed_set(self.random_seed)
        
        def get_key(color):
            """Convert color components to KDTree coordinates"""
            if self.mapping_criterion == 'RGB':
                return color[0], color[1], color[2]
            h, s, v = rgb_to_hsv(color[0], color[1], color[2])
            h = 0 if not self.use_h else h
            s = 0 if not self.use_s else s
            v = 0 if not self.use_v else v
            return h, s, v
        
        # Check each color in the palette
        palette = bpy.data.palettes[self.palette_name]
        kdt = kdtree.KDTree(3 * len(palette.colors))
        palette_color_set = set()
        dst_stat = [[], [], []]
        for i,color in enumerate(palette.colors):
            # Set up KDTree for color lookup
            linear_color = [srgb_to_linear(color.color[j]) for j in range(3)]
            k = get_key(linear_color)
            if self.use_h:
                kdt.insert((k[0]-1.0, k[1], k[2]), i)
                kdt.insert((k[0]+1.0, k[1], k[2]), i)
            kdt.insert(k, i)
            # Collect statistics: do not count repeated colors which may be placeholders
            if rgb_to_hex_code(linear_color) not in palette_color_set:
                palette_color_set.add(rgb_to_hex_code(linear_color))
                for j in range(3):
                    dst_stat[j].append(k[j])
        kdt.balance()
        dst_stat = np.array(dst_stat)
        dst_mean, dst_std = dst_stat.mean(axis=-1), dst_stat.std(axis=-1)

        # Go through all selected points for the first time to collect statistics
        src_stat = [[], [], []]
        for frame in frames_to_process:
            for stroke in get_input_strokes(gp_obj, frame):
                if stroke.select:
                    fill_rgba = get_mixed_color(gp_obj, stroke, to_linear=True)
                    fill_key = list(get_key(fill_rgba))
                    for i,point in enumerate(stroke.points):
                        if is_fill_tintable(self, gp_obj, stroke):
                            for j in range(3):
                                src_stat[j].append(fill_key[j])
                        if is_point_tintable(self, gp_obj, stroke, point):
                            line_rgba = get_mixed_color(gp_obj, stroke, i, to_linear=True)
                            line_key = list(get_key(line_rgba))
                            for j in range(3):
                                src_stat[j].append(line_key[j])                            
        src_stat = np.array(src_stat)
        src_mean, src_std = src_stat.mean(axis=-1), src_stat.std(axis=-1)

        def recolor_single(target, rgb, noise):
            # Find the closest color in the palette
            key = Vector(get_key(rgb))
            key = key * (1-self.random_factor) + noise * self.random_factor
            for j in range(3):
                if not self.use_h or j!=0:
                    normalized = (key[j] - src_mean[j])/max(src_std[j], 1e-5)*dst_std[j] + dst_mean[j]
                    normalized = np.clip(normalized, 0, 1)
                    key[j] = normalized * self.normalize + key[j] * (1-self.normalize)
            _, p_idx, _ = kdt.find(key)
            c = [srgb_to_linear(palette.colors[p_idx].color[j]) for j in range(3)]

            # Mix the saturation and brightness
            h, s, v = rgb_to_hsv(c[0], c[1], c[2])
            _, s0, v0 = rgb_to_hsv(rgb[0], rgb[1], rgb[2])
            s = s * (1-self.keep_s) + s0 * self.keep_s
            v = v * (1-self.keep_v) + v0 * self.keep_v
            r, g, b = hsv_to_rgb(h, s, v)
            target[:] = [r,g,b,1]

        # Go through all selected points for the second time to apply color changes
        for frame in frames_to_process:
            for stroke in get_input_strokes(gp_obj, frame):
                line_noise = Vector((noise.random(), noise.random(), noise.random()))
                fill_noise = Vector((noise.random(), noise.random(), noise.random()))
                # Process fill color
                if stroke.select:
                    if is_fill_tintable(self, gp_obj, stroke):
                        r, g, b, _ = get_mixed_color(gp_obj, stroke, to_linear=True)
                        recolor_single(stroke.vertex_color_fill, (r,g,b), fill_noise)
                    # Process line color
                    if self.tint_mode != 'FILL':
                        for i,point in enumerate(stroke.points):
                            if is_point_tintable(self, gp_obj, stroke, point):
                                r, g, b, _ = get_mixed_color(gp_obj, stroke, i, to_linear=True)
                                recolor_single(point.vertex_color, (r,g,b), line_noise)                     
        
        return {'FINISHED'}