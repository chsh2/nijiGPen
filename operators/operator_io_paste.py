import bpy
import os
import numpy as np
from mathutils import *
from ..utils import *
from ..resources import get_cache_folder
from ..api_router import *

def make_filename(text):
    """Convert a string provided by user to a legal file name"""
    return "".join(x for x in text if x.isalnum() or x in ". _-")

def save_clipboard_text(filename):
    text = bpy.context.window_manager.clipboard
    path = os.path.join( get_cache_folder(), filename)
    try:
        fd = open(path, "w")
    except:
        path = os.path.join(bpy.app.tempdir, filename)
        fd = open(path, "w")
    fd.write(text)
    fd.close()
    return path

class PasteSVGOperator(bpy.types.Operator):
    """Convert SVG codes in the clipboard to strokes and insert them in the current Grease Pencil object"""
    bl_idname = "gpencil.nijigp_paste_svg"
    bl_label = "Paste SVG Codes"
    bl_category = 'View'
    bl_options = {'REGISTER', 'UNDO'}

    svg_name: bpy.props.StringProperty(
            name='Name',
            default='SVG_From_Clipboard',
            description='Rename the layer and materials of the pasted shape'
    )
    svg_resolution: bpy.props.IntProperty(
            name='Resolution',
            min=1, max=50, default=10,
            description='Resolution of pasted SVG',
    )
    svg_scale: bpy.props.FloatProperty(
            name='Scale',
            min=0, max=100, default=50,
            description='Scale of pasted SVG',
    )
    auto_holdout: bpy.props.BoolProperty(
            name='Auto Holdout',
            default=False,
            description='Change materials of holes (SVG polygons with negative area) to holdout and move holes to front'
    )
    reuse_materials: bpy.props.BoolProperty(
            name='Reuse Materials',
            default=False,
            description='Share the materials among all appended SVG instead of creating new materials'
    )     

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "svg_name")
        layout.label(text = "Geometry Options:")
        box1 = layout.box()
        box1.prop(self, "svg_resolution")
        box1.prop(self, "svg_scale")
        layout.label(text = "Material Options:")
        box2 = layout.box()
        box2.prop(self, "auto_holdout")
        box2.prop(self, "reuse_materials")

    def execute(self, context):
        fname = make_filename(self.svg_name)
        svg_path = save_clipboard_text(fname)
        bpy.ops.gpencil.nijigp_append_svg(
            filepath = svg_path,
            directory = os.path.dirname(svg_path),
            files = [{"name": fname}],
            svg_resolution = self.svg_resolution,
            svg_scale = self.svg_scale,
            auto_holdout = self.auto_holdout,
            image_sequence = False,
            reuse_materials = self.reuse_materials
        )
        return {'FINISHED'}

class PasteSwatchOperator(bpy.types.Operator):
    """Parse an XML message or a Hex code list in the clipboard to create a new palette"""
    bl_idname = "gpencil.nijigp_paste_swatch"
    bl_label = "Paste Swatches"
    bl_category = 'View'
    bl_options = {'REGISTER', 'UNDO'}

    name: bpy.props.StringProperty(
            name='Name',
            default='Palette_From_Clipboard',
            description='Name the imported palette if the format does not contain a name itself'
    )
    tints_level: bpy.props.IntProperty(
            name='Tints and Shades',
            min=0, max=10, default=0,
            description='Extend the palette by generating tints and shades colors based on existing ones',
    )

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "name")
        layout.prop(self, "tints_level", text = "Tints and Shades")

    def execute(self, context):
        fname = make_filename(self.name)
        swatch_path = save_clipboard_text(fname)
        bpy.ops.gpencil.nijigp_import_swatch(
            filepath = swatch_path,
            directory = os.path.dirname(swatch_path),
            files = [{"name": fname}],
            ignore_placeholders = False,
            tints_level = self.tints_level
        )
        context.scene.tool_settings.gpencil_paint.palette = bpy.data.palettes[-1]
        return {'FINISHED'}
