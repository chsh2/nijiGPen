import bpy
import os
import numpy as np
from mathutils import *
from ..utils import *
from ..resources import get_cache_folder

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
            default='',
            description='Rename the layer and materials of the pasted shape if not empty'
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

    def draw(self, context):
        layout = self.layout
        row = layout.row()
        row.prop(self, "svg_name")
        row = layout.row()
        row.prop(self, "svg_resolution", text = "Resolution")
        row = layout.row()
        row.prop(self, "svg_scale", text = "Scale")
        row = layout.row()
        row.prop(self, "auto_holdout", text = "Auto Holdout")

    def execute(self, context):
        current_gp_obj = context.object
        current_material_idx = context.object.active_material_index
        num_layers = len(context.object.data.layers)
        t_mat, inv_mat = get_transformation_mat(mode=context.scene.nijigp_working_plane, gp_obj=current_gp_obj)

        # Convert clipboard data to SVG file
        svg_path = save_clipboard_text("clipboard.svg")

        # Import SVG file
        bpy.ops.object.mode_set(mode='OBJECT')
        if bpy.app.version > (3, 3, 0):
            svg_dirname, svg_filename = os.path.split(svg_path)
            bpy.ops.wm.gpencil_import_svg("EXEC_DEFAULT", filepath=svg_path, directory=svg_dirname, files=[{"name":svg_filename}], resolution=self.svg_resolution, scale=self.svg_scale)
        else:
            bpy.ops.wm.gpencil_import_svg("EXEC_DEFAULT", filepath=svg_path, resolution=self.svg_resolution, scale = self.svg_scale)
        new_gp_obj = context.object

        if new_gp_obj == current_gp_obj:
            self.report({"WARNING"}, "Cannot write temporary files. Import failed.")
            return {'FINISHED'}
        
        if len(new_gp_obj.data.layers) > 0:
            # Rename the layer and materials of the pasted figure
            if len(self.svg_name)>0:
                new_gp_obj.data.layers.active.info = self.svg_name
                fig_name = self.svg_name
            else:
                fig_name = new_gp_obj.data.layers.active.info
            for slot in new_gp_obj.material_slots:
                slot.material.name = fig_name

            # Copy all strokes to the existing GP object
            current_gp_obj.select_set(True)
            bpy.ops.gpencil.layer_duplicate_object(mode='ALL', only_active=False)
        else:
            self.report({"WARNING"}, "Cannot recognize the pasted content. No data is imported.")

        # Delete the new GP object and switch back to the existing one
        bpy.ops.object.select_all(action='DESELECT')
        new_gp_obj.select_set(True)
        bpy.ops.object.delete()
        context.view_layer.objects.active = current_gp_obj
        bpy.ops.object.mode_set(mode='EDIT_GPENCIL')
        bpy.ops.gpencil.select_all(action='DESELECT')

        # Select pasted strokes
        bpy.ops.gpencil.select_all(action='DESELECT')
        for i in range(len(context.object.data.layers) - num_layers):
            for stroke in context.object.data.layers[i].active_frame.strokes:
                stroke.select = True
        current_gp_obj.active_material_index = current_material_idx

        # Transform the figure to the working 2D plane
        # Default plane is X-Z for imported SVG. Convert it to X-Y first
        z_to_y_mat = Matrix([(1,0,0), (0,0,1), (0,1,0)])
        for i in range(len(context.object.data.layers) - num_layers):
            for stroke in context.object.data.layers[i].active_frame.strokes:
                for point in stroke.points:
                    point.co = inv_mat @ z_to_y_mat @ point.co
        if context.scene.tool_settings.gpencil_stroke_placement_view3d == 'CURSOR':
            bpy.ops.transform.translate(value=context.scene.cursor.location)

        if self.auto_holdout:
            bpy.ops.gpencil.nijigp_hole_processing(rearrange=True, separate_colors=True)

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
        swatch_path = save_clipboard_text(self.name)
        bpy.ops.gpencil.nijigp_import_swatch(
            filepath = swatch_path,
            directory = os.path.dirname(swatch_path),
            files = [{"name": self.name}],
            ignore_placeholders = False,
            tints_level = self.tints_level
        )
        context.scene.tool_settings.gpencil_paint.palette = bpy.data.palettes[-1]
        return {'FINISHED'}
