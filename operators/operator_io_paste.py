import bpy
import os
import numpy as np
from mathutils import *
from ..utils import *
from ..resources import get_cache_folder

class PasteSVGOperator(bpy.types.Operator):
    """Convert SVG codes in the clipboard to strokes and insert them in the current Grease Pencil object"""
    bl_idname = "gpencil.nijigp_paste_svg"
    bl_label = "Paste SVG Codes"
    bl_category = 'View'
    bl_options = {'REGISTER', 'UNDO'}

    # Define properties
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
        svg_str = context.window_manager.clipboard
        svg_path = os.path.join( get_cache_folder(), "clipboard.svg")
        svg_file = open(svg_path, "w")
        svg_file.write(svg_str)
        svg_file.close()

        # Import SVG file
        bpy.ops.object.mode_set(mode='OBJECT')
        if bpy.app.version > (3, 3, 0):
            svg_dirname, svg_filename = os.path.split(svg_path)
            bpy.ops.wm.gpencil_import_svg("EXEC_DEFAULT", filepath=svg_path, directory=svg_dirname, files=[{"name":svg_filename}], resolution = self.svg_resolution, scale = self.svg_scale)
        else:
            bpy.ops.wm.gpencil_import_svg("EXEC_DEFAULT", filepath = svg_path, resolution = self.svg_resolution, scale = self.svg_scale)
        new_gp_obj = context.object

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
        bpy.ops.transform.rotate(value=math.pi/2, orient_axis='X', orient_type='LOCAL')
        for i in range(len(context.object.data.layers) - num_layers):
            for stroke in context.object.data.layers[i].active_frame.strokes:
                for point in stroke.points:
                    point.co = inv_mat @ point.co
        if context.scene.tool_settings.gpencil_stroke_placement_view3d == 'CURSOR':
            bpy.ops.transform.translate(value=context.scene.cursor.location)

        if self.auto_holdout:
            bpy.ops.gpencil.nijigp_hole_processing(rearrange=True, separate_colors=True)

        return {'FINISHED'}

class PasteXMLOperator(bpy.types.Operator):
    """Parse XML messages in the clipboard to create a new palette"""
    bl_idname = "gpencil.nijigp_paste_xml_palette"
    bl_label = "Paste XML Palette"
    bl_category = 'View'
    bl_options = {'REGISTER', 'UNDO'}

    tints_level: bpy.props.IntProperty(
            name='Tints and Shades',
            min=0, max=10, default=0,
            description='Automatically generate tints and shades from existing colors',
            )

    def draw(self, context):
        layout = self.layout
        row = layout.row()
        row.prop(self, "tints_level", text = "Tints and Shades")

    def execute(self, context):
        import xml.etree.ElementTree as ET
        clipboard_str = context.window_manager.clipboard

        palette_name = 'Pasted_Palette'
        colors_to_add = []
        try:
            root = ET.fromstring(clipboard_str)
            entries = root.findall('color')
            for color_entry in entries:
                info = color_entry.attrib
                if 'r' in info:
                    r = int(info['r']) / 255.0
                    g = int(info['g']) / 255.0
                    b = int(info['b']) / 255.0
                    colors_to_add.append(Color([r,g,b]))
                elif 'rgb' in info:
                    h = int(info['rgb'], 16)
                    colors_to_add.append(hex_to_rgb(h))
                if 'name' in info:
                    palette_name = info['name']
        except:
            # If the string is not XML, regard it as a list and try to extract hex codes
            alnum_str = "".join(filter(str.isalnum, clipboard_str))
            palette_name = clipboard_str
            i = 0
            while(i+5<len(alnum_str)):
                hex_str = alnum_str[i:i+6]
                h = int(hex_str, 16)
                colors_to_add.append(hex_to_rgb(h))
                i += 6

        if len(colors_to_add) == 0:
            self.report({"INFO"}, "No available colors found in the clipboard.")
        else:
            new_palette = bpy.data.palettes.new(palette_name)
            for color in colors_to_add:
                new_palette.colors.new()
                new_palette.colors[-1].color = color

            # Generate tints and shades
            if self.tints_level > 0:
                # Add padding slots for better alignment
                padding_slots = (10 - len(colors_to_add))%10
                for _ in range(padding_slots):
                    new_palette.colors.new()

                padding_slots = (5 - len(colors_to_add))%5
                for i in range(self.tints_level):
                    factor = (i+1) / (self.tints_level + 1)
                    # Add tints
                    for color in colors_to_add:
                        new_palette.colors.new()
                        new_palette.colors[-1].color = color * (1-factor) + Color((1,1,1)) * factor  
                    for _ in range(padding_slots):
                        new_palette.colors.new()
                    # Add shades
                    for color in colors_to_add:
                        new_palette.colors.new()
                        new_palette.colors[-1].color = color * (1-factor) + Color((0,0,0)) * factor  
                    for _ in range(padding_slots):
                        new_palette.colors.new()
                                      
            total_num = len(colors_to_add) * (1+2*self.tints_level)
            self.report({"INFO"}, "A new palette is created with " + str(total_num) + " colors.")

        return {'FINISHED'}
