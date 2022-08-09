import bpy
import os
import math
from .utils import *

class PasteSVGOperator(bpy.types.Operator):
    """Convert SVG codes in the clipboard to strokes and insert them in the current Grease Pencil object"""
    bl_idname = "nijigp.paste_svg"
    bl_label = "Paste SVG Codes"
    bl_category = 'View'
    bl_options = {'REGISTER', 'UNDO'}

    # Define properties
    svg_resolution: bpy.props.IntProperty(
            name='Resolution',
            min=1, max=50, default=10,
            description='Resolution of pasted SVG',
            )
    svg_scale: bpy.props.FloatProperty(
            name='Scale',
            min=0, max=100, default=10,
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
        row.prop(self, "svg_resolution", text = "Resolution")
        row = layout.row()
        row.prop(self, "svg_scale", text = "Scale")
        row = layout.row()
        row.prop(self, "auto_holdout", text = "Auto Holdout")

    def execute(self, context):

        #preferences = context.preferences.addons[__package__].preferences
        preferences = context.preferences
        current_gp_obj = context.object
        current_material_idx = context.object.active_material_index
        num_layers = len(context.object.data.layers)

        # Convert clipboard data to SVG file
        svg_str = context.window_manager.clipboard
        svg_path = os.path.join(preferences.filepaths.temporary_directory, "clipboard.svg")
        svg_file = open(svg_path, "w")
        svg_file.write(svg_str)
        svg_file.close()

        # Import SVG file
        bpy.ops.object.mode_set(mode='OBJECT')
        bpy.ops.wm.gpencil_import_svg("EXEC_DEFAULT", filepath = svg_path, resolution = self.svg_resolution, scale = self.svg_scale)
        new_gp_obj = context.object

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

        # Holes processing
        if self.auto_holdout:
            # Create a new material with fill holdout
            holdout_material = bpy.data.materials.new("SVG_Holdout")
            bpy.data.materials.create_gpencil_data(holdout_material)
            holdout_material.grease_pencil.show_stroke = False
            holdout_material.grease_pencil.show_fill = True
            holdout_material.grease_pencil.use_fill_holdout = True
            context.object.data.materials.append(holdout_material)
            holdout_material_index = len(context.object.data.materials) -1

            # Emitting rays vertically to get winding numbers
            def pos_H_intersect(point, seg):
                if seg[0][0]>point[0] and seg[1][0]>point[0]:
                    return False
                if seg[0][0]<point[0] and seg[1][0]<point[0]:
                    return False
                if math.isclose(seg[0][0], point[0]):
                    return False
                if math.isclose(seg[1][0], point[0]):
                    return seg[1][1] > point[1]
                ratio = (point[0] - seg[0][0]) / (seg[1][0] - seg[0][0])
                h_intersect = seg[0][1] + ratio * (seg[1][1] - seg[0][1])
                return h_intersect > point[1]

            # Detect holes using even-odd rule (roughly)
            for i in range(len(context.object.data.layers) - num_layers):
                strokes = context.object.data.layers[i].active_frame.strokes
                to_process = []
                outer_shapes = []
                for stroke in strokes:
                    to_process.append(stroke)
                    outer_shapes.append(stroke)
                is_hole = False

                while len(to_process)>0 and len(outer_shapes)>0:
                    winding_number_list = []
                    is_hole = (not is_hole)

                    # Judge whether a stroke is inside another one
                    # Roughly computed by sampling only one point on each curve to avoid too much calculation
                    for stroke_src in to_process:
                        sample_point = vec3_to_vec2(stroke_src.points[len(stroke_src.points)//2].co)
                        winding_number = 0
                        for stroke_dst in outer_shapes:
                            if stroke_dst != stroke_src:
                                poly_list, _ = stroke_to_poly([stroke_dst])
                                co_list = poly_list[0]
                                for j, co in enumerate(co_list):
                                    if j!= 0:
                                        seg = [co_list[j], co_list[j-1]]
                                        winding_number += pos_H_intersect(sample_point, seg)       
                        winding_number_list.append(winding_number)  

                    # Process the inner strokes
                    outer_shapes = []
                    for j,winding_number in enumerate(winding_number_list):
                        if winding_number % 2 == 1:
                            if is_hole:
                                to_process[j].material_index = holdout_material_index
                            to_process[j].select = True
                            outer_shapes.append(to_process[j]) 
                    for stroke in outer_shapes:
                        to_process.remove(stroke)              
                    bpy.ops.gpencil.stroke_arrange("EXEC_DEFAULT", direction='TOP')
                    bpy.ops.gpencil.select_all(action='DESELECT')      
              


        # Select pasted strokes
        bpy.ops.gpencil.select_all(action='DESELECT')
        for i in range(len(context.object.data.layers) - num_layers):
            for stroke in context.object.data.layers[i].active_frame.strokes:
                stroke.select = True
        current_gp_obj.active_material_index = current_material_idx

        if bpy.context.scene.nijigp_working_plane == 'X-Y':
            bpy.ops.transform.rotate(value=math.pi/2, orient_axis='X', orient_type='LOCAL')
        if bpy.context.scene.nijigp_working_plane == 'Y-Z':
            bpy.ops.transform.rotate(value=-math.pi/2, orient_axis='Z', orient_type='LOCAL')

        return {'FINISHED'}

class PasteXMLOperator(bpy.types.Operator):
    """Parse XML messages in the clipboard to create a new palette"""
    bl_idname = "nijigp.paste_xml_palette"
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
        xml_str = context.window_manager.clipboard

        root = ET.fromstring(xml_str)
        palette_name = 'Pasted_Palette'
        colors_to_add = []

        entries = root.findall('color')
        for color_entry in entries:
            info = color_entry.attrib
            if 'r' in info:
                r = int(info['r']) / 255.0
                g = int(info['g']) / 255.0
                b = int(info['b']) / 255.0
                colors_to_add.append(Color([r,g,b]))
                if 'name' in info:
                    palette_name = info['name']

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
