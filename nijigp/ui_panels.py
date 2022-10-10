from ctypes import alignment
import bpy

class NIJIGP_PT_draw_panel_setting(bpy.types.Panel):
    bl_idname = 'NIJIGP_PT_draw_panel_setting'
    bl_label = "Global Setting"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "NijiGP"
    bl_context = "greasepencil_paint"

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        obj = context.object
        layout.label(text="Working Plane:")
        row = layout.row()
        row.prop(scene, "nijigp_working_plane", text='')

class NIJIGP_PT_edit_panel_setting(bpy.types.Panel):
    bl_idname = 'NIJIGP_PT_edit_panel_setting'
    bl_label = "Global Setting"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "NijiGP"
    bl_context = "greasepencil_edit"

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        obj = context.object
        layout.label(text="Working Plane:")
        row = layout.row()
        row.prop(scene, "nijigp_working_plane", text='')

class NIJIGP_PT_draw_panel_io(bpy.types.Panel):
    bl_idname = 'NIJIGP_PT_draw_panel_io'
    bl_label = "Import/Export"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "NijiGP"
    bl_context = "greasepencil_paint"

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        obj = context.object

        layout.label(text="Shapes from Clipboard:")
        row = layout.row()
        row.operator("nijigp.paste_svg", text="Paste SVG Codes", icon="PASTEDOWN")

        layout.label(text="Palettes and Colors:")
        row = layout.row()
        row.operator("nijigp.paste_xml_palette", text="Paste XML Palette", icon="PASTEDOWN")
        

class NIJIGP_PT_edit_panel_io(bpy.types.Panel):
    bl_idname = 'NIJIGP_PT_edit_panel_io'
    bl_label = "Import/Export"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "NijiGP"
    bl_context = "greasepencil_edit"

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        obj = context.object

        layout.label(text="Shapes from Clipboard:")
        row = layout.row()
        row.operator("nijigp.paste_svg", text="Paste SVG Codes", icon="PASTEDOWN")

        layout.label(text="Palettes and Colors:")
        row = layout.row()
        row.operator("nijigp.paste_xml_palette", text="Paste XML Palette", icon="PASTEDOWN")

class NIJIGP_PT_draw_panel_polygon(bpy.types.Panel):
    bl_idname = 'NIJIGP_PT_draw_panel_polygon'
    bl_label = "Polygon Operations"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "NijiGP"
    bl_context = "greasepencil_paint"

    def draw(self, context):
        layout = self.layout
        scene = context.scene

        layout.label(text="Stroke Operations:")
        row = layout.row()
        row.operator("nijigp.bool_last", text="Boolean with Last Stroke", icon="MOD_BOOLEAN")
        layout.label(text="Affect only:")
        row = layout.row()
        row.prop(scene, "nijigp_draw_bool_material_constraint", text = "Same Material")
        row = layout.row()
        row.prop(scene, "nijigp_draw_bool_fill_constraint", text = "Strokes with Fills")


class NIJIGP_PT_edit_panel_polygon(bpy.types.Panel):
    bl_idname = 'NIJIGP_PT_edit_panel_polygon'
    bl_label = "Polygon Operations"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "NijiGP"
    bl_context = "greasepencil_edit"

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        obj = context.object

        layout.label(text="Stroke Operations:")
        row = layout.row()
        row.operator("nijigp.offset_selected", text="Offset Selected", icon="MOD_SKIN")
        row = layout.row()
        row.operator("nijigp.bool_selected", text="Boolean of Selected", icon="MOD_BOOLEAN")
        row = layout.row()

class NIJIGP_PT_edit_panel_mesh(bpy.types.Panel):
    bl_idname = 'NIJIGP_PT_edit_panel_mesh'
    bl_label = "Strokes to Meshes"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "NijiGP"
    bl_context = "greasepencil_edit"

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        obj = context.object
        layout.label(text="Preprocessing:")
        row = layout.row()
        row.operator("gpencil.stroke_sample", text="Resample")
        row.operator("gpencil.stroke_smooth", text="Smooth")
        layout.label(text="Generation Methods:")
        row = layout.row()
        row.operator("nijigp.mesh_generation_offset", text="Offset to Mesh", icon="CONE")
        row = layout.row()
        row.operator("nijigp.mesh_generation_normal", text="Planar with Normals", icon="NORMALS_FACE")

