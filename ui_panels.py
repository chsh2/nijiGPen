import bpy
from .api_router import *

class RenderAndVectorizeMenu(bpy.types.Menu):
    bl_label = "Render and Convert Scene/Mesh"
    bl_idname = "NIJIGP_MT_render_and_vectorize"
    def draw(self, context):
        self.layout.operator("gpencil.nijigp_render_and_vectorize")

# Panels that appear in more than one mode
def panel_global_setting(panel, context):
    layout = panel.layout
    scene = context.scene
    row = layout.row()
    row.label(text="Working Plane:")
    row.prop(scene, "nijigp_working_plane", text='')
    if scene.nijigp_working_plane == 'VIEW' or scene.nijigp_working_plane == 'AUTO':
        row = layout.row()
        row.prop(scene, "nijigp_working_plane_layer_transform", text='Use Transform of Active Layer')

def panel_io(panel, context):
    layout = panel.layout

    layout.label(text="Paste from Clipboard:")
    row = layout.row()
    row.operator("gpencil.nijigp_paste_svg", text="SVG Code", icon="PASTEDOWN")
    row.operator("gpencil.nijigp_paste_swatch", text="Swatches", icon="PASTEDOWN")

    layout.label(text="Image Vectorization:")
    row = layout.split()
    row.operator("gpencil.nijigp_import_lineart", text="Line Art", icon="LINE_DATA")
    sub_row = row.row(align=True)
    sub_row.operator("gpencil.nijigp_import_color_image", text="Flat Color", icon="IMAGE")
    sub_row.menu("NIJIGP_MT_render_and_vectorize", text="", icon="TRIA_DOWN")

    layout.label(text="Asset Import:")
    row = layout.row()
    row.operator("gpencil.nijigp_import_brush", text="Brushes", icon="BRUSH_DATA")
    row.operator("gpencil.nijigp_import_swatch", text="Swatches", icon="COLOR")
    row = layout.row()
    row.operator("gpencil.nijigp_append_svg", text="Append SVG", icon="FILE_TICK")
    
    layout.label(text="Image Export:")
    row = layout.row()
    row.operator("gpencil.nijigp_multilayer_render", text="Multi-Layer PSD Render", icon="RENDERLAYERS")

class NIJIGP_PT_draw_panel_setting(bpy.types.Panel):
    bl_idname = 'NIJIGP_PT_draw_panel_setting'
    bl_label = "Global Setting"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "NijiGP"
    bl_context = get_bl_context_str('paint')
    bl_order = 0

    def draw(self, context):
        panel_global_setting(self, context)

class NIJIGP_PT_edit_panel_setting(bpy.types.Panel):
    bl_idname = 'NIJIGP_PT_edit_panel_setting'
    bl_label = "Global Setting"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "NijiGP"
    bl_context = get_bl_context_str('edit')
    bl_order = 0

    def draw(self, context):
        panel_global_setting(self, context)

class NIJIGP_PT_weight_panel_setting(bpy.types.Panel):
    bl_idname = 'NIJIGP_PT_weight_panel_setting'
    bl_label = "Global Setting"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "NijiGP"
    bl_context = get_bl_context_str('weight')
    bl_order = 0

    def draw(self, context):
        panel_global_setting(self, context)

class NIJIGP_PT_draw_panel_io(bpy.types.Panel):
    bl_idname = 'NIJIGP_PT_draw_panel_io'
    bl_label = "Import/Export"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "NijiGP"
    bl_context = get_bl_context_str('paint')
    bl_order = 5

    def draw(self, context):
        panel_io(self, context)
         
class NIJIGP_PT_edit_panel_io(bpy.types.Panel):
    bl_idname = 'NIJIGP_PT_edit_panel_io'
    bl_label = "Import/Export"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "NijiGP"
    bl_context = get_bl_context_str('edit')
    bl_order = 5

    def draw(self, context):
        panel_io(self, context)

class NIJIGP_PT_draw_panel_polygon(bpy.types.Panel):
    bl_idname = 'NIJIGP_PT_draw_panel_polygon'
    bl_label = "Polygon Operators"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "NijiGP"
    bl_context = get_bl_context_str('paint')
    bl_order = 2

    def draw(self, context):
        layout = self.layout
        scene = context.scene

        row = layout.row()
        row.operator("gpencil.nijigp_bool_last", text="Boolean with Last Stroke")
        row = layout.row()
        row.operator("gpencil.nijigp_bool_last", text="+", icon="SELECT_EXTEND").operation_type = 'UNION'
        row.operator("gpencil.nijigp_bool_last", text="-", icon="SELECT_SUBTRACT").operation_type = 'DIFFERENCE'
        row.operator("gpencil.nijigp_bool_last", text="×", icon="SELECT_INTERSECT").operation_type = 'INTERSECTION'
        layout.label(text="Affected Strokes:")
        layout.prop(scene, "nijigp_draw_bool_material_constraint", text = "")
        layout.prop(scene, "nijigp_draw_bool_fill_constraint", text = "")
        layout.prop(scene, "nijigp_draw_bool_selection_constraint", text = "Selected Only", icon = "GP_SELECT_STROKES")

class NIJIGP_PT_edit_panel_polygon(bpy.types.Panel):
    bl_idname = 'NIJIGP_PT_edit_panel_polygon'
    bl_label = "Polygon Operators"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "NijiGP"
    bl_context = get_bl_context_str('edit')
    bl_order = 2

    def draw(self, context):
        layout = self.layout

        row = layout.row()
        row.operator("gpencil.nijigp_hole_processing", text="Hole Holdout", icon="MESH_TORUS")
        row = layout.row()
        row.operator("gpencil.nijigp_offset_selected", text="Offset", icon="MOD_SKIN")
        row.operator("gpencil.nijigp_sweep_selected", text="Sweep", icon="TRACKING_REFINE_FORWARDS")
        row = layout.row()
        row.operator("gpencil.nijigp_bool_selected", text="Boolean of Selected")
        row = layout.row()
        row.operator("gpencil.nijigp_bool_selected", text="+", icon="SELECT_EXTEND").operation_type = 'UNION'
        row.operator("gpencil.nijigp_bool_selected", text="-", icon="SELECT_SUBTRACT").operation_type = 'DIFFERENCE'
        row.operator("gpencil.nijigp_bool_selected", text="×", icon="SELECT_INTERSECT").operation_type = 'INTERSECTION'
        row = layout.row()
        row.operator("gpencil.nijigp_fracture_selected", text="Fracture Selected", icon="MOD_EXPLODE")
        row = layout.row()
        row.operator("gpencil.nijigp_hatch_fill", text="Hatch Fill", icon="ALIGN_JUSTIFY")
        row = layout.row()
        row.operator("gpencil.nijigp_shade_selected", text="Calculate Shading", icon="SHADING_RENDERED")

class NIJIGP_PT_edit_panel_mesh(bpy.types.Panel):
    bl_idname = 'NIJIGP_PT_edit_panel_mesh'
    bl_label = "Strokes to Meshes"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "NijiGP"
    bl_context = get_bl_context_str('edit')
    bl_order = 3

    def draw(self, context):
        layout = self.layout
        
        layout.label(text="Generation Methods:")
        row = layout.row()
        row.operator("gpencil.nijigp_mesh_generation_offset", text="Frustum", icon="CONE")
        row.operator("gpencil.nijigp_mesh_generation_normal", text="Interpolation", icon="NORMALS_FACE")

        layout.label(text="Mesh Management:")
        row = layout.row()
        row.operator("gpencil.nijigp_mesh_management", text="Hide").action = 'HIDE'
        row.operator("gpencil.nijigp_mesh_management", text="Show").action = 'SHOW'
        row.operator("gpencil.nijigp_mesh_management", text="Clear").action = 'CLEAR'

class NIJIGP_PT_draw_panel_mesh(bpy.types.Panel):
    bl_idname = 'NIJIGP_PT_draw_panel_mesh'
    bl_label = "Strokes to Meshes"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "NijiGP"
    bl_context = get_bl_context_str('paint')
    bl_order = 3

    def draw(self, context):
        layout = self.layout

        layout.label(text="Mesh Management:")
        row = layout.row()
        row.operator("gpencil.nijigp_mesh_management", text="Hide").action = 'HIDE'
        row.operator("gpencil.nijigp_mesh_management", text="Show").action = 'SHOW'
        row.operator("gpencil.nijigp_mesh_management", text="Clear").action = 'CLEAR'

class NIJIGP_PT_edit_panel_line(bpy.types.Panel):
    bl_idname = 'NIJIGP_PT_edit_panel_line'
    bl_label = "Line Operators"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "NijiGP"
    bl_context = get_bl_context_str('edit')
    bl_order = 1

    def draw(self, context):
        layout = self.layout

        layout.label(text="Line Cleanup by Fitting:")
        row = layout.row()
        row.operator("gpencil.nijigp_fit_selected", text="Single-Line", icon="MOD_SMOOTH")
        row.operator("gpencil.nijigp_cluster_and_fit", text="Multi-Line", icon="CURVES")
        layout.label(text="Line Utilities:")
        row = layout.row()
        row.operator("gpencil.nijigp_cluster_select", text="Cluster Select", icon="SELECT_SET")
        row = layout.row()
        row.operator("gpencil.nijigp_pinch", text="Pinch", icon="HANDLE_VECTOR")
        row.operator("gpencil.nijigp_taper_selected", text="Taper", icon="GP_ONLY_SELECTED")
     
class NIJIGP_PT_draw_panel_line(bpy.types.Panel):
    bl_idname = 'NIJIGP_PT_draw_panel_line'
    bl_label = "Line Operators"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "NijiGP"
    bl_context = get_bl_context_str('paint')
    bl_order = 1

    def draw(self, context):
        layout = self.layout

        row = layout.row()
        row.operator("gpencil.nijigp_fit_last", icon='MOD_SMOOTH')
        row = layout.row()
        row.operator("gpencil.nijigp_smart_fill", icon='SHADING_SOLID')

class NIJIGP_PT_weight_panel_rig(bpy.types.Panel):
    bl_idname = 'NIJIGP_PT_weight_panel_rig'
    bl_label = "Rig Operators"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "NijiGP"
    bl_context = get_bl_context_str('weight')
    bl_order = 1

    def draw(self, context):
        layout = self.layout
        layout.label(text="Generate Bone Weights:")
        row = layout.row()
        row.operator("gpencil.nijigp_rig_by_transfer_weights", text="Weights From Meshes", icon='MOD_DATA_TRANSFER')
        row = layout.row()
        row.operator("gpencil.nijigp_rig_by_pin_hints", text="Pins From Hints", icon='PINNED')
        layout.label(text="Utilities:")
        row = layout.row()
        row.operator("gpencil.nijigp_bake_rigging_animation", icon='KEYFRAME')
        row = layout.row()
        row.operator("gpencil.nijigp_vertex_group_clear", icon='X')

class NIJIGP_PT_edit_panel_misc(bpy.types.Panel):
    bl_idname = 'NIJIGP_PT_edit_panel_misc'
    bl_label = "Other Utilities"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "NijiGP"
    bl_context = get_bl_context_str('edit')
    bl_order = 4

    def draw(self, context):
        layout = self.layout
        layout.label(text="Shape Preprocessing:")
        row = layout.row()
        # Resample operator is missing in Blender 4.3
        sample_ops_str = get_ops_str("gpencil.stroke_sample")
        if sample_ops_str:
            row.operator(sample_ops_str, text="Resample").length = 0.02
        row.operator(get_ops_str("gpencil.stroke_smooth"), text="Smooth")
        
        layout.label(text="Color Utilities:")
        row = layout.row()
        row.operator("gpencil.nijigp_color_tint", text="Tint", icon='MOD_TINT')
        row.operator("gpencil.nijigp_recolor", text="Recolor", icon='SEQ_CHROMA_SCOPE')
        
class NIJIGP_PT_edit_subpanel_palette(bpy.types.Panel):
    bl_idname = 'NIJIGP_PT_edit_subpanel_palette'
    bl_label = "Palette Preview"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "NijiGP"
    bl_context = get_bl_context_str('edit')
    bl_parent_id = "NIJIGP_PT_edit_panel_misc"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        layout.prop(context.tool_settings.gpencil_vertex_paint, "palette")
        layout.template_palette(context.tool_settings.gpencil_vertex_paint, "palette", color=True)