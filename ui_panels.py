import bpy

# Panels that appear in more than one mode
def draw_global_setting(panel, context):
    layout = panel.layout
    scene = context.scene
    row = layout.row()
    row.label(text="Working Plane:")
    row.prop(scene, "nijigp_working_plane", text='')
    if scene.nijigp_working_plane == 'VIEW' or scene.nijigp_working_plane == 'AUTO':
        row = layout.row()
        row.prop(scene, "nijigp_working_plane_layer_transform", text='Use Transform of Active Layer')

def draw_io(panel, context):
    layout = panel.layout

    layout.label(text="Paste from Clipboard:")
    row = layout.row()
    row.operator("gpencil.nijigp_paste_svg", text="SVG", icon="PASTEDOWN")
    row.operator("gpencil.nijigp_paste_xml_palette", text="XML/Hex", icon="COLOR")

    layout.label(text="Image Vectorization:")
    row = layout.row()
    row.operator("gpencil.nijigp_import_lineart", text="Line Art", icon="LINE_DATA")
    row.operator("gpencil.nijigp_import_color_image", text="Flat Color", icon="IMAGE")

    layout.label(text="Asset Import:")
    row = layout.row()
    row.operator("gpencil.nijigp_import_brush", text="ABR/GBR Brushes", icon="BRUSH_DATA")
    
    layout.label(text="Image Export:")
    row = layout.row()
    row.operator("gpencil.nijigp_multilayer_render", text="Multi-Layer PSD Render", icon="RENDERLAYERS")

class NIJIGP_PT_draw_panel_setting(bpy.types.Panel):
    bl_idname = 'NIJIGP_PT_draw_panel_setting'
    bl_label = "Global Setting"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "NijiGP"
    bl_context = "greasepencil_paint"
    bl_order = 0

    def draw(self, context):
        draw_global_setting(self, context)

class NIJIGP_PT_edit_panel_setting(bpy.types.Panel):
    bl_idname = 'NIJIGP_PT_edit_panel_setting'
    bl_label = "Global Setting"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "NijiGP"
    bl_context = "greasepencil_edit"
    bl_order = 0

    def draw(self, context):
         draw_global_setting(self, context)

class NIJIGP_PT_weight_panel_setting(bpy.types.Panel):
    bl_idname = 'NIJIGP_PT_weight_panel_setting'
    bl_label = "Global Setting"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "NijiGP"
    bl_context = "greasepencil_weight"
    bl_order = 0

    def draw(self, context):
        draw_global_setting(self, context)

class NIJIGP_PT_draw_panel_io(bpy.types.Panel):
    bl_idname = 'NIJIGP_PT_draw_panel_io'
    bl_label = "Import/Export"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "NijiGP"
    bl_context = "greasepencil_paint"
    bl_order = 4

    def draw(self, context):
        draw_io(self, context)
         
class NIJIGP_PT_edit_panel_io(bpy.types.Panel):
    bl_idname = 'NIJIGP_PT_edit_panel_io'
    bl_label = "Import/Export"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "NijiGP"
    bl_context = "greasepencil_edit"
    bl_order = 4

    def draw(self, context):
        draw_io(self, context)

class NIJIGP_PT_draw_panel_polygon(bpy.types.Panel):
    bl_idname = 'NIJIGP_PT_draw_panel_polygon'
    bl_label = "Polygon Operators"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "NijiGP"
    bl_context = "greasepencil_paint"
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
        layout.label(text="Affect only:")
        row = layout.row()
        row.prop(scene, "nijigp_draw_bool_material_constraint", text = "Same Material")
        row = layout.row()
        row.prop(scene, "nijigp_draw_bool_fill_constraint", text = "Strokes with Fills")

class NIJIGP_PT_edit_panel_polygon(bpy.types.Panel):
    bl_idname = 'NIJIGP_PT_edit_panel_polygon'
    bl_label = "Polygon Operators"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "NijiGP"
    bl_context = "greasepencil_edit"
    bl_order = 2

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        obj = context.object

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

class NIJIGP_PT_edit_panel_mesh(bpy.types.Panel):
    bl_idname = 'NIJIGP_PT_edit_panel_mesh'
    bl_label = "Strokes to Meshes"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "NijiGP"
    bl_context = "greasepencil_edit"
    bl_order = 3

    def draw(self, context):
        layout = self.layout
        
        layout.label(text="Preprocessing:")
        row = layout.row()
        row.operator("gpencil.stroke_sample", text="Resample").length = 0.02
        row.operator("gpencil.stroke_smooth", text="Smooth")

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
    bl_context = "greasepencil_paint"
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
    bl_context = "greasepencil_edit"
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
    bl_context = "greasepencil_paint"
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
    bl_context = "greasepencil_weight"
    bl_order = 1

    def draw(self, context):
        layout = self.layout
        layout.label(text="Generate Bone Weights:")
        row = layout.row()
        row.operator("gpencil.nijigp_rig_by_transfer_weights", text="Transfer From Meshes", icon='MOD_DATA_TRANSFER')
        layout.label(text="Utilities:")
        row = layout.row()
        row.operator("gpencil.nijigp_bake_rigging_animation", icon='KEYFRAME')
        row = layout.row()
        row.operator("object.vertex_group_remove", icon='X').all = True