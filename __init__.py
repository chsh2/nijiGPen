# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTIBILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

bl_info = {
    "name" : "nijiGPen",
    "author" : "https://github.com/chsh2/nijiGPen",
    "description" : "A Grease Pencil toolbox for 2D graphic design and illustrations",
    "blender" : (3, 3, 0),
    "version" : (0, 14, 1),
    "location" : "View3D > Sidebar > NijiGP, in specific modes of Grease Pencil objects",
    "doc_url": "https://chsh2.github.io/nijigp/",
    "wiki_url": "https://chsh2.github.io/nijigp/",
    "tracker_url": "https://github.com/chsh2/nijiGPen/issues",
    "category" : "Object"
}

_needs_reload = "bpy" in locals()

import bpy
from . import (
    ui_panels,
    ui_viewport_tools,
    preferences,
    operators,
    api_router,
)

if _needs_reload:
    import importlib

    ui_panels = importlib.reload(ui_panels)
    ui_viewport_tools = importlib.reload(ui_viewport_tools)
    preferences = importlib.reload(preferences)
    operators = importlib.reload(operators)
    api_router = importlib.reload(api_router)

def _get_all_classes():
    return (
        # ui_panels
        ui_panels.RenderAndVectorizeMenu,
        ui_panels.NIJIGP_PT_draw_panel_setting,
        ui_panels.NIJIGP_PT_edit_panel_setting,
        ui_panels.NIJIGP_PT_weight_panel_setting,
        ui_panels.NIJIGP_PT_draw_panel_io,
        ui_panels.NIJIGP_PT_edit_panel_io,
        ui_panels.NIJIGP_PT_draw_panel_polygon,
        ui_panels.NIJIGP_PT_edit_panel_polygon,
        ui_panels.NIJIGP_PT_edit_panel_mesh,
        ui_panels.NIJIGP_PT_draw_panel_mesh,
        ui_panels.NIJIGP_PT_edit_panel_line,
        ui_panels.NIJIGP_PT_draw_panel_line,
        ui_panels.NIJIGP_PT_weight_panel_rig,
        ui_panels.NIJIGP_PT_edit_panel_misc,
        ui_panels.NIJIGP_PT_edit_subpanel_palette,
        # ui_viewport_tools
        ui_viewport_tools.BooleanModalOperator,
        ui_viewport_tools.SmartFillModalOperator,
        ui_viewport_tools.SweepModalOperator,
        ui_viewport_tools.OffsetModalOperator,
        ui_viewport_tools.RollViewModalOperator,
        ui_viewport_tools.ArrangeModalOperator,
        ui_viewport_tools.ViewportShortcuts,
        ui_viewport_tools.RefreshGizmoOperator,
        # preferences
        preferences.ClearLogs,
        preferences.ApplyCustomLibPath,
        preferences.DetectDependencies,
        preferences.InstallDependency,
        preferences.RemoveDependency,
        preferences.NijiGPAddonPreferences,
    )

def register_classes():
    for cls in _get_all_classes():
        bpy.utils.register_class(cls)

def unregister_classes():
    for cls in reversed(_get_all_classes()):
        bpy.utils.unregister_class(cls)

def register():
    register_classes()
    operators.register_classes()
    bpy.types.Scene.nijigp_working_plane = bpy.props.EnumProperty(
                        name='Working Plane',
                        items=[('X-Z', 'Front (X-Z)', ''),
                                ('Y-Z', 'Side (Y-Z)', ''),
                                ('X-Y', 'Top (X-Y)', ''),
                                ('VIEW', 'View', 'Use the current view as the 2D working plane'),
                                ('AUTO', 'Auto', 'Calculate the 2D plane automatically based on input points and view angle')],
                        default='AUTO',
                        description='The 2D (local) plane that most add-on operators are working on'
                        )
    bpy.types.Scene.nijigp_working_plane_layer_transform = bpy.props.BoolProperty(
                        default=True, 
                        description="Taking the active layer's transform into consideration when calculating the view angle"
                        )
    bpy.types.Scene.nijigp_draw_bool_material_constraint = bpy.props.EnumProperty(
                        name='Material Filter',
                        items=[('ALL', 'All Materials', ''),
                               ('SAME', 'Same Material', ''),
                               ('DIFF', 'Different Material', '')],
                        default='ALL',
                        description="Boolean operations in Draw mode only apply to strokes with materials satisfying this requirement"
                        )
    bpy.types.Scene.nijigp_draw_bool_fill_constraint = bpy.props.EnumProperty(
                        name='Stroke Filter',
                        items=[('ALL', 'Stroke & Fill', ''),
                               ('FILL', 'Fill Only', '')],
                        default='FILL',
                        description="Boolean operations in Draw mode apply to either all strokes or only strokes with fills"
                        )
    bpy.types.Scene.nijigp_draw_bool_selection_constraint = bpy.props.BoolProperty(
                        default=False, 
                        description="Boolean operations in Draw mode apply to selected strokes only"
                        )
    ui_viewport_tools.register_viewport_tools()
    api_router.register_alternative_api_paths()
    
    custom_lib_path = bpy.context.preferences.addons[__package__].preferences.custom_lib_path
    if len(custom_lib_path) > 0:
        import sys
        import os
        sys.path.append(custom_lib_path)
        dll_paths = [
            os.path.join(custom_lib_path, "numpy.libs"),
            os.path.join(custom_lib_path, "numpy", ".libs"),
            os.path.join(custom_lib_path, "numpy", ".dylibs")]
        for dll_path in dll_paths:
            if os.path.exists(dll_path) and hasattr(os, "add_dll_directory"):
                os.add_dll_directory(dll_path)

def unregister():
    api_router.unregister_alternative_api_paths()
    ui_viewport_tools.unregister_viewport_tools()
    operators.unregister_classes()
    unregister_classes()
