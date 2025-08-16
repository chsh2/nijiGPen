import bpy
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
    "version" : (0, 12, 2),
    "location" : "View3D > Sidebar > NijiGP, in specific modes of Grease Pencil objects",
    "doc_url": "https://chsh2.github.io/nijigp/",
    "wiki_url": "https://chsh2.github.io/nijigp/",
    "tracker_url": "https://github.com/chsh2/nijiGPen/issues",
    "category" : "Object"
}

from . import auto_load
from .ui_viewport_tools import *
from .api_router import register_alternative_api_paths, unregister_alternative_api_paths

auto_load.init()

def register():
    auto_load.register()
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
    register_viewport_tools()
    register_alternative_api_paths()
    
    custom_lib_path = bpy.context.preferences.addons[__package__].preferences.custom_lib_path
    if len(custom_lib_path) > 0:
        import sys
        sys.path.append(custom_lib_path)

def unregister():
    auto_load.unregister()
    unregister_viewport_tools()
    unregister_alternative_api_paths()
