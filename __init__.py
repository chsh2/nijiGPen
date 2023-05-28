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
    "description" : "Tools modifying/generating Grease Pencil strokes in a 2D plane",
    "blender" : (3, 3, 0),
    "version" : (0, 5, 0),
    "location" : "View3D > Sidebar > NijiGP, in Draw and Edit mode of Grease Pencil objects",
    "warning" : "This addon is still in an early stage of development",
    "category" : "Object"
}

from . import auto_load
from .ui_viewport_tools import *

auto_load.init()


def register():
    auto_load.register()
    bpy.types.Scene.nijigp_working_plane = bpy.props.EnumProperty(
                        name='Working Plane',
                        items=[('X-Z', 'Front (X-Z)', ''),
                                ('Y-Z', 'Side (Y-Z)', ''),
                                ('X-Y', 'Top (X-Y)', ''),
                                ('VIEW', 'View (Debug: DO NOT USE)', ''),
                                ('AUTO', 'Auto (Debug: DO NOT USE)', '')],
                        default='X-Z',
                        description='The 2D (local) plane that most add-on operators are working on'
                        )
    bpy.types.Scene.nijigp_draw_bool_material_constraint = bpy.props.BoolProperty(
                        default=False, 
                        description="Boolean operations in Draw mode only apply to strokes with the same material"
                        )
    bpy.types.Scene.nijigp_draw_bool_fill_constraint = bpy.props.BoolProperty(
                        default=True, 
                        description="Boolean operations in Draw mode only apply to strokes showing fills"
                        )
    register_viewport_tools()

def unregister():
    auto_load.unregister()
    unregister_viewport_tools()
