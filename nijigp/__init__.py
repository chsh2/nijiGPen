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
    "description" : "Tools modifying Grease Pencil strokes in a 2D plane",
    "blender" : (3, 3, 0),
    "version" : (0, 3, 2),
    "location" : "View3D > Sidebar > NijiGP, in Draw and Edit mode of Grease Pencil objects",
    "warning" : "This addon is still in an early stage of development",
    "category" : "Object"
}

from . import auto_load

auto_load.init()

def draw_shortcuts(self, context):
    if not context.preferences.addons[__package__].preferences.extra_buttons:
        return
    if context.mode == 'PAINT_GPENCIL' or context.mode == 'SCULPT_GPENCIL':
        self.layout.operator("ed.undo", text='', icon='TRIA_LEFT')
        self.layout.operator("ed.redo", text='', icon='TRIA_RIGHT')


def register():
    auto_load.register()
    bpy.types.Scene.nijigp_working_plane = bpy.props.EnumProperty(
                        name='Working Plane',
                        items=[('X-Z', 'Front (X-Z)', ''),
                                ('Y-Z', 'Side (Y-Z)', ''),
                                ('X-Y', 'Top (X-Y)', '')],
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
    bpy.types.PROPERTIES_PT_navigation_bar.prepend(draw_shortcuts)

def unregister():
    auto_load.unregister()
    bpy.types.PROPERTIES_PT_navigation_bar.remove(draw_shortcuts)
