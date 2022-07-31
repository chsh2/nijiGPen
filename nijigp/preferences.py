import bpy
import os

class InstallDependencies(bpy.types.Operator):
    """
    Enable pip and install required Python packages
    """
    bl_idname = "nijigp.install_dependencies"
    bl_label = "Install Dependencies"
    bl_description = ("Enable pip and install required packages")
    bl_options = {"REGISTER", "INTERNAL"}

    def execute(self, context):
        import subprocess
        import sys
        python_exe = sys.executable

        res = subprocess.call([python_exe, '-m', 'ensurepip', '--upgrade'])
        if res > 0:
            self.report({"ERROR"}, "Cannot ensure pip.")
        res = subprocess.call([python_exe, '-m', 'pip', 'install', 'pyclipper'])
        if res > 0:
            self.report({"ERROR"}, "Cannot install the package PyClipper.")

        self.report({"INFO"}, "Python packages installed successfully.")
        return {"FINISHED"}


class NijiGPAddonPreferences(bpy.types.AddonPreferences):
    bl_idname = __package__

    extra_buttons: bpy.props.BoolProperty(
        name='Show Extra Undo/Redo Buttons',
        description='Show extra undo and redo buttons in the sidebar of Properties panel',
        default=True
    )

    def draw(self, context):
        layout = self.layout
        row = layout.row()
        row.prop(self, 'extra_buttons')
        row = layout.row()
        row.operator("nijigp.install_dependencies", text="Install Dependencies", icon="CONSOLE")