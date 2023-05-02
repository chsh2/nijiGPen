import bpy
import os

def modify_package(command, option, name):
    """
    Install or remove a Python package through pip
    """    
    import subprocess
    import sys
    python_exe = sys.executable

    res = subprocess.call([python_exe, '-m', 'ensurepip', '--upgrade'])
    if res > 0:
        return False
    res = subprocess.call([python_exe, '-m', 'pip', command, option, name])
    if res > 0:
        return False
    bpy.ops.nijigp.check_dependencies()
    return True
    

class DetectDependencies(bpy.types.Operator):
    """
    Check if required Python packages are installed
    """
    bl_idname = "nijigp.check_dependencies"
    bl_label = "Check Dependencies"
    bl_description = ("Check if required Python packages are installed")
    bl_options = {"REGISTER", "INTERNAL"}

    def execute(self, context):
        preferences = context.preferences.addons[__package__].preferences
        preferences.package_pyclipper = True
        preferences.package_triangle = True
        preferences.package_skimage = True
        try:
            import pyclipper
        except:
            preferences.package_pyclipper = False
        try:
            import triangle
        except:
            preferences.package_triangle = False
        try:
            import skimage
        except:
            preferences.package_skimage = False
        return {"FINISHED"}

class InstallPyClipper(bpy.types.Operator):
    bl_idname = "nijigp.dependencies_pyclipper_install"
    bl_label = "Install"
    bl_description = ("Manage packages through pip")
    bl_options = {"REGISTER", "INTERNAL"}

    def execute(self, context):
        res = modify_package('install','--no-input','pyclipper')
        if res:
            self.report({"INFO"}, "Python package installed successfully.")
        else:
            self.report({"ERROR"}, "Cannot install the required package.")
        return {"FINISHED"}

class InstallTriangle(bpy.types.Operator):
    bl_idname = "nijigp.dependencies_triangle_install"
    bl_label = "Install"
    bl_description = ("Manage packages through pip")
    bl_options = {"REGISTER", "INTERNAL"}

    def execute(self, context):
        res = modify_package('install','--no-input','triangle')
        if res:
            self.report({"INFO"}, "Python package installed successfully.")
        else:
            self.report({"ERROR"}, "Cannot install the required package.")
        return {"FINISHED"}

class InstallSkimage(bpy.types.Operator):
    bl_idname = "nijigp.dependencies_skimage_install"
    bl_label = "Install"
    bl_description = ("Manage packages through pip")
    bl_options = {"REGISTER", "INTERNAL"}

    def execute(self, context):
        res = modify_package('install','--no-input','scikit-image')
        if res:
            self.report({"INFO"}, "Python package installed successfully.")
        else:
            self.report({"ERROR"}, "Cannot install the required package.")
        return {"FINISHED"}

class RemovePyClipper(bpy.types.Operator):
    bl_idname = "nijigp.dependencies_pyclipper_remove"
    bl_label = "Remove"
    bl_description = ("Manage packages through pip")
    bl_options = {"REGISTER", "INTERNAL"}

    def execute(self, context):
        res = modify_package('uninstall','-y','pyclipper')
        self.report({"INFO"}, "Blender needs to be restarted.")
        return {"FINISHED"}

class RemoveTriangle(bpy.types.Operator):
    bl_idname = "nijigp.dependencies_triangle_remove"
    bl_label = "Remove"
    bl_description = ("Manage packages through pip")
    bl_options = {"REGISTER", "INTERNAL"}

    def execute(self, context):
        res = modify_package('uninstall','-y','triangle')
        self.report({"INFO"}, "Blender needs to be restarted.")
        return {"FINISHED"}

class RemoveSkimage(bpy.types.Operator):
    bl_idname = "nijigp.dependencies_skimage_remove"
    bl_label = "Remove"
    bl_description = ("Manage packages through pip")
    bl_options = {"REGISTER", "INTERNAL"}

    def execute(self, context):
        res = modify_package('uninstall','-y','scikit-image')
        self.report({"INFO"}, "Blender needs to be restarted.")
        return {"FINISHED"}


class NijiGPAddonPreferences(bpy.types.AddonPreferences):
    bl_idname = __package__

    cache_folder: bpy.props.StringProperty(
        name='Cache Folder',
        subtype='DIR_PATH',
        description='Location storing temporary files. Use the default temporary folder when empty',
        default=''
    )

    shortcut_button_enabled: bpy.props.BoolProperty(
        name='Enable Shortcut Buttons',
        description='Add a group of buttons at the bottom of the 3D view that brings better touchscreen control',
        default=True
    )
    shortcut_button_size: bpy.props.FloatProperty(
        name='Button Size',
        default=18, min=10, max=40
    )
    shortcut_button_location: bpy.props.FloatProperty(
        name='Button Location',
        description='The position of the shortcut buttons. Zero means buttons in the center. Positive/negative values means buttons in the right/left',
        default=0, soft_min=-2000, soft_max=2000
    )

    package_pyclipper: bpy.props.BoolProperty(
        name='PyClipper Installed',
        default=False
    )
    package_triangle: bpy.props.BoolProperty(
        name='Triangle Installed',
        default=False
    )
    package_skimage: bpy.props.BoolProperty(
        name='Scikit-Image Installed',
        default=False
    )

    def draw(self, context):
        layout = self.layout

        # Dependency manager
        box1 = layout.box()
        row = box1.row()
        row.label(text = "Dependency Management")
        row.separator()
        row.operator("nijigp.check_dependencies", text="Check", icon="FILE_REFRESH")

        table_key = ['Package', 'Type', 'Status', 'Actions','']
        packages = [{'name': 'PyClipper', 'type': 'Essential', 
                        'signal': self.package_pyclipper, 
                        'operator1':"nijigp.dependencies_pyclipper_install", 
                        'operator2':"nijigp.dependencies_pyclipper_remove"},
                    {'name': 'Scikit-Image', 'type': 'Optional', 
                        'signal': self.package_skimage, 
                        'operator1':"nijigp.dependencies_skimage_install", 
                        'operator2':"nijigp.dependencies_skimage_remove"},
                    {'name': 'Triangle', 'type': 'Optional', 
                        'signal': self.package_triangle, 
                        'operator1':"nijigp.dependencies_triangle_install", 
                        'operator2':"nijigp.dependencies_triangle_remove"},]
        column = box1.column()
        row = column.row()
        for key in table_key:
            row.label(text=key)
        for p in packages:
            row = column.row()
            row.label(text=p['name'])
            row.label(text=p['type'])
            if p['signal']:
                row.label(text='OK')
            else:
                row.label(text='Not Installed')
            row.operator(p['operator1'])
            row.operator(p['operator2'])

        # Other options
        layout.prop(self, 'cache_folder')
        layout.label(text='UI Setting:')
        row = layout.row()
        row.prop(self, 'shortcut_button_enabled')
        row.operator("gpencil.refresh_gizmo", text='Apply')
        if self.shortcut_button_enabled:
            row = layout.row()
            row.prop(self, 'shortcut_button_size')
            row.prop(self, 'shortcut_button_location')
