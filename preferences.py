import bpy
import sys

def log_append(str):
    bpy.context.preferences.addons[__package__].preferences.captured_logs.append(str)

def run_command(commands = [], output_log = True, return_full_result = False):
    from subprocess import Popen, PIPE
    texts = []
    with Popen(commands, stdout=PIPE, shell=False) as p:
        while p.poll() is None:
            text = p.stdout.readline().decode("utf-8")
            if len(text) > 0:
                texts.append(text)
                print(text)
                if output_log:
                    log_append(text)
                    bpy.context.region.tag_redraw()
        return texts if return_full_result else p.returncode

def modify_package(command, option, name):
    """
    Install or remove a Python package through pip
    """    
    python_exe = sys.executable

    res = run_command([python_exe, '-m', 'ensurepip', '--upgrade'])
    if res > 0:
        return False
    # Use an alternative source for triangle in MacOS with Apple Silicon
    import platform
    if name == 'triangle' and command == 'install' and platform.machine() == 'arm64':
        res = run_command([python_exe, '-m', 'pip', command, option, 'triangle2'])
    else:
        res = run_command([python_exe, '-m', 'pip', command, option, name])
    if res > 0:
        return False
    bpy.ops.nijigp.check_dependencies(output_log=False)
    return True

class ClearLogs(bpy.types.Operator):
    """
    Clear the captured logs from the Preferences panel
    """
    bl_idname = "nijigp.clear_logs"
    bl_label = "Clear Logs"
    bl_description = ("Clear the captured logs from the Preferences panel")
    bl_options = {"REGISTER", "INTERNAL"} 
    
    def execute(self, context):
        bpy.context.preferences.addons[__package__].preferences.captured_logs.clear()
        return {"FINISHED"}

class ApplyCustomLibPath(bpy.types.Operator):
    """
    Add the custom site-package path to the package search path
    """
    bl_idname = "nijigp.apply_custom_lib_path"
    bl_label = "Apply Custom Package Path"
    bl_description = ("Add the custom site-package path to the package search path")
    bl_options = {"REGISTER", "INTERNAL"} 
    
    output_log: bpy.props.BoolProperty(default = True)
    
    def execute(self, context):
        custom_lib_path = bpy.context.preferences.addons[__package__].preferences.custom_lib_path
        if len(custom_lib_path) > 0 and custom_lib_path not in sys.path:
            sys.path.append(custom_lib_path)
        if self.output_log:
            log_append("[NijiGPen Info] Package Search Paths Updated:")
            for path in sys.path:
                log_append(path)
        return {"FINISHED"}

class DetectDependencies(bpy.types.Operator):
    """
    Check if required Python packages are installed
    """
    bl_idname = "nijigp.check_dependencies"
    bl_label = "Check Dependencies"
    bl_description = ("Check if required Python packages are installed")
    bl_options = {"REGISTER", "INTERNAL"}

    output_log: bpy.props.BoolProperty(default = True)

    def execute(self, context):
        preferences = context.preferences.addons[__package__].preferences
        preferences.package_pyclipper = True
        preferences.package_triangle = True
        preferences.package_skimage = True
        
        try:
            import pyclipper
            if self.output_log:
                log_append("[NijiGPen Info] Package PyClipper:")
                log_append("  Version: "+str(pyclipper.__version__))
                log_append("  Location: "+str(pyclipper.__file__))
        except:
            preferences.package_pyclipper = False
        
        try:
            import scipy
            if self.output_log:
                log_append("[NijiGPen Info] Package SciPy:")
                log_append("  Version: "+str(scipy.__version__))
                log_append("  Location: "+str(scipy.__file__))
        except:
            preferences.package_skimage = False
        
        try:
            import skimage
            if self.output_log:
                log_append("[NijiGPen Info] Package Scikit-Image:")
                log_append("  Version: "+str(skimage.__version__))
                log_append("  Location: "+str(skimage.__file__))
        except:
            preferences.package_skimage = False
        
        try:
            import triangle
            if self.output_log:
                log_append("[NijiGPen Info] Package Triangle:")
                log_append("  Version: "+str(triangle.__version__))
                log_append("  Location: "+str(triangle.__file__))
        except:
            preferences.package_triangle = False
        return {"FINISHED"}

class InstallDependency(bpy.types.Operator):
    bl_idname = "nijigp.dependencies_install"
    bl_label = "Install"
    bl_description = ("Manage packages through pip")
    bl_options = {"REGISTER", "INTERNAL"}

    package_name: bpy.props.StringProperty()
    def execute(self, context):
        res = modify_package('install','--no-input', self.package_name)
        if res:
            self.report({"INFO"}, "Python package installed successfully.")
            log_append("[NijiGPen Info] Python package installed successfully.")
            
            # Check if the custom site-package path needs to be updated
            installed_package_info = run_command([sys.executable, '-m', 'pip', 'show', self.package_name], output_log=False, return_full_result=True)
            custom_lib_path = bpy.context.preferences.addons[__package__].preferences.custom_lib_path
            for str in installed_package_info:
                if str.startswith("Location:"):
                    site_packages_path = str[9:].strip()
                    if site_packages_path not in sys.path:
                        bpy.context.preferences.addons[__package__].preferences.custom_lib_path = site_packages_path
                        bpy.ops.nijigp.apply_custom_lib_path(output_log=False)
                        log_append("[NijiGPen Info] Package Search Path Updated Automatically.")
                        return {"FINISHED"}
                        
        else:
            self.report({"ERROR"}, "Cannot install the required package.")
            log_append("[NijiGPen Error] Cannot install the required package.")
            
        return {"FINISHED"}

class RemoveDependency(bpy.types.Operator):
    bl_idname = "nijigp.dependencies_remove"
    bl_label = "Remove"
    bl_description = ("Manage packages through pip")
    bl_options = {"REGISTER", "INTERNAL"}

    package_name: bpy.props.StringProperty()
    def execute(self, context):
        res = modify_package('uninstall','-y', self.package_name)
        self.report({"INFO"}, "Please restart Blender to apply the changes.")
        log_append("[NijiGPen Info] Please restart Blender to apply the changes.")
        return {"FINISHED"}

class NijiGPAddonPreferences(bpy.types.AddonPreferences):
    bl_idname = __package__

    custom_lib_path: bpy.props.StringProperty(
        name='Custom Site-Packages Path',
        subtype='DIR_PATH',
        description='An additional directory that the add-on will try to load packages from',
        default=''
    )
    cache_folder: bpy.props.StringProperty(
        name='Cache Folder',
        subtype='DIR_PATH',
        description='Location storing temporary files. Use the default temporary folder when empty',
        default=''
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
    show_full_logs: bpy.props.BoolProperty(
        name='Show Full Logs',
        default=True
    )
    captured_logs = []
    
    shortcut_button_enabled: bpy.props.BoolProperty(
        name='Enable Viewport Gizmos',
        description='Add a group of buttons at the bottom of the 3D view that brings better touchscreen control',
        default=True
    )
    shortcut_button_style: bpy.props.EnumProperty(
            name='Display Location',
            items=[ ('TOP', 'Top', ''),
                    ('BOTTOM', 'Bottom', ''),
                    ('RIGHT', 'Right', ''),
                    ('LEFT', 'Left', '')],
            default='BOTTOM',
            description='The way of displaying the button group'
    )
    shortcut_button_size: bpy.props.FloatProperty(
        name='Button Size',
        default=18, min=10, max=40
    )
    shortcut_button_spacing: bpy.props.FloatProperty(
        name='Button Spacing',
        default=4, min=1, max=8,
        description='Distance between buttons'
    )
    shortcut_button_location: bpy.props.FloatVectorProperty(
        name='Location Offset',
        description='The position of the shortcut buttons. Zero means buttons in the center. Positive/negative values means buttons in the right/left',
        default=(0,0), soft_min=-2000, soft_max=2000, size=2
    )
    tool_shortcut_confirm: bpy.props.EnumProperty(
            name='Confirm',
            items=[ ('MIDDLEMOUSE', 'Middle Click', ''),
                    ('BUTTON4MOUSE', 'Mouse4 (Back)', ''),
                    ('BUTTON5MOUSE', 'Mouse5 (Forward)', ''),
                    ('BUTTON6MOUSE', 'Mouse6', ''),
                    ('BUTTON7MOUSE', 'Mouse7', '')],
            default='BUTTON5MOUSE',
            description='The mouse button to confirm the result of a viewport tool'
    )
    tool_shortcut_cancel: bpy.props.EnumProperty(
            name='Cancel',
            items=[ ('MIDDLEMOUSE', 'Middle Click', ''),
                    ('BUTTON4MOUSE', 'Mouse4 (Back)', ''),
                    ('BUTTON5MOUSE', 'Mouse5 (Forward)', ''),
                    ('BUTTON6MOUSE', 'Mouse6', ''),
                    ('BUTTON7MOUSE', 'Mouse7', '')],
            default='BUTTON4MOUSE',
            description='The mouse button to cancel the ongoing operation of a viewport tool'
    )

    def draw(self, context):
        layout = self.layout
        wiki_url = "https://chsh2.github.io/nijigp/docs/get_started/installation/"

        # Dependency manager
        row = layout.row()
        row.label(text="Dependency Management", icon="PREFERENCES")
        row.separator()
        row.operator("wm.url_open", text="Help", icon="HELP").url = wiki_url
        box1 = layout.box()
        
        # Custom package path
        row = box1.row()
        row.label(text='Custom Package Path:', icon='DECORATE_KEYFRAME')
        row.separator()
        row.operator("nijigp.apply_custom_lib_path", text='Apply', icon="FILE_REFRESH")
        column = box1.box().column(align=True)
        column.prop(self, "custom_lib_path", text='Site-Packages')

        # Summary table
        row = box1.row()
        row.label(text='Summary:', icon='DECORATE_KEYFRAME')
        row.separator()
        row.operator("nijigp.check_dependencies", text="Check", icon="FILE_REFRESH")
        table_key = ['[Package]', '[Status]', '[Actions]','']
        packages = [{'name': 'PyClipper',
                        'signal': self.package_pyclipper, 
                        'package':"pyclipper"},
                    {'name': 'Skimage & SciPy', 
                        'signal': self.package_skimage, 
                        'package':"scikit-image"},
                    {'name': 'Triangle',
                        'signal': self.package_triangle, 
                        'package':"triangle"},]
        column = box1.box().column(align=True)
        row = column.row()
        for key in table_key:
            row.label(text=key)
        for p in packages:
            row = column.row()
            row.label(text=p['name'])
            if p['signal']:
                row.label(text='OK')
            else:
                row.label(text='Not Installed')
            row.operator("nijigp.dependencies_install").package_name = p['package']
            row.operator("nijigp.dependencies_remove").package_name = p['package']
        
        # Show captured logs
        row = box1.row()
        row.label(text='Logs:', icon='DECORATE_KEYFRAME')
        row.separator()
        row.prop(self, 'show_full_logs')
        row.operator("nijigp.clear_logs", text='Clear', icon='TRASH')
        column = box1.box().column(align=True)
        oldest_log = 0 if self.show_full_logs else max(0,len(self.captured_logs)-5)
        if oldest_log > 0:
            column.label(text="...")
        for i in range(oldest_log, len(self.captured_logs)):
            column.label(text=self.captured_logs[i])

        # UI setting
        layout.separator(factor=2)
        row = layout.row()
        row.label(text='UI Management', icon='PREFERENCES')
        box2 = layout.box()
        row = box2.row()
        row.label(text='Navigation Shortcuts:', icon='DECORATE_KEYFRAME')
        row.separator()
        row.operator("gpencil.nijigp_refresh_gizmo", text='Apply', icon="FILE_REFRESH")
        subbox = box2.box()
        row = subbox.row()
        row.prop(self, 'shortcut_button_enabled')
        row.separator()
        row.prop(self, 'shortcut_button_style', text='')
        if self.shortcut_button_enabled:
            row = subbox.row()
            row.label(text='Button Display:')
            row.separator()
            row.prop(self, 'shortcut_button_size', text='Size')
            row.prop(self, 'shortcut_button_spacing', text='Spacing')
            row = subbox.row()
            row.prop(self, 'shortcut_button_location')
            
        row = box2.row()
        row.label(text='Interactive Tool Keymap:', icon='DECORATE_KEYFRAME')
        subbox = box2.box()
        row = subbox.row()
        row.prop(self, 'tool_shortcut_confirm')
        row = subbox.row()
        row.prop(self, 'tool_shortcut_cancel')
        
        layout.separator(factor=2)
        row = layout.row()
        row.label(text='Cache Folder', icon='PREFERENCES')
        row.separator()            
        row.prop(self, 'cache_folder', text='')
