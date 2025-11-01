import bpy
import sys
import site
import os

def log_append(str):
    bpy.context.preferences.addons[__package__].preferences.captured_logs.append(str)

def run_command(commands, output_log=True):
    from subprocess import Popen, PIPE
    with Popen(commands, stdout=PIPE, shell=False) as p:
        while p.poll() is None:
            text = p.stdout.readline().decode("utf-8")
            if len(text) > 0:
                print(text)
                if output_log:
                    log_append(text)
                    bpy.context.region.tag_redraw()
        return p.returncode

def is_writable(path, op=None):
    """Check if this add-on has the write access to a specific path"""

    def handle_exception():
        error_message = f'Cannot write to the current path. Please select a proper Custom Package Path.'
        log_append("[NijiGPen Error] "+error_message)
        if op is not None:
            op.report({"ERROR"}, error_message)
            
    try:
        os.makedirs(path, exist_ok=True)
    except:
        handle_exception()
        return False

    # This method may return incorrect results. Therefore, also try to write a temporary file
    if not os.access(path, os.W_OK):
        handle_exception()
        return False
    try:
        test_file = os.path.join(path, "._nijigp_perm_test")
        with open(test_file, "wb") as f:
            f.write(b"test")
        os.remove(test_file)
        return True
    except:
        handle_exception()
        return False

def pin_numpy_version():
    """
    Generate a constraint file to make sure Blender's NumPy will not be overwritten
    """
    import importlib.metadata
    numpy_version = importlib.metadata.version("numpy")
    constraint_file = os.path.join(bpy.app.tempdir, "nijigp_numpy_constraint.txt")
    fd = open(constraint_file, "w")
    fd.write(f"numpy=={numpy_version}\n")
    fd.close()
    return constraint_file

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
        if len(custom_lib_path) < 1:
            self.report({"INFO"}, "Please select a path.")
            return {"FINISHED"}
        if not is_writable(custom_lib_path, self):
            return {"FINISHED"}
        if custom_lib_path not in sys.path:
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
        bpy.ops.nijigp.apply_custom_lib_path(output_log=False)
        import importlib.util
        
        spec = importlib.util.find_spec("pyclipper")
        if spec is not None:
            if self.output_log:
                log_append("[NijiGPen Info] Package PyClipper:")
                log_append("  Location: "+spec.origin)
        else:
            preferences.package_pyclipper = False

        spec = importlib.util.find_spec("scipy")
        if spec is not None:
            if self.output_log:
                log_append("[NijiGPen Info] Package SciPy:")
                log_append("  Location: "+spec.origin)
        else:
            preferences.package_skimage = False
                    
        spec = importlib.util.find_spec("skimage")
        if spec is not None:
            if self.output_log:
                log_append("[NijiGPen Info] Package Scikit-Image:")
                log_append("  Location: "+spec.origin)
        else:
            preferences.package_skimage = False
        
        spec = importlib.util.find_spec("triangle")
        if spec is not None:
            if self.output_log:
                log_append("[NijiGPen Info] Package Triangle:")
                log_append("  Location: "+spec.origin)
        else:
            preferences.package_triangle = False

        return {"FINISHED"}

class InstallDependency(bpy.types.Operator):
    bl_idname = "nijigp.dependencies_install"
    bl_label = "Install"
    bl_description = ("Manage packages through pip")
    bl_options = {"REGISTER", "INTERNAL"}

    package_name: bpy.props.EnumProperty(
            items=[ ('PYCLIPPER', '', ''),
                    ('TRIANGLE', '', ''),
                    ('SKIMAGE', '', '')],
            default='PYCLIPPER',
    )
    def execute(self, context):
        # Fetch necessary information
        package_versions = {
            'PYCLIPPER': ['pyclipper<1.4'],
            'TRIANGLE': ['triangle3==20250811.1'],
            'SKIMAGE': ['scipy<1.16', 'scikit-image<0.26']
        }
        default_lib_path = site.getsitepackages()[0]
        custom_lib_path = bpy.context.preferences.addons[__package__].preferences.custom_lib_path
        use_custom_lib_path = bpy.context.preferences.addons[__package__].preferences.use_custom_lib_path
        use_custom_lib_path = (use_custom_lib_path and len(custom_lib_path)>0)
        python_exe = sys.executable

        # Activate pip, which should not fail
        res = run_command([python_exe, '-m', 'ensurepip', '--upgrade'], output_log=False)
        if res > 0:
            self.report({"ERROR"}, "Pip cannot be enabled. Please run Blender as Administrator and try again.")
            log_append("[NijiGPen Error] Pip cannot be enabled. Please run Blender as Administrator and try again.")
            return {"FINISHED"}

        # Without write access to the path, suggest the user to set a custom path and then stop
        if not is_writable(custom_lib_path if use_custom_lib_path else default_lib_path, self):
            return {"FINISHED"}

        constraint_file = pin_numpy_version()
        commands = [python_exe, '-m', 'pip', 'install', '--no-input', '--force-reinstall', '--no-cache-dir', '--upgrade-strategy', 'only-if-needed', '--constraint', constraint_file]
        if use_custom_lib_path:
            commands += ['--target', custom_lib_path]
            bpy.ops.nijigp.apply_custom_lib_path(output_log=False)

        res = run_command(commands + package_versions[self.package_name])
        if res == 0:
            self.report({"INFO"}, "Python package installed successfully.")
            log_append("[NijiGPen Info] Python package installed successfully.")    
        else:
            self.report({"ERROR"}, "Cannot install the required package.")
            log_append("[NijiGPen Error] Cannot install the required package.")
        bpy.ops.nijigp.check_dependencies(output_log=False)
        return {"FINISHED"}

class RemoveDependency(bpy.types.Operator):
    bl_idname = "nijigp.dependencies_remove"
    bl_label = "Remove"
    bl_description = ("Manage packages through pip")
    bl_options = {"REGISTER", "INTERNAL"}

    package_name: bpy.props.EnumProperty(
            items=[ ('PYCLIPPER', '', ''),
                    ('TRIANGLE', '', ''),
                    ('SKIMAGE', '', '')],
            default='PYCLIPPER',
    )
    def execute(self, context):
        package_versions = {
            'PYCLIPPER': ['pyclipper'],
            'TRIANGLE': ['triangle3', 'triangle2', 'triangle'], # This package has multiple entries in PyPI
            'SKIMAGE': ['scipy', 'scikit-image']
        }
        custom_lib_path = bpy.context.preferences.addons[__package__].preferences.custom_lib_path
        use_custom_lib_path = bpy.context.preferences.addons[__package__].preferences.use_custom_lib_path
        use_custom_lib_path = (use_custom_lib_path and len(custom_lib_path)>0)
        python_exe = sys.executable

        # Pip cannot specify the target folder for uninstallation. The user should delete files manually instead.
        if use_custom_lib_path and custom_lib_path != site.getusersitepackages():
            self.report({"INFO"}, "Please manually remove the files in your Custom Package Path.")
            log_append("[NijiGPen Info] Please manually remove the files in your Custom Package Path.")
            return {"FINISHED"}

        res = run_command([python_exe, '-m', 'ensurepip', '--upgrade'], output_log=False)
        if res > 0:
            self.report({"ERROR"}, "Pip cannot be enabled. Please run Blender as Administrator and try again.")
            log_append("[NijiGPen Error] Pip cannot be enabled. Please run Blender as Administrator and try again.")
            return {"FINISHED"}

        commands = [python_exe, '-m', 'pip', 'uninstall', '-y']
        res = run_command(commands + package_versions[self.package_name])
        if res == 0:
                self.report({"INFO"}, "Python package uninstalled.")
                log_append("[NijiGPen Info] Python package uninstalled.")   
        else:
            self.report({"ERROR"}, "Cannot uninstall the package.")
            log_append("[NijiGPen Error] Cannot uninstall the package.")
        return {"FINISHED"}

def common_lib_path_search_func(self, context, edit_text):
    """
    Show common locations of Python site-packages to the user.
    Currently, only indicate the USER_SITE path
    """
    usersite_dir = site.getusersitepackages()
    #addon_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'site-packages')
    return [usersite_dir, edit_text]

class NijiGPAddonPreferences(bpy.types.AddonPreferences):
    bl_idname = __package__

    custom_lib_path: bpy.props.StringProperty(
        name='Custom Site-Packages Path',
        subtype='DIR_PATH',
        description='An additional directory that the add-on will try to load packages from',
        default='',
        search=common_lib_path_search_func
    )
    use_custom_lib_path: bpy.props.BoolProperty(
        name='Install New Packages in Custom Path',
        description="When enabled and the custom path is set above, install Python packages there. Otherwise, install them in Blender's folder",
        default=True
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
    tool_shortcut_pan: bpy.props.EnumProperty(
            name='Pan',
            items=[ ('MIDDLEMOUSE', 'Middle Click', ''),
                    ('BUTTON4MOUSE', 'Mouse4 (Back)', ''),
                    ('BUTTON5MOUSE', 'Mouse5 (Forward)', ''),
                    ('BUTTON6MOUSE', 'Mouse6', ''),
                    ('BUTTON7MOUSE', 'Mouse7', '')],
            default='MIDDLEMOUSE',
            description='The mouse button to pan the view of a viewport tool if possible'
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
        column.prop(self, "custom_lib_path", text='Package Folder')
        column.prop(self, "use_custom_lib_path")

        # Summary table
        row = box1.row()
        row.label(text='Summary:', icon='DECORATE_KEYFRAME')
        row.separator()
        row.operator("nijigp.check_dependencies", text="Check", icon="FILE_REFRESH")
        table_key = ['[Package]', '[Status]', '[Actions]','']
        packages = [{'name': 'PyClipper',
                        'signal': self.package_pyclipper, 
                        'package':"PYCLIPPER"},
                    {'name': 'Skimage & SciPy', 
                        'signal': self.package_skimage, 
                        'package':"SKIMAGE"},
                    {'name': 'Triangle',
                        'signal': self.package_triangle, 
                        'package':"TRIANGLE"},]
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
        row = subbox.row()
        row.prop(self, 'tool_shortcut_pan')
        
        layout.separator(factor=2)
        row = layout.row()
        row.label(text='Cache Folder', icon='PREFERENCES')
        row.separator()            
        row.prop(self, 'cache_folder', text='')
