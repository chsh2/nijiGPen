import bpy
import os

MATERIAL_PREFIX = {'MESH': 'NijiGP_Mesh: ',
                   'NORMAL': 'NijiGP_Normal: '}

def get_cache_folder():
    preferences = bpy.context.preferences.addons[__package__].preferences
    if len(preferences.cache_folder)>0:
        return preferences.cache_folder
    else:
        return bpy.app.tempdir

def get_workspace_tool_icon(file_name):
    """
    Get a pre-defined .dat file as the icon of a workspace tool
    """
    script_file = os.path.realpath(__file__)
    return os.path.join(os.path.dirname(script_file), 'res/icons', file_name)
    
def get_library_blend_file():
    file_name = 'res/library.blend'
    script_file = os.path.realpath(__file__)
    directory = os.path.dirname(script_file)
    file_path = os.path.join(directory, file_name)
    return file_path

def get_material_list(mesh_type, engine):
    if mesh_type == 'MESH' and engine!='CYCLES':
        return ['Principled BSDF',
                'Simple Toon',
                'Simple Comic',
                'MatCap Liquid',
                'MatCap Metal',
                'Isoline Map']
    elif mesh_type == 'MESH' and engine=='CYCLES':
        return ['Principled BSDF',
                'Cycles Toon',
                'MatCap Liquid',
                'MatCap Metal',
                'Isoline Map']
    elif mesh_type == 'NORMAL' and engine!='CYCLES':
        return ['Principled BSDF',
                'Simple Toon',
                'Simple Comic',
                'MatCap Liquid',
                'MatCap Metal',
                'Normal Map']
    elif mesh_type == 'NORMAL' and engine=='CYCLES':
        return ['Principled BSDF',
                'Cycles Toon',
                'MatCap Liquid',
                'MatCap Metal',
                'Normal Map']
    
def append_material(context, mesh_type, material_name, reuse = True, operator = None):
    """
    Import a pre-defined material from the library blend file
    """
    true_name = MATERIAL_PREFIX[mesh_type] + material_name
    if reuse and true_name in bpy.data.materials:
        return bpy.data.materials[true_name]
    
    # Record the status before appending to check what is appended later
    material_set = set(bpy.data.materials[:])
    mode = context.object.mode
    bpy.ops.object.mode_set(mode='OBJECT')

    file_path = get_library_blend_file()
    inner_path = 'Material'
    bpy.ops.wm.append(
        filepath=os.path.join(file_path, inner_path, true_name),
        directory=os.path.join(file_path, inner_path),
        filename=true_name
    )
    bpy.ops.object.mode_set(mode=mode)
    new_materials = set(bpy.data.materials[:])-material_set

    if len(new_materials)>0:
        return list(new_materials)[0]
    else:
        if operator:
            operator.report({"WARNING"}, "Material not found. Please select the material again.")
        return

def append_geometry_nodes(context, node_tree_name='NijiGP Stop Motion'):
    if node_tree_name in bpy.data.node_groups:
        return bpy.data.node_groups[node_tree_name]
    
    mode = context.object.mode
    bpy.ops.object.mode_set(mode='OBJECT')

    file_path = get_library_blend_file()
    inner_path = 'NodeTree'
    bpy.ops.wm.append(
        filepath=os.path.join(file_path, inner_path, node_tree_name),
        directory=os.path.join(file_path, inner_path),
        filename=node_tree_name
    )

    bpy.ops.object.mode_set(mode=mode)
    return bpy.data.node_groups[node_tree_name]