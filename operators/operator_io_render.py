import os
import bpy
from mathutils import Vector
from bpy_extras.io_utils import ExportHelper
from ..utils import *
from ..api_router import *
from ..resources import get_cache_folder
from ..file_formats import PsdFileWriter, PsdLayer

class MultiLayerRenderOperator(bpy.types.Operator, ExportHelper):
    """Render to image formats (e.g., PSD) that preserve the layer structure of the active Grease Pencil object"""
    bl_idname = "gpencil.nijigp_multilayer_render"
    bl_label = "Multi-Layer Render"
    bl_category = 'View'
    bl_options = {'REGISTER'}

    filename_ext = ".psd"
    filter_glob: bpy.props.StringProperty(
        default="*.psd",
        options={'HIDDEN'},
        maxlen=255,  # Max internal buffer length, longer would be clamped.
    )
    render_target: bpy.props.EnumProperty(            
            name='Render Target',
            items=[ ('ACTIVE', 'Active Object', ''),
                    ('ALL', 'All GPencil Objects', '')],
            default='ALL'
    )
    bake_masks: bpy.props.BoolProperty(
            name='Bake Layer Masks',
            default=True,
            description='Clip the layer image according to its masks. Please notice that inverted masks are not supported and will be ignored'
    ) 
    
    def draw(self, context):
        layout = self.layout
        row = layout.row()
        row.label(text='Render Target:')
        row.prop(self, 'render_target', text='')
        layout.prop(self, 'bake_masks')
    
    def execute(self, context):
        import numpy as np
        import uuid
        
        scene = context.scene
        if self.render_target=='ACTIVE':
            gp_obj_list = [context.object] if obj_is_gp(context.object) else []
        else:
            gp_obj_list = [obj for obj in bpy.data.objects if obj_is_gp(obj)]
            # Sort objects by the depth to the active camera if feasible
            if scene.camera:
                camera_loc = scene.camera.matrix_world.translation
                camera_vec = scene.camera.matrix_world.to_quaternion() @ Vector((0.0, 0.0, -1.0))
                gp_obj_list.sort(key=lambda o: camera_vec.dot(o.matrix_world.translation - camera_loc), reverse=True)
        
        def render_and_load(filepath):
            """Render an image and read its pixels into a matrix"""
            scene.render.filepath = filepath
            bpy.ops.render.render(write_still=True)
            img_obj = bpy.data.images.load(filepath)
            img_W, img_H = img_obj.size[0], img_obj.size[1]
            img_mat = np.array(img_obj.pixels).reshape(img_H,img_W, img_obj.channels)
            img_mat = np.flipud(img_mat) * 255
            bpy.data.images.remove(img_obj)
            return img_mat
        
        # Save current render-related configurations
        render_setting = {}
        hidden_objects = {}
        gplayers_info = []
        layer_group_info = []
        for obj in bpy.data.objects:
            hidden_objects[obj.name] = obj.hide_render
        for gp_obj in gp_obj_list:
            gplayers_info.append([])
            for layer in gp_obj.data.layers:
                gplayers_info[-1].append((layer.hide, layer.blend_mode, layer.opacity))
            layer_group_info.append({})
            if is_gpv3():
                for group in gp_obj.data.layer_groups:
                    layer_group_info[-1][group.name] = group.hide
                    group.hide = False
        render_setting['film_transparent'] = scene.render.film_transparent
        render_setting['filepath'] = scene.render.filepath
        render_setting['file_format'] = scene.render.image_settings.file_format
        render_setting['color_mode'] = scene.render.image_settings.color_mode
        render_setting['color_depth'] = scene.render.image_settings.color_depth
        render_setting['compression'] = scene.render.image_settings.compression
        
        # Initialize a new render configuration
        scene.render.image_settings.file_format = 'PNG'
        scene.render.image_settings.color_mode = 'RGBA'
        scene.render.image_settings.color_depth = '8'
        scene.render.image_settings.compression = 0
        
        # Render the merged image
        cache_folder = get_cache_folder()
        merged_img_mat = render_and_load(os.path.join(cache_folder,'Merged.png'))
        
        # Render the scene without target GPencil objects
        for gp_obj in gp_obj_list:
            gp_obj.hide_render = True
        scene.render.film_transparent = True
        scene_img_mat = render_and_load(os.path.join(cache_folder,'Scene.png'))
        img_W, img_H = scene_img_mat.shape[1], scene_img_mat.shape[0]
        
        # Hide all objects and render the background
        scene.render.film_transparent = False
        for obj in bpy.data.objects:
            if obj.type!='CAMERA' and obj.type!='LIGHT':
                obj.hide_render = True
        background_img_mat = render_and_load(os.path.join(cache_folder,'Background.png'))
        scene.render.film_transparent = True
        
        # Render each GPencil layer and load the output image
        layer_img_mats = []
        for i,gp_obj in enumerate(gp_obj_list):
            name_to_idx = {}
            gp_obj.hide_render = False
            for layer in gp_obj.data.layers:
                layer.hide = True
            layer_img_mats.append([])
            for layer in gp_obj.data.layers:
                layer.hide, layer.blend_mode, layer.opacity = False, 'REGULAR', 1
                layer_img_mats[-1].append( render_and_load(os.path.join(cache_folder,str(uuid.uuid4())+'.png')) )
                layer.hide = True
                name_to_idx[layer.info] = len(layer_img_mats[-1]) - 1
            gp_obj.hide_render = True
        
            # Adjust the alpha channel according to the masking relationship
            if self.bake_masks:
                layer_masks = []
                # Calculate masks in the first loop
                for layer in gp_obj.data.layers:
                    layer_idx = name_to_idx[layer.info]
                    mask_mat = np.zeros((img_H, img_W))
                    is_mask_valid = False
                    if layer.use_mask_layer:
                        for mask in layer.mask_layers:
                            mask_idx = name_to_idx[mask.name]
                            # Currently, invert masks are not supported
                            if (not gplayers_info[i][mask_idx][0]) and (not mask.hide) and (not mask.invert):
                                is_mask_valid = True
                                mask_mat += (layer_img_mats[-1][mask_idx][:,:,3] > 0)
                    if is_mask_valid:
                        layer_masks.append(mask_mat)
                    else:
                        layer_masks.append(np.ones((img_H, img_W)) * 255)
                # Apply masks in the second loop
                for j,layer in enumerate(gp_obj.data.layers):
                    layer_idx = name_to_idx[layer.info]    
                    layer_img_mats[-1][layer_idx][:,:,3][layer_masks[j] < 1] = 0

        # Convert Blender image objects to PSD format
        psd_template = PsdFileWriter(height=img_H, width=img_W)
        psd_template.set_merged_img(merged_img_mat)
        psd_template.append_layer(PsdLayer(background_img_mat, name='Background', hide=render_setting['film_transparent']))
        psd_template.append_layer(PsdLayer(scene_img_mat, name='3D Scene'))
             
        for i,gp_obj in enumerate(gp_obj_list):
            psd_template.append_layer(PsdLayer(np.zeros((1, 1, 4)), name=gp_obj.name, divider_type=3))
            
            # For GPv2, just add each layer following the original order
            # For GPv3, need to check if each layer belongs to groups and add separators
            layer_ancestors = [[] for _ in range(len(gp_obj.data.layers))]
            folders_for_layer = [[] for _ in range(len(gp_obj.data.layers))]
            separators_for_layer = [[] for _ in range(len(gp_obj.data.layers))]
            if is_gpv3():
                for j,layer in enumerate(gp_obj.data.layers):
                    p = layer
                    while p.parent_group is not None:
                        group_name = p.parent_group.name
                        # Add a separator at the end of a folder
                        if j == 0 or p.parent_group not in layer_ancestors[j-1]:
                            separators_for_layer[j].append(PsdLayer(np.zeros((1, 1, 4)), name=group_name, divider_type=3))
                        # Add the header of the folder if this is the top layer
                        if j == len(gp_obj.data.layers)-1:
                            folders_for_layer[j].append(PsdLayer(np.zeros((1, 1, 4)), name=group_name, 
                                                                   hide=layer_group_info[i][p.parent_group.name], divider_type=1))
                        layer_ancestors[j].append(p.parent_group)
                        p = p.parent_group
                    # Check the previous layer for unpaired folders which need headers
                    if j > 0:
                        for group in layer_ancestors[j-1]:
                            if group not in layer_ancestors[j]:
                                folders_for_layer[j-1].append(PsdLayer(np.zeros((1, 1, 4)), name=group.name, 
                                                                    hide=layer_group_info[i][group.name], divider_type=1))
                    
            # Writing each layer to PSD from bottom to top: seperator -> layer -> folder
            for j,layer in enumerate(gp_obj.data.layers):
                for seperator in reversed(separators_for_layer[j]):
                    psd_template.append_layer(seperator)
                psd_template.append_layer(PsdLayer(layer_img_mats[i][j],
                                                name=layer.info,
                                                hide=gplayers_info[i][j][0],
                                                opacity=gplayers_info[i][j][2],
                                                blend_mode_key=gplayers_info[i][j][1]))
                for folder in folders_for_layer[j]:
                    psd_template.append_layer(folder)
            # Finally, put the whole GPencil object into a bigger folder
            psd_template.append_layer(PsdLayer(np.zeros((1, 1, 4)), name=gp_obj.name, divider_type=1))
        psd_fd = open(self.filepath, 'wb')
        psd_fd.write(psd_template.get_file_bytes())
        psd_fd.close()
        
        # Restore configuration options
        for obj in bpy.data.objects:
            obj.hide_render = hidden_objects[obj.name]
        for i,gp_obj in enumerate(gp_obj_list):
            for j,layer in enumerate(gp_obj.data.layers):
                layer.hide, layer.blend_mode, layer.opacity = gplayers_info[i][j]
            if is_gpv3():
                for group in gp_obj.data.layer_groups:
                    group.hide = layer_group_info[i][group.name]
        scene.render.film_transparent = render_setting['film_transparent']
        scene.render.filepath = render_setting['filepath']
        scene.render.image_settings.file_format = render_setting['file_format']
        scene.render.image_settings.color_mode = render_setting['color_mode']
        scene.render.image_settings.color_depth = render_setting['color_depth']
        scene.render.image_settings.compression = render_setting['compression']
                           
        return {'FINISHED'}