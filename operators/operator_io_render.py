import os
import bpy
from bpy_extras.io_utils import ExportHelper
from ..utils import *
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
    
    def execute(self, context):
        import numpy as np
        gp_obj = context.object
        scene = context.scene
        
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
        for obj in bpy.data.objects:
            hidden_objects[obj.name] = obj.hide_render
        for layer in gp_obj.data.layers:
            gplayers_info.append((layer.hide, layer.blend_mode, layer.opacity))
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
        
        # Render the scene without active GPencil
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
        gp_obj.hide_render = False
        for layer in gp_obj.data.layers:
            layer.hide = True
        for i,layer in enumerate(gp_obj.data.layers):
            layer.hide, layer.blend_mode, layer.opacity = False, 'REGULAR', 1
            layer_img_mats.append( render_and_load(os.path.join(cache_folder,str(layer.info)+'.png')) )
            layer.hide = True
        
        # Convert Blender image objects to PSD format
        psd_template = PsdFileWriter(height=img_H, width=img_W)
        psd_template.set_merged_img(merged_img_mat)
        psd_template.append_layer(PsdLayer(background_img_mat, name='Background', hide=render_setting['film_transparent']))
        psd_template.append_layer(PsdLayer(scene_img_mat, name='3D Scene'))
        for i,layer in enumerate(gp_obj.data.layers):
            psd_template.append_layer(PsdLayer(layer_img_mats[i],
                                               name=layer.info,
                                               hide=gplayers_info[i][0],
                                               opacity=gplayers_info[i][2],
                                               blend_mode_key=gplayers_info[i][1]))
        psd_fd = open(self.filepath, 'wb')
        psd_fd.write(psd_template.get_file_bytes())
        psd_fd.close()
        
        # Restore configuration options
        for obj in bpy.data.objects:
            obj.hide_render = hidden_objects[obj.name]
        for i,layer in enumerate(gp_obj.data.layers):
            layer.hide, layer.blend_mode, layer.opacity = gplayers_info[i]
        scene.render.film_transparent = render_setting['film_transparent']
        scene.render.filepath = render_setting['filepath']
        scene.render.image_settings.file_format = render_setting['file_format']
        scene.render.image_settings.color_mode = render_setting['color_mode']
        scene.render.image_settings.color_depth = render_setting['color_depth']
        scene.render.image_settings.compression = render_setting['compression']
                           
        return {'FINISHED'}