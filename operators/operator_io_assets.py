import os
import bpy
import struct
from bpy_extras.io_utils import ImportHelper
from ..file_formats import GbrParser, Abr1Parser, Abr6Parser
from ..resources import get_cache_folder

class ImportBrushOperator(bpy.types.Operator, ImportHelper):
    """Extract textures of ABR or GBR brushes and append them to the current file"""
    bl_idname = "gpencil.nijigp_import_brush"
    bl_label = "Import Brushes"
    bl_category = 'View'
    bl_options = {'REGISTER', 'UNDO'}

    directory: bpy.props.StringProperty(subtype='DIR_PATH')
    files: bpy.props.CollectionProperty(type=bpy.types.OperatorFileListElement)
    filepath = bpy.props.StringProperty(name="File Path", subtype='FILE_PATH')
    filter_glob: bpy.props.StringProperty(
        default='*.gbr;*.abr',
        options={'HIDDEN'}
    )

    texture_usage: bpy.props.EnumProperty(
            name='Texture Usage',
            items=[('IMAGE', 'New Images', ''),
                    ('MATERIAL', 'New Materials', ''),
                    ('BRUSH', 'New Brushes', '')],
            default='BRUSH'
    )
    color_mode: bpy.props.EnumProperty(
            name='Color Mode',
            items=[('WHITE', 'White', ''),
                    ('BLACK', 'Black', ''),
                    ('GRAYSCALE', 'Grayscale', '')],
            default='WHITE'
    )
    icon_save_path: bpy.props.EnumProperty(
            name='Icon Folder',
            items=[('PROJECT', 'Folder of Blend File', ''),
                    ('BRUSH', 'Folder of Brush File', ''),
                    ('TMP', 'Temporary Folder', '')],
            default='BRUSH'
    )
    alpha_clip: bpy.props.BoolProperty(
            name='Alpha Clip',
            default=False,
            description='If applied, the transparency of the brush pixels will be either 0 or 1'
    )
    keep_aspect_ratio: bpy.props.BoolProperty(
            name='Keep Aspect Ratio',
            default=True,
            description='If applied, pads the texture to a square to display it without distortion.'
    )
    template_brush: bpy.props.StringProperty(
            name='Template Brush',
            description='When creating new brushes, copy attributes from the selected brush',
            default='Airbrush',
            search=lambda self, context, edit_text: [brush.name for brush in bpy.data.brushes if brush.use_paint_grease_pencil and brush.gpencil_tool=='DRAW']
    )
    uv_randomness: bpy.props.FloatProperty(
            name='UV Randomness',
            default=1, min=0, max=1,
            description='Rotate the brush texture randomly for each stroke point'
    )
    hardness: bpy.props.FloatProperty(
            name='Hardness',
            default=1, min=0, max=1,
            description='Whether adding opacity gradient to the texture'
    )
    input_samples: bpy.props.IntProperty(
            name='Input Samples',
            default=0, min=0, max=10,
            description='Whether generating intermediate points for fast movement'
    )
    override_uv_randomness: bpy.props.BoolProperty(name='UV Randomness', default=True)
    override_hardness: bpy.props.BoolProperty(name='Hardness', default=True)
    override_input_samples: bpy.props.BoolProperty(name='Input Samples', default=False)

    def draw(self, context):
        layout = self.layout
        row = layout.row()
        row.label(text = 'Import Brushes as: ')
        row.prop(self, "texture_usage", text="")
        layout.label(text = 'Texture Options:')
        box1 = layout.box()
        row = box1.row()
        row.label(text = 'Brush Color: ')
        row.prop(self, "color_mode", text="")    
        row = box1.row()
        row.label(text = 'Alpha Clip: ')
        row.prop(self, "alpha_clip", text="")
        row = box1.row()
        row.label(text = 'Keep Aspect Ratio: ')
        row.prop(self, "keep_aspect_ratio", text="")
        if self.texture_usage == "BRUSH":
            layout.label(text = 'Brush Options:')
            box2 = layout.box()
            row = box2.row()
            row.label(text = 'Template Brush: ')
            row.prop(self, "template_brush", text="", icon='BRUSH_DATA')
            box2.label(text = 'Override Parameters: ')
            row = box2.row()
            row.prop(self, 'override_uv_randomness')
            row.prop(self, 'uv_randomness', text='')
            row = box2.row()
            row.prop(self, 'override_hardness')
            row.prop(self, 'hardness', text='')
            row = box2.row()
            row.prop(self, 'override_input_samples')
            row.prop(self, 'input_samples', text='')
            row = box2.row()
            row.label(text = 'Save Icons to: ')
            row.prop(self, "icon_save_path", text="")

    def execute(self, context):
        import numpy as np

        # Determine the location to save icons. Create a new folder if necessary
        if self.texture_usage == 'BRUSH':
            if self.icon_save_path=='PROJECT' and len(bpy.path.abspath('//'))>0:
                icon_dir = bpy.path.abspath('//')
            elif self.icon_save_path=='BRUSH':
                icon_dir = self.directory
            else:
                icon_dir = get_cache_folder()
            icon_dir =   os.path.join(icon_dir, 'gp_brush_icons')
            if not os.path.exists(icon_dir):
                os.makedirs(icon_dir)
            
        for f in self.files:
            filename = os.path.join(self.directory, f.name)
            fd = open(filename, 'rb')

            # Determine the software that generates the brush file
            if f.name.split('.')[-1] == 'gbr':
                parser = GbrParser(fd.read())
            else:
                bytes = fd.read()
                major_version = struct.unpack_from('>H',bytes)[0]
                if major_version > 5:
                    parser = Abr6Parser(bytes)
                else:
                    parser = Abr1Parser(bytes)
            if not parser.check():
                self.report({"ERROR"}, "The file format of the brush cannot be recognized.")
                return {'FINISHED'}
            
            parser.parse()
            for i,brush_mat in enumerate(parser.brush_mats):
                if len(parser.brush_mats) == 1:
                    brush_name = f.name.split('.')[0]
                else:
                    brush_name = f.name.split('.')[0] + '_' + str(i)
                img_H, img_W = brush_mat.shape[0], brush_mat.shape[1]

                # Extract and convert an image texture
                if len(brush_mat.shape)==3:     # RGBA brush, for GBR only
                    image_mat = brush_mat.copy()
                else:
                    image_mat = brush_mat.reshape((img_H, img_W, 1)).repeat(4, axis=2)
                if self.color_mode == 'WHITE':
                    image_mat[:,:,0] = (image_mat[:,:,3] > 0) * 255
                    image_mat[:,:,1] = (image_mat[:,:,3] > 0) * 255
                    image_mat[:,:,2] = (image_mat[:,:,3] > 0) * 255
                elif self.color_mode == 'BLACK':
                    image_mat[:,:,0] = (image_mat[:,:,3] < 1) * 255
                    image_mat[:,:,1] = (image_mat[:,:,3] < 1) * 255
                    image_mat[:,:,2] = (image_mat[:,:,3] < 1) * 255
                    
                if self.alpha_clip:
                    image_mat[:,:,3] = (image_mat[:,:,3] > 127) * 255
                    
                if self.keep_aspect_ratio:
                    img_L = max(img_H, img_W)
                    offset_H, offset_W = (img_L-img_H)//2, (img_L-img_W)//2
                    square_img_mat = np.zeros((img_L, img_L, 4))
                    square_img_mat[offset_H:offset_H+img_H, offset_W:offset_W+img_W, :] = image_mat
                    image_mat, img_H, img_W = square_img_mat, img_L, img_L
                    
                # Convert texture to Blender data block
                img_obj = bpy.data.images.new(brush_name, img_W, img_H, alpha=True, float_buffer=False)
                img_obj.pixels = np.flipud(image_mat).ravel() / 255.0
                img_obj.pack()
                
                # Create GPencil material
                if self.texture_usage != 'IMAGE':
                    new_material = bpy.data.materials.new(brush_name)
                    bpy.data.materials.create_gpencil_data(new_material)
                    new_material.grease_pencil.show_stroke = True
                    new_material.grease_pencil.mode = 'BOX'
                    new_material.grease_pencil.stroke_style = 'TEXTURE'
                    new_material.grease_pencil.mix_stroke_factor = 1
                    new_material.grease_pencil.stroke_image = img_obj
                
                # Create GPencil draw brush
                if self.texture_usage == 'BRUSH':
                    new_brush: bpy.types.Brush = bpy.data.brushes[self.template_brush].copy()
                    new_brush.name = brush_name
                    new_brush.use_custom_icon = True
                    new_brush.gpencil_settings.use_material_pin = True
                    new_brush.gpencil_settings.material = new_material
                    if self.override_uv_randomness:
                        new_brush.gpencil_settings.use_settings_random = (self.uv_randomness > 0)
                        new_brush.gpencil_settings.uv_random = self.uv_randomness
                    if self.override_input_samples:
                        new_brush.gpencil_settings.input_samples = self.input_samples
                    if self.override_hardness:
                        new_brush.gpencil_settings.hardness = self.hardness

                    # Create an icon by scaling the brush texture down
                    icon_obj = img_obj.copy()
                    icon_obj.name = "icon_"+brush_name
                    icon_filepath = os.path.join(icon_dir, brush_name+'.png')
                    icon_obj.filepath = icon_filepath
                    icon_obj.scale(256,256)
                    icon_obj.save()
                    new_brush.icon_filepath = icon_filepath
                    bpy.data.images.remove(icon_obj)
                    
            fd.close()

        return {'FINISHED'}