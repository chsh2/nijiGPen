import os
import bpy
import struct
from bpy_extras.io_utils import ImportHelper
from mathutils import Color
from ..file_formats import GbrParser, GihParser, Abr1Parser, Abr6Parser, BrushsetParser, SutParser, PaletteParser
from ..resources import get_cache_folder
from ..utils import *
from ..api_router import *

def set_brush_color_randomness(brush, attribute, value):
    """Depending on Blender verions, color randomness is available in different modes and also has different attribute names"""
    new_attr = f'{attribute}_jitter'
    if hasattr(brush, new_attr):
        setattr(brush, new_attr, value)

    legacy_attr = f'random_{attribute}_factor'
    if hasattr(brush, 'gpencil_settings') and hasattr(brush.gpencil_settings, legacy_attr):
        setattr(brush.gpencil_settings, legacy_attr, value)

class ImportBrushOperator(bpy.types.Operator, ImportHelper):
    """Extract textures of brushes exported from painting software and append them to the current file"""
    bl_idname = "gpencil.nijigp_import_brush"
    bl_label = "Import Brushes"
    bl_category = 'View'
    bl_options = {'REGISTER', 'UNDO'}

    directory: bpy.props.StringProperty(subtype='DIR_PATH')
    files: bpy.props.CollectionProperty(type=bpy.types.OperatorFileListElement)
    filepath = bpy.props.StringProperty(name="File Path", subtype='FILE_PATH')
    filter_glob: bpy.props.StringProperty(
        default='*.gbr;*.gih;*.abr;*.brushset;*.brush;*.sut',
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
            default='GRAYSCALE'
    )
    icon_save_path: bpy.props.EnumProperty(
            name='Icon Folder',
            items=[('PROJECT', 'Folder of Blend File', ''),
                    ('BRUSH', 'Folder of Brush File', ''),
                    ('TMP', 'Temporary Folder', '')],
            default='BRUSH'
    )
    invert_alpha: bpy.props.BoolProperty(
            name='Invert Alpha',
            default=False,
            description='If applied, treat white as transparency instead of black for single-channel images'
    )
    alpha_clip: bpy.props.BoolProperty(
            name='Alpha Clip',
            default=False,
            description='If applied, set the alpha value of brush pixels to either 0 or 1'
    )
    keep_aspect_ratio: bpy.props.BoolProperty(
            name='Keep Aspect Ratio',
            default=True,
            description='If applied, pads the texture to a square to display it without distortion'
    )
    template_brush: bpy.props.StringProperty(
            name='Template Brush',
            description='When creating new brushes, copy attributes from the selected brush',
            default='',
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
    convert_orig_params: bpy.props.BoolProperty(
            name='Parse Brush Parameters',
            default=True,
            description='Attempt to parse the original parameters in the brush file and convert them to Grease Pencil options. '
                        'Please note that not all brush formats are supported, and the conversion may not perfectly replicate the original brush look'
    )

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
        row.prop(self, "alpha_clip")
        row.prop(self, "invert_alpha")
        row = box1.row()
        row.prop(self, "keep_aspect_ratio")
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
        layout.prop(self, 'convert_orig_params')

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
            
        total_brushes = 0
        failures = 0
        for f in self.files:
            # Determine the software that generates the brush file
            filename = os.path.join(self.directory, f.name)
            fd = open(filename, 'rb')
            parser = None
            
            try:
                if f.name.endswith('.gbr'):  
                    parser = GbrParser(fd.read())
                elif f.name.endswith('.gih'):
                    parser = GihParser(fd.read())
                elif f.name.endswith('.abr'):
                    bytes = fd.read()
                    major_version = struct.unpack_from('>H',bytes)[0]
                    if major_version > 5:
                        parser = Abr6Parser(bytes)
                    else:
                        parser = Abr1Parser(bytes)
                elif f.name.endswith('.brushset') or f.name.endswith('.brush'):
                    parser = BrushsetParser(filename)
                elif f.name.endswith('.sut'):
                    parser = SutParser(filename)

                if not parser or not parser.check():
                    self.report({"ERROR"}, f"The brush file {f.name} cannot be recognized.")
                    continue
                parser.parse()
                
            except Exception as e:
                self.report({"ERROR"}, f"Failed to parse the brush file {f.name}: {e}")
                failures += 1
                continue
            
            total_brushes += len(parser.brush_mats)
            for i,brush_mat in enumerate(parser.brush_mats):
                if len(parser.brush_mats) == 1:
                    brush_name = f.name.split('.')[0]
                else:
                    brush_name = f.name.split('.')[0] + '_' + str(i)
                img_H, img_W = brush_mat.shape[0], brush_mat.shape[1]

                # Attempt to read original parameters data
                # Currently supporting only Procreate and SUT brushes
                orig_name, orig_params, orig_type = None, None, None
                if hasattr(parser, 'get_params'):
                    orig_name, orig_params = parser.get_params(i)
                    if orig_name:
                        brush_name = orig_name
                if hasattr(parser, 'is_tex_grain') and parser.is_tex_grain[i]:
                    orig_type = 'GRAIN'

                # Extract and convert an image texture
                if len(brush_mat.shape)==3:             # RGBA brush such as SUT and some GBR
                    image_mat = brush_mat.copy()
                    if isinstance(parser, SutParser):   # SUT brushes use black patterns while others use white
                        image_mat[:,:,:3] = 255 - image_mat[:,:,:3]
                else:
                    image_mat = brush_mat.reshape((img_H, img_W, 1)).repeat(4, axis=2)
                    # In some cases, the texture alpha needs to be inverted
                    if self.invert_alpha:
                        image_mat = 255 - image_mat
                    if orig_params:
                        if orig_type == 'GRAIN' and \
                            'textureInverted' in orig_params and \
                            orig_params['textureInverted']:
                                image_mat = 255 - image_mat
                        if orig_type != 'GRAIN' and \
                            'shapeInverted' in orig_params and \
                            orig_params['shapeInverted']:
                                image_mat = 255 - image_mat

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
                    if orig_type == 'GRAIN':
                        brush_name = '(Grain) ' + brush_name
                        new_material = bpy.data.materials.new(brush_name)
                        bpy.data.materials.create_gpencil_data(new_material)
                        new_material.grease_pencil.show_stroke = False
                        new_material.grease_pencil.show_fill = True
                        new_material.grease_pencil.fill_style = 'TEXTURE'
                        new_material.grease_pencil.mix_factor = 1
                        new_material.grease_pencil.fill_image = img_obj
                    else:
                        new_material = bpy.data.materials.new(brush_name)
                        bpy.data.materials.create_gpencil_data(new_material)
                        new_material.grease_pencil.show_stroke = True
                        new_material.grease_pencil.mode = 'BOX'
                        new_material.grease_pencil.stroke_style = 'TEXTURE'
                        new_material.grease_pencil.mix_stroke_factor = 1
                        new_material.grease_pencil.stroke_image = img_obj
                
                # Create GPencil draw brush
                if self.texture_usage == 'BRUSH':
                    template_brush_name = self.template_brush
                    if self.template_brush != '':
                        new_brush = bpy.data.brushes[template_brush_name].copy()
                    else:
                        new_brush = new_gp_brush(brush_name)
                    new_brush.name = brush_name
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
                    icon_obj.name = f"icon_{f.name.split('.')[0]}_{i}"
                    icon_filepath = os.path.join(icon_dir, icon_obj.name+'.png')
                    icon_obj.filepath_raw = icon_filepath
                    icon_obj.scale(128,128)
                    icon_obj.save()

                    # Setting icon for Blender 3.x/4.x
                    if hasattr(new_brush, 'use_custom_icon') and hasattr(new_brush, 'icon_filepath'):
                        new_brush.use_custom_icon = True
                        new_brush.icon_filepath = icon_filepath
                        new_brush.asset_generate_preview()

                    new_brush.asset_mark()
                    new_brush.asset_data.description = f'Original brush file: {f.name}'

                    # Setting icon for Blender 5.x
                    if bpy.app.version >= (5, 0, 0):
                        with bpy.context.temp_override(id=new_brush):
                            bpy.ops.ed.lib_id_load_custom_preview(filepath=icon_filepath)
                    bpy.data.images.remove(icon_obj)

                # Override parameters by parsing original Procreate brush data
                if self.convert_orig_params and isinstance(parser, BrushsetParser) and orig_params:
                    if self.texture_usage != 'IMAGE':
                        if 'textureScale' in orig_params:
                            new_material.grease_pencil.texture_scale = (orig_params['textureScale'], orig_params['textureScale'])
                    if self.texture_usage == 'BRUSH':
                        if 'paintSize' in orig_params:
                            new_brush.size = int(500.0 * orig_params['paintSize'])
                        if 'plotJitter' in orig_params:
                            new_brush.gpencil_settings.pen_jitter = orig_params['plotJitter']
                        if 'plotSpacing' in orig_params:
                            new_brush.gpencil_settings.input_samples = int(10 - 10 * orig_params['plotSpacing'])
                        if 'paintOpacity' in orig_params:
                            new_brush.gpencil_settings.pen_strength = orig_params['paintOpacity']
                        if 'shapeRandomise' in orig_params:
                            new_brush.gpencil_settings.uv_random = round(orig_params['shapeRandomise'])
                        if 'dynamicsJitterSize' in orig_params:
                            new_brush.gpencil_settings.random_pressure = orig_params['dynamicsJitterSize']
                        if 'dynamicsJitterOpacity' in orig_params:
                            new_brush.gpencil_settings.random_strength = orig_params['dynamicsJitterOpacity']
                        if 'dynamicsJitterHue' in orig_params:
                            set_brush_color_randomness(new_brush, 'hue', orig_params['dynamicsJitterHue'])
                        if 'dynamicsJitterStrokeSaturation' in orig_params:
                            set_brush_color_randomness(new_brush, 'saturation', orig_params['dynamicsJitterStrokeSaturation'])
                        if 'dynamicsJitterStrokeDarkness' in orig_params:
                            set_brush_color_randomness(new_brush, 'value', orig_params['dynamicsJitterStrokeDarkness'])
                            
                # Overrider parameters by parsing original SUT brush data
                if self.convert_orig_params and isinstance(parser, SutParser) and orig_params:
                    if self.texture_usage != 'IMAGE':
                        if 'TextureScale2' in orig_params:
                            new_material.grease_pencil.texture_scale = (orig_params['TextureScale2']/100.0, orig_params['TextureScale2']/100.0)
                        if 'BrushRotation' in orig_params:
                            if orig_params['BrushRotation'] > 1.0:
                                new_material.grease_pencil.alignment_rotation = (orig_params['BrushRotation'] % 1.0) * np.pi / 2.0
                            else:
                                new_material.grease_pencil.alignment_rotation = orig_params['BrushRotation'] * np.pi / 2.0
                    if self.texture_usage == 'BRUSH':
                        if 'BrushSize' in orig_params:
                            new_brush.size = int(orig_params['BrushSize'])
                        if 'Opacity' in orig_params:
                            new_brush.gpencil_settings.pen_strength = orig_params['Opacity'] / 100.0
                        if 'BrushHardness' in orig_params:
                            new_brush.gpencil_settings.hardness = orig_params['BrushHardness'] / 100.0
                        if 'BrushInterval' in orig_params:
                            new_brush.gpencil_settings.input_samples = int(orig_params['BrushInterval'] / 2000.0)
                        if 'BrushChangePatternColor' in orig_params and orig_params['BrushChangePatternColor'] > 0:
                            if 'BrushHueChange' in orig_params:
                                set_brush_color_randomness(new_brush, 'hue', orig_params['BrushHueChange'] / 360.0)
                            if 'BrushSaturationChange' in orig_params:
                                set_brush_color_randomness(new_brush, 'saturation', orig_params['BrushSaturationChange'] / 100.0)
                            if 'BrushValueChange' in orig_params:
                                set_brush_color_randomness(new_brush, 'value', orig_params['BrushValueChange'] / 100.0)
            fd.close()
            
        if failures == 0:
            self.report({"INFO"}, f'Imported {total_brushes} brush texture(s).') 
        else:
            self.report({"WARNING"}, f'Imported {total_brushes} brush texture(s). Failed to recognize {failures} brush file(s).') 
        return {'FINISHED'}
    
class ImportSwatchOperator(bpy.types.Operator, ImportHelper):
    """Import palette or swatch files. Currently supported formats: .swatches, .aco, .xml, .txt"""
    bl_idname = "gpencil.nijigp_import_swatch"
    bl_label = "Import Swatches"
    bl_category = 'View'
    bl_options = {'REGISTER', 'UNDO'}

    directory: bpy.props.StringProperty(subtype='DIR_PATH')
    files: bpy.props.CollectionProperty(type=bpy.types.OperatorFileListElement)
    filepath = bpy.props.StringProperty(name="File Path", subtype='FILE_PATH')
    filter_glob: bpy.props.StringProperty(
        default='*.swatches;*.aco;*.xml;*.txt',
        options={'HIDDEN'}
    )
    ignore_placeholders: bpy.props.BoolProperty(
        name='Ignore Placeholders',
        default=False,
        description='Some palette files contain empty color slots, which will be ignored when this option is enabled'
    )
    tints_level: bpy.props.IntProperty(
        name='Tints and Shades',
        min=0, max=10, default=0,
        description='Extend the palette by generating tints and shades colors based on existing ones',
    )
        
    def draw(self, context):
        layout = self.layout
        layout.prop(self, 'tints_level')
        layout.prop(self, 'ignore_placeholders')
    
    def execute(self, context):
        total_palettes = 0
        for f in self.files:
            filename = os.path.join(self.directory, f.name)
            parser = PaletteParser()
            
            ret = 0
            if f.name.endswith('.swatches'):
                ret = parser.parse_from_swatches(filename, self.ignore_placeholders)
            elif f.name.endswith('.aco'):  
                ret = parser.parse_from_aco(filename)
            elif f.name.endswith('.xml'):  
                ret = parser.parse_from_xml(filename)
            elif f.name.endswith('.txt'):  
                ret = parser.parse_from_hex(filename)
            else:
                ret = parser.parse_auto(filename)
            if ret > 0:
                self.report({"ERROR"}, f'The file {f.name} belongs to an unknown format and cannot be parsed.')
                return {'FINISHED'}
                
            # Create a new palette in Blender
            if len(parser.colors) > 0:
                total_palettes += 1
                new_palette = bpy.data.palettes.new(parser.name)
                for color in parser.colors:
                    new_palette.colors.new()
                    new_palette.colors[-1].color = color
                    
            # Generate tints and shades
            if self.tints_level > 0:
                # Add padding slots for better alignment
                padding_slots = (10 - len(parser.colors))%10 if not self.ignore_placeholders else 0
                for _ in range(padding_slots):
                    new_palette.colors.new()
                padding_slots = (5 - len(parser.colors))%5 if not self.ignore_placeholders else 0
                for i in range(self.tints_level):
                    factor = (i+1) / (self.tints_level + 1)
                    # Add tints
                    for color in parser.colors:
                        new_palette.colors.new()
                        new_palette.colors[-1].color = color * (1-factor) + Color((1,1,1)) * factor  
                    for _ in range(padding_slots):
                        new_palette.colors.new()
                    # Add shades
                    for color in parser.colors:
                        new_palette.colors.new()
                        new_palette.colors[-1].color = color * (1-factor) + Color((0,0,0)) * factor  
                    for _ in range(padding_slots):
                        new_palette.colors.new()

        self.report({"INFO"}, f'Finish importing {total_palettes} palette(s).')
        return {'FINISHED'}
    
class AppendSVGOperator(bpy.types.Operator, ImportHelper):
    """Similar to the native SVG import tool, but append SVG to the active object instead of creating a new one"""
    bl_idname = "gpencil.nijigp_append_svg"
    bl_label = "Append SVG"
    bl_category = 'View'
    bl_options = {'REGISTER', 'UNDO'}

    directory: bpy.props.StringProperty(subtype='DIR_PATH')
    files: bpy.props.CollectionProperty(type=bpy.types.OperatorFileListElement)
    filepath = bpy.props.StringProperty(name="File Path", subtype='FILE_PATH')
    filter_glob: bpy.props.StringProperty(
        default='*.svg',
        options={'HIDDEN'}
    )
    
    svg_resolution: bpy.props.IntProperty(
            name='Resolution',
            min=1, max=50, default=10,
            description='Resolution of pasted SVG',
    )
    svg_scale: bpy.props.FloatProperty(
            name='Scale',
            min=0, max=100, default=50,
            description='Scale of pasted SVG',
    )
    auto_holdout: bpy.props.BoolProperty(
            name='Auto Holdout',
            default=False,
            description='Change materials of holes (SVG polygons with negative area) to holdout and move holes to front'
    )
    image_sequence: bpy.props.BoolProperty(
            name='Image Sequence',
            default=False,
            description='Import each SVG as a new frame rather than a new layer'
    )
    frame_step: bpy.props.IntProperty(
            name='Frame Step',
            default=1, min=1,
            description='The number of frames between two generated keyframes'
    )
    reuse_materials: bpy.props.BoolProperty(
            name='Reuse Materials',
            default=False,
            description='Share the materials among all appended SVG instead of creating new materials'
    ) 
    
    def draw(self, context):
        layout = self.layout
        layout.label(text = "Geometry Options:")
        box1 = layout.box()
        box1.prop(self, "svg_resolution")
        box1.prop(self, "svg_scale")
        layout.label(text = "Material Options:")
        box2 = layout.box()
        box2.prop(self, "auto_holdout")
        box2.prop(self, "reuse_materials")
        layout.label(text = "Animation Options:")
        box3 = layout.box()
        box3.prop(self, "image_sequence")
        if self.image_sequence:
            box3.prop(self, "frame_step")
            
    def execute(self, context):
        current_gp_obj = context.object
        current_material_idx = context.object.active_material_index
        current_frame_number = context.scene.frame_current
        use_multiedit = get_multiedit(current_gp_obj)
        t_mat, inv_mat = get_transformation_mat(mode=context.scene.nijigp_working_plane, gp_obj=current_gp_obj)
            

        frame_number = current_frame_number
        set_multiedit(current_gp_obj, False)
        target_layer = None
        multiframe_svg_name = None
        for f in self.files:
            # Determine the destination layer and frame to paste strokes
            if target_layer == None or not self.image_sequence:
                target_layer = current_gp_obj.data.layers.new(f.name)
            target_frame = target_layer.frames.new(frame_number)
            
            filename = os.path.join(self.directory, f.name)
            if not multiframe_svg_name:
                multiframe_svg_name = f.name
            
            # Call the native SVG import operator
            bpy.ops.object.mode_set(mode='OBJECT')
            op_import_svg(filename, self.directory, [{"name":f.name}], self.svg_resolution, self.svg_scale)

            new_gp_obj: bpy.types.Object = context.object
            if new_gp_obj == current_gp_obj or len(new_gp_obj.data.layers) < 1:
                self.report({"ERROR"}, "No data can be imported. Import failed.")
                return {'FINISHED'}            

            # Reassign materials and copy all strokes
            op_layer_merge(mode='ALL')
            mat_name_prefix = 'SVG' if self.reuse_materials else \
                              multiframe_svg_name if self.image_sequence else f.name
            for slot in new_gp_obj.material_slots:
                gp_mat = slot.material.grease_pencil
                if gp_mat.show_fill and not gp_mat.show_stroke:
                    mat_name = f'{mat_name_prefix}_Fill'
                elif gp_mat.show_stroke and not gp_mat.show_fill:
                    mat_name = f'{mat_name_prefix}_Line'
                else:
                    mat_name = f'{mat_name_prefix}_Both'
                    
                if mat_name in bpy.data.materials:
                    slot.material = bpy.data.materials[mat_name]
                else:
                    slot.material.name = mat_name
            bpy.ops.object.mode_set(mode=get_obj_mode_str('EDIT'))
            op_select_all()
            op_copy_strokes()

            # Paste all strokes to the existing GP object
            bpy.ops.object.mode_set(mode='OBJECT')
            bpy.ops.object.select_all(action='DESELECT')
            context.view_layer.objects.active = current_gp_obj
            bpy.ops.object.mode_set(mode=get_obj_mode_str('EDIT'))
            op_paste_strokes()
            
            # Delete the new object
            bpy.ops.object.mode_set(mode='OBJECT')
            bpy.ops.object.select_all(action='DESELECT')
            new_gp_obj.select_set(True)
            bpy.ops.object.delete()
            
            # Go back to the existing GP object
            bpy.ops.object.mode_set(mode='OBJECT')
            bpy.ops.object.select_all(action='DESELECT')
            current_gp_obj.select_set(True)
            context.view_layer.objects.active = current_gp_obj
            bpy.ops.object.mode_set(mode=get_obj_mode_str('EDIT'))
            op_deselect()
            
            # Transform the figure to the working 2D plane
            # Default plane is X-Z for imported SVG. Convert it to X-Y first
            # TODO: transform point handles for GPv3 Bezier curves
            z_to_y_mat = Matrix([(1,0,0), (0,0,1), (0,1,0)])
            if not is_gpv3():
                for stroke in context.object.data.layers.active.active_frame.nijigp_strokes:
                    for point in stroke.points:
                        point.co = inv_mat @ z_to_y_mat @ point.co
                                                
            # Set animation data
            if self.image_sequence:
                frame_number += self.frame_step
                context.scene.frame_set(frame_number)
        
        # Select imported strokes
        for i in range(len(self.files)):
            if self.image_sequence and i > 0:
                break
            for frame in context.object.data.layers[-i-1].frames:
                for stroke in frame.nijigp_strokes:
                    stroke.select = True
                if is_gpv3():
                    # GPv3 may convert SVG to Bezier curves. Ensure that all control points are selected
                    context.scene.frame_set(frame.frame_number)
                    bpy.ops.grease_pencil.set_selection_mode(mode='POINT')
                    bpy.ops.grease_pencil.select_linked()
        
        if self.auto_holdout:
            set_multiedit(current_gp_obj, self.image_sequence)
            bpy.ops.gpencil.nijigp_hole_processing(rearrange=True, separate_colors=True)
        
        # Recover the state
        context.object.active_material_index = current_material_idx
        context.scene.frame_set(current_frame_number)
        set_multiedit(current_gp_obj, use_multiedit)

        return {'FINISHED'}