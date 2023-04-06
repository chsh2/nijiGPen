import os
import bpy
from bpy_extras.io_utils import ImportHelper
from bpy_extras import image_utils
from ..utils import *

class ImportLineImageOperator(bpy.types.Operator, ImportHelper):
    """Generate strokes from a raster image of line art using medial axis algorithm"""
    bl_idname = "gpencil.nijigp_import_lineart"
    bl_label = "Import Line Art from Image"
    bl_category = 'View'
    bl_options = {'REGISTER', 'UNDO'}

    directory: bpy.props.StringProperty(subtype='DIR_PATH')
    files: bpy.props.CollectionProperty(type=bpy.types.OperatorFileListElement)
    filepath = bpy.props.StringProperty(name="File Path", subtype='FILE_PATH')
    filter_glob: bpy.props.StringProperty(
        default='*.jpg;*.jpeg;*.png;*.tif;*.tiff;*.bmp',
        options={'HIDDEN'}
    )

    image_sequence: bpy.props.BoolProperty(
            name='Image Sequence',
            default=False,
            description='Process multiple images as a sequence'
    ) 
    frame_step: bpy.props.IntProperty(
            name='Frame Step',
            default=1, min=1,
            description='The number of frames between two generated line art keyframes'
    )  
    threshold: bpy.props.FloatProperty(
            name='Color Threshold',
            default=0.75, min=0, max=1,
            description='Threshold on lightness, below which the pixels are regarded as a part of a stroke'
    )
    median_radius: bpy.props.IntProperty(
            name='Median Filter Radius',
            default=0, min=0, soft_max=15,
            description='Denoise the image with a median filter. Disabled when the value is 0'
    )  
    size: bpy.props.FloatProperty(
            name='Size',
            default=2, min=0.001, soft_max=10,
            unit='LENGTH',
            description='Dimension of the generated strokes'
    )  
    sample_length: bpy.props.IntProperty(
            name='Sample Length',
            default=4, min=1, soft_max=32,
            description='Number of pixels of the original image between two generated stroke points'
    )
    min_length: bpy.props.IntProperty(
            name='Min Stroke Length',
            default=4, min=0, soft_max=32,
            description='Number of pixels of a line, below which a stroke will not be generated'
    )
    max_length: bpy.props.IntProperty(
            name='Max Stroke Length',
            default=512, min=0, soft_max=2048,
            description='Number of pixels of a line, above which a stroke may be cut into two or more'
    )
    generate_color: bpy.props.BoolProperty(
            name='Generate Vertex Color',
            default=False,
            description='Extract color information from the image and apply it to generated strokes'
    )
    generate_strength: bpy.props.BoolProperty(
            name='Generate Pen Strength',
            default=False,
            description='Convert the lightness of the image as strength of stroke points'
    )  
    output_material: bpy.props.StringProperty(
        name='Output Material',
        description='Draw the new stroke using this material. If empty, use the active material',
        default='',
        search=lambda self, context, edit_text: [material.name for material in context.object.data.materials if material]
    )

    def draw(self, context):
        layout = self.layout
        layout.label(text = "Image Options:")
        box1 = layout.box()
        box1.prop(self, "threshold")
        box1.prop(self, "median_radius")
        box1.prop(self, "image_sequence")
        if self.image_sequence:
            box1.prop(self, "frame_step")
        layout.label(text = "Stroke Options:")
        box2 = layout.box()
        box2.prop(self, "size")
        box2.prop(self, "sample_length")
        box2.prop(self, "min_length")
        box2.prop(self, "max_length")
        box2.prop(self, "output_material", text='Material', icon='MATERIAL')
        box2.prop(self, "generate_color")
        box2.prop(self, "generate_strength")

    def execute(self, context):
        gp_obj = context.object
        gp_layer = gp_obj.data.layers.active

        try:
            import skimage.morphology
            import skimage.filters
            import numpy as np
        except:
            self.report({"ERROR"}, "Please install Scikit-Image in the Preferences panel.")
            return {'FINISHED'}

        # Find the material slot
        output_material_idx = gp_obj.active_material_index
        if len(self.output_material) > 0:
            for i,material_slot in enumerate(gp_obj.material_slots):
                if material_slot.material and material_slot.material.name == self.output_material:
                    output_material_idx = i

        # Get or generate the starting frame
        if not gp_layer.active_frame:
            starting_frame = gp_layer.frames.new(context.scene.frame_current)
        else:
            starting_frame = gp_layer.active_frame

        # Process file paths in the case of multiple input images
        img_filepaths = []
        for f in self.files:
            img_filepaths.append(os.path.join(self.directory, f.name))
        img_filepaths.sort()

        # For image sequences, find all frames where strokes will be generated
        frame_dict = {}
        if self.image_sequence:
            for f in gp_layer.frames:
                frame_dict[f.frame_number] = f

        def process_single_image(img_filepath, frame):
            """
            Extract line art from a specific image and generate strokes in a given frame
            """
            img_obj = image_utils.load_image(img_filepath, check_existing=True) # type: bpy.types.Image
            img_W = img_obj.size[0]
            img_H = img_obj.size[1]
            img_mat = np.array(img_obj.pixels).reshape(img_H,img_W, img_obj.channels)
            img_mat = np.flipud(img_mat)

            # Preprocessing: binarization and denoise
            lumi_mat = img_mat
            if img_obj.channels > 2:
                lumi_mat = 0.2126 * img_mat[:,:,0] + 0.7152 * img_mat[:,:,1] + 0.0722 * img_mat[:,:,2]
            bin_mat = lumi_mat < self.threshold
            if img_obj.channels > 3:
                bin_mat = bin_mat * (img_mat[:,:,3]>0)

            denoised_mat = img_mat
            denoised_lumi_mat = lumi_mat
            denoised_bin_mat = bin_mat
            if self.median_radius > 0:
                footprint = skimage.morphology.disk(self.median_radius)
                denoised_mat = np.zeros(img_mat.shape)
                for channel in range(img_obj.channels):
                    denoised_mat[:,:,channel] = skimage.filters.median(img_mat[:,:,channel], footprint)
                denoised_lumi_mat = skimage.filters.median(lumi_mat, footprint)
                denoised_bin_mat = skimage.filters.median(bin_mat, footprint)

            # Get skeleton and distance information
            skel_mat, dist_mat = skimage.morphology.medial_axis(denoised_bin_mat, return_distance=True)
            line_thickness = dist_mat.max()
            dist_mat /= line_thickness

            # Convert skeleton into line segments
            search_mat = np.zeros(skel_mat.shape)
            def line_point_dfs(v, u):
                """
                Traverse a 2D matrix to get connected pixels as a line
                """
                def get_info(v, u):
                    if v<0 or v>=img_H:
                        return None
                    if u<0 or u>=img_W:
                        return None
                    if search_mat[v,u]>0:
                        return None
                    if skel_mat[v,u]==0:
                        return None
                    return (v,u)
                    
                line_points = []
                # Search along the same direction if possible, otherwise choose a similar direction
                deltas = ((0,-1), (-1,-1), (-1,0), (-1,1), (0,1), (1,1), (1,0), (1,-1))
                search_indices = (0, 1, -1, 2, -2, 3, -3, 4)
                idx0 = 0
                pos = (v,u)
                next_pos = None
                while len(line_points) <= self.max_length:
                    line_points.append(pos)
                    search_mat[pos[0],pos[1]] = 1
                    for idx1 in search_indices:
                        true_idx = (idx0+idx1)%8
                        d = deltas[true_idx]
                        ret = get_info(pos[0]+d[0], pos[1]+d[1])
                        if ret:
                            next_pos = ret
                            idx0 = true_idx
                            break
                    if not next_pos:
                        break

                    pos = next_pos
                    next_pos = None
            
                return line_points
            
            lines = []
            for v in range(img_H):
                for u in range(img_W):
                    if search_mat[v,u]==0 and skel_mat[v,u]>0:
                        lines.append(line_point_dfs(v,u))

            # Generate strokes according to line segments
            frame_strokes = frame.strokes
            scale_factor = min(img_H, img_W) / self.size
            for line in lines:
                if len(line) < self.min_length:
                    continue
                point_count = len(line) // self.sample_length
                if len(line)%self.sample_length != 1:
                    point_count += 1

                frame_strokes.new()
                frame_strokes[-1].line_width = int(line_thickness / scale_factor * LINE_WIDTH_FACTOR)
                frame_strokes[-1].material_index = output_material_idx
                frame_strokes[-1].points.add(point_count)

                for i,point in enumerate(frame_strokes[-1].points):
                    img_co = line[min(i*self.sample_length, len(line)-1)]
                    point.co = vec2_to_vec3( (img_co[1] - img_W/2, img_co[0] - img_H/2), 0, scale_factor)
                    point.pressure = dist_mat[img_co]
                    if self.generate_strength:
                        point.strength = 1 - denoised_lumi_mat[img_co]
                    if self.generate_color:
                        point.vertex_color[3] = 1
                        point.vertex_color[0] = srgb_to_linear(denoised_mat[img_co[0], img_co[1], min(0, img_obj.channels-1)])
                        point.vertex_color[1] = srgb_to_linear(denoised_mat[img_co[0], img_co[1], min(1, img_obj.channels-1)])
                        point.vertex_color[2] = srgb_to_linear(denoised_mat[img_co[0], img_co[1], min(2, img_obj.channels-1)])
                frame_strokes[-1].select = True

        if not self.image_sequence:
            process_single_image(self.filepath, starting_frame)
        else:
            for frame_idx, img_filepath in enumerate(img_filepaths):
                target_frame_number = starting_frame.frame_number + frame_idx * self.frame_step
                if target_frame_number in frame_dict:
                    process_single_image(img_filepath, frame_dict[target_frame_number])
                else:
                    process_single_image(img_filepath, gp_layer.frames.new(target_frame_number))

        return {'FINISHED'}

class ImportColorImageOperator(bpy.types.Operator, ImportHelper):
    """Generate strokes from a raster image by quantizing its colors"""
    bl_idname = "gpencil.nijigp_import_color_image"
    bl_label = "Import Color Image"
    bl_category = 'View'
    bl_options = {'REGISTER', 'UNDO'}

    directory: bpy.props.StringProperty(subtype='DIR_PATH')
    files: bpy.props.CollectionProperty(type=bpy.types.OperatorFileListElement)
    filepath = bpy.props.StringProperty(name="File Path", subtype='FILE_PATH')
    filter_glob: bpy.props.StringProperty(
        default='*.jpg;*.jpeg;*.png;*.tif;*.tiff;*.bmp',
        options={'HIDDEN'}
    )

    image_sequence: bpy.props.BoolProperty(
            name='Image Sequence',
            default=False,
            description='Process multiple images as a sequence'
    ) 
    frame_step: bpy.props.IntProperty(
            name='Frame Step',
            default=1, min=1,
            description='The number of frames between two generated line art keyframes'
    )  
    num_colors: bpy.props.IntProperty(
            name='Number of Colors',
            default=8, min=2, max=32,
            description='Color quantization in order to convert the image to Grease Pencil strokes'
    )
    median_radius: bpy.props.IntProperty(
            name='Median Filter Radius',
            default=3, min=0, soft_max=15,
            description='Denoise the image with a median filter. Disabled when the value is 0'
    )  
    size: bpy.props.FloatProperty(
            name='Size',
            default=2, min=0.001, soft_max=10,
            unit='LENGTH',
            description='Dimension of the generated strokes'
    )  
    sample_length: bpy.props.IntProperty(
            name='Sample Length',
            default=8, min=1, soft_max=64,
            description='Number of pixels of the original image between two generated stroke points'
    )
    min_length: bpy.props.IntProperty(
            name='Min Stroke Length',
            default=16, min=0, soft_max=512,
            description='Number of pixels, a contour smaller than which will not be converted to a stroke'
    )
    color_mode: bpy.props.EnumProperty(            
            name='Color Mode',
            items=[ ('VERTEX', 'Vertex Color', ''),
                    ('MATERIAL', 'New Materials', '')],
            default='VERTEX',
            description='Whether using an existing material with vertex colors, or creating new materials'
    )
    set_line_color: bpy.props.BoolProperty(
            name='Generate Line Color',
            default=True,
            description='Set the line color besides the fill color'
    )
    output_material: bpy.props.StringProperty(
        name='Output Material',
        description='Draw the new stroke using this material. If empty, use the active material',
        default='',
        search=lambda self, context, edit_text: [material.name for material in context.object.data.materials if material]
    )

    def draw(self, context):
        layout = self.layout
        layout.label(text = "Image Options:")
        box1 = layout.box()
        box1.prop(self, "num_colors")
        box1.prop(self, "median_radius")
        box1.prop(self, "image_sequence")
        if self.image_sequence:
            box1.prop(self, "frame_step")
        layout.label(text = "Stroke Options:")
        box2 = layout.box()
        box2.prop(self, "size")
        box2.prop(self, "sample_length")
        box2.prop(self, "min_length")
        box2.prop(self, "color_mode")
        if self.color_mode == 'VERTEX':
            box2.prop(self, "output_material", text='Material', icon='MATERIAL')
        box2.prop(self, "set_line_color")

    def execute(self, context):
        gp_obj: bpy.types.Object = context.object
        gp_layer = gp_obj.data.layers.active
        current_mode = context.mode
        use_multiedit = gp_obj.data.use_multiedit

        try:
            from skimage import morphology, filters, measure
            from scipy import cluster
            import numpy as np
            import pyclipper
        except:
            self.report({"ERROR"}, "Please install Scikit-Image in the Preferences panel.")
            return {'FINISHED'}

        gp_obj.data.use_multiedit = self.image_sequence
        bpy.ops.object.mode_set(mode='EDIT_GPENCIL')
        bpy.ops.gpencil.select_all(action='DESELECT')

        # Get or generate the starting frame
        if not gp_layer.active_frame:
            starting_frame = gp_layer.frames.new(context.scene.frame_current)
        else:
            starting_frame = gp_layer.active_frame

        # Process file paths in the case of multiple input images
        img_filepaths = []
        for f in self.files:
            img_filepaths.append(os.path.join(self.directory, f.name))
        img_filepaths.sort()

        # For image sequences, find all frames where strokes will be generated
        frame_dict = {}
        if self.image_sequence:
            for f in gp_layer.frames:
                frame_dict[f.frame_number] = f

        def process_single_image(img_filepath, frame):
            """
            Quantize colors of a specific image and generate strokes in a given frame
            """
            img_obj = image_utils.load_image(img_filepath, check_existing=True) # type: bpy.types.Image
            img_W = img_obj.size[0]
            img_H = img_obj.size[1]
            img_mat = np.array(img_obj.pixels).reshape(img_H,img_W, img_obj.channels)
            img_mat = np.flipud(img_mat)

            # Preprocessing: alpha and denoise
            if img_obj.channels > 3:
                color_mat = img_mat[:,:,:3]
                alpha_mask = img_mat[:,:,3]>0
                num_color_channel = 3
            else:
                color_mat = img_mat
                alpha_mask = np.ones((img_H,img_W))
                num_color_channel = img_obj.channels

            if self.median_radius > 0:
                footprint = morphology.disk(self.median_radius)
                footprint = np.repeat(footprint[:, :, np.newaxis], num_color_channel, axis=2)
                color_mat = filters.median(color_mat, footprint)

            # Quantization with K-mean
            pixels_1d = color_mat.reshape(-1,num_color_channel)
            palette, label = cluster.vq.kmeans2(pixels_1d, self.num_colors, minit='++', seed=0)
            label = label.reshape((img_H, img_W))

            # Get contours of each color
            contours, contour_color, contour_label = [], [], []
            label_count = [0] * len(palette)
            global_mask = np.zeros((img_H,img_W))
            global_mask[1:-1,1:-1] = 1
            global_mask *= alpha_mask
            for i,color in enumerate(palette):
                res = measure.find_contours( (label==i)*global_mask, 0.5, positive_orientation='high')
                for contour in res:
                    if pyclipper.Area(contour)>self.min_length**2:
                        contours.append(contour[::self.sample_length,:])
                        contour_color.append(color)
                        contour_label.append(i)
                        label_count[i] += 1

            # Vertex color mode: find the material slot 
            if self.color_mode == 'VERTEX':
                output_material_idx = [gp_obj.active_material_index] * len(palette)
                if len(self.output_material) > 0:
                    for i,material_slot in enumerate(gp_obj.material_slots):
                        if material_slot.material and material_slot.material.name == self.output_material:
                            output_material_idx = [i] * len(palette)
                            break
            # New material mode
            elif self.color_mode == 'MATERIAL':
                # Either find or create a material
                material_names = []
                for i,color in enumerate(palette):
                    if label_count[i] == 0:
                        material_names.append(None)
                        continue
                    hex_code = rgb_to_hex_code(color)
                    material_name = 'GP_Line-Fill'+hex_code if self.set_line_color else 'GP_Fill'+hex_code
                    material_names.append(material_name)
                    if material_name not in bpy.data.materials:
                        mat = bpy.data.materials.new(material_name)
                        bpy.data.materials.create_gpencil_data(mat)
                        mat.grease_pencil.fill_color = [srgb_to_linear(color[0]),
                                            srgb_to_linear(color[1]),
                                            srgb_to_linear(color[2]),1]
                        mat.grease_pencil.show_fill = True
                        mat.grease_pencil.show_stroke = self.set_line_color
                        if self.set_line_color:
                            mat.grease_pencil.color = [srgb_to_linear(color[0]),
                                              srgb_to_linear(color[1]),
                                              srgb_to_linear(color[2]),1]
                # Either find or create a slot
                output_material_idx = []
                for name in material_names:
                    if not name:
                        output_material_idx.append(None)
                        continue
                    for i,material_slot in enumerate(gp_obj.material_slots):
                        if material_slot.material and material_slot.material.name == name:
                            output_material_idx.append(i)
                            break
                    else:
                        gp_obj.data.materials.append(bpy.data.materials[name])
                        output_material_idx.append(len(gp_obj.material_slots)-1)

            # Generate strokes
            line_width = context.tool_settings.gpencil_paint.brush.size
            strength = context.tool_settings.gpencil_paint.brush.gpencil_settings.pen_strength
            frame_strokes = frame.strokes
            scale_factor = min(img_H, img_W) / self.size

            for i,path in enumerate(contours):
                color, label = contour_color[i], contour_label[i]
                frame_strokes.new()
                stroke: bpy.types.GPencilStroke = frame_strokes[-1]
                stroke.line_width = line_width
                stroke.use_cyclic = True
                stroke.material_index = output_material_idx[label]
                if self.color_mode == 'VERTEX':
                    stroke.vertex_color_fill = [srgb_to_linear(color[0]),
                                                srgb_to_linear(color[1]),
                                                srgb_to_linear(color[2]),1]
                stroke.points.add(len(path))
                for i,point in enumerate(frame_strokes[-1].points):
                    point.co = vec2_to_vec3( (path[i][1] - img_W/2, path[i][0] - img_H/2), 0, scale_factor)
                    point.strength = strength
                    if self.color_mode == 'VERTEX' and self.set_line_color:
                        point.vertex_color = [srgb_to_linear(color[0]),
                                              srgb_to_linear(color[1]),
                                              srgb_to_linear(color[2]),1]
                stroke.select = True
            
            if self.image_sequence:
                frame.select = True

        if not self.image_sequence:
            process_single_image(self.filepath, starting_frame)
        else:
            for frame_idx, img_filepath in enumerate(img_filepaths):
                target_frame_number = starting_frame.frame_number + frame_idx * self.frame_step
                if target_frame_number in frame_dict:
                    process_single_image(img_filepath, frame_dict[target_frame_number])
                else:
                    process_single_image(img_filepath, gp_layer.frames.new(target_frame_number))

        # Refresh the generated strokes, otherwise there might be display errors
        bpy.ops.transform.translate()
        bpy.ops.gpencil.nijigp_hole_processing(rearrange=True, apply_holdout=False)

        # Recover context
        gp_obj.data.use_multiedit = use_multiedit
        bpy.ops.object.mode_set(mode=current_mode)

        return {'FINISHED'}