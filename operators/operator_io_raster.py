import os
import bpy
import numpy as np
from bpy_extras.io_utils import ImportHelper
from bpy_extras import image_utils
from mathutils import *
from .common import *
from ..utils import *
from ..api_router import *

class CameraPlaneProjector:
    """
    Calculate the coordinates of 2D pixels projected to the camera plane. May be useful when vectorizing rendered results
    """
    def __init__(self, gp_obj, camera_obj, scene=None):
        self.ref_gp: bpy.types.Object = gp_obj
        self.ref_cam: bpy.types.Object = camera_obj
        
        # Calculate distance relationships in the camera space
        gp_center = (self.ref_cam.matrix_world.inverted_safe()) @ (self.ref_gp.matrix_world.translation)
        camera_corners = self.ref_cam.data.view_frame(scene=scene)
        camera_x_vec = camera_corners[0] - camera_corners[1]
        camera_y_vec = camera_corners[0] - camera_corners[3]
        # Focal distance is negative according to Blender convention
        camera_norm = np.cross(camera_x_vec, camera_y_vec)
        dist_focal_to_cam = -geometry.distance_point_to_plane( (0,0,0), camera_corners[0], camera_norm)
        dist_gp_to_focal = geometry.distance_point_to_plane(gp_center, camera_corners[0], camera_norm)
        dist_gp_to_cam = dist_focal_to_cam + dist_gp_to_focal
        
        # Currently, support orthographic and perspective cameras
        self.projected_corners = []
        for corner in camera_corners:
            if self.ref_cam.data.type == 'ORTHO':
                self.projected_corners.append(Vector((corner[0], corner[1], -dist_gp_to_cam)))
            else:
                self.projected_corners.append(Vector(corner / dist_focal_to_cam * dist_gp_to_cam))
        # Convert to GPencil local space
        gp_inv_mat = self.ref_gp.matrix_world.inverted_safe()
        for i in range(4):
            self.projected_corners[i] = gp_inv_mat @ self.ref_cam.matrix_world @ self.projected_corners[i]
        
    def get_co(self, norm_x, norm_y):
        """Convert normalized 2D pixel coordinates to world 3D coordinates"""
        return Vector(self.projected_corners[2]) + \
                Vector(norm_x * (self.projected_corners[1]-self.projected_corners[2])) + \
                Vector(norm_y * (self.projected_corners[3]-self.projected_corners[2]))

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
    fit_to_camera: bpy.props.BoolProperty(
            name='Fit to Camera',
            default=False,
            description='Adjust the size of imported image automatically to fill the whole view of the scene camera'
    ) 
    sample_length: bpy.props.IntProperty(
            name='Sample Length',
            default=4, min=1, soft_max=32,
            description='Number of pixels of the original image between two generated stroke points'
    )
    min_length: bpy.props.IntProperty(
            name='Min Stroke Length',
            default=4, min=1, soft_max=32,
            description='Number of pixels of a line, below which a stroke will not be generated'
    )
    smooth_level: bpy.props.IntProperty(
            name='Smooth Level',
            default=1, min=0, soft_max=10,
            description='Perform smoothing to reduce the alias of pixels'
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
        box2.prop(self, "fit_to_camera")
        if not self.fit_to_camera:
            box2.prop(self, "size")
        box2.prop(self, "sample_length")
        box2.prop(self, "min_length")
        box2.prop(self, "smooth_level")
        box2.prop(self, "output_material", text='Material', icon='MATERIAL')
        box2.prop(self, "generate_color")
        box2.prop(self, "generate_strength")

    def execute(self, context):
        gp_obj = context.object
        gp_layer = gp_obj.data.layers.active
        current_mode = gp_obj.mode
        bpy.ops.object.mode_set(mode=get_obj_mode_str('EDIT'))
        op_deselect()
        t_mat, inv_mat = get_transformation_mat(mode=context.scene.nijigp_working_plane,
                                                gp_obj=gp_obj)
        try:
            import skimage.morphology
            import skimage.filters
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
            img_obj = image_utils.load_image(img_filepath) # type: bpy.types.Image
            img_W = img_obj.size[0]
            img_H = img_obj.size[1]
            img_mat = np.array(img_obj.pixels).reshape(img_H,img_W, img_obj.channels)
            img_mat = np.flipud(img_mat)
            plane_projector = CameraPlaneProjector(gp_obj, bpy.context.scene.camera, bpy.context.scene) \
                        if self.fit_to_camera else None 
                        
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

            # Detect all line segments from the skeleton
            search_mat = np.zeros(skel_mat.shape)
            def line_point_dfs(v, u):
                """
                Traverse a 2D matrix to get connected pixels as a line
                """
                def get_info(v, u):
                    if v<0 or v>=img_H or u<0 or u>=img_W:
                        return 1
                    if skel_mat[v,u]==0:
                        return 1
                    return search_mat[v,u] # 0: not searched, 1: searched non-end point, 2: end point
                    
                line_points = []
                deltas = ((0,-1), (-1,-1), (-1,0), (-1,1), (0,1), (1,1), (1,0), (1,-1))
                pos = (v,u)
                is_starting_point = True
                while True:
                    line_points.append(pos)
                    search_mat[pos[0],pos[1]] = 1
                    num_paths = 0
                    next_pos = None
                    for d in deltas:
                        ret = get_info(pos[0]+d[0], pos[1]+d[1])
                        if ret==0:
                            next_pos = (pos[0]+d[0], pos[1]+d[1])
                        if ret==0 or ret==2:
                            num_paths += 1
                    if (num_paths!=1 and not is_starting_point) or not next_pos:
                        break
                    pos = next_pos
                    is_starting_point = False
                search_mat[line_points[0]] = 2
                search_mat[line_points[-1]] = 2
                return line_points
            
            segments = []
            for v in range(img_H):
                for u in range(img_W):
                    if search_mat[v,u]==0 and skel_mat[v,u]>0:
                        segments.append(line_point_dfs(v,u))

            # Record information of each tip point: a KDTree
            #   , a list: [(point_co, direction)]
            #   , and a map: {point_co: segment_index}
            tip_factor = 0.25
            kdt = kdtree.KDTree(2*len(segments))
            tip_info, tip_map = [], {}
            for i,seg in enumerate(segments):
                if len(seg) > 1:
                    tip_length = int(max(1, len(seg)*tip_factor))
                    start_point_co = Vector((seg[0][0], seg[0][1], 0))
                    start_direction = start_point_co - Vector((seg[tip_length][0], seg[tip_length][1], 0))
                    kdt.insert(start_point_co, len(tip_info))
                    tip_info.append(( (seg[0][0], seg[0][1]), start_direction.normalized()))
                    tip_map[(seg[0][0], seg[0][1])] = i

                    end_point_co = Vector((seg[-1][0], seg[-1][1], 0))
                    end_direction = end_point_co - Vector((seg[-1-tip_length][0], seg[-1-tip_length][1], 0))
                    kdt.insert(end_point_co, len(tip_info))
                    tip_info.append(( (seg[-1][0], seg[-1][1]), end_direction.normalized()))      
                    tip_map[(seg[-1][0], seg[-1][1])] = i           
            kdt.balance()

            # Calculate the similarity between each pair of segments: (tip1, tip2, dot of direction)
            joint_info = []
            tip_pair_set = set()
            for i,tip1 in enumerate(tip_info):
                candidates = kdt.find_range((tip1[0][0],tip1[0][1],0), max(dist_mat[tip1[0]], 1.5) )
                for candidate in candidates:
                    tip2 = tip_info[candidate[1]]
                    if tip1[0]!=tip2[0] and (tip1[0], tip2[0]) not in tip_pair_set and (tip2[0], tip1[0]) not in tip_pair_set:
                        tip_pair_set.add(((tip1[0], tip2[0])))
                        joint_info.append( (tip1[0], tip2[0], tip1[1].dot(tip2[1])) )
            joint_info.sort(key=lambda x:x[2])

            # Join segments based on similarity
            merged_tip = set()
            parent_map = {}
            for joint in joint_info:
                if (joint[0] in merged_tip or joint[1] in merged_tip
                    or joint[2]>0 ):
                    continue
                # Determine the indices of segments to merge
                seg_idx1 = tip_map[joint[0]]
                seg_idx2 = tip_map[joint[1]]
                while seg_idx1 in parent_map:
                    seg_idx1 = parent_map[seg_idx1]
                while seg_idx2 in parent_map:
                    seg_idx2 = parent_map[seg_idx2]
                if seg_idx1 == seg_idx2:
                    continue
                # Merge process: 4 cases - head/tail to head/tail
                if joint[0]==segments[seg_idx1][0]:
                    segments[seg_idx1].reverse()
                if joint[1]==segments[seg_idx2][-1]:
                    segments[seg_idx2].reverse()
                segments[seg_idx1] += segments[seg_idx2]
                segments[seg_idx2] = []
                # State updates
                parent_map[seg_idx2] = seg_idx1
                merged_tip.add(joint[0])
                merged_tip.add(joint[1])

            # Generate strokes according to line segments
            frame_strokes = frame.nijigp_strokes
            scale_factor = min(img_H, img_W) / self.size
            line_thickness = dist_mat.max()
            dist_mat /= line_thickness
            line_thickness = int(line_thickness / scale_factor * LINE_WIDTH_FACTOR)

            for line in segments:
                if len(line) < self.min_length:
                    continue
                point_count = len(line) // self.sample_length
                if self.sample_length > 1 and len(line)%self.sample_length != 1:
                    point_count += 1

                frame_strokes.new()
                frame_strokes[-1].line_width = line_thickness
                frame_strokes[-1].material_index = output_material_idx
                frame_strokes[-1].points.add(point_count)

                for i,point in enumerate(frame_strokes[-1].points):
                    img_co = line[min(i*self.sample_length, len(line)-1)]
                    if self.fit_to_camera:
                        point.co = plane_projector.get_co(img_co[1]/img_W, 1-img_co[0]/img_H)
                    else:                    
                        point.co = restore_3d_co((img_co[1]-img_W/2, -img_co[0]+img_H/2, 0), 0, inv_mat, scale_factor)
                    set_point_radius(point, dist_mat[img_co], line_thickness)
                    if self.generate_strength:
                        point.strength = 1 - denoised_lumi_mat[img_co]
                    if self.generate_color:
                        point.vertex_color[3] = 1
                        point.vertex_color[0] = srgb_to_linear(denoised_mat[img_co[0], img_co[1], min(0, img_obj.channels-1)])
                        point.vertex_color[1] = srgb_to_linear(denoised_mat[img_co[0], img_co[1], min(1, img_obj.channels-1)])
                        point.vertex_color[2] = srgb_to_linear(denoised_mat[img_co[0], img_co[1], min(2, img_obj.channels-1)])
                frame_strokes[-1].select = True
                
                if self.smooth_level > 0 and point_count > 2:
                    smooth_stroke_attributes(frame_strokes[-1], self.smooth_level, 
                                             {'co':3, 'strength':1, 'pressure':1, 'vertex_color':4})

        # Main processing loop
        processed_frame_numbers = []
        if not self.image_sequence:
            process_single_image(self.filepath, starting_frame)
            processed_frame_numbers.append(starting_frame.frame_number)
        else:
            for frame_idx, img_filepath in enumerate(img_filepaths):
                target_frame_number = starting_frame.frame_number + frame_idx * self.frame_step
                context.scene.frame_set(target_frame_number)
                if target_frame_number in frame_dict:
                    process_single_image(img_filepath, frame_dict[target_frame_number])
                else:
                    process_single_image(img_filepath, gp_layer.frames.new(target_frame_number))
                processed_frame_numbers.append(target_frame_number)
        # Post-processing
        refresh_strokes(gp_obj, processed_frame_numbers)
        bpy.ops.object.mode_set(mode=current_mode)
        return {'FINISHED'}

class ImportColorImageConfig:
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
    auto_resize: bpy.props.BoolProperty(
            name='Auto Resize',
            default=True,
            description='When the image resolution is high, downsize the image to speed up the processing'
    )
    num_colors: bpy.props.IntProperty(
            name='Number of Colors',
            default=8, min=1, max=32,
            description='Color quantization in order to convert the image to Grease Pencil strokes'
    )
    color_source: bpy.props.EnumProperty(            
            name='Color Source',
            items=[ ('FIRST_FRAME', 'First Frame', ''),
                    ('EVERY_FRAME', 'Every Frame', ''),
                    ('PALETTE_RGB', 'Palette by RGB Distance', ''),
                    ('PALETTE_AREA', 'Palette by Area', '')],
            default='FIRST_FRAME',
            description='Where colors are picked for generated strokes'
    )
    reference_palette_name: bpy.props.StringProperty(
            name='Reference Palette',
            default='',
            search=lambda self, context, edit_text: [palette.name for palette in bpy.data.palettes if palette]
    )
    size: bpy.props.FloatProperty(
            name='Size',
            default=2, min=0.001, soft_max=10,
            unit='LENGTH',
            description='Dimension of the generated strokes'
    )  
    fit_to_camera: bpy.props.BoolProperty(
            name='Fit to Camera',
            default=False,
            description='Adjust the size of imported image automatically to fill the whole view of the scene camera'
    )
    sample_length: bpy.props.IntProperty(
            name='Sample Length',
            default=4, min=1, soft_max=64,
            description='Number of pixels of the original image between two generated stroke points'
    )
    smooth_level: bpy.props.IntProperty(
            name='Smooth',
            default=1, min=0, soft_max=10,
            description='Number of smooth operations performed on the extracted contours'
    )
    min_area: bpy.props.IntProperty(
            name='Min Stroke Area',
            default=16, min=0, soft_max=2048,
            description='Number of pixels, a contour with its area smaller than which will be merged to adjacent shapes'
    )
    color_mode: bpy.props.EnumProperty(            
            name='Color Mode',
            items=[ ('VERTEX', 'Vertex Color', ''),
                    ('MATERIAL', 'New Materials', '')],
            default='MATERIAL',
            description='Whether using an existing material with vertex colors, or creating new materials'
    )
    set_line_color: bpy.props.BoolProperty(
            name='Generate Line Color',
            default=True,
            description='Set the line color besides the fill color'
    )
    new_palette: bpy.props.BoolProperty(
            name='Generate Palette',
            default=False,
            description='Generate a new palette based on colors extracted from this image'
    )
    output_material: bpy.props.StringProperty(
            name='Output Material',
            description='Draw the new stroke using this material. If empty, use the active material',
            default='',
            search=lambda self, context, edit_text: [material.name for material in context.object.data.materials if material]
    )

class ImportColorImageOperator(bpy.types.Operator, ImportHelper, ImportColorImageConfig):
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

    def draw(self, context):
        layout = self.layout
        layout.label(text = "Image Options:")
        box1 = layout.box()
        box1.prop(self, "num_colors")
        box1.prop(self, "image_sequence")
        box1.prop(self, "auto_resize")
        if self.image_sequence:
            box1.prop(self, "frame_step")
        box1.prop(self, "color_source")
        if self.color_source=='PALETTE_RGB' or self.color_source=='PALETTE_AREA':
            box1.prop(self, "reference_palette_name")

        layout.label(text = "Stroke Options:")
        box2 = layout.box()
        box2.prop(self, "fit_to_camera")
        if not self.fit_to_camera:
            box2.prop(self, "size")
        box2.prop(self, "min_area")
        box2.prop(self, "sample_length")
        box2.prop(self, "smooth_level")
        box2.prop(self, "color_mode")
        if self.color_mode == 'VERTEX':
            box2.prop(self, "output_material", text='Material', icon='MATERIAL')
        box2.prop(self, "set_line_color")
        if self.color_source!='PALETTE_RGB' and self.color_source!='PALETTE_AREA':
            layout.prop(self, "new_palette")

    def execute(self, context):
        gp_obj: bpy.types.Object = context.object
        gp_layer = gp_obj.data.layers.active
        current_mode = gp_obj.mode
        use_multiedit = get_multiedit(gp_obj)
        t_mat, inv_mat = get_transformation_mat(mode=context.scene.nijigp_working_plane,
                                                gp_obj=gp_obj)
        try:
            from skimage import measure, segmentation, transform
            import skimage.color
            from scipy import cluster
            from ..solvers.measure import simplify_contour_path, multicolor_contour_find, merge_small_areas
        except:
            self.report({"ERROR"}, "Please install SciPy and Scikit-Image in the Preferences panel.")
            return {'FINISHED'}

        set_multiedit(gp_obj, self.image_sequence)
        bpy.ops.object.mode_set(mode=get_obj_mode_str('EDIT'))
        op_deselect()

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

        def process_single_image(img_filepath, frame, given_colors = []):
            """
            Quantize colors of a specific image and generate strokes in a given frame
            """
            img_obj = image_utils.load_image(img_filepath) # type: bpy.types.Image
            img_W = img_obj.size[0]
            img_H = img_obj.size[1]
            img_mat = np.array(img_obj.pixels).reshape(img_H,img_W, img_obj.channels)
            img_mat = np.flipud(img_mat)

            if self.auto_resize:
                max_W, max_H = context.scene.render.resolution_x * 0.5, context.scene.render.resolution_y * 0.5
                factor = min(max_H/img_H, max_W/img_W)
                if factor < 1:
                    img_mat = transform.rescale(img_mat, factor, anti_aliasing=False, channel_axis=2)
                    img_H, img_W = img_mat.shape[0], img_mat.shape[1]

            # Preprocessing: alpha and denoise
            if img_obj.channels == 4:
                color_mat = img_mat[:,:,:3]
                alpha_mask = img_mat[:,:,3]>0
                num_color_channel = 3
            elif img_obj.channels == 3:
                color_mat = img_mat
                alpha_mask = np.ones((img_H,img_W))
                num_color_channel = img_obj.channels
            else:
                self.report({"WARNING"}, "Only RGB or RGBA images are supported.")
                return {'FINISHED'}   

            # To balance between efficiency and quality, adopt the following steps: 
            #      LAB k-means -> felzenszwalb -> 2nd k-means
            color_mat_lab = skimage.color.rgb2lab(color_mat)
            num_colors = self.num_colors if len(given_colors)==0 else len(given_colors)
            _, label = cluster.vq.kmeans2(color_mat_lab.reshape(-1,3).astype('float'), num_colors*2, minit='++', seed=0)
            label = label.reshape((img_H, img_W))
            color_mat_lab = skimage.color.label2rgb(label+1, color_mat_lab, kind='avg')
            
            label = segmentation.felzenszwalb(color_mat_lab, scale=1, min_size=self.min_area)
            color_mat_lab = skimage.color.label2rgb(label+1, color_mat_lab, kind='avg')

            # Final color quantization in different modes
            pixels_1d = color_mat_lab.reshape(-1,num_color_channel)
            if len(given_colors)==0:
                palette, label = cluster.vq.kmeans2(pixels_1d, self.num_colors, minit='++', seed=0)
                palette = skimage.color.lab2rgb(palette)
            # If given colors, sort colors by either RGB distance or number of colored pixels
            elif self.color_source == 'PALETTE_AREA':
                palette, label = cluster.vq.kmeans2(pixels_1d, given_colors.shape[0], minit='++', seed=0)
                color_seq = np.argsort(-np.bincount(label))
                palette[color_seq] = given_colors
            else:
                palette = skimage.color.rgb2lab(given_colors)
                lab_diff = pixels_1d.reshape(-1,1,num_color_channel) - palette.reshape(1,-1,num_color_channel)
                lab_dist = np.sum(np.square(lab_diff), axis=2)
                label = np.argmin(lab_dist, axis=1)
                palette = skimage.color.lab2rgb(palette)
            # Use Label 0 as transparent areas
            label = label.reshape((img_H, img_W)) + 1
            
            # Get contours of each color
            spatial_label = measure.label(label*alpha_mask, connectivity=1)
            merge_small_areas(spatial_label, label, self.min_area)    
            contours_info = multicolor_contour_find(spatial_label, label)
                        
            # Record colors in a Blender palette, skipping colors that are not used
            label_count = [0] * len(palette)
            for _, _, color_label in contours_info:
                if color_label > 0:
                    label_count[color_label-1] += 1
            if len(given_colors)==0 and palette_to_write:
                for i, color in enumerate(palette):
                    if label_count[i]>0:
                        palette_to_write.colors.new()
                        palette_to_write.colors[-1].color = Color(color)

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
            active_brush = context.tool_settings.gpencil_paint.brush
            line_width = active_brush.size if active_brush else 20
            strength = active_brush.gpencil_settings.pen_strength if active_brush else 1.0
            frame_strokes = frame.nijigp_strokes
            scale_factor = min(img_H, img_W) / self.size
            plane_projector = CameraPlaneProjector(gp_obj, bpy.context.scene.camera, bpy.context.scene) \
                        if self.fit_to_camera else None 

            for path, critical_idx, color_label in contours_info:
                if color_label == 0:
                    continue
                color = Color(palette[color_label-1])
                path = simplify_contour_path(path, critical_idx, self.smooth_level, self.sample_length)
                if len(path) < 3:
                    continue
                frame_strokes.new()
                stroke = frame_strokes[-1]
                stroke.line_width = line_width
                stroke.use_cyclic = True
                stroke.material_index = output_material_idx[color_label-1]
                if self.color_mode == 'VERTEX':
                    stroke.vertex_color_fill = [srgb_to_linear(color[0]),
                                                srgb_to_linear(color[1]),
                                                srgb_to_linear(color[2]),1]
                stroke.points.add(len(path))
                for i,point in enumerate(stroke.points):
                    if self.fit_to_camera:
                        point.co = plane_projector.get_co(path[i][1]/img_W, 1-path[i][0]/img_H)
                    else:
                        point.co = restore_3d_co((path[i][1]-img_W/2, -path[i][0]+img_H/2, 0), 0, inv_mat, scale_factor)
                    point.strength = strength
                    set_point_radius(point, 1)
                    if self.color_mode == 'VERTEX' and self.set_line_color:
                        point.vertex_color = [srgb_to_linear(color[0]),
                                              srgb_to_linear(color[1]),
                                              srgb_to_linear(color[2]),1]
                stroke.select = True

            if self.image_sequence:
                frame.select = True
            return palette

        # Color palette operations: either read or write, not both
        given_colors = []
        palette_to_write = None
        if self.color_source == 'PALETTE_RGB' or self.color_source == 'PALETTE_AREA':
            palette_to_read = bpy.data.palettes[self.reference_palette_name]
            for slot in palette_to_read.colors:
                given_colors.append(slot.color)
            given_colors = np.array(given_colors)[:self.num_colors]
        elif self.new_palette:
            palette_to_write = bpy.data.palettes.new(self.files[0].name)

        # Main processing loop
        processed_frame_numbers = []
        if not self.image_sequence:
            process_single_image(self.filepath, starting_frame, given_colors)
            processed_frame_numbers.append(starting_frame.frame_number)
        else:
            for frame_idx, img_filepath in enumerate(img_filepaths):
                target_frame_number = starting_frame.frame_number + frame_idx * self.frame_step
                context.scene.frame_set(target_frame_number)
                if target_frame_number in frame_dict:
                    updated_palette = process_single_image(img_filepath, frame_dict[target_frame_number], given_colors)
                else:
                    updated_palette = process_single_image(img_filepath, gp_layer.frames.new(target_frame_number), given_colors)
                if self.color_source == 'FIRST_FRAME' and len(given_colors)==0:
                    given_colors = updated_palette
                processed_frame_numbers.append(target_frame_number)

        # Post-processing and context recovery
        refresh_strokes(gp_obj, processed_frame_numbers)
        set_multiedit(gp_obj, use_multiedit)
        bpy.ops.object.mode_set(mode=current_mode)

        return {'FINISHED'}
    
class RenderAndVectorizeOperator(bpy.types.Operator, ImportColorImageConfig):
    """Render the current scene or objects inside it, and vectorize the image by quantizing the colors"""
    bl_idname = "gpencil.nijigp_render_and_vectorize"
    bl_label = "Render and Convert Scene/Mesh"
    bl_category = 'View'
    bl_options = {'REGISTER', 'UNDO'}

    source_type: bpy.props.EnumProperty(            
        name='Source Type',
        items=[ ('SCENE', 'Scene', ''),
                ('OBJECT', 'Object', ''),
                ('COLLECTION', 'Collection', '')],
        default='SCENE'
    )
    source_obj: bpy.props.StringProperty(
        name='Object',
        search=lambda self, context, edit_text: [object.name for object in context.scene.objects if object.type!='GPENCIL']
    )
    source_coll: bpy.props.StringProperty(
        name='Collection',
        search=lambda self, context, edit_text: [coll.name for coll in bpy.data.collections]
    )
    render_animation: bpy.props.BoolProperty(
        name='Render Animation',
        default=False,
    )
    frame_start: bpy.props.IntProperty(
        name='Start Frame',
        default=1, min=1,
    )
    frame_end: bpy.props.IntProperty(
        name='End Frame',
        default=1, min=1,
    )
    bake_lineart: bpy.props.BoolProperty(
        name='Project and Bake Line Art',
        default=False,
        description='If there are Line Art modifiers on the same object/collection, project line arts to the same 2D plane as the color fills'
    )
    
    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self, width=400)
    
    def draw(self, context):
        layout = self.layout
        layout.label(text = "Input Options:")
        box1 = layout.box()
        box1.prop(self, "source_type")
        if self.source_type == 'COLLECTION':
            box1.prop(self, "source_coll", icon='OUTLINER_COLLECTION')
        if self.source_type == 'OBJECT':
            box1.prop(self, "source_obj", icon='OBJECT_DATA')
        box1.prop(self, "render_animation")   
        if self.render_animation:
            split = box1.split(align=True)
            split.prop(self, "frame_start", text="Start")
            split.prop(self, "frame_end", text="End")  
            split.prop(self, "frame_step", text="Step")  
        
        layout.label(text = "Image Options:")
        box2 = layout.box()
        box2.prop(self, "num_colors")
        box2.prop(self, "auto_resize")
        
        layout.label(text = "Stroke Options:")
        box3 = layout.box()
        box3.prop(self, "min_area")
        box3.prop(self, "sample_length")
        box3.prop(self, "smooth_level")
        box3.prop(self, "color_mode")
        if self.color_mode == 'VERTEX':
            box3.prop(self, "output_material", text='Material', icon='MATERIAL')
        box3.prop(self, "set_line_color")
        layout.prop(self, "bake_lineart")
        
    def execute(self, context):
        scene = bpy.context.scene
        gp_obj = bpy.context.active_object
                
        # Save the current scene state
        is_render_hidden = {}
        for obj in scene.objects:
            is_render_hidden[obj] = obj.hide_render
        multiedit = get_multiedit(gp_obj) 
        is_background_transparent = scene.render.film_transparent
        render_format = scene.render.image_settings.file_format
        render_color_mode = scene.render.image_settings.color_mode
        render_path = scene.render.filepath
        frame_start = scene.frame_start
        frame_end = scene.frame_end
        
        # Change scene setting and object visibility for rendering
        bpy.ops.object.mode_set(mode=get_obj_mode_str('EDIT'))
        bpy.context.space_data.region_3d.view_perspective = 'CAMERA'
        scene.render.film_transparent = True
        scene.render.image_settings.file_format = 'PNG'
        scene.render.image_settings.color_mode = 'RGBA'
        set_multiedit(gp_obj, False)
        scene.frame_start = self.frame_start if self.render_animation else scene.frame_current
        scene.frame_end = self.frame_end if self.render_animation else scene.frame_current

        for obj in scene.objects:
            if obj.type in {'LIGHT', 'LIGHT_PROBE', 'CAMERA'}:
                continue
            if obj.type == 'GPENCIL':
                obj.hide_render = True
            if self.source_type == 'OBJECT' and obj.name != self.source_obj:
                obj.hide_render = True
            if self.source_type == 'COLLECTION' and (self.source_coll in bpy.data.collections) and (obj.name not in bpy.data.collections[self.source_coll].objects):
                obj.hide_render = True
                        
        # Render and vectorize the mesh
        op_filepath = None
        op_files = []
        frame_numbers = range(scene.frame_start, scene.frame_end + 1, self.frame_step)
        for frame_number in frame_numbers:
            scene.frame_set(frame_number)
            img_path = f'{bpy.app.tempdir}/{str(frame_number).zfill(6)}.png'
            scene.render.filepath = img_path
            op_filepath = img_path
            op_files.append({'name': f'{str(frame_number).zfill(6)}.png'})
            bpy.ops.render.render(write_still=True)
        scene.frame_set(scene.frame_start)
        bpy.ops.gpencil.nijigp_import_color_image(filepath=op_filepath, directory=bpy.app.tempdir, files=op_files, 
                                                num_colors=self.num_colors, auto_resize=self.auto_resize,
                                                min_area=self.min_area, smooth_level=self.smooth_level, sample_length=self.sample_length,
                                                color_mode=self.color_mode, output_material=self.output_material, set_line_color=self.set_line_color,
                                                image_sequence=True, frame_step=self.frame_step, fit_to_camera=True)
        
        # Find line art modifiers: with the same source and being valid
        if self.bake_lineart:
            lineart_modifier = None
            for modifier in get_gp_modifiers(gp_obj):
                if self.source_type != modifier.source_type or not modifier.target_layer or not modifier.target_material:
                    continue
                if self.source_type == 'COLLECTION' and modifier.source_collection and self.source_coll == modifier.source_collection.name and not modifier.use_invert_collection:
                    lineart_modifier = modifier
                    break
                if self.source_type == 'SCENE' or (self.source_type == 'OBJECT' and modifier.source_object and self.source_obj == modifier.source_object.name):
                    lineart_modifier = modifier
                    break
                
            if lineart_modifier is not None:
                # Bake the line art and get all frames
                lineart_layer = gp_obj.data.layers[modifier.target_layer]
                if lineart_layer == gp_obj.data.layers.active:
                    self.report({'WARNING'}, "Line Art modifier uses the same layer as color fills. Data may be overwritten. Please consider switching to another layer.")
                bpy.ops.object.lineart_bake_strokes()
                get_gp_modifiers(gp_obj).remove(lineart_modifier)
                lineart_frames = {}
                for frame in lineart_layer.frames:
                    lineart_frames[int(frame.frame_number)] = frame
                    
                # Project the line art to the same plane, or delete the frame if not needed
                frame_number_set = set(frame_numbers)
                for frame_number in range(scene.frame_start, scene.frame_end + 1):
                    scene.frame_set(frame_number)
                    op_deselect()
                    for stroke in lineart_frames[frame_number].nijigp_strokes:
                        stroke.select = True
                    if frame_number in frame_number_set:
                        op_reproject()
                    else:
                        remove_frame(lineart_layer.frames, lineart_frames[frame_number])

        # Recover the scene state
        for obj in scene.objects:
            obj.hide_render = is_render_hidden[obj]
        scene.render.film_transparent = is_background_transparent
        scene.render.image_settings.file_format = render_format
        scene.render.image_settings.color_mode = render_color_mode
        scene.render.filepath = render_path
        scene.frame_start = frame_start
        scene.frame_end = frame_end
        set_multiedit(gp_obj, multiedit)
                
        return {'FINISHED'}