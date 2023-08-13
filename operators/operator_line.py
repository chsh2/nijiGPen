import bpy
import math
import numpy as np
from mathutils import *
from .common import *
from ..utils import *

def stroke_to_kdtree(co_list):
    n = len(co_list)
    kdt = kdtree.KDTree(n)
    for i in range(n):
        kdt.insert(xy0(co_list[i]), i)
    kdt.balance()
    return kdt

def fit_2d_strokes(strokes, search_radius, smoothness_factor = 1, pressure_delta = 0, closed = False, operator = None, t_mat = [], inv_mat = []):
    '''
    Fit points from multiple strokes to a single curve, by executing the following operations:
        1. Delaunay triangulation
        2. Euclidean minimum spanning tree
        3. Longest path in the tree
        4. Offset based on points in the neighborhood
        5. Post-processing: vertex smooth or B-spline fitting

    The function will return positions and attributes of points in the following sequence:
        2D coordinates, accumulated pressure, base pressure, strength, vertex color
    '''
    empty_result = None, None, None, None, None, None, None, None
    try:
        from scipy.interpolate import splprep, splev
        from ..solvers.graph import TriangleMst
    except:
        if operator:
            operator.report({"ERROR"}, "Please install Scipy in the Preferences panel.")
        return empty_result

    if len(t_mat)<1:
        t_mat, inv_mat = get_transformation_mat(mode=bpy.context.scene.nijigp_working_plane,
                                                gp_obj=bpy.context.active_object,
                                                strokes=strokes, operator=operator)
    poly_list, depth_list, _ = get_2d_co_from_strokes(strokes, t_mat, scale=False)
    
    total_point_count = 0
    for i,stroke in enumerate(strokes):
        if len(stroke.points)<2:
            continue
        total_point_count += len(stroke.points)

    if total_point_count<3:
        return empty_result
    
    # Create a KDTree for point attribute lookup
    kdt = kdtree.KDTree(total_point_count)
    kdt_tangent_list = []
    kdt_stroke_list = []
    kdt_point_list = []
    kdt_depth_list = []
    kdt_idx = 0
    co_set = set()

    # Record point attributes and prepare the input of Step 1
    # Ignore single-point strokes and merge points with the same coordinates
    tr_input = dict(vertices = [])
    for i,stroke in enumerate(strokes):
        if len(stroke.points)<2:
            continue
        for j,point in enumerate(stroke.points):
            if (poly_list[i][j][0],poly_list[i][j][1]) not in co_set:
                co_set.add((poly_list[i][j][0],poly_list[i][j][1]))
                kdt.insert( xy0(poly_list[i][j]), kdt_idx)
                kdt_idx += 1
                if j>0:
                    kdt_tangent_list.append((poly_list[i][j][0]-poly_list[i][j-1][0],
                                            poly_list[i][j][1]-poly_list[i][j-1][1]))
                else:
                    kdt_tangent_list.append((poly_list[i][j+1][0]-poly_list[i][j][0],
                                            poly_list[i][j+1][1]-poly_list[i][j][1]))    
                kdt_stroke_list.append(i)
                kdt_point_list.append(point)
                kdt_depth_list.append(depth_list[i][j])
                tr_input['vertices'].append(poly_list[i][j])                    
    kdt.balance()

    # Triangulation and spanning tree conversion
    tr_output = {}
    tr_output['vertices'], _, tr_output['triangles'], _,_,_ = geometry.delaunay_2d_cdt(tr_input['vertices'], [], [], 0, 1e-9)
    mst_builder = TriangleMst()
    mst_builder.build_mst(tr_output)
    total_length, path_whole = mst_builder.get_longest_path()
    
    # The fitting method needs at least 4 points
    if len(path_whole)<4:
        return empty_result
    
    # Get the points in the tree as the input of postprocessing
    co_raw = np.zeros((len(path_whole), 2))
    for i,key in enumerate(path_whole):
        co = tr_output['vertices'][key]
        co_raw[i][0] = co[0]
        co_raw[i][1] = co[1]

    # Initialize lists of each point attributes
    accumulated_pressure_raw = np.zeros(len(path_whole))
    inherited_pressure_raw = np.zeros(len(path_whole))
    inherited_strength_raw = np.zeros(len(path_whole))
    inherited_depth_raw = np.zeros(len(path_whole))
    inherited_color = np.zeros((len(path_whole), 4))
    inherited_uv_rotation = np.zeros(len(path_whole))

    # Apply offsets in the normal direction if there are points in the neighborhood
    # At the same time, inherit the point attributes
    for i,co in enumerate(co_raw):
        self_vec ,self_idx,_ = kdt.find(xy0(co))
        self_vec = self_vec.xy
        self_stroke = kdt_stroke_list[self_idx]
        unit_normal_vector = Vector(kdt_tangent_list[self_idx]).orthogonal().normalized()
        sum_normal_offset = 0
        neighbors = kdt.find_range(xy0(co), search_radius)
        for neighbor in neighbors:
            if kdt_stroke_list[neighbor[1]]!=self_stroke:
                normal_dist = neighbor[0].xy - self_vec
                normal_dist = normal_dist.dot(unit_normal_vector)
                sum_normal_offset += normal_dist
                accumulated_pressure_raw[i] += pressure_delta
            # Inherit each attribute
            inherited_pressure_raw[i] += kdt_point_list[neighbor[1]].pressure
            inherited_strength_raw[i] += kdt_point_list[neighbor[1]].strength
            inherited_depth_raw[i] += kdt_depth_list[neighbor[1]]
            inherited_color[i] += np.array(kdt_point_list[neighbor[1]].vertex_color)
            inherited_uv_rotation[i] += kdt_point_list[neighbor[1]].uv_rotation

        sum_normal_offset /= len(neighbors)
        inherited_pressure_raw[i] /= len(neighbors)
        inherited_strength_raw[i] /= len(neighbors)
        inherited_depth_raw[i] /= len(neighbors)
        inherited_color[i] /= len(neighbors)
        inherited_uv_rotation[i] /= len(neighbors)
        co_raw[i] += unit_normal_vector * sum_normal_offset

    # Postprocessing: B-spline fitting
    if smoothness_factor is None:
        return inv_mat, co_raw, accumulated_pressure_raw, inherited_pressure_raw, inherited_strength_raw, inherited_color, inherited_uv_rotation, inherited_depth_raw

    attributes_index = np.linspace(0,1,len(path_whole))
    if closed:
        co_raw = np.append(co_raw, [co_raw[0]], axis=0)
        attributes_index = np.append(attributes_index, 0)
        accumulated_pressure_raw = np.append(accumulated_pressure_raw, accumulated_pressure_raw[0])
        inherited_pressure_raw = np.append(inherited_pressure_raw, inherited_pressure_raw[0])
        inherited_strength_raw = np.append(inherited_strength_raw, inherited_strength_raw[0])
        inherited_depth_raw = np.append(inherited_depth_raw, inherited_depth_raw[0])
        inherited_color = np.append(inherited_color, [inherited_color[0]], axis=0)
        inherited_uv_rotation = np.append(inherited_uv_rotation, inherited_uv_rotation[0])
    tck, u = splprep([co_raw[:,0], co_raw[:,1]], s=total_length**2 * smoothness_factor * 0.001, per=closed)
    co_fit = np.array(splev(u, tck)).transpose()    
    tck2, u2 = splprep([attributes_index, accumulated_pressure_raw], per=closed)
    accumulated_pressure_fit = np.array(splev(u2, tck2))[1]
    tck3, u3 = splprep([attributes_index, inherited_pressure_raw], per=closed)
    inherited_pressure_fit = np.array(splev(u3, tck3))[1]
    tck4, u4 = splprep([attributes_index, inherited_strength_raw], per=closed)
    inherited_strength_fit = np.array(splev(u4, tck4))[1]
    tck5, u5 = splprep([attributes_index, inherited_depth_raw], s=total_length**2 * smoothness_factor * 0.001, per=closed)
    inherited_depth_fit = np.array(splev(u5, tck5))[1]

    return inv_mat, co_fit, accumulated_pressure_fit, inherited_pressure_fit, inherited_strength_fit, inherited_color, inherited_uv_rotation, inherited_depth_fit

def distance_to_another_stroke(co_list1, co_list2, kdt2 = None, angular_tolerance = math.pi/4, correct_orientation = True):
    '''
    Calculating the similarity between two lines
    '''
    # Some algorithms do not support infinity; use a finite number instead.
    no_similarity = 65535

    n1 = len(co_list1)
    n2 = len(co_list2)
    if n1<2 or n2<2:
        return no_similarity

    # Generate the KDTrees if not provided
    if not kdt2:
        kdt2 = kdtree.KDTree(n2)
        for i in range(n2):
            kdt2.insert(xy0(co_list2[i]), i)
        kdt2.balance()        

    # Get point-wise distance values
    idx_arr = np.zeros(n1, dtype='int')
    dist_arr = np.zeros(n1)
    for i in range(n1):
        _, idx_arr[i], dist_arr[i] = kdt2.find(xy0(co_list1[i]))

    # Calculate the orientation difference of two lines
    contact_idx1 = np.argmin(dist_arr)
    contact_idx2 = min(idx_arr[contact_idx1], n2-2)
    contact_idx1 = min(contact_idx1, n1-2)
    direction1 = Vector(co_list1[contact_idx1+1]) - Vector(co_list1[contact_idx1])
    direction2 = Vector(co_list2[contact_idx2+1]) - Vector(co_list2[contact_idx2])
    if math.isclose(direction1.length, 0) or math.isclose(direction2.length, 0):
        angle_diff = 0
    else:
        angle_diff = direction1.angle(direction2)

    # Three cases of directions: similar, opposite or different
    end2 = n2-1
    if correct_orientation and angle_diff > math.pi / 2:
        angle_diff = math.pi - angle_diff
        end2 = 0
    if angle_diff > angular_tolerance:
        return no_similarity
    
    # Calculate the total cost
    total_cost, total_count = 0.0, 0.0
    for i in range(n1):
        total_cost += dist_arr[i]
        total_count += 1
        if idx_arr[i] == end2:
            break
    return total_cost/total_count
      
class CommonFittingConfig:
    """
    Options shared by all line-fitting operators
    """
    line_sampling_size: bpy.props.IntProperty(
            name='Line Spacing',
            description='Strokes with gap smaller than this may be merged',
            default=50, min=1, soft_max=100, subtype='PIXEL'
    )
    closed: bpy.props.BoolProperty(
            name='Closed Stroke',
            default=False,
            description='Treat selected strokes as a closed shape'
    )
    pressure_variance: bpy.props.FloatProperty(
            name='Pressure Accumulation',
            description='Increase the point radius at positions where lines are repeatedly drawn',
            default=5, soft_max=20, min=0, subtype='PERCENTAGE'
    )
    max_delta_pressure: bpy.props.FloatProperty(
            name='Max Accumulated Pressure',
            description='Upper bound of the additional point radius',
            default=50, soft_max=100, min=0, subtype='PERCENTAGE'
    )
    line_width: bpy.props.IntProperty(
            name='Base Line Width',
            description='The minimum width of the newly generated stroke',
            default=10, min=1, soft_max=100, subtype='PIXEL'
    )   
    postprocessing_method: bpy.props.EnumProperty(
        name='Methods',
        description='Algorithms to generate a smooth stroke',
        options={'ENUM_FLAG'},
        items = [
            ('SPLPREP', 'B-Spline', ''),
            ('RESAMPLE', 'Resample', '')
            ],
        default={'SPLPREP'}
    )
    b_smoothness: bpy.props.FloatProperty(
            name='B-Spline Smoothness',
            description='Smoothness factor when applying the B-spline fitting algorithm',
            default=1, soft_max=100, min=0
    )
    resample_length: bpy.props.FloatProperty(
            name='Resample Length',
            description='',
            default=0.02, min=0
    )
    smooth_repeat: bpy.props.IntProperty(
            name='Smooth Repeat',
            description='',
            default=2, min=1, max=1000
    )
    inherited_attributes: bpy.props.EnumProperty(
        name='Inherited Attributes',
        description='Inherit point attributes from the original strokes',
        options={'ENUM_FLAG'},
        items = [
            ('STRENGTH', 'Strength', ''),
            ('PRESSURE', 'Pressure', ''),
            ('COLOR', 'Color', ''),
            ('UV', 'UV', '')
            ],
        default=set(['UV', 'COLOR'])
    )
    output_layer: bpy.props.StringProperty(
        name='Output Layer',
        description='Draw the new stroke in this layer. If empty, draw to the active layer',
        default='',
        search=lambda self, context, edit_text: [layer.info for layer in context.object.data.layers]
    )
    output_material: bpy.props.StringProperty(
        name='Output Material',
        description='Draw the new stroke using this material. If empty, use the active material',
        default='',
        search=lambda self, context, edit_text: [material.name for material in context.object.data.materials if material]
    )
    keep_original: bpy.props.BoolProperty(
            name='Keep Original',
            default=True,
            description='Do not delete the original stroke'
    )
    
class FitSelectedOperator(CommonFittingConfig, bpy.types.Operator):
    """Fit select strokes or points to a new stroke"""
    bl_idname = "gpencil.nijigp_fit_selected"
    bl_label = "Single-Line Fit"
    bl_category = 'View'
    bl_options = {'REGISTER', 'UNDO'}    

    def draw(self, context):
        layout = self.layout
        layout.label(text = "Input Options:")
        box1 = layout.box()
        box1.prop(self, "line_sampling_size")
        box1.prop(self, "closed")
        
        layout.label(text = "Post-Processing Options:")
        box2 = layout.box()
        row = box2.row()
        row.prop(self, "postprocessing_method")  
        if 'SPLPREP' in self.postprocessing_method:
            box2.prop(self, "b_smoothness")
        if 'RESAMPLE' in self.postprocessing_method:
            box2.prop(self, "resample_length")
            box2.prop(self, "smooth_repeat")

        layout.label(text = "Output Options:")
        box3 = layout.box()   
        box3.prop(self, "line_width")
        box3.prop(self, "pressure_variance")
        box3.prop(self, "max_delta_pressure")
        box3.label(text='Inherit Point Attribtues:')
        row = box3.row()
        row.prop(self, "inherited_attributes")
        box3.prop(self, "output_layer", text='Layer', icon='OUTLINER_DATA_GP_LAYER')
        box3.prop(self, "output_material", text='Material', icon='MATERIAL')
        box3.prop(self, "keep_original")

    def execute(self, context):

        # Get input strokes
        gp_obj = context.object
        stroke_list = []
        frames_to_process = get_input_frames(gp_obj, multiframe=False)
        for frame in frames_to_process:
            stroke_list += get_input_strokes(gp_obj, frame)
        
        # Execute the fitting function
        b_smoothness = self.b_smoothness if 'SPLPREP' in self.postprocessing_method else None
        inv_mat, co_list, pressure_accumulation, pressure_list, strength_list, color_list, uv_list, depth_list = fit_2d_strokes(stroke_list, 
                                                                            search_radius=self.line_sampling_size/LINE_WIDTH_FACTOR, 
                                                                            smoothness_factor=b_smoothness,
                                                                            pressure_delta=self.pressure_variance*0.01, 
                                                                            closed = self.closed,
                                                                            operator=self)
        if not self.keep_original:
            bpy.ops.gpencil.delete()

        if co_list is None:
            bpy.ops.gpencil.select_all(action='DESELECT')
            return {'FINISHED'}

        # Turn fitting output to a new stroke
        output_layer = gp_obj.data.layers.active
        if len(self.output_layer) > 0:
            for layer in gp_obj.data.layers:
                if layer.info == self.output_layer:
                    output_layer = layer
        if not output_layer.active_frame:
            output_frame = output_layer.frames.new(context.scene.frame_current)
        else:
            output_frame = output_layer.active_frame

        output_material_idx = gp_obj.active_material_index
        if len(self.output_material) > 0:
            for i,material_slot in enumerate(gp_obj.material_slots):
                if material_slot.material and material_slot.material.name == self.output_material:
                    output_material_idx = i

        # Gather stroke attributes from input strokes
        new_stroke: bpy.types.GPencilStroke = output_frame.strokes.new()
        new_stroke.material_index = output_material_idx
        new_stroke.line_width = self.line_width
        copy_stroke_attributes(new_stroke, stroke_list,
                               copy_color = 'COLOR' in self.inherited_attributes,
                               copy_uv = 'UV' in self.inherited_attributes)
        new_stroke.points.add(co_list.shape[0])
        for i,point in enumerate(new_stroke.points):
            point.co = restore_3d_co(co_list[i], depth_list[i], inv_mat)
            point.pressure = pressure_list[i] if 'PRESSURE' in self.inherited_attributes else 1
            point.strength = strength_list[i] if 'STRENGTH' in self.inherited_attributes else 1
            point.uv_rotation = uv_list[i] if 'UV' in self.inherited_attributes else 0
            point.vertex_color = color_list[i] if 'COLOR' in self.inherited_attributes else (0,0,0,0)
            point.pressure *= (1 + min(pressure_accumulation[i], self.max_delta_pressure*0.01) )
        bpy.ops.gpencil.select_all(action='DESELECT')
        new_stroke.use_cyclic = self.closed
        new_stroke.select = True
        bpy.ops.transform.translate()

        # Post-processing
        if 'RESAMPLE' in self.postprocessing_method:
            bpy.ops.gpencil.stroke_sample(length=self.resample_length)
            bpy.ops.gpencil.stroke_smooth(repeat=self.smooth_repeat, smooth_strength=True)

        return {'FINISHED'}
    
class SelectSimilarOperator(bpy.types.Operator):
    """Find similar strokes with the selected ones that may belong to the same part of the drawing"""
    bl_idname = "gpencil.nijigp_cluster_select"
    bl_label = "Cluster Select"
    bl_category = 'View'
    bl_options = {'REGISTER', 'UNDO'}  

    line_sampling_size: bpy.props.IntProperty(
            name='Line Spacing',
            description='Strokes with gap smaller than this may be regarded similar',
            default=50, min=1, soft_max=100, subtype='PIXEL'
    )
    angular_tolerance: bpy.props.FloatProperty(
            name='Angular Tolerance',
            description='Two lines will not be regarded as similar with their directions deviated more than this value',
            default=math.pi/3, min=math.pi/18, max=math.pi/2, unit='ROTATION'
    )
    same_material: bpy.props.BoolProperty(
            name='Same Material',
            description='Ignore strokes with materials different from selected ones',
            default=True
    )
    repeat: bpy.props.BoolProperty(
            name='Repeat',
            description='Keep expanding selection until no new similar strokes found',
            default=False
    )

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "line_sampling_size")
        layout.prop(self, "angular_tolerance")
        layout.prop(self, "same_material")
        layout.prop(self, "repeat")

    def execute(self, context):
        gp_obj: bpy.types.Object = context.object
        frames_to_process = get_input_frames(gp_obj, gp_obj.data.use_multiedit)
        
        def process_one_frame(frame):
            stroke_list = get_input_strokes(gp_obj, frame)
            stroke_set = set()
            t_mat, inv_mat = get_transformation_mat(mode=context.scene.nijigp_working_plane,
                                                    gp_obj=gp_obj, operator=self)
            poly_list, _, _ = get_2d_co_from_strokes(stroke_list, t_mat, scale=False)
            kdt_list = []
            for i,stroke in enumerate(stroke_list):
                kdt_list.append(stroke_to_kdtree(poly_list[i]))
                
            # Add new strokes to selection whenever possible
            while True:
                new_strokes = []
                for stroke in frame.strokes:
                    tmp, _, _ = get_2d_co_from_strokes([stroke], t_mat, scale = False)
                    co_list = tmp[0]
                    kdt = stroke_to_kdtree(co_list)
                    for i,src_stroke in enumerate(stroke_list):
                        if self.same_material and src_stroke.material_index != stroke.material_index:
                            continue
                        if not stroke_bound_box_overlapping(stroke, src_stroke, t_mat):
                            continue
                        # Determine similarity based on the distance function
                        line_dist1 = distance_to_another_stroke(co_list, poly_list[i], kdt_list[i], self.angular_tolerance)
                        line_dist2 = distance_to_another_stroke(poly_list[i], co_list, kdt, self.angular_tolerance)
                        if min(line_dist1, line_dist2) < self.line_sampling_size / LINE_WIDTH_FACTOR:
                            stroke.select = True
                            # In repeat selection mode, add the stroke's information for the next iteration
                            if stroke not in stroke_set and self.repeat:
                                stroke_set.add(stroke)
                                new_strokes.append(stroke)
                                poly_list.append(co_list)
                                kdt_list.append(kdt)
                            break
                if len(new_strokes) < 1:
                    break
                stroke_list += new_strokes
        
        for frame in frames_to_process:
            process_one_frame(frame)
        return {'FINISHED'}
    
class ClusterAndFitOperator(CommonFittingConfig, bpy.types.Operator):
    """Dividing select strokes into clusters and fit each of them to a new stroke"""
    bl_idname = "gpencil.nijigp_cluster_and_fit"
    bl_label = "Multi-Line Fit"
    bl_category = 'View'
    bl_options = {'REGISTER', 'UNDO'}    

    cluster_criterion: bpy.props.EnumProperty(
            name='Criterion',
            items=[ ('DIST', 'By Distance (Absolute)', ''),
                    ('RATIO', 'By Distance (Relative)', ''),
                    ('NUM', 'By Number', '')],
            default='RATIO',
            description='The criterion determining how many clusters selected strokes will be divided into'
    )
    cluster_dist: bpy.props.FloatProperty(
            name='Absolute Distance',
            default=0.05, min=0,
            unit='LENGTH',
            description='The mininum distance between two clusters'
    ) 
    cluster_ratio: bpy.props.FloatProperty(
            name='Relative Distance',
            default=5, min=0, soft_max=100,
            subtype='PERCENTAGE',
            description='The mininum relative distance (compared to the stroke length) between two clusters'
    )
    cluster_num: bpy.props.IntProperty(
            name='Max Number',
            default=5, min=1,
            description='The maximum number of clusters'
    )

    def draw(self, context):
        layout = self.layout

        layout.label(text = "Clustering Options:")
        box0 = layout.box()
        box0.prop(self, "cluster_criterion")
        if self.cluster_criterion == 'DIST':
            box0.prop(self, "cluster_dist")
        elif self.cluster_criterion == 'NUM':
            box0.prop(self, "cluster_num")
        else:
            box0.prop(self, "cluster_ratio")

        layout.label(text = "Input Options:")
        box1 = layout.box()
        box1.prop(self, "line_sampling_size")
        box1.prop(self, "closed")
        
        layout.label(text = "Post-Processing Options:")
        box2 = layout.box()
        row = box2.row()
        row.prop(self, "postprocessing_method")  
        if 'SPLPREP' in self.postprocessing_method:
            box2.prop(self, "b_smoothness")
        if 'RESAMPLE' in self.postprocessing_method:
            box2.prop(self, "resample_length")
            box2.prop(self, "smooth_repeat")

        layout.label(text = "Output Options:")
        box3 = layout.box()   
        box3.prop(self, "line_width")
        box3.prop(self, "pressure_variance")
        box3.prop(self, "max_delta_pressure")
        box3.label(text='Inherit Point Attribtues:')
        row = box3.row()
        row.prop(self, "inherited_attributes")
        box3.prop(self, "output_layer", text='Layer', icon='OUTLINER_DATA_GP_LAYER')
        box3.prop(self, "output_material", text='Material', icon='MATERIAL')
        box3.prop(self, "keep_original")

    def execute(self, context):
        try:
            from scipy.cluster.hierarchy import linkage, fcluster
        except ImportError:
            self.report({"ERROR"}, "Please install dependencies in the Preferences panel.")
            return {'FINISHED'}

        # Get input strokes
        gp_obj = context.object
        stroke_list = []
        frames_to_process = get_input_frames(gp_obj, multiframe=False)
        for frame in frames_to_process:
            stroke_list += get_input_strokes(gp_obj, frame)
        if len(stroke_list)<2:
            self.report({"INFO"}, "Please select at least two strokes.")
            return {'FINISHED'}

        # Get stroke information
        t_mat, inv_mat = get_transformation_mat(mode=context.scene.nijigp_working_plane,
                                                gp_obj=gp_obj, strokes=stroke_list, operator=self)
        poly_list, _, _ = get_2d_co_from_strokes(stroke_list, t_mat, scale=False)
        kdt_list = []
        length_list = []
        for co_list in poly_list:
            kdt_list.append(stroke_to_kdtree(co_list))
            length_list.append(get_stroke_length(co_list=co_list))

        # Get stroke distance matrix
        dist_mat = []
        for i,co_list1 in enumerate(poly_list):
            for j,co_list2 in enumerate(poly_list):
                if i<j:
                    dist1 = distance_to_another_stroke(poly_list[i], poly_list[j], kdt_list[j])
                    dist2 = distance_to_another_stroke(poly_list[j], poly_list[i], kdt_list[i])
                    dist_mat.append(min(dist1,dist2))
                    if self.cluster_criterion == 'RATIO':
                        dist_mat[-1] /= 0.5 * (length_list[i] + length_list[j])

        # Hierarchy clustering algorithm
        linkage_mat = linkage(dist_mat, method='single')
        if self.cluster_criterion == 'DIST':
            cluster_res = fcluster(linkage_mat, self.cluster_dist, criterion='distance')
        elif self.cluster_criterion == 'RATIO':
            cluster_res = fcluster(linkage_mat, self.cluster_ratio/100.0, criterion='distance')
        else:
            cluster_res = fcluster(linkage_mat, self.cluster_num, criterion='maxclust')
                 
        # Place strokes in clusters
        cluster_map = {}
        for i,stroke in enumerate(stroke_list):
            cluster_idx = cluster_res[i]
            if cluster_idx not in cluster_map:
                cluster_map[cluster_idx] = []
            cluster_map[cluster_idx].append(stroke)

        # Process each cluster one by one
        generated_strokes = []
        for cluster in cluster_map:
            bpy.ops.gpencil.select_all(action='DESELECT')
            for stroke in cluster_map[cluster]:
                stroke.select = True
            bpy.ops.gpencil.nijigp_fit_selected(line_sampling_size = self.line_sampling_size,
                                            closed = self.closed,
                                            pressure_variance = self.pressure_variance,
                                            max_delta_pressure = self.max_delta_pressure,
                                            line_width = self.line_width,
                                            postprocessing_method = self.postprocessing_method,
                                            b_smoothness = self.b_smoothness,
                                            resample_length = self.resample_length,
                                            smooth_repeat = self.smooth_repeat,
                                            inherited_attributes = self.inherited_attributes,
                                            output_layer = self.output_layer,
                                            output_material = self.output_material,
                                            keep_original = self.keep_original)
            # Record the stroke selection status
            for frame in frames_to_process:
                generated_strokes += get_input_strokes(gp_obj, frame)
        
        # Select all generated strokes
        for stroke in generated_strokes:
            stroke.select = True
        return {'FINISHED'}
    
class FitLastOperator(CommonFittingConfig, bpy.types.Operator):
    """Fit the latest drawn stroke according to nearby strokes in the reference layer"""
    bl_idname = "gpencil.nijigp_fit_last"
    bl_label = "Fit Last Stroke"
    bl_category = 'View'
    bl_options = {'REGISTER', 'UNDO'}  

    cluster_criterion: bpy.props.EnumProperty(
            name='Criterion',
            items=[ ('DIST', 'Absolute', ''),
                    ('RATIO', 'Relative', '')],
            default='RATIO',
            description='The criterion determining which strokes will be divided into'
    )
    cluster_ratio: bpy.props.FloatProperty(
            name='Detection Ratio',
            default=5, min=0, soft_max=100,
            subtype='PERCENTAGE',
            description='Search strokes in the reference layer if it is close enough to the last drawn stroke'
    )
    cluster_dist: bpy.props.FloatProperty(
            name='Detection Radius',
            default=0.05, min=0,
            unit='LENGTH',
            description='Search strokes in the reference layer if it is close enough to the last drawn stroke'
    )
    resample_output: bpy.props.BoolProperty(
            name='Resample',
            default=True,
            description='Resample the generated stroke to keep the number of points similar to the original stroke'
    )
    trim_ends: bpy.props.BoolProperty(
            name='Trim Ends',
            default=True,
            description='Move ends of the drawn stroke as little as possible'
    )
    reference_layer: bpy.props.StringProperty(
        name='Reference Layer',
        description='The layer with draft strokes which are used to guide the fitting of the newly drawn stroke',
        default='',
        search=lambda self, context, edit_text: [layer.info for layer in context.object.data.layers]
    ) 

    def draw(self, context):
        layout = self.layout
        layout.label(text = "Input Options:")
        box1 = layout.box()
        row = box1.row()
        row.label(text = "Reference Layer:")
        row.prop(self, "reference_layer", icon='OUTLINER_DATA_GP_LAYER', text='')
        box1.prop(self, "cluster_criterion")
        if self.cluster_criterion == 'DIST':
            box1.prop(self, "cluster_dist")
        else:
            box1.prop(self, "cluster_ratio")
        box1.prop(self, "line_sampling_size")
        
        layout.label(text = "Post-Processing Options:")
        box2 = layout.box()
        box2.prop(self, "b_smoothness")
        row = box2.row()
        row.prop(self, "resample_output")
        row.prop(self, "trim_ends")

        layout.label(text = "Output Options:")
        box3 = layout.box()   
        box3.prop(self, "pressure_variance")
        box3.prop(self, "max_delta_pressure")

    def execute(self, context):
        # Get input and context
        gp_obj = context.object
        if len(self.reference_layer) > 0:
            reference_layer = gp_obj.data.layers[self.reference_layer]
        else:
            return {'FINISHED'}
        drawing_layer = gp_obj.data.layers.active
        if not reference_layer.active_frame:
            self.report({"INFO"}, "The reference layer has no stroke data.")
            return {'FINISHED'} 
        if reference_layer == drawing_layer:
            self.report({"INFO"}, "Please draw in a layer other than the reference layer.")
            return {'FINISHED'}  
                 
        # Get stroke information from the input
        if not drawing_layer.active_frame or len(drawing_layer.active_frame.strokes)<1:
            return {'FINISHED'} 
        stroke_index = 0 if context.scene.tool_settings.use_gpencil_draw_onback else -1
        src_stroke: bpy.types.GPencilStroke = drawing_layer.active_frame.strokes[stroke_index]
        
        t_mat, inv_mat = get_transformation_mat(mode=context.scene.nijigp_working_plane,
                                                gp_obj=gp_obj, operator=self)
        tmp, tmp2, _ = get_2d_co_from_strokes([src_stroke], t_mat, scale=False)
        src_co_list = tmp[0]
        depth_lookup_tree = DepthLookupTree(tmp, tmp2)
        src_kdt = stroke_to_kdtree(src_co_list)
        src_stroke_length = get_stroke_length(co_list=src_co_list)
        
        # Check each stroke in the reference layer for similarity
        stroke_list = []
        threshold = (self.cluster_dist if self.cluster_criterion == 'DIST' else
                     self.cluster_ratio / 100.0 * src_stroke_length)
        for stroke in reference_layer.active_frame.strokes:
            if not stroke_bound_box_overlapping(stroke, src_stroke, t_mat):
                continue
            tmp, _, _ = get_2d_co_from_strokes([stroke], t_mat, scale=False)
            co_list = tmp[0]
            kdt = stroke_to_kdtree(co_list)
            line_dist1 = distance_to_another_stroke(co_list, src_co_list, src_kdt)
            line_dist2 = distance_to_another_stroke(src_co_list, co_list, kdt)
            if min(line_dist1, line_dist2) < threshold:
                stroke_list.append(stroke)

        if len(stroke_list)<1:
            return {'FINISHED'}  
        
        # Execute the fitting function
        b_smoothness = self.b_smoothness
        _, new_co_list, pressure_accumulation, _, _, _, _, _ = fit_2d_strokes(stroke_list, 
                                                                        search_radius=self.line_sampling_size/LINE_WIDTH_FACTOR, 
                                                                        smoothness_factor=b_smoothness,
                                                                        pressure_delta=self.pressure_variance*0.01, 
                                                                        closed=src_stroke.use_cyclic,
                                                                        operator=self,
                                                                        t_mat = t_mat,
                                                                        inv_mat = inv_mat)
        # Orientation correction and trimming
        src_direction = Vector(src_co_list[-1]) - Vector(src_co_list[0])
        new_direction = Vector(new_co_list[-1]) - Vector(new_co_list[0])
        angle_diff = src_direction.angle(new_direction)
        if angle_diff > math.pi/2:
            new_co_list = np.flipud(new_co_list)
            
        if self.trim_ends:
            start_idx, end_idx = 0, len(new_co_list)-1
            for i,co in enumerate(new_co_list):
                vec = Vector(co)-Vector(src_co_list[0])
                if vec.length < threshold:
                    start_idx = i
                    break
            for i,co in enumerate(np.flipud(new_co_list)):
                vec = Vector(co)-Vector(src_co_list[-1])
                if vec.length < threshold:
                    end_idx = len(new_co_list)-i
                    break
            if start_idx < end_idx:
                new_co_list = new_co_list[start_idx:end_idx]
            
        # Remove the last drawn stroke and generate a new one
        new_stroke: bpy.types.GPencilStroke = drawing_layer.active_frame.strokes.new()
        copy_stroke_attributes(new_stroke, [src_stroke],
                            copy_hardness=True, copy_linewidth=True,
                            copy_cap=True, copy_cyclic=True,
                            copy_uv=True, copy_material=True, copy_color=True)
        new_stroke.points.add(new_co_list.shape[0])
        for i,point in enumerate(new_stroke.points):
            point.co = restore_3d_co(new_co_list[i], depth_lookup_tree.get_depth(new_co_list[i]), inv_mat)
            attr_idx = int( float(i) / (len(new_stroke.points)-1) * (len(src_stroke.points)-1) )
            point.pressure = src_stroke.points[attr_idx].pressure
            point.strength = src_stroke.points[attr_idx].strength
            point.vertex_color = src_stroke.points[attr_idx].vertex_color
            point.uv_rotation = src_stroke.points[attr_idx].uv_rotation
            point.pressure *= (1 + min(pressure_accumulation[i], self.max_delta_pressure*0.01) )

        # Resample the generated stroke
        if self.resample_output:
            resample_length = get_stroke_length(stroke=new_stroke)/(len(src_stroke.points)+.5)
            select_map = save_stroke_selection(gp_obj)
            bpy.ops.gpencil.select_all(action='DESELECT')
            new_stroke.select = True
            bpy.ops.gpencil.stroke_sample(length=resample_length)
            load_stroke_selection(gp_obj, select_map)

        drawing_layer.active_frame.strokes.remove(src_stroke)
        return {'FINISHED'}  

class PinchSelectedOperator(bpy.types.Operator):
    """Pinch and join strokes with their ends close to each other"""
    bl_idname = "gpencil.nijigp_pinch"
    bl_label = "Pinch Together"
    bl_category = 'View'
    bl_options = {'REGISTER', 'UNDO'}    

    threshold: bpy.props.FloatProperty(
            name='Distance Threshold',
            default=0.05, min=0,
            unit='LENGTH',
            description='If either end of a stroke is close to another stroke end, the two strokes will be pinched together'
    ) 
    transition_length: bpy.props.FloatProperty(
            name='Transition',
            default=0.5, min=0.1, max=1,
            description='The portion of the stroke where points will be moved'
    )
    mix_factor: bpy.props.FloatProperty(
            name='Factor',
            default=1, min=0, max=1,
            description='1 means filling the gap completely, and 0 means no movement'
    )
    end_to_end: bpy.props.BoolProperty(
            name='Consider End Points',
            default=True,
            description='Move the stroke when its end point is close to an end point on another stroke'
    )
    end_to_intermediate: bpy.props.BoolProperty(
            name='Consider Intermediate Points',
            default=False,
            description='Also move the stroke when its end point is close to an intermediate point on another stroke'
    )
    contact_pressure: bpy.props.FloatProperty(
            name='Contact Pressure',
            default=0.5, min=0, max=1,
            description='Increase the pressure of the contact point to achieve a shadow-like effect'
    ) 
    contact_length: bpy.props.IntProperty(
            name='Contact Length',
            default=5, min=0, soft_max=10,
            description='Area of the contact shadow'
    )  
    join_strokes: bpy.props.BoolProperty(
            name='Join Strokes',
            default=False,
            description='Join pinched strokes together if possible'
    )

    def draw(self, context):
        layout = self.layout

        layout.prop(self, "threshold")
        layout.prop(self, "transition_length")
        layout.prop(self, "mix_factor")
        layout.prop(self, "end_to_end")
        layout.prop(self, "end_to_intermediate")
        if self.end_to_intermediate:
            layout.prop(self, "contact_pressure")
            layout.prop(self, "contact_length")
        if self.end_to_end:    
            layout.prop(self, "join_strokes")

    def execute(self, context):

        # Get input strokes
        gp_obj = context.object
        stroke_list = []
        frames_to_process = get_input_frames(gp_obj, multiframe=False)
        for frame in frames_to_process:
            stroke_list += get_input_strokes(gp_obj, frame)
        stroke_list = [stroke for stroke in stroke_list
                       if len(stroke.points)>1 and not stroke.use_cyclic]
        
        if len(stroke_list)<2:
            self.report({"INFO"}, "Please select at least two open strokes.")
            return {'FINISHED'}
        
        # Variables related to transformation
        t_mat, inv_mat = get_transformation_mat(mode=context.scene.nijigp_working_plane,
                                                gp_obj=gp_obj, strokes=stroke_list, operator=self)
        poly_list, depth_list, _ = get_2d_co_from_strokes(stroke_list, t_mat, scale=False)
        depth_lookup_tree = DepthLookupTree(poly_list, depth_list)
        trans2d = lambda co: (t_mat @ co).xy
        
        # Detect and eliminate the gap between two end points
        point_processed = set()
        if self.end_to_end:
            num_chains = 0
            stroke_chain_idx = {}
            point_offset = {}
            for i,stroke in enumerate(stroke_list):
                for point in [stroke.points[0], stroke.points[-1]]:
                    if point in point_offset:
                        continue
                    for j,stroke0 in enumerate(stroke_list):
                        if point in point_offset:
                            break
                        for point0 in [stroke0.points[0], stroke0.points[-1]]:
                            # The case where two points are close and not paired
                            if i<j and point0 not in point_offset and (trans2d(point0.co)-trans2d(point.co)).length < self.threshold:
                                # Assign the stroke to a chain
                                if stroke0 in stroke_chain_idx and stroke in stroke_chain_idx:
                                    pass
                                elif stroke0 in stroke_chain_idx:
                                    stroke_chain_idx[stroke] = stroke_chain_idx[stroke0]
                                elif stroke in stroke_chain_idx:
                                    stroke_chain_idx[stroke0] = stroke_chain_idx[stroke]
                                else:
                                    stroke_chain_idx[stroke] = num_chains
                                    stroke_chain_idx[stroke0] = num_chains
                                    num_chains += 1

                                # Calculate the offset value
                                center = 0.5 * (trans2d(point0.co)+trans2d(point.co))
                                point_offset[point] = center - trans2d(point.co)
                                point_offset[point0] = center - trans2d(point0.co)
                                break
            # Padding zeros for ends without offsets
            for i,stroke in enumerate(stroke_list):
                for point in [stroke.points[0], stroke.points[-1]]:
                    if point not in point_offset:
                        point_offset[point] = Vector((0,0))
                    else:
                        point_processed.add(point)

            # Apply offsets
            for i,stroke in enumerate(stroke_list):
                for j,point in enumerate(stroke.points):
                    w1, w2 = 1-j/(len(stroke.points)-1), j/(len(stroke.points)-1)
                    J = len(stroke.points)-1
                    w1 = 1 - j/J/self.transition_length
                    w2 = 1 - (J-j)/J/self.transition_length
                    new_co = trans2d(point.co)
                    new_co += self.mix_factor * (point_offset[stroke.points[0]] * smoothstep(w1) 
                                                 + point_offset[stroke.points[-1]] * smoothstep(w2))
                    point.co = restore_3d_co(new_co, depth_lookup_tree.get_depth(new_co), inv_mat)

        # Detect and eliminate the gap between an endpoint and a non-endpoint
        def prolong_to_stroke(stroke, end_type, ray_length, stroke0):
            '''
            Add length to the end of a stroke to see if it crosses another stroke
            '''
            if end_type == 'start':
                p1 = trans2d(stroke.points[0].co)
                delta = (p1 - trans2d(stroke.points[1].co)).normalized() * ray_length
            else:
                p1 = trans2d(stroke.points[-1].co)
                delta = (p1 - trans2d(stroke.points[-2].co)).normalized() * ray_length
            min_dist = None
            ret = None, None
            for i,p3 in enumerate(stroke0.points):
                if i==0:
                    continue
                p3 = trans2d(p3.co)
                p4 = trans2d(stroke0.points[i-1].co)
                for p2 in (p1-delta, p1+delta):
                    intersect = geometry.intersect_line_line_2d(p1,p2,p3,p4)
                    if intersect:
                        dist = (intersect-p1).length
                        if not min_dist or dist<min_dist:
                            min_dist = dist
                            ret = intersect, (i if (intersect-p3).length<(intersect-p4).length else i-1)
            return ret

        if self.end_to_intermediate:
            for i,stroke in enumerate(stroke_list):
                for end_type, point in {'start':stroke.points[0], 'end':stroke.points[-1]}.items():
                    if point in point_processed:
                        continue
                    for j,stroke0 in enumerate(stroke_list):
                        if i==j:
                            continue
                        intersect, contact_idx = prolong_to_stroke(stroke, end_type, self.threshold, stroke0)
                        if intersect:
                            # Move the end of the stroke immediately
                            offset = intersect - trans2d(point.co)
                            L = math.ceil((len(stroke.points)-1)*self.transition_length)
                            for l in range(L+1):
                                target_point = stroke.points[l] if end_type=='start' else stroke.points[-l-1]
                                new_co = trans2d(target_point.co) + offset * (1-l/L) * self.mix_factor
                                target_point.co = restore_3d_co(new_co, depth_lookup_tree.get_depth(new_co), inv_mat)

                            # Thicken the contact points
                            contact_pressure = self.contact_pressure * self.mix_factor
                            stroke0.points[contact_idx].pressure += contact_pressure
                            (stroke.points[0] if end_type=='start' else stroke.points[-1]).pressure += contact_pressure
                            for delta_idx in range(1, self.contact_length):
                                for idx in (contact_idx+delta_idx, contact_idx-delta_idx):
                                    if idx>=0 and idx<len(stroke0.points):
                                        stroke0.points[idx].pressure += contact_pressure*(1-delta_idx/self.contact_length)
                                idx = delta_idx if end_type=='start' else len(stroke.points)-1-delta_idx
                                if idx>=0 and idx<len(stroke.points):
                                    stroke.points[idx].pressure += contact_pressure*(1-delta_idx/self.contact_length)
        # Apply join
        if self.end_to_end and self.join_strokes:
            for i in range(num_chains):
                bpy.ops.gpencil.select_all(action='DESELECT')
                for stroke in stroke_chain_idx:
                    if stroke_chain_idx[stroke]==i:
                        stroke.select = True
                bpy.ops.gpencil.stroke_join()

        return {'FINISHED'}
    
class TaperSelectedOperator(bpy.types.Operator):
    """Reshape the ends of selected strokes"""
    bl_idname = "gpencil.nijigp_taper_selected"
    bl_label = "Taper Selected Strokes"
    bl_category = 'View'
    bl_options = {'REGISTER', 'UNDO'}  

    start_length: bpy.props.FloatProperty(
            name='Taper Start',
            default=1, min=0, max=1,
            description='Factor of stroke points that will be impacted'
    ) 
    end_length: bpy.props.FloatProperty(
            name='Taper End',
            default=1, min=0, max=1,
            description='Factor of stroke points that will be impacted'
    )
    curvature_factor: bpy.props.FloatProperty(
            name='Curvature Factor',
            default=0.05, min=0, soft_max=0.1,
            description='Shape the stroke based on its curvature/concavity'
    )
    curvature_limit: bpy.props.FloatProperty(
            name='Max Change',
            default=0.6, min=0, soft_max=1,
            description='The change of attributes due to curvature cannot exceed this value'
    )
    smooth_level: bpy.props.IntProperty(
            name='Smooth Level',
            default=5, min=0, soft_max=10,
            description='Smooth the curvature values along the stroke'
    )
    invert_leftturn: bpy.props.BoolProperty(
            name='Invert Left Turn', default = False,
            description='Decrease the attribute value at left turns rather than increase it'
    )
    invert_rightturn: bpy.props.BoolProperty(
            name='Invert Right Turn', default = False,
            description='Decrease the attribute value at right turns rather than increase it'
    )
    target_attributes: bpy.props.EnumProperty(
        name='Affected Attributes',
        description='Point attributes to modify',
        options={'ENUM_FLAG'},
        items = [
            ('PRESSURE', 'Pressure', ''),
            ('STRENGTH', 'Strength', '')
            ],
        default={'PRESSURE'}
    )
    operation: bpy.props.EnumProperty(
        name='Operation',
        description='How to modify the attributes',
        items = [
            ('MULTIPLY', 'Multiply', ''),
            ('REPLACE', 'Replace', '')
            ],
        default='MULTIPLY'
    )
    line_width: bpy.props.IntProperty(
            name='Uniform Line Width',
            description='Set the same base line width value to all selected strokes. 0 means not to change the line width',
            default=0, min=0, soft_max=100, subtype='PIXEL'
    )
    min_radius: bpy.props.FloatProperty(name='Min Radius', default=0, min=0, soft_max=1) 
    max_radius: bpy.props.FloatProperty(name='Max Radius', default=1, min=0, soft_max=1) 

    def draw(self, context):
        layout = self.layout
        layout.label(text = "End Points:")
        box1 = layout.box()
        box1.prop(self, "start_length")
        box1.prop(self, "end_length")
        row = box1.row(align=True)
        row.label(text="Radius Range")
        row.prop(self, "min_radius", text="")
        row.prop(self, "max_radius", text="")
        
        layout.label(text = "Turning Points:")
        box2 = layout.box()
        box2.prop(self, "curvature_factor")
        box2.prop(self, "curvature_limit")
        box2.prop(self, "smooth_level")
        row = box2.row(align=True)
        row.label(text="Invert at:")
        row.prop(self, "invert_leftturn", text="Left Turns")
        row.prop(self, "invert_rightturn", text="Right Turns")

        layout.label(text = "Output Options:")
        box3 = layout.box()
        row = box3.row()
        row.prop(self, "target_attributes")
        if "PRESSURE" in self.target_attributes:
            box3.prop(self, "line_width")
        box3.prop(self, "operation")

    def execute(self, context):

        def process_one_stroke(stroke: bpy.types.GPencilStroke, co_list):
            if self.line_width > 0:
                stroke.line_width = self.line_width
            L = len(stroke.points)
            L1 = int(L*self.start_length)
            L2 = int(L*self.end_length)
            factor_arr = np.ones(L)

            # Calculate impacts of tapers
            factor_arr[:L1] *= np.linspace(0, 1, L1)
            factor_arr[L-L2:L] *= np.linspace(1, 0, L2)

            # Scale the array according to the range
            factor_range = (np.min(factor_arr), np.max(factor_arr))
            if math.isclose(factor_range[1], factor_range[0]):
                factor_arr[:] = self.max_radius
            else:
                factor_arr = ( (factor_arr - factor_range[0])
                                    / (factor_range[1] - factor_range[0])
                                    * (self.max_radius - self.min_radius) 
                                    + self.min_radius)

            # Calculate impacts of curvature/concavity
            additional_factor_arr = np.ones(L)
            if L>2:
                for i in range(1, L-1):
                    co0 = Vector(co_list[i-1])
                    co1 = Vector(co_list[i])
                    co2 = Vector(co_list[i+1])
                    curvature, center = get_concyclic_info(co0[0],co0[1],co1[0],co1[1],co2[0],co2[1])
                    direction = 0 if not center else (co2-co1).angle_signed(center-co1)
                    delta = min(curvature * self.curvature_factor, self.curvature_limit)
                    if (direction<0 and self.invert_rightturn) or (direction>0 and self.invert_leftturn):
                        delta *= -1
                    additional_factor_arr[i] += delta

                # Smooth the curvature by convolution
                kernel = np.array([1.0/3, 1.0/3, 1.0/3])
                for i in range(self.smooth_level):
                    additional_factor_arr = np.convolve(additional_factor_arr, kernel, mode='same')
                factor_arr *= additional_factor_arr

            # Apply final modification results
            for i,point in enumerate(stroke.points):
                if 'PRESSURE' in self.target_attributes:
                    point.pressure = point.pressure * factor_arr[i] if self.operation=='MULTIPLY' else factor_arr[i]
                if 'STRENGTH' in self.target_attributes:
                    point.strength = point.strength * factor_arr[i] if self.operation=='MULTIPLY' else factor_arr[i]

        gp_obj = context.object
        frames_to_process = get_input_frames(gp_obj, gp_obj.data.use_multiedit)
        stroke_list = []
        for frame in frames_to_process:
            stroke_list += get_input_strokes(gp_obj, frame)
        t_mat, inv_mat = get_transformation_mat(mode=context.scene.nijigp_working_plane,
                                                    gp_obj=gp_obj, strokes=stroke_list, operator=self)
        poly_list, _, _ = get_2d_co_from_strokes(stroke_list, t_mat, scale=False)
        for i, stroke in enumerate(stroke_list):
            process_one_stroke(stroke, poly_list[i])

        return {'FINISHED'}
