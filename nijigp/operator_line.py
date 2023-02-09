import bpy
import math
from mathutils import *
from .utils import *

def stroke_to_kdtree(co_list):
    n = len(co_list)
    kdt = kdtree.KDTree(n)
    for i in range(n):
        kdt.insert(vec2_to_vec3(co_list[i]), i)
    kdt.balance()
    return kdt

def fit_2d_strokes(strokes, search_radius, smoothness_factor = 1, pressure_delta = 0, closed = False, operator = None):
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
    empty_result = None, None, None, None, None
    try:
        import triangle as tr
        from scipy.sparse.csgraph import minimum_spanning_tree
        from scipy.interpolate import splprep, splev
    except:
        if operator:
            operator.report({"ERROR"}, "Please install dependencies in the Preferences panel.")
        return empty_result
    import numpy as np

    # Create a KDTree for point attribute lookup
    poly_list, scale_factor = stroke_to_poly(strokes)
    total_point_count = 0
    for i,stroke in enumerate(strokes):
        if len(stroke.points)<2:
            continue
        total_point_count += len(stroke.points)

    if total_point_count<3:
        return empty_result
    kdt = kdtree.KDTree(total_point_count)
    kdt_tangent_list = []
    kdt_stroke_list = []
    kdt_point_list = []
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
                kdt.insert(vec2_to_vec3(poly_list[i][j],0,1), kdt_idx)
                kdt_idx += 1
                if j>0:
                    kdt_tangent_list.append((poly_list[i][j][0]-poly_list[i][j-1][0],
                                            poly_list[i][j][1]-poly_list[i][j-1][1]))
                else:
                    kdt_tangent_list.append((poly_list[i][j+1][0]-poly_list[i][j][0],
                                            poly_list[i][j+1][1]-poly_list[i][j][1]))    
                kdt_stroke_list.append(i)
                kdt_point_list.append(point)
                tr_input['vertices'].append(poly_list[i][j])                    
    kdt.balance()

    # Triangulation and spanning tree conversion
    tr_output = tr.triangulate(tr_input, '')
    def e_dist(i,j):
        src = tr_output['vertices'][i]
        dst = tr_output['vertices'][j]
        return np.sqrt((dst[0]-src[0])**2 + (dst[1]-src[1])**2)
    num_vert = len(tr_output['vertices'])
    dist = np.zeros((num_vert, num_vert))
    for f in tr_output['triangles']:
        dist[f[1], f[0]] = e_dist(f[0], f[1])
        dist[f[2], f[0]] = e_dist(f[0], f[2])
        dist[f[1], f[2]] = e_dist(f[2], f[1])
        dist[f[0], f[1]] = e_dist(f[0], f[1])
        dist[f[0], f[2]] = e_dist(f[0], f[2])
        dist[f[2], f[1]] = e_dist(f[2], f[1])
    mst = minimum_spanning_tree(dist).toarray()
    mst = np.maximum(mst, mst.transpose())

    # Find the longest path in the tree by executing DFS twice
    def tree_dfs(mat, node, parent):
        dist_sum = 0
        path = []
        for i, value in enumerate(mat[node]):
            if value > 0 and i!=parent:
                child_sum, child_path = tree_dfs(mat, i, node)
                if dist_sum < child_sum + value:
                    dist_sum = child_sum + value
                    path = child_path
        return dist_sum, [node]+path    
    _, path_half = tree_dfs(mst, 0, None)
    total_length, path_whole = tree_dfs(mst, path_half[-1], None)    
    
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
    inherited_color = np.zeros((len(path_whole), 3))

    # Apply offsets in the normal direction if there are points in the neighborhood
    # At the same time, inherit the point attributes
    for i,co in enumerate(co_raw):
        self_vec ,self_idx,_ = kdt.find(vec2_to_vec3(co,0,1))
        self_vec = vec3_to_vec2(self_vec)
        self_stroke = kdt_stroke_list[self_idx]
        unit_normal_vector = Vector(kdt_tangent_list[self_idx]).orthogonal().normalized()
        sum_normal_offset = 0
        neighbors = kdt.find_range(vec2_to_vec3(co,0,1), search_radius)
        for neighbor in neighbors:
            if kdt_stroke_list[neighbor[1]]!=self_stroke:
                normal_dist = vec3_to_vec2(neighbor[0]) - self_vec
                normal_dist = normal_dist.dot(unit_normal_vector)
                sum_normal_offset += normal_dist
                accumulated_pressure_raw[i] += pressure_delta
            # Inherit each attribute
            inherited_pressure_raw[i] += kdt_point_list[neighbor[1]].pressure
            inherited_strength_raw[i] += kdt_point_list[neighbor[1]].strength
            inherited_color[i] += np.array(kdt_point_list[neighbor[1]].vertex_color)[:3] * kdt_point_list[neighbor[1]].vertex_color[3]

        sum_normal_offset /= len(neighbors)
        inherited_pressure_raw[i] /= len(neighbors)
        inherited_strength_raw[i] /= len(neighbors)
        inherited_color[i] /= len(neighbors)
        co_raw[i] += unit_normal_vector * sum_normal_offset

    # Postprocessing: B-spline fitting
    if smoothness_factor is None:
        return co_raw, accumulated_pressure_raw, inherited_pressure_raw, inherited_strength_raw, inherited_color

    attributes_index = np.linspace(0,1,len(path_whole))
    if closed:
        co_raw = np.append(co_raw, [co_raw[0]], axis=0)
        attributes_index = np.append(attributes_index, 0)
        accumulated_pressure_raw = np.append(accumulated_pressure_raw, accumulated_pressure_raw[0])
        inherited_pressure_raw = np.append(inherited_pressure_raw, inherited_pressure_raw[0])
        inherited_strength_raw = np.append(inherited_strength_raw, inherited_strength_raw[0])
        inherited_color = np.append(inherited_color, [inherited_color[0]], axis=0)
    tck, u = splprep([co_raw[:,0], co_raw[:,1]], s=total_length**2 * smoothness_factor * 0.001, per=closed)
    co_fit = np.array(splev(u, tck)).transpose()    
    tck2, u2 = splprep([attributes_index, accumulated_pressure_raw], per=closed)
    accumulated_pressure_fit = np.array(splev(u2, tck2))[1]
    tck3, u3 = splprep([attributes_index, inherited_pressure_raw], per=closed)
    inherited_pressure_fit = np.array(splev(u3, tck3))[1]
    tck4, u4 = splprep([attributes_index, inherited_strength_raw], per=closed)
    inherited_strength_fit = np.array(splev(u4, tck4))[1]

    return co_fit, accumulated_pressure_fit, inherited_pressure_fit, inherited_strength_fit, inherited_color

def distance_to_another_stroke(co_list1, co_list2, kdt2 = None, angular_tolerance = math.pi/4, correct_orientation = True):
    '''
    Calculating the similarity between two lines
    '''
    import numpy as np
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
            kdt2.insert(vec2_to_vec3(co_list2[i]), i)
        kdt2.balance()        

    # Get point-wise distance values
    idx_arr = np.zeros(n1, dtype='int')
    dist_arr = np.zeros(n1)
    for i in range(n1):
        _, idx_arr[i], dist_arr[i] = kdt2.find(vec2_to_vec3(co_list1[i]))

    # Calculate the orientation difference of two lines
    contact_idx1 = np.argmin(dist_arr)
    contact_idx2 = min(idx_arr[contact_idx1], n2-2)
    contact_idx1 = min(contact_idx1, n1-2)
    direction1 = Vector(co_list1[contact_idx1+1]) - Vector(co_list1[contact_idx1])
    direction2 = Vector(co_list2[contact_idx2+1]) - Vector(co_list2[contact_idx2])
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
            ('COLOR', 'Color', '')
            ],
        default=set()
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
        import numpy as np

        # Get input strokes
        gp_obj = context.object
        stroke_list = []
        for i,layer in enumerate(gp_obj.data.layers):
            if layer.active_frame and not layer.lock:
                for stroke in layer.active_frame.strokes:
                    if stroke.select:
                        stroke_list.append(stroke)
        
        # Execute the fitting function
        b_smoothness = self.b_smoothness if 'SPLPREP' in self.postprocessing_method else None
        co_list, pressure_accumulation, pressure_list, strength_list, color_list = fit_2d_strokes(stroke_list, 
                                                                        search_radius=self.line_sampling_size/LINE_WIDTH_FACTOR, 
                                                                        smoothness_factor=b_smoothness,
                                                                        pressure_delta=self.pressure_variance*0.01, 
                                                                        closed = self.closed,
                                                                        operator=self)
        if co_list is None:
            bpy.ops.gpencil.select_all(action='DESELECT')
            return {'FINISHED'}

        if not self.keep_original:
            bpy.ops.gpencil.delete()

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

        new_stroke: bpy.types.GPencilStroke = output_frame.strokes.new()
        new_stroke.material_index = output_material_idx
        new_stroke.line_width = self.line_width
        new_stroke.points.add(co_list.shape[0])
        for i,point in enumerate(new_stroke.points):
            point.co = vec2_to_vec3(co_list[i], depth=0, scale_factor=1)
            point.pressure = pressure_list[i] if 'PRESSURE' in self.inherited_attributes else 1
            point.strength = strength_list[i] if 'STRENGTH' in self.inherited_attributes else 1
            point.vertex_color[3] = 1 if 'COLOR' in self.inherited_attributes else 0
            point.vertex_color[0] = color_list[i][0] if 'COLOR' in self.inherited_attributes else 0
            point.vertex_color[1] = color_list[i][1] if 'COLOR' in self.inherited_attributes else 0
            point.vertex_color[2] = color_list[i][2] if 'COLOR' in self.inherited_attributes else 0
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
    bl_idname = "gpencil.nijigp_select_similar"
    bl_label = "Select Similar"
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

        # Get the scope of searching
        frame_list, frame_set = [], set()
        stroke_list, stroke_set = [], set()
        for layer in gp_obj.data.layers:
            for frame in layer.frames:
                for stroke in frame.strokes:
                    if stroke.select and not is_stroke_locked(stroke, gp_obj):
                        stroke_list.append(stroke)
                        frame_list.append(frame)
                        stroke_set.add(stroke)
                        frame_set.add(frame)

        # Initialization
        poly_list, _ = stroke_to_poly(stroke_list, scale = False, correct_orientation = False)
        kdt_list = []

        for i,stroke in enumerate(stroke_list):
            kdt_list.append(stroke_to_kdtree(poly_list[i]))

        # Check every stroke in target frames
        while True:
            new_strokes = []
            new_stroke_frames= []
            for frame in frame_set:
                for stroke in frame.strokes:
                    tmp, _ = stroke_to_poly([stroke], scale = False, correct_orientation = False)
                    co_list = tmp[0]
                    kdt = stroke_to_kdtree(co_list)
                    for i,src_stroke in enumerate(stroke_list):
                        if frame_list[i] != frame:
                            continue
                        if self.same_material and src_stroke.material_index != stroke.material_index:
                            continue
                        if not overlapping_bounding_box(stroke, src_stroke):
                            continue

                        line_dist1 = distance_to_another_stroke(co_list, poly_list[i], kdt_list[i], self.angular_tolerance)
                        line_dist2 = distance_to_another_stroke(poly_list[i], co_list, kdt, self.angular_tolerance)
                        if min(line_dist1, line_dist2) < self.line_sampling_size / LINE_WIDTH_FACTOR:
                            stroke.select = True
                            # In repeat selection mode, add the stroke's information for the next iteration
                            if stroke not in stroke_set and self.repeat:
                                stroke_set.add(stroke)
                                new_strokes.append(stroke)
                                new_stroke_frames.append(frame)
                                poly_list.append(co_list)
                                kdt_list.append(kdt)
                            break
            if len(new_strokes) < 1:
                break
            stroke_list += new_strokes
            frame_list += new_stroke_frames

        return {'FINISHED'}
    
class ClusterAndFitOperator(CommonFittingConfig, bpy.types.Operator):
    """Dividing select strokes into clusters and fit each of them to a new stroke"""
    bl_idname = "gpencil.nijigp_cluster_and_fit"
    bl_label = "Multi-Line Fit"
    bl_category = 'View'
    bl_options = {'REGISTER', 'UNDO'}    

    cluster_criterion: bpy.props.EnumProperty(
            name='Criterion',
            items=[ ('DIST', 'By Distance', ''),
                    ('NUM', 'By Number', '')],
            default='DIST',
            description='The criterion determining how many clusters selected strokes will be divided into'
    )
    cluster_dist: bpy.props.FloatProperty(
            name='Min Distance',
            default=0.05, min=0,
            unit='LENGTH',
            description='The mininum distance between two clusters'
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
        else:
            box0.prop(self, "cluster_num")

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
        import numpy as np
        try:
            from scipy.cluster.hierarchy import linkage, fcluster
        except ImportError:
            self.report({"ERROR"}, "Please install dependencies in the Preferences panel.")
            return {'FINISHED'}

        # Get input strokes
        gp_obj = context.object
        stroke_list = []
        for i,layer in enumerate(gp_obj.data.layers):
            if layer.active_frame and not layer.lock:
                for stroke in layer.active_frame.strokes:
                    if stroke.select:
                        stroke_list.append(stroke)
        if len(stroke_list)<2:
            self.report({"INFO"}, "Please select at least two strokes.")
            return {'FINISHED'}

        # Get stroke information
        poly_list, _ = stroke_to_poly(stroke_list, scale = False, correct_orientation = False)
        kdt_list = []
        for co_list in poly_list:
            kdt_list.append(stroke_to_kdtree(co_list))

        # Get stroke distance matrix
        dist_mat = []
        for i,co_list1 in enumerate(poly_list):
            for j,co_list2 in enumerate(poly_list):
                if i<j:
                    dist1 = distance_to_another_stroke(poly_list[i], poly_list[j], kdt_list[j])
                    dist2 = distance_to_another_stroke(poly_list[j], poly_list[i], kdt_list[i])
                    dist_mat.append(min(dist1,dist2))

        # Hierarchy clustering algorithm
        linkage_mat = linkage(dist_mat, method='single')
        if self.cluster_criterion == 'DIST':
            cluster_res = fcluster(linkage_mat, self.cluster_dist, criterion='distance')
        else:
            cluster_res = fcluster(linkage_mat, self.cluster_num, criterion='maxclust')
                 
        # Place strokes in clusters
        cluster_map = {}
        for i,stroke in enumerate(stroke_list):
            cluster_idx = cluster_res[i]
            if cluster_idx not in cluster_map:
                cluster_map[cluster_idx] = []
            cluster_map[cluster_idx].append(stroke)

        # For debugging: mark clusters with colors
        '''
        print(cluster_res)
        for cluster in cluster_map:
            color_mark = [np.random.rand(),np.random.rand(),np.random.rand()]
            for stroke in cluster_map[cluster]:
                for point in stroke.points:
                    point.vertex_color[0] = color_mark[0]
                    point.vertex_color[1] = color_mark[1]
                    point.vertex_color[2] = color_mark[2]
                    point.vertex_color[3] = 1
                    point.strength = 1
        '''

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
            for i,layer in enumerate(gp_obj.data.layers):
                if layer.active_frame and not layer.lock:
                    for stroke in layer.active_frame.strokes:
                        if stroke.select:
                            generated_strokes.append(stroke)
        
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

    cluster_dist: bpy.props.FloatProperty(
            name='Detection Radius',
            default=0.05, min=0,
            unit='LENGTH',
            description='Search strokes in the reference layer if it is close enough to the last drawn stroke'
    ) 

    def draw(self, context):
        layout = self.layout
        layout.label(text = "Input Options:")
        box1 = layout.box()
        box1.prop(self, "cluster_dist")
        box1.prop(self, "line_sampling_size")
        
        layout.label(text = "Post-Processing Options:")
        box2 = layout.box()
        box2.prop(self, "b_smoothness")

        layout.label(text = "Output Options:")
        box3 = layout.box()   
        box3.prop(self, "pressure_variance")
        box3.prop(self, "max_delta_pressure")

    def execute(self, context):
        import numpy as np

        # Get input and context
        gp_obj = context.object
        reference_layer = None
        drawing_layer = gp_obj.data.layers.active
        for layer in gp_obj.data.layers:
            if layer.info == context.scene.nijigp_draw_fit_reference_layer:
                reference_layer = layer
        if not reference_layer:
            self.report({"INFO"}, "Please select a reference layer.")
            return {'FINISHED'}
        if not reference_layer.active_frame:
            self.report({"INFO"}, "The reference layer has no stroke data.")
            return {'FINISHED'} 
        if reference_layer == drawing_layer:
            self.report({"INFO"}, "Please draw in a layer other than the reference layer.")
            return {'FINISHED'}  
                 
        # Get stroke information from the input
        if not drawing_layer.active_frame or len(drawing_layer.active_frame.strokes)<1:
            return {'FINISHED'} 
        src_stroke: bpy.types.GPencilStroke = drawing_layer.active_frame.strokes[-1]
        tmp, _ = stroke_to_poly([src_stroke], scale = False, correct_orientation = False)
        src_co_list = tmp[0]
        src_kdt = stroke_to_kdtree(src_co_list)
        
        # Check each stroke in the reference layer for similarity
        stroke_list = []
        for stroke in reference_layer.active_frame.strokes:
            if not overlapping_bounding_box(stroke, src_stroke):
                continue
            tmp, _ = stroke_to_poly([stroke], scale = False, correct_orientation = False)
            co_list = tmp[0]
            kdt = stroke_to_kdtree(co_list)
            line_dist1 = distance_to_another_stroke(co_list, src_co_list, src_kdt)
            line_dist2 = distance_to_another_stroke(src_co_list, co_list, kdt)
            if min(line_dist1, line_dist2) < self.cluster_dist:
                stroke_list.append(stroke)
        if len(stroke_list)<1:
            return {'FINISHED'}  
        
        # Execute the fitting function
        b_smoothness = self.b_smoothness
        new_co_list, pressure_accumulation, _, _, _ = fit_2d_strokes(stroke_list, 
                                                                        search_radius=self.line_sampling_size/LINE_WIDTH_FACTOR, 
                                                                        smoothness_factor=b_smoothness,
                                                                        pressure_delta=self.pressure_variance*0.01, 
                                                                        closed=src_stroke.use_cyclic,
                                                                        operator=self)
        # Correct the orientation of the generated stroke
        src_direction = Vector(src_co_list[-1]) - Vector(src_co_list[0])
        new_direction = Vector(new_co_list[-1]) - Vector(new_co_list[0])
        angle_diff = src_direction.angle(new_direction)
        if angle_diff > math.pi/2:
            new_co_list = np.flipud(new_co_list)

        # Remove the last drawn stroke and generate a new one
        new_stroke: bpy.types.GPencilStroke = drawing_layer.active_frame.strokes.new()
        new_stroke.material_index, new_stroke.line_width, new_stroke.use_cyclic = src_stroke.material_index, src_stroke.line_width, src_stroke.use_cyclic
        new_stroke.hardness, new_stroke.start_cap_mode, new_stroke.end_cap_mode, new_stroke.vertex_color_fill = src_stroke.hardness, src_stroke.start_cap_mode, src_stroke.end_cap_mode, src_stroke.vertex_color_fill
        new_stroke.points.add(new_co_list.shape[0])
        for i,point in enumerate(new_stroke.points):
            point.co = vec2_to_vec3(new_co_list[i], depth=0, scale_factor=1)
            attr_idx = int( float(i) / (len(new_stroke.points)-1) * (len(src_stroke.points)-1) )
            point.pressure = src_stroke.points[attr_idx].pressure
            point.strength = src_stroke.points[attr_idx].strength
            point.vertex_color = src_stroke.points[attr_idx].vertex_color
            point.pressure *= (1 + min(pressure_accumulation[i], self.max_delta_pressure*0.01) )

        drawing_layer.active_frame.strokes.remove(src_stroke)
        return {'FINISHED'}  

class PinchSelectedOperator(bpy.types.Operator):
    """Pinch and join strokes with their ends close to each other"""
    bl_idname = "gpencil.nijigp_pinch"
    bl_label = "Pinch Together"
    bl_category = 'View'
    bl_options = {'REGISTER', 'UNDO'}    

    threshold: bpy.props.FloatProperty(
            name='Threshold',
            default=0.05, min=0,
            unit='LENGTH',
            description='If either end of a stroke is close to another stroke end, the two strokes will be pinched together'
    ) 
    join_strokes: bpy.props.BoolProperty(
            name='Join Strokes',
            default=False,
            description='Join pinched strokes together if possible'
    )

    def draw(self, context):
        layout = self.layout

        layout.prop(self, "threshold")
        layout.prop(self, "join_strokes")

    def execute(self, context):

        # Get input strokes
        gp_obj = context.object
        stroke_list = []
        for i,layer in enumerate(gp_obj.data.layers):
            if layer.active_frame and not layer.lock:
                for stroke in layer.active_frame.strokes:
                    if stroke.select and len(stroke.points)>1 and not stroke.use_cyclic:
                        stroke_list.append(stroke)
        if len(stroke_list)<2:
            self.report({"INFO"}, "Please select at least two strokes.")
            return {'FINISHED'}

        # Calculate the relationship of strokes
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
                        if i!=j and point0 not in point_offset and (vec3_to_vec2(point0.co)-vec3_to_vec2(point.co)).length < self.threshold:
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
                            center = 0.5 * (vec3_to_vec2(point0.co)+vec3_to_vec2(point.co))
                            point_offset[point] = center - vec3_to_vec2(point.co)
                            point_offset[point0] = center - vec3_to_vec2(point0.co)
                            break
        # Padding zeros for ends without offsets
        for i,stroke in enumerate(stroke_list):
            for point in [stroke.points[0], stroke.points[-1]]:
                if point not in point_offset:
                    point_offset[point] = Vector((0,0))

        # Apply offsets
        for i,stroke in enumerate(stroke_list):
            for j,point in enumerate(stroke.points):
                w1, w2 = 1-j/(len(stroke.points)-1), j/(len(stroke.points)-1)
                new_co = vec3_to_vec2(point.co)
                new_co += point_offset[stroke.points[0]] * w1 + point_offset[stroke.points[-1]] * w2
                set_vec2(point, new_co)

        # Apply join
        if self.join_strokes:
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
    radius: bpy.props.FloatProperty(
            name='Radius',
            default=0, min=0, soft_max=5,
            description='The radius of both end points after this operation'
    )
    line_width: bpy.props.IntProperty(
            name='Uniform Thickness',
            description='Set the same width value to all selected strokes. 0 means not to change the width',
            default=0, min=0, soft_max=100, subtype='PIXEL'
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
        default='REPLACE'
    )    

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "start_length")
        layout.prop(self, "end_length")
        layout.prop(self, "radius")
        layout.prop(self, "line_width")
        layout.label(text='Affect:')
        row = layout.row()
        row.prop(self, "target_attributes")
        layout.prop(self, "operation")

    def execute(self, context):
        import numpy as np

        def process_one_stroke(stroke: bpy.types.GPencilStroke):
            if self.line_width > 0:
                stroke.line_width = self.line_width
            L = len(stroke.points)
            L1 = int(L*self.start_length)
            L2 = int(L*self.end_length)
            factor_arr = np.ones(L)
            factor_arr[:L1] *= np.linspace(self.radius, 1, L1)
            factor_arr[L-L2:L] *= np.linspace(1, self.radius, L2)
            for i,point in enumerate(stroke.points):
                if 'PRESSURE' in self.target_attributes:
                    point.pressure = point.pressure * factor_arr[i] if self.operation=='MULTIPLY' else factor_arr[i]
                if 'STRENGTH' in self.target_attributes:
                    point.strength = point.strength * factor_arr[i] if self.operation=='MULTIPLY' else factor_arr[i]

        for layer in context.object.data.layers:
            if not layer.lock:
                for frame in layer.frames:
                    for stroke in frame.strokes:
                        if stroke.select and not is_stroke_locked(stroke, context.object):
                            process_one_stroke(stroke)

        return {'FINISHED'}
