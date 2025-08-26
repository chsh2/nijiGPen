import bpy
import math
import numpy as np
from mathutils import *
from .common import *
from ..utils import *
from ..api_router import *

def stroke_to_kdtree(co_list):
    n = len(co_list)
    kdt = kdtree.KDTree(n)
    for i in range(n):
        kdt.insert(xy0(co_list[i]), i)
    kdt.balance()
    return kdt

def fit_2d_strokes(fitter, strokes, frame_number=-1, ignore_transparent=False, search_radius=0, pressure_delta=0, resample=None, t_mat=[]):
    '''
    Fit points from multiple strokes to a single curve, by executing the following operations:
        1. Delaunay triangulation
        2. Euclidean minimum spanning tree
        3. Longest path in the tree
        4. Offset based on points in the neighborhood
        5. Fit and smooth points to a B-Spline
    '''
    from ..solvers.graph import MstSolver

    poly_list, depth_list, _ = get_2d_co_from_strokes(strokes, t_mat, scale=False)
    
    total_point_count = 0
    for i,stroke in enumerate(strokes):
        if len(stroke.points)<2:
            continue
        total_point_count += len(stroke.points)

    if total_point_count<3:
        return 1
    
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
            if ignore_transparent and point.strength < 1e-3:
                continue
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
    if len(kdt_point_list)<3:
        return 1

    # Triangulation and spanning tree conversion
    tr_output = {}
    tr_output['vertices'], _, tr_output['triangles'], _,_,_ = geometry.delaunay_2d_cdt(tr_input['vertices'], [], [], 0, 1e-9)
    mst_builder = MstSolver()
    mst_builder.mst_from_triangles(tr_output)
    total_length, path_whole = mst_builder.get_longest_path()
    
    # The fitting method needs at least 4 points
    if len(path_whole)<4:
        return 1
    
    # Get the points in the tree as the input of postprocessing
    co_raw = np.zeros((len(path_whole), 2))
    for i,key in enumerate(path_whole):
        co = tr_output['vertices'][key]
        co_raw[i][0] = co[0]
        co_raw[i][1] = co[1]

    # Initialize lists of each point attributes
    accumulated_pressure = np.zeros(len(path_whole))
    inherited_pressure = np.zeros(len(path_whole))
    inherited_strength = np.zeros(len(path_whole))
    inherited_depth = np.zeros(len(path_whole))
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
                accumulated_pressure[i] += pressure_delta
            # Inherit each attribute
            inherited_pressure[i] += kdt_point_list[neighbor[1]].pressure
            inherited_strength[i] += kdt_point_list[neighbor[1]].strength
            inherited_depth[i] += kdt_depth_list[neighbor[1]]
            inherited_color[i] += np.array(kdt_point_list[neighbor[1]].vertex_color)
            inherited_uv_rotation[i] += kdt_point_list[neighbor[1]].uv_rotation

        sum_normal_offset /= len(neighbors)
        inherited_pressure[i] /= len(neighbors)
        inherited_strength[i] /= len(neighbors)
        inherited_depth[i] /= len(neighbors)
        inherited_color[i] /= len(neighbors)
        inherited_uv_rotation[i] /= len(neighbors)
        co_raw[i] += unit_normal_vector * sum_normal_offset

    fitter.set_coordinates(frame_number, co_raw, total_length)
    fitter.set_attribute_data(frame_number, 'pressure', inherited_pressure)
    fitter.set_attribute_data(frame_number, 'extra_pressure', accumulated_pressure)
    fitter.set_attribute_data(frame_number, 'strength', inherited_strength)
    fitter.set_attribute_data(frame_number, 'depth', inherited_depth)
    fitter.set_attribute_data(frame_number, 'r', inherited_color[:,0])
    fitter.set_attribute_data(frame_number, 'g', inherited_color[:,1])
    fitter.set_attribute_data(frame_number, 'b', inherited_color[:,2])
    fitter.set_attribute_data(frame_number, 'a', inherited_color[:,3])
    fitter.set_attribute_data(frame_number, 'uv_rotation', inherited_uv_rotation)
    if resample is not None:
        num_points = max(7, int(total_length / resample))
        fitter.input_u[frame_number] = np.linspace(0, 1, num_points, endpoint=True)
    return 0

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
    is_sequence: bpy.props.BoolProperty(
            name='Animation Sequence',
            default=False,
            description='Make the output stroke morph smoothly between keyframes. Otherwise, process each frame independently'
    )
    frame_step: bpy.props.IntProperty(
            name='Interpolation Step',
            default=0,min=0,
            description='Interpolate between keyframes when non-zero'
    )
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
    ignore_transparent: bpy.props.BoolProperty(
            name='Ignore Transparent',
            default=False,
            description='Do not fit points with zero opacity'
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
    b_smoothness: bpy.props.FloatProperty(
            name='Smooth (B-Spline)',
            description='Smoothness factor of the B-spline fitting algorithm',
            default=1, soft_max=100, min=0
    )
    resample: bpy.props.BoolProperty(
            name='Resample',
            default=False,
            description='Make generated points evenly distributed'
    )
    resample_length: bpy.props.FloatProperty(
            name='Length',
            description='',
            default=0.02, min=0.002
    )
    smooth_repeat: bpy.props.IntProperty(
            name='Smooth (Vertex)',
            description='Smooth level of vertex averaging',
            default=0, min=0, max=500
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
    
nijigp_generated_fit_strokes = []  # Save the states across multiple operator calls for the clustering fit
class FitSelectedOperator(CommonFittingConfig, bpy.types.Operator):
    """Fit select strokes or points to a new stroke"""
    bl_idname = "gpencil.nijigp_fit_selected"
    bl_label = "Single-Line Fit"
    bl_category = 'View'
    bl_options = {'REGISTER', 'UNDO'}    

    save_output_state: bpy.props.BoolProperty(default=False)

    def draw(self, context):
        layout = self.layout
        layout.label(text = "Input Options:")
        box1 = layout.box()
        box1.prop(self, "line_sampling_size")
        box1.prop(self, "closed")
        box1.prop(self, "ignore_transparent")
        if get_multiedit(context.active_object):
            box1.prop(self, "is_sequence")
            if self.is_sequence:
                row = box1.row()
                row.prop(self, "frame_step")
        
        layout.label(text = "Post-Processing Options:")
        box2 = layout.box()
        row = box2.row()
        row.prop(self, "resample")  
        if self.resample:
            row.prop(self, "resample_length")
        box2.prop(self, "b_smoothness")
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
            from ..solvers.graph import MstSolver
            from ..solvers.fit import CurveFitter
        except:
            self.report({"ERROR"}, "Please install Scipy in the Preferences panel.")
            return {'FINISHED'}
                
        # Get input strokes frame by frame
        gp_obj = context.object
        if not gp_obj.data.layers.active:
            self.report({"INFO"}, "Please select a layer.")
            return {'FINISHED'}
        stroke_frame_map = {}
        stroke_list = []
        frames_to_process = get_input_frames(gp_obj,
                                             multiframe = get_multiedit(gp_obj),
                                             return_map = True)
        for frame_number, layer_frame_map in frames_to_process.items():
            stroke_frame_map[frame_number] = []
            for _, item in layer_frame_map.items():
                stroke_frame_map[frame_number] += get_input_strokes(gp_obj, item[0])
            stroke_list += stroke_frame_map[frame_number]
            
        t_mat, inv_mat = get_transformation_mat(mode=bpy.context.scene.nijigp_working_plane,
                                                gp_obj=gp_obj, strokes=stroke_list)
        
        # Put data of each frame into fitting algorithm
        resample_length = self.resample_length if self.resample else None
        fitter = CurveFitter(self.closed)
        frame_with_output = set()
        for frame_number in frames_to_process:
            err = fit_2d_strokes(fitter, stroke_frame_map[frame_number], 
                                    frame_number = frame_number,
                                    search_radius=self.line_sampling_size/LINE_WIDTH_FACTOR, 
                                    ignore_transparent=self.ignore_transparent,
                                    pressure_delta=self.pressure_variance*0.01, 
                                    resample=resample_length,
                                    t_mat=t_mat)
            if not err:
                frame_with_output.add(frame_number)
        
        # Do spatial fit and (optional) temporal fit
        fitter.fit_spatial(self.b_smoothness*0.001, self.b_smoothness*10)
        has_temporal_fit = False
        if self.is_sequence and get_multiedit(gp_obj) and len(frames_to_process) > 1:
            fitter.fit_temporal()
            has_temporal_fit = True

        # For GPv2, remove input strokes before generating new ones
        if not self.keep_original and not is_gpv3():
            bpy.ops.gpencil.delete(type='STROKES')

        # Prepare for output
        op_deselect()
        output_layer = gp_obj.data.layers.active
        if len(self.output_layer) > 0:
            for i,layer in enumerate(gp_obj.data.layers):
                if layer.info == self.output_layer:
                    output_layer = layer
        output_material_idx = gp_obj.active_material_index
        if len(self.output_material) > 0:
            for i,material_slot in enumerate(gp_obj.material_slots):
                if material_slot.material and material_slot.material.name == self.output_material:
                    output_material_idx = i
        output_frames = {}
        for frame in output_layer.frames:
            output_frames[int(frame.frame_number)] = frame
            
        # Get output frame numbers
        if not get_multiedit(gp_obj):
            target_frames = [context.scene.frame_current] if context.scene.frame_current in frame_with_output else []
        else:
            target_frames = frame_with_output.copy()
            if has_temporal_fit and self.frame_step > 0:
                for i in range(min(frame_with_output), max(frame_with_output), self.frame_step):
                    target_frames.add(i)
            
        # Use fitting results of each frame to generate new strokes
        stroke_set = set(stroke_list)
        for frame_number in target_frames:
            co_fit, attr_fit = fitter.eval_temporal(frame_number) if has_temporal_fit else fitter.eval_spatial(frame_number)
            if frame_number not in output_frames:
                output_frame = output_layer.frames.new(frame_number)
            else:
                output_frame = get_layer_frame_by_number(output_layer, frame_number)
            output_frame.select = True
            
            # Gather stroke attributes from input strokes
            new_stroke = output_frame.nijigp_strokes.new()
            new_stroke.material_index = output_material_idx
            new_stroke.line_width = self.line_width
            copy_stroke_attributes(new_stroke, stroke_list,
                                copy_color = 'COLOR' in self.inherited_attributes,
                                copy_uv = 'UV' in self.inherited_attributes)
            new_stroke.points.add(co_fit.shape[0])
            for i,point in enumerate(new_stroke.points):
                point.co = restore_3d_co(co_fit[i], attr_fit['depth'][i], inv_mat)
                point.strength = attr_fit['strength'][i] if 'STRENGTH' in self.inherited_attributes else 1
                point.uv_rotation = attr_fit['uv_rotation'][i] if 'UV' in self.inherited_attributes else 0
                point.vertex_color = (attr_fit['r'][i], attr_fit['g'][i], attr_fit['b'][i], attr_fit['a'][i]) if 'COLOR' in self.inherited_attributes else (0,0,0,0)
                # Consider differences between GPv2 and GPv3 when processing pressure values
                if 'PRESSURE' in self.inherited_attributes:
                    new_pressure = attr_fit['pressure'][i]
                    new_pressure *= (1 + min(attr_fit['extra_pressure'][i], self.max_delta_pressure*0.01) )
                    point.pressure = new_pressure
                else:
                    new_pressure = 1 + min(attr_fit['extra_pressure'][i], self.max_delta_pressure*0.01)
                    set_point_radius(point, new_pressure, self.line_width)
            new_stroke.use_cyclic = self.closed
            new_stroke.select = True
            if self.save_output_state:
                nijigp_generated_fit_strokes.append(new_stroke)

            # Post-processing
            if self.smooth_repeat > 0:
                smooth_stroke_attributes(new_stroke, self.smooth_repeat, attr_map={'co':3, 'pressure':1, 'strength':1})    
            # For GPv3, remove input strokes at the end
            if not self.keep_original and is_gpv3():
                to_remove = [stroke for stroke in output_frame.nijigp_strokes if stroke in stroke_set]
                for stroke in to_remove:
                    output_frame.nijigp_strokes.remove(stroke)
                
        refresh_strokes(bpy.context.active_object)
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
        frames_to_process = get_input_frames(gp_obj, get_multiedit(gp_obj))
        
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
                for stroke in frame.nijigp_strokes:
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
    cluster_by_color: bpy.props.BoolProperty(
            name='Separate by Vertex Color',
            description='Make sure that strokes with different vertex colors are assigned to different clusters',
            default=False
    )

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self, width=300)
    
    def draw(self, context):
        layout = self.layout

        layout.label(text = "Clustering Options:")
        box0 = layout.box()
        box0.prop(self, "cluster_criterion")
        if self.cluster_criterion == 'DIST':
            box0.prop(self, "cluster_dist")
        elif self.cluster_criterion == 'NUM':
            box0.prop(self, "cluster_num")
        elif self.cluster_criterion == 'RATIO':
            box0.prop(self, "cluster_ratio")
        box0.prop(self, "cluster_by_color")

        layout.label(text = "Input Options:")
        box1 = layout.box()
        box1.prop(self, "line_sampling_size")
        box1.prop(self, "closed")
        box1.prop(self, "ignore_transparent")
        if get_multiedit(context.active_object):
            box1.prop(self, "is_sequence")
            if self.is_sequence:
                row = box1.row()
                row.prop(self, "frame_step")

        layout.label(text = "Post-Processing Options:")
        box2 = layout.box()
        row = box2.row()
        row.prop(self, "resample")  
        if self.resample:
            row.prop(self, "resample_length")
        box2.prop(self, "b_smoothness")
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
        
        # Get input strokes frame by frame
        gp_obj = context.object
        stroke_list = []
        stroke_frame_map = {}
        frames_to_process = get_input_frames(gp_obj,
                                             multiframe = get_multiedit(gp_obj),
                                             return_map = True)
        skipped_frames = []
        for frame_number, layer_frame_map in frames_to_process.items():
            stroke_frame_map[frame_number] = []
            for _, item in layer_frame_map.items():
                stroke_frame_map[frame_number] += get_input_strokes(gp_obj, item[0])
                
            if len(stroke_frame_map[frame_number])<2:
                skipped_frames.append(frame_number)
            else:
                stroke_list += stroke_frame_map[frame_number]
                
        for frame_number in skipped_frames:
            frames_to_process.pop(frame_number)
        
        t_mat, inv_mat = get_transformation_mat(mode=context.scene.nijigp_working_plane,
                                                gp_obj=gp_obj, strokes=stroke_list, operator=self)

        # Cluster strokes in each frame
        cluster_map = {}
        num_cluster = 65535
        for frame_number in frames_to_process:
            poly_list, _, _ = get_2d_co_from_strokes(stroke_frame_map[frame_number], t_mat, scale=False)
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
            # Add a large value to the distance for strokes with different vertex colors
            if self.cluster_by_color:
                mean_colors = []
                for stroke in stroke_frame_map[frame_number]:
                    v_colors = [point.vertex_color[:3] for point in stroke.points if point.vertex_color[3]>1e-3]
                    mean_colors.append(0 if len(v_colors)<1 else np.mean(v_colors, axis=0))
                dist_index = 0
                for i,mean_color1 in enumerate(mean_colors):
                    for j,mean_color2 in enumerate(mean_colors):
                        if i<j:
                            dist_mat[dist_index] += (np.linalg.norm(mean_color1 - mean_color2) > 1e-3)
                            dist_index += 1

            # Hierarchy clustering algorithm: assigning cluster ID starting from 1 to each stroke
            linkage_mat = linkage(dist_mat, method='single')
            if self.cluster_criterion == 'DIST':
                cluster_res = fcluster(linkage_mat, self.cluster_dist, criterion='distance')
            elif self.cluster_criterion == 'RATIO':
                cluster_res = fcluster(linkage_mat, self.cluster_ratio/100.0, criterion='distance')
            else:
                # Skip clustering if there are too few input strokes
                if self.cluster_num >= len(stroke_frame_map[frame_number]):
                    cluster_res = [i+1 for i in range(len(stroke_frame_map[frame_number]))]
                else:
                    cluster_res = fcluster(linkage_mat, self.cluster_num, criterion='maxclust')
            
            # Reorder the cluster indices according to drawing sequence; also make ID start from 0
            cluster_drawing_seq = {}
            for i,stroke in enumerate(stroke_frame_map[frame_number]):
                if (cluster_res[i]-1) not in cluster_drawing_seq:
                    cluster_drawing_seq[cluster_res[i]-1] = i
            cluster_sorted = sorted(list(cluster_drawing_seq), key=lambda _:cluster_drawing_seq[_])
            cluster_sorted = [cluster_sorted.index(i) for i in range(len(cluster_sorted))]

            # Place strokes in clusters
            for i,stroke in enumerate(stroke_frame_map[frame_number]):
                cluster_idx = cluster_sorted[cluster_res[i]-1]
                if cluster_idx not in cluster_map:
                    cluster_map[cluster_idx] = []
                cluster_map[cluster_idx].append(stroke)

        # Process each cluster one by one
        global nijigp_generated_fit_strokes
        nijigp_generated_fit_strokes = []
        for cluster in range(len(cluster_map)):
            # Set frame selection
            for layer_idx,layer in enumerate(gp_obj.data.layers):
                for frame in layer.frames:
                    frame.select = (frame.frame_number in frames_to_process) and (layer_idx in frames_to_process[frame.frame_number])
            # Set stroke selection
            op_deselect()
            for stroke in cluster_map[cluster]:
                stroke.select = True
            bpy.ops.gpencil.nijigp_fit_selected(line_sampling_size = self.line_sampling_size,
                                            closed = self.closed,
                                            is_sequence = self.is_sequence,
                                            frame_step = self.frame_step,
                                            ignore_transparent = self.ignore_transparent,
                                            pressure_variance = self.pressure_variance,
                                            max_delta_pressure = self.max_delta_pressure,
                                            line_width = self.line_width,
                                            resample = self.resample,
                                            b_smoothness = self.b_smoothness,
                                            resample_length = self.resample_length,
                                            smooth_repeat = self.smooth_repeat,
                                            inherited_attributes = self.inherited_attributes,
                                            output_layer = self.output_layer,
                                            output_material = self.output_material,
                                            keep_original = self.keep_original,
                                            save_output_state = True)
        # Select all output strokes
        op_deselect()
        for stroke in nijigp_generated_fit_strokes:
            stroke.select = True
        nijigp_generated_fit_strokes = []
        return {'FINISHED'}
    
class FitLastOperator(CommonFittingConfig, bpy.types.Operator):
    """Fit the recently drawn stroke(s) to a smooth one"""
    bl_idname = "gpencil.nijigp_fit_last"
    bl_label = "Fit Last Stroke"
    bl_category = 'View'
    bl_options = {'REGISTER', 'UNDO'}  

    target: bpy.props.EnumProperty(
            name='Target',
            items=[ ('SINGLE', 'Single Stroke', 'Smooth the last drawn stroke only'),
                    ('COLOR', 'Same Vertex Color', 'Fit all recent strokes with the same vertex color to a single smooth one'),
                    ('REF', 'Reference Layer', 'Fit strokes from a separate draft layer that are close enough to the last drawn stroke')],
            default='SINGLE',
            description='Different modes that determine which strokes will be fitted'
    )
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
    trim_ends: bpy.props.BoolProperty(
            name='Trim Ends',
            default=True,
            description='Move ends of the drawn stroke as little as possible'
    )
    reference_layer: bpy.props.StringProperty(
        name='Reference Layer',
        description='The layer with draft strokes which are used to guide the fitting of the newly drawn stroke',
        default='',
        search=multilayer_search_func
    ) 

    def draw(self, context):
        layout = self.layout
        layout.label(text = "Input Options:")
        box1 = layout.box()
        box1.prop(self, "target")
        if self.target == 'REF':
            row = box1.row()
            row.label(text = "Reference Layer:")
            row.prop(self, "reference_layer", icon='OUTLINER_DATA_GP_LAYER', text='')
            box1.prop(self, "cluster_criterion")
            if self.cluster_criterion == 'DIST':
                box1.prop(self, "cluster_dist")
            else:
                box1.prop(self, "cluster_ratio")
        box1.prop(self, "line_sampling_size")
        box1.prop(self, "ignore_transparent")
        
        layout.label(text = "Post-Processing Options:")
        box2 = layout.box()
        row = box2.row()
        row.prop(self, "resample")  
        if self.resample:
            row.prop(self, "resample_length")
        box2.prop(self, "b_smoothness")
        box2.prop(self, "smooth_repeat")
        if self.target == 'REF':
            row = box2.row()
            row.prop(self, "trim_ends")

        layout.label(text = "Output Options:")
        box3 = layout.box()   
        box3.prop(self, "pressure_variance")
        box3.prop(self, "max_delta_pressure")

    def execute(self, context):
        try:
            from ..solvers.graph import MstSolver
            from ..solvers.fit import CurveFitter
        except:
            self.report({"ERROR"}, "Please install Scipy in the Preferences panel.")
            return {'FINISHED'}
        
        # Get input and context
        gp_obj = context.object
        if not gp_obj.data.layers.active:
            self.report({"INFO"}, "Please select a layer.")
            return {'FINISHED'}
        if len(self.reference_layer) > 0:
            reference_layers, _ = multilayer_search_decode(self.reference_layer)
            if gp_obj.data.layers.active in reference_layers:
                self.report({"WARNING"}, "Reference layer cannot be the active layer.")
                return {'FINISHED'} 
        elif self.target == 'REF':
            self.report({"INFO"}, "Please select a layer containing draft strokes.")
            return {'FINISHED'}
        drawing_layer = gp_obj.data.layers.active 
                 
        # Get stroke information from the input
        if not drawing_layer.active_frame or len(drawing_layer.active_frame.nijigp_strokes)<1:
            return {'FINISHED'} 
        stroke_index = 0 if context.scene.tool_settings.use_gpencil_draw_onback else len(drawing_layer.active_frame.nijigp_strokes)-1
        
        # For vertex color mode, find the first stroke with non-zero vertex color
        while stroke_index>=0 and stroke_index < len(drawing_layer.active_frame.nijigp_strokes):
            src_stroke = drawing_layer.active_frame.nijigp_strokes[stroke_index]
            if self.target != 'COLOR' or max([point.vertex_color[3] for point in src_stroke.points]) > 1e-3:
                break
            stroke_index += 1 if context.scene.tool_settings.use_gpencil_draw_onback else -1
        if stroke_index<0 or stroke_index >= len(drawing_layer.active_frame.nijigp_strokes):
            return {'FINISHED'}
        
        t_mat, inv_mat = get_transformation_mat(mode=context.scene.nijigp_working_plane,
                                                gp_obj=gp_obj, operator=self)
        tmp, tmp2, _ = get_2d_co_from_strokes([src_stroke], t_mat, scale=False)
        src_co_list = tmp[0]
        depth_lookup_tree = DepthLookupTree(tmp, tmp2)
        src_kdt = stroke_to_kdtree(src_co_list)
        src_stroke_length = get_stroke_length(co_list=src_co_list)
        
        # Check each stroke in the reference layer for similarity
        stroke_list = []
        if self.target == 'REF':
            threshold = (self.cluster_dist if self.cluster_criterion == 'DIST' else
                        self.cluster_ratio / 100.0 * src_stroke_length)
            for reference_layer in reference_layers:
                for stroke in reference_layer.active_frame.nijigp_strokes:
                    if not stroke_bound_box_overlapping(stroke, src_stroke, t_mat):
                        continue
                    tmp, _, _ = get_2d_co_from_strokes([stroke], t_mat, scale=False)
                    co_list = tmp[0]
                    kdt = stroke_to_kdtree(co_list)
                    line_dist1 = distance_to_another_stroke(co_list, src_co_list, src_kdt)
                    line_dist2 = distance_to_another_stroke(src_co_list, co_list, kdt)
                    if min(line_dist1, line_dist2) < threshold:
                        stroke_list.append(stroke)
        # Check all recent strokes until the vertex color becomes different
        elif self.target == 'COLOR':
            stroke_list = []
            src_color = np.mean([point.vertex_color for point in src_stroke.points if point.vertex_color[3]>1e-3], axis=0)
            while stroke_index>=0 and stroke_index < len(drawing_layer.active_frame.nijigp_strokes):
                stroke = drawing_layer.active_frame.nijigp_strokes[stroke_index]
                v_colors = [point.vertex_color for point in stroke.points if point.vertex_color[3]>1e-3]
                if len(v_colors) < 1 or np.linalg.norm(np.mean(v_colors, axis=0) - src_color) > 1e-3:
                    break
                stroke_list.append(stroke)
                stroke_index += 1 if context.scene.tool_settings.use_gpencil_draw_onback else -1
        elif self.target == 'SINGLE':
            stroke_list = [src_stroke]

        if len(stroke_list)<1:
            return {'FINISHED'}  
        
        # Execute the fitting function
        fitter = CurveFitter(src_stroke.use_cyclic)
        resample_length = self.resample_length if self.resample else None
        err = fit_2d_strokes(fitter, stroke_list, 
                                search_radius=self.line_sampling_size/LINE_WIDTH_FACTOR, 
                                ignore_transparent=self.ignore_transparent,
                                pressure_delta=self.pressure_variance*0.01, 
                                resample=resample_length,
                                t_mat = t_mat)
        if err != 0:
            return {'FINISHED'}
        fitter.fit_spatial(self.b_smoothness*0.001, self.b_smoothness*10)
        co_fit, attr_fit = fitter.eval_spatial(-1)
        
        # Orientation correction and trimming
        src_direction = Vector(src_co_list[-1]) - Vector(src_co_list[0])
        new_direction = Vector(co_fit[-1]) - Vector(co_fit[0])
        angle_diff = 0 if new_direction.length < 1e-6 else src_direction.angle(new_direction)
        if angle_diff > math.pi/2:
            co_fit = np.flipud(co_fit)
            for key in attr_fit:
                attr_fit[key] = np.flipud(attr_fit[key])
            
        if self.target == 'REF' and self.trim_ends:
            start_idx, end_idx = 0, len(co_fit)-1
            for i,co in enumerate(co_fit):
                vec = Vector(co)-Vector(src_co_list[0])
                if vec.length < threshold:
                    start_idx = i
                    break
            for i,co in enumerate(np.flipud(co_fit)):
                vec = Vector(co)-Vector(src_co_list[-1])
                if vec.length < threshold:
                    end_idx = len(co_fit)-i
                    break
            if start_idx < end_idx:
                co_fit = co_fit[start_idx:end_idx]
            
        # Remove the last drawn stroke and generate a new one
        new_stroke = drawing_layer.active_frame.nijigp_strokes.new()
        copy_stroke_attributes(new_stroke, [src_stroke],
                            copy_hardness=True, copy_linewidth=True,
                            copy_cap=True, copy_cyclic=True,
                            copy_uv=True, copy_material=True, copy_color=True)
        new_stroke.points.add(co_fit.shape[0])
        for i,point in enumerate(new_stroke.points):
            point.co = restore_3d_co(co_fit[i], depth_lookup_tree.get_depth(co_fit[i]), inv_mat)
            attr_idx = int( float(i) / (len(new_stroke.points)-1) * (len(src_stroke.points)-1) )
            if self.target == 'COLOR':
                point.pressure = attr_fit['pressure'][i]
                point.strength = attr_fit['strength'][i]
                point.vertex_color = (0,0,0,0)
                point.uv_rotation = attr_fit['uv_rotation'][i]
            else:
                point.pressure = src_stroke.points[attr_idx].pressure
                point.strength = src_stroke.points[attr_idx].strength
                point.vertex_color = src_stroke.points[attr_idx].vertex_color
                point.uv_rotation = src_stroke.points[attr_idx].uv_rotation
            point.pressure *= (1 + min(attr_fit['extra_pressure'][i], self.max_delta_pressure*0.01) )
        if self.target == 'COLOR':
            for stroke in stroke_list:
                drawing_layer.active_frame.nijigp_strokes.remove(stroke)
        else:
            drawing_layer.active_frame.nijigp_strokes.remove(src_stroke)
        
        # Post-processing
        if self.smooth_repeat > 0:
            smooth_stroke_attributes(new_stroke, self.smooth_repeat, attr_map={'co':3, 'pressure':1, 'strength':1})
        refresh_strokes(bpy.context.active_object)
        
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

        # Get input strokes frame by frame
        gp_obj = context.object
        stroke_list = []
        stroke_frame_map = {}
        frames_to_process = get_input_frames(gp_obj,
                                             multiframe = get_multiedit(gp_obj),
                                             return_map = True)
        skipped_frames = []
        for frame_number, layer_frame_map in frames_to_process.items():
            stroke_frame_map[frame_number] = []
            for _, item in layer_frame_map.items():
                stroke_frame_map[frame_number] += get_input_strokes(gp_obj, item[0])
            stroke_frame_map[frame_number] = [stroke for stroke in stroke_frame_map[frame_number]
                                                if len(stroke.points)>1 and not stroke.use_cyclic]
            if len(stroke_frame_map[frame_number])<2:
                self.report({"INFO"}, "Please select at least two strokes in each frame, otherwise the frame will be skipped.")
                skipped_frames.append(frame_number)
            else:
                stroke_list += stroke_frame_map[frame_number]
                
        for frame_number in skipped_frames:
            frames_to_process.pop(frame_number)    
                        
        t_mat, inv_mat = get_transformation_mat(mode=context.scene.nijigp_working_plane,
                                                gp_obj=gp_obj, strokes=stroke_list, operator=self)

        # Process each frame
        for frame_number in frames_to_process:
            poly_list, depth_list, _ = get_2d_co_from_strokes(stroke_frame_map[frame_number], t_mat, scale=False)
            depth_lookup_tree = DepthLookupTree(poly_list, depth_list)
            trans2d = lambda co: (t_mat @ co).xy
            
            # Detect and eliminate the gap between two end points
            point_processed = set()     # key: (stroke, point_index)
            if self.end_to_end:
                num_chains = 0
                stroke_chain_idx = {}   # key: stroke
                point_offset = {}       # key: (stroke, point_index)
                for i,stroke in enumerate(stroke_frame_map[frame_number]):
                    for p_idx in (0, -1):
                        if (stroke, p_idx) in point_offset:
                            continue
                        point = stroke.points[p_idx]
                        
                        for j,stroke0 in enumerate(stroke_frame_map[frame_number]):
                            for p_idx0 in (0, -1):
                                point0 = stroke0.points[p_idx0]
                                if i<j and (stroke0, p_idx0) not in point_offset and (trans2d(point0.co)-trans2d(point.co)).length < self.threshold:
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
                                    point_offset[(stroke, p_idx)] = center - trans2d(point.co)
                                    point_offset[(stroke0, p_idx0)] = center - trans2d(point0.co)
                                    break                                 
                # Padding zeros for ends without offsets
                for i,stroke in enumerate(stroke_frame_map[frame_number]):
                    for p_idx in (0, -1):
                        if (stroke, p_idx) not in point_offset:
                            point_offset[(stroke, p_idx)] = Vector((0,0))
                        else:
                            point_processed.add((stroke, p_idx))
                # Apply offsets
                for i,stroke in enumerate(stroke_frame_map[frame_number]):
                    for j,point in enumerate(stroke.points):
                        w1, w2 = 1-j/(len(stroke.points)-1), j/(len(stroke.points)-1)
                        J = len(stroke.points)-1
                        w1 = 1 - j/J/self.transition_length
                        w2 = 1 - (J-j)/J/self.transition_length
                        new_co = trans2d(point.co)
                        new_co += self.mix_factor * (point_offset[(stroke, 0)] * smoothstep(w1) 
                                                    + point_offset[(stroke, -1)] * smoothstep(w2))
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
                for i,stroke in enumerate(stroke_frame_map[frame_number]):
                    for end_type, p_idx in {'start':0, 'end':-1}.items():
                        if (stroke, p_idx) in point_processed:
                            continue
                        point = stroke.points[p_idx]
                        for j,stroke0 in enumerate(stroke_frame_map[frame_number]):
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
                                stroke0.points[contact_idx].pressure *= (1+contact_pressure)
                                (stroke.points[0] if end_type=='start' else stroke.points[-1]).pressure *= (1+contact_pressure)
                                for delta_idx in range(1, self.contact_length):
                                    for idx in (contact_idx+delta_idx, contact_idx-delta_idx):
                                        if idx>=0 and idx<len(stroke0.points):
                                            stroke0.points[idx].pressure *= (1+contact_pressure*(1-delta_idx/self.contact_length))
                                    idx = delta_idx if end_type=='start' else len(stroke.points)-1-delta_idx
                                    if idx>=0 and idx<len(stroke.points):
                                        stroke.points[idx].pressure *= (1+contact_pressure*(1-delta_idx/self.contact_length))
            # Apply join
            if self.end_to_end and self.join_strokes:
                for i in range(num_chains):
                    op_deselect()
                    for stroke in stroke_chain_idx:
                        if stroke_chain_idx[stroke]==i:
                            stroke.select = True
                    op_join_strokes()

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

        def process_one_stroke(stroke, co_list):
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
                    additional_factor_arr[1:-1] = np.convolve(additional_factor_arr, kernel, mode='same')[1:-1]
                factor_arr *= additional_factor_arr

            # Apply final modification results
            for i,point in enumerate(stroke.points):
                if 'PRESSURE' in self.target_attributes:
                    if self.operation=='MULTIPLY':
                        point.pressure = point.pressure * factor_arr[i]
                    else:
                        set_point_radius(point, factor_arr[i], self.line_width)
                if 'STRENGTH' in self.target_attributes:
                    point.strength = point.strength * factor_arr[i] if self.operation=='MULTIPLY' else factor_arr[i]

        gp_obj = context.object
        frames_to_process = get_input_frames(gp_obj, get_multiedit(gp_obj))
        stroke_list = []
        for frame in frames_to_process:
            stroke_list += get_input_strokes(gp_obj, frame)
        t_mat, inv_mat = get_transformation_mat(mode=context.scene.nijigp_working_plane,
                                                    gp_obj=gp_obj, strokes=stroke_list, operator=self)
        poly_list, _, _ = get_2d_co_from_strokes(stroke_list, t_mat, scale=False)
        for i, stroke in enumerate(stroke_list):
            process_one_stroke(stroke, poly_list[i])

        return {'FINISHED'}
