import bpy
import math
from .utils import *

def fit_2d_strokes(strokes, search_radius, smoothness_factor = 1, pressure_delta = 0, operator = None):
    '''
    Fit points from multiple strokes to a single curve, by executing the following operations:
        1. Delaunay triangulation
        2. Euclidean minimum spanning tree
        3. Longest path in the tree
        4. Offset based on points in the neighborhood
        5. Post-processing: vertex smooth or B-spline fitting
    '''
    try:
        import triangle as tr
        from scipy.sparse.csgraph import minimum_spanning_tree
        from scipy.interpolate import splprep, splev
    except:
        if operator:
            operator.report({"ERROR"}, "Please install dependencies in the Preferences panel.")
        return None, None
    import numpy as np

    # Create a KDTree for point attribute lookup
    poly_list, scale_factor = stroke_to_poly(strokes)
    total_point_count = 0
    for i,stroke in enumerate(strokes):
        if len(stroke.points)<2:
            continue
        for j,point in enumerate(stroke.points):
            if point.select:
                total_point_count += 1
    if total_point_count<3:
        return None, None
    kdt = kdtree.KDTree(total_point_count)
    kdt_tangent_list = []
    kdt_stroke_list = []
    kdt_idx = 0
    co_set = set()

    # Record point attributes and prepare the input of Step 1
    # Ignore single-point strokes and merge points with the same coordinates
    tr_input = dict(vertices = [])
    for i,stroke in enumerate(strokes):
        if len(stroke.points)<2:
            continue
        for j,point in enumerate(stroke.points):
            if point.select and (poly_list[i][j][0],poly_list[i][j][1]) not in co_set:
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

    # Get the points in the tree as the input of postprocessing
    co_raw = np.zeros((len(path_whole), 2))
    pressure_raw = np.ones(len(path_whole))
    for i,key in enumerate(path_whole):
        co = tr_output['vertices'][key]
        co_raw[i][0] = co[0]
        co_raw[i][1] = co[1]

    # Apply offsets in the normal direction if there are points in the neighborhood
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
                pressure_raw[i] += pressure_delta
        sum_normal_offset /= len(neighbors)
        co_raw[i] += unit_normal_vector * sum_normal_offset

    # Postprocessing: B-spline fitting
    tck, u = splprep([co_raw[:,0], co_raw[:,1]], s=total_length**2 * smoothness_factor * 0.001)
    co_fit = np.array(splev(u, tck)).transpose()
    tck2, u2 = splprep([np.linspace(0,1,len(path_whole)), pressure_raw])
    pressure_fit = np.array(splev(u2, tck2))[1]

    return co_fit, pressure_fit

class FitSelectedOperator(bpy.types.Operator):
    """Fit select strokes or points to a new stroke"""
    bl_idname = "gpencil.nijigp_fit_selected"
    bl_label = "Fit Selected Strokes"
    bl_category = 'View'
    bl_options = {'REGISTER', 'UNDO'}    

    detection_radius: bpy.props.IntProperty(
            name='Detection Radius',
            description='A point will try to merge with points from other strokes if the distance is smaller than this value',
            default=50, min=1, soft_max=100, subtype='PIXEL'
    )
    pressure_variance: bpy.props.FloatProperty(
            name='Pressure Variance',
            description='Increase the point radius at positions where lines are repeatedly drawn',
            default=5, soft_max=20, min=0, subtype='PERCENTAGE'
    )
    max_pressure: bpy.props.FloatProperty(
            name='Maximum Pressure',
            description='Upper bound of the point radius',
            default=150, soft_max=200, min=100, subtype='PERCENTAGE'
    )
    line_width: bpy.props.IntProperty(
            name='Base Line Width',
            description='The minimum width of the newly generated stroke',
            default=10, min=1, soft_max=100, subtype='PIXEL'
    )   
    smoothness: bpy.props.FloatProperty(
            name='Smoothness',
            description='Smoothness factor when applying the B-spline fitting algorithm',
            default=2, soft_max=100, min=0
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

    def draw(self, context):
        layout = self.layout
        layout.label(text = "Fitting Options:")
        box1 = layout.box()
        box1.prop(self, "detection_radius")
        box1.prop(self, "smoothness")
        layout.label(text = "Output Options:")
        box2 = layout.box()        
        box2.prop(self, "line_width")
        box2.prop(self, "pressure_variance")
        box2.prop(self, "max_pressure")
        box2.prop(self, "output_layer", text='Layer', icon='OUTLINER_DATA_GP_LAYER')
        box2.prop(self, "output_material", text='Material', icon='MATERIAL')
        box2.prop(self, "keep_original")

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
        co_list, pressure_list = fit_2d_strokes(stroke_list, 
                                                search_radius=self.detection_radius/LINE_WIDTH_FACTOR, 
                                                smoothness_factor=self.smoothness, 
                                                pressure_delta=self.pressure_variance * 0.01,
                                                operator=self)
        if co_list is None:
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
            point.pressure = min(pressure_list[i], self.max_pressure*0.01)
        bpy.ops.gpencil.select_all(action='DESELECT')
        new_stroke.select = True
        bpy.ops.transform.translate()

        return {'FINISHED'}