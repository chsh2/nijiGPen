import bpy
import math
import bmesh
from mathutils import *
from .common import *
from ..utils import *
from ..resources import *
from ..api_router import *

MAX_DEPTH = 4096

def merge_poly_points(poly, min_dist):
    """Remove points of a 2D polygon that are too close to each other"""
    if len(poly) < 1:
        return []
    new_poly = [poly[0]]
    for i in range(1,len(poly)):
        dist = Vector(poly[i]) - Vector(new_poly[-1])
        if dist.length > min_dist:
            new_poly.append(poly[i])
    dist = Vector(poly[0]) - Vector(new_poly[-1])
    if dist.length <= min_dist:
        new_poly = new_poly[1:]
    return new_poly

def apply_mirror_in_depth(obj, inv_mat, loc = (0,0,0)):
    """Apply a Mirror modifier in the Object mode with given center and axis"""
    obj.modifiers.new(name="nijigp_Mirror", type='MIRROR')
    obj.modifiers["nijigp_Mirror"].use_axis = (False, False, True)
    
    # Create an empty object as the mirror reference
    empty_object = bpy.data.objects.new("Empty", None)
    bpy.context.collection.objects.link(empty_object)
    empty_object.location = loc
    empty_object.parent = obj
    empty_object.rotation_mode = 'QUATERNION'
    empty_object.rotation_quaternion = inv_mat.to_quaternion()
    obj.modifiers["nijigp_Mirror"].mirror_object = empty_object
    
    # Clean-up
    bpy.ops.object.modifier_apply("EXEC_DEFAULT", modifier = "nijigp_Mirror")
    bpy.data.objects.remove(empty_object)

class CommonMeshConfig:
    postprocess_double_sided: bpy.props.BoolProperty(
            name='Double-Sided',
            default=True,
            description='Make the mesh symmetric to the working plane'
    )
    stacked: bpy.props.BoolProperty(
            name='Stacked',
            default=True,
            description='Resolve the collision with previously generated meshes and move the new mesh accordingly'
    )
    vertical_gap: bpy.props.FloatProperty(
            name='Min Gap',
            default=0.01, min=0,
            unit='LENGTH',
            description='Additional vertical space between generated meshes'
    )   
    overlap_tolerance: bpy.props.IntProperty(
            name='Overlap Tolerance',
            description='The maximum percentage of vertices allowed to overlap with other meshes',
            default=15, max=100, min=0, subtype='PERCENTAGE'
    ) 
    ignore_mode: bpy.props.EnumProperty(
            name='Ignore',
            items=[('NONE', 'None', ''),
                    ('LINE', 'All Lines', ''),
                    ('OPEN', 'All Open Lines', '')],
            default='NONE',
            description='Skip strokes without fill'
    )
    reuse_material: bpy.props.BoolProperty(
            name='Reuse Materials',
            default=True,
            description='Do not create a new material if it exists'
    )
    stop_motion_animation: bpy.props.BoolProperty(
            name='Stop-Motion Animation',
            default=False,
            description='Hide the mesh at the next keyframe'
    )
    keep_original: bpy.props.BoolProperty(
            name='Keep Original',
            default=True,
            description='Do not delete the original stroke'
    )

class MeshManagement(bpy.types.Operator):
    """Manage mesh objects generated from the active GPencil object"""
    bl_idname = "gpencil.nijigp_mesh_management"
    bl_label = "Generated Mesh Management"
    bl_category = 'View'
    bl_options = {'REGISTER', 'UNDO'}

    action: bpy.props.EnumProperty(
            name='Action',
            items=[('HIDE', 'Hide', ''),
                    ('SHOW', 'Show', ''),
                    ('CLEAR', 'Clear', '')],
            default='SHOW',
            description='Actions to perform on children meshes'
    )  

    def draw(self, context):
        layout = self.layout
        row = layout.row()
        row.prop(self, "action", text = "Action")

    def execute(self, context):
        current_gp_obj = context.object
        for obj in current_gp_obj.children:
            if 'nijigp_mesh' in obj:
                obj.hide_set(self.action == 'HIDE')
                obj.hide_render = (self.action == 'HIDE')
                if self.action == 'CLEAR':
                    bpy.data.objects.remove(obj, do_unlink=True)

        return {'FINISHED'}

class MeshGenerationByNormal(CommonMeshConfig, bpy.types.Operator):
    """Generate a planar mesh with an interpolated normal map calculated from the selected strokes"""
    bl_idname = "gpencil.nijigp_mesh_generation_normal"
    bl_label = "Convert to Meshes by Normal Interpolation"
    bl_category = 'View'
    bl_options = {'REGISTER', 'UNDO'}

    mesh_type: bpy.props.EnumProperty(
            name='Mesh Type',
            items=[('NORMAL', 'Planar with Normals', ''),
                    ('MESH', '3D Mesh', '')],
            default='NORMAL',
            description='Generate either a normal map or real 3D structure'
    )
    mesh_style: bpy.props.EnumProperty(
            name='Mesh Style',
            items=[('TRI', 'Delaunay Triangulation', ''),
                    ('QUAD', 'Grid', '')],
            default='TRI',
            description='Method of creating faces inside the stroke shape'
    )
    contour_subdivision: bpy.props.IntProperty(
            name='Contour Subdivision',
            default=2, min=0, soft_max=5,
            description='Generate denser mesh near the contour for better quality shading'
    )
    use_native_triangulation: bpy.props.BoolProperty(
            name='Use Native Method',
            default=False,
            description='Do not use the external library for triangulation. Some advanced features will be disabled'
    )
    contour_trim: bpy.props.BoolProperty(
            name='Contour Trim',
            default=True,
            description='Dissolve points on the contour to reduce the number of n-gons'
    )
    resolution: bpy.props.IntProperty(
            name='Resolution',
            default=20, min=2, soft_max=100,
            description='Relative dimension of polygons of the generated mesh'
    )
    max_vertical_angle: bpy.props.FloatProperty(
            name='Max Vertical Angle',
            default=math.pi / 2, min= -math.pi / 2, max=math.pi / 2,
            unit='ROTATION',
            description='Vertical angle of the normal vector at the boundary vertices'
    ) 
    vertical_scale: bpy.props.FloatProperty(
            name='Vertical Scale',
            default=1, soft_max=5, soft_min=-5,
            description='Scale the vertical component of generated normal vectors. Negative values result in concave shapes'
    )
    closed_double_sided: bpy.props.BoolProperty(
            name='Closed Mesh',
            default=False,
            description='Leave no seams when mirroring the mesh'
    )
    excluded_group: bpy.props.StringProperty(
        name='Vertex Group',
        description='Points in this group will be regarded floating',
        default='',
        search=lambda self, context, edit_text: [group.name for group in context.object.vertex_groups]
    )
    fade_out: bpy.props.BoolProperty(
            name='Fade Out',
            default=False,
            description='Making the open area transparent'
    )
    transition_length: bpy.props.FloatProperty(
            name='Transition Length',
            default=0.1, min=0.001,
            unit='LENGTH'
    ) 
    advanced_solver: bpy.props.BoolProperty(
            name='Advanced Solver',
            default=False,
            description='Calculate more precise depth values through L-BFGS-B optimization, while being slower and relying on SciPy'
    )
    solver_max_iter: bpy.props.IntProperty(
            name='Iteration',
            default=100, min=10, soft_max=300,
            description='Maximum number of iterations of running the optimization algorithm'
    )
    mesh_material: bpy.props.StringProperty(
        name='Material',
        description='The material applied to generated mesh. Principled BSDF by default',
        default='Principled BSDF',
        search=lambda self, context, edit_text: get_material_list(self.mesh_type, bpy.context.engine)
    )
    vertex_color_mode: bpy.props.EnumProperty(
            name='Color',
            items=[('LINE', 'Use Line Color', ''),
                    ('FILL', 'Use Fill Color', '')],
            default='FILL',
            description='Source of vertex colors of the generated mesh'
    )

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self, width=300)

    def draw(self, context):
        layout = self.layout
        layout.label(text = "Multi-Object Alignment:")
        box1 = layout.box()
        box1.prop(self, "mesh_type")
        row = box1.row()
        row.prop(self, "stacked")
        if self.stacked:
            row.prop(self, "vertical_gap")
            if self.mesh_type == 'MESH':
                box1.prop(self, "overlap_tolerance")
        if self.mesh_type == 'MESH':
            row = box1.row()
            row.prop(self, "postprocess_double_sided")
            if self.postprocess_double_sided:
                row.prop(self, "closed_double_sided")
            row = box1.row()
            row.prop(self, "advanced_solver")
            if self.advanced_solver:
                row.prop(self, "solver_max_iter")
        box1.prop(self, "ignore_mode")

        layout.label(text = "Geometry Options:")
        box2 = layout.box()
        box2.label(text = "Mesh Style:")
        box2.prop(self, "mesh_style", text = "")   
        if self.mesh_style == 'TRI':
            box2.prop(self, "use_native_triangulation")
            box2.prop(self, "contour_subdivision")
        elif self.mesh_style == 'QUAD':
            box2.prop(self, "contour_trim", text = "Contour Trim")
        box2.prop(self, "resolution", text = "Resolution")
        box2.prop(self, "max_vertical_angle")
        box2.prop(self, "vertical_scale")
        
        row = box2.row()
        row.label(text = 'Open Area Group:')
        row.prop(self, "excluded_group", text='', icon='GROUP_VERTEX')
        if len(self.excluded_group)>0:
            row = box2.row()
            row.prop(self, "fade_out")
            row.prop(self, "transition_length", text = "Length")

        layout.prop(self, "mesh_material", text='Material', icon='MATERIAL')
        layout.prop(self, "vertex_color_mode")
        layout.prop(self, "reuse_material")
        layout.prop(self, "stop_motion_animation")
        layout.prop(self, "keep_original", text = "Keep Original")

    def execute(self, context):
        import numpy as np

        try:
            import pyclipper
        except ImportError:
            self.report({"ERROR"}, "Please install PyClipper in the Preferences panel.")
            return {'FINISHED'}
        
        if self.mesh_style == 'TRI' and not self.use_native_triangulation:
            try:
                import triangle as tr
            except ImportError:
                self.report({"INFO"}, "Triangle package is not installed. Switch to the native method.")
                self.use_native_triangulation = True
        
        if self.mesh_type == 'MESH' and self.advanced_solver:
            try:
                from ..solvers.optimizer import MeshDepthSolver
            except ImportError:
                self.report({"WARNING"}, "SciPy is not installed. Advanced solver is disabled.")
                self.advanced_solver = False
        
        # Get input information & resources
        current_gp_obj = context.object
        current_frame_number = context.scene.frame_current
        mesh_material = append_material(context, self.mesh_type, self.mesh_material, self.reuse_material, self)
        excluded_group_idx = -1
        if self.excluded_group in current_gp_obj.vertex_groups:
            excluded_group_idx = current_gp_obj.vertex_groups[self.excluded_group].index
        weight_helper = GPv3WeightHelper(current_gp_obj)
            
        frames_to_process = get_input_frames(current_gp_obj,
                                             multiframe = get_multiedit(current_gp_obj),
                                             return_map = True)
        
        # Process selected strokes frame by frame
        for frame_number, layer_frame_map in frames_to_process.items():
            stroke_info, stroke_list = [], []
            mask_info, mask_list = [], []
            mesh_names = []
            context.scene.frame_set(frame_number)
            if excluded_group_idx >= 0:
                weight_helper.setup()
            
            for layer_idx, item in layer_frame_map.items():
                frame = item[0]
                if is_frame_valid(frame):
                    for j,stroke in enumerate(frame.nijigp_strokes):
                        if stroke.select:
                            if self.ignore_mode == 'LINE' and is_stroke_line(stroke, current_gp_obj):
                                continue
                            if self.ignore_mode == 'OPEN' and is_stroke_line(stroke, current_gp_obj) and not stroke.use_cyclic:
                                continue
                            if is_stroke_hole(stroke, current_gp_obj):
                                mask_info.append([stroke, layer_idx, j, frame])
                                mask_list.append(stroke)
                            else:
                                stroke_info.append([stroke, layer_idx, j, frame])
                                stroke_list.append(stroke)
                                mesh_names.append('Planar_' + current_gp_obj.data.layers[layer_idx].info + '_' + str(j))
            t_mat, inv_mat = get_transformation_mat(mode=context.scene.nijigp_working_plane,
                                                    gp_obj=current_gp_obj, strokes=stroke_list, operator=self,
                                                    requires_layer=False)
            poly_list, depth_list, poly_inverted, scale_factor = get_2d_co_from_strokes(stroke_list, t_mat, 
                                                                        scale=True, correct_orientation=True, return_orientation=True)
            mask_poly_list, mask_depth_list, mask_inverted, _ = get_2d_co_from_strokes(mask_list, t_mat, 
                                                                scale=True, correct_orientation=True,
                                                                scale_factor=scale_factor, return_orientation=True)
            trans2d = lambda co: (t_mat @ co).xy

            generated_objects = get_generated_meshes(current_gp_obj)
            if len(poly_list) < 1:
                continue
            # Holes should have an opposite direction, and coordinate of an inside point is needed for triangle input
            mask_hole_points = []
            for mask_co_list in mask_poly_list:
                mask_co_list.reverse()
                mask_hole_points.append(get_an_inside_co(mask_co_list))

            def process_single_stroke(i, co_list, mask_indices = []):
                # Initialize the mesh to be generated
                new_mesh = bpy.data.meshes.new(mesh_names[i])
                bm = bmesh.new()
                vertex_color_layer = bm.verts.layers.color.new('Color')
                normal_map_layer = bm.verts.layers.float_vector.new('NormalMap')
                vertex_start_frame_layer = bm.verts.layers.int.new('start_frame')
                vertex_end_frame_layer = bm.verts.layers.int.new('end_frame')
                frame_range = layer_frame_map[stroke_info[i][1]][1]
                depth_layer = bm.verts.layers.float.new('Depth')
                uv_layer = bm.loops.layers.uv.new('UVMap')

                # Initialize the mask information
                local_mask_polys, local_mask_list, local_mask_inverted, hole_points = [], [], [], []
                for idx in mask_indices:
                    local_mask_polys.append(np.array(mask_poly_list[idx]))
                    local_mask_list.append(mask_list[idx])
                    local_mask_inverted.append(mask_inverted[idx])
                    if mask_hole_points[idx]:
                        hole_points.append(mask_hole_points[idx])

                # Calculate the normal and attributes of the original stroke points
                # Map for fast lookup; arrays for inner-production of weighted sum
                contour_co_array = []
                contour_normal_array = []
                contour_color_array = []
                contour_normal_map = {}
                contour_color_map = {}
                contour_excluded_set = set()
                                
                for j,co in enumerate(co_list):
                    point_idx = j if not poly_inverted[i] else len(co_list)-j-1
                    is_floating = False
                    if excluded_group_idx>=0:
                        try:
                            weight = stroke_list[i].points.weight_get(vertex_group_index=excluded_group_idx, point_index=point_idx)
                            is_floating = weight > 0
                        except:
                            pass    # TODO: Is there a better method that does not lead to exceptions?
                    if not is_floating:
                        _co = co_list[j-1]
                        norm = Vector([co[1]-_co[1], -co[0]+_co[0], 0]).normalized()
                        norm = norm * math.sin(self.max_vertical_angle) + Vector((0, 0, math.cos(self.max_vertical_angle)))
                        contour_co_array.append(co)
                        contour_normal_array.append(norm)
                        contour_normal_map[(int(co[0]),int(co[1]))] = norm
                        
                        point_color = get_mixed_color(current_gp_obj, stroke_list[i], point_idx)
                        contour_color_array.append(point_color)
                        contour_color_map[(int(co[0]),int(co[1]))] = point_color
                    else:
                        contour_excluded_set.add((int(co[0]),int(co[1])))
                if len(contour_co_array)<2:     # Case where too many points are excluded
                    bpy.ops.object.mode_set(mode='OBJECT')
                    return
                # Same process but for masks
                for mask_idx,poly in enumerate(local_mask_polys):
                    for j,co in enumerate(poly):
                        point_idx = j if local_mask_inverted[mask_idx] else len(poly)-j-1
                        is_floating = False
                        if excluded_group_idx>=0:
                            try:
                                weight = local_mask_list[mask_idx].points.weight_get(vertex_group_index=excluded_group_idx, point_index=point_idx)
                                is_floating = weight > 0
                            except:
                                pass
                        if not is_floating:
                            _co = poly[j-1]
                            norm = Vector([co[1]-_co[1], -co[0]+_co[0], 0]).normalized()
                            norm = norm * math.sin(self.max_vertical_angle) + Vector((0, 0, math.cos(self.max_vertical_angle)))
                            contour_co_array.append(co)
                            contour_normal_array.append(norm)
                            contour_normal_map[(int(co[0]),int(co[1]))] = norm
                            
                            point_color = get_mixed_color(current_gp_obj, local_mask_list[mask_idx], point_idx)
                            contour_color_array.append(point_color)
                            contour_color_map[(int(co[0]),int(co[1]))] = point_color
                        else:
                            contour_excluded_set.add((int(co[0]),int(co[1])))
                contour_color_array = np.array(contour_color_array)
                contour_normal_array = np.array(contour_normal_array)
                contour_co_array = np.array(contour_co_array)
                kdt_excluded = kdtree.KDTree(len(contour_excluded_set))
                for j,co in enumerate(contour_excluded_set):
                    kdt_excluded.insert(xy0(co),j)
                kdt_excluded.balance()

                co_list = np.array(co_list)
                u_min, u_max = np.min(co_list[:,0]), np.max(co_list[:,0])
                v_min, v_max = np.min(co_list[:,1]), np.max(co_list[:,1])
                
                # Generate vertices and faces in BMesh
                # Method 1: use Knife Project with a 2D grid
                mean_depth = np.mean(np.array(depth_list[i]))
                if self.mesh_style=='QUAD':
                    # Convert the stroke curve to a temporary mesh
                    bm_cut = bmesh.new()
                    for stroke in [stroke_list[i]]+local_mask_list:
                        for point in stroke.points:
                            bm_cut.verts.new(point.co)
                        bm_cut.verts.ensure_lookup_table()
                        for j in range(len(stroke.points)):
                            vert_src = bm_cut.verts[-1-j]
                            vert_dst = bm_cut.verts[-1] if j==len(stroke.points)-1 else bm_cut.verts[-2-j]
                            bm_cut.edges.new([vert_src, vert_dst])
                    cut_mesh = bpy.data.meshes.new('stroke_cut')
                    bm_cut.to_mesh(cut_mesh)
                    bm_cut.free()
                    cut_obj = bpy.data.objects.new('stroke_cut', cut_mesh)
                    cut_obj.parent = current_gp_obj
                    bpy.context.collection.objects.link(cut_obj)

                    # Generate a grid and cut it with the Knife Project operator
                    bpy.ops.object.mode_set(mode='OBJECT')
                    grid_size = max((u_max-u_min), (v_max-v_min))/scale_factor
                    margin_size = grid_size/self.resolution * 0.5
                    grid_loc = restore_3d_co([(u_max+u_min)/2, (v_max+v_min)/2], mean_depth, inv_mat, scale_factor)
                    bpy.ops.mesh.primitive_grid_add(x_subdivisions=self.resolution,
                                                    y_subdivisions=self.resolution,
                                                    size= grid_size + margin_size,  
                                                    align='WORLD',
                                                    location=grid_loc, 
                                                    scale=(1, 1, 1))
                    grid_obj = bpy.context.object
                    grid_obj.rotation_mode = 'QUATERNION'
                    grid_obj.rotation_quaternion = inv_mat.to_quaternion() 
                    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
                    grid_obj = bpy.context.object
                    grid_obj.parent = current_gp_obj
                    bpy.ops.object.mode_set(mode='EDIT')
                    cut_obj.select_set(True)
                    bpy.ops.mesh.knife_project()
                    bpy.ops.mesh.select_mode(type='FACE')
                    bpy.ops.mesh.select_all(action='INVERT')
                    bpy.ops.mesh.delete(type='FACE')
                    bpy.ops.object.mode_set(mode='OBJECT')
                    bm.from_mesh(grid_obj.data)
                    bpy.ops.object.delete()
                    bpy.context.view_layer.objects.active = current_gp_obj

                    # Transform coordinates to 2D for now
                    for vert in bm.verts:
                        vert.co = xy0(trans2d(vert.co) * scale_factor)
                    # Trim the boundary in BMesh
                    if self.contour_trim:
                        to_trim = []
                        for vert in bm.verts:
                            if vert.is_boundary and len(vert.link_edges)==2:
                                to_trim.append(vert)
                        bmesh.ops.dissolve_verts(bm, verts=to_trim)
                    bm.verts.ensure_lookup_table()
                    bm.verts.index_update()

                # Method 2: use Delaunay triangulation
                elif self.mesh_style=='TRI':
                    segs = []
                    verts= []
                    num_verts = 0
                    offset_size = -min((u_max-u_min),(v_max-v_min))/self.resolution
                    offset_values = [0]

                    # Preprocessing the polygon using Clipper
                    # The triangle library may crash in several cases, which should be avoided with every effort
                    # https://www.cs.cmu.edu/~quake/triangle.trouble.html
                    clipper = pyclipper.PyclipperOffset()
                    clipper.AddPaths([co_list]+local_mask_polys, join_type = pyclipper.JT_ROUND, end_type = pyclipper.ET_CLOSEDPOLYGON)
                    
                    for _ in range(self.contour_subdivision):
                        new_offset = offset_values[-1] + offset_size / max(1, self.contour_subdivision)
                        offset_values.append(new_offset)
                    # If using the native method for triangulation, do offset for more times to achieve the target resolution
                    if self.use_native_triangulation:
                        for _ in range(self.resolution-1):
                            new_offset = offset_values[-1] + offset_size
                            offset_values.append(new_offset)
                    
                    for value in offset_values:
                        poly_results = clipper.Execute(value)
                        for poly_result in poly_results:
                            if value < 0:
                                poly_result = merge_poly_points(poly_result, -offset_size * 0.25)
                            for j,co in enumerate(poly_result):
                                verts.append(co)
                                segs.append( [j + num_verts, (j+1)%len(poly_result) + num_verts] )
                            num_verts = len(verts)

                    # Refer to: https://rufat.be/triangle/API.html
                    if len(verts)<3:
                        bpy.ops.object.mode_set(mode='OBJECT')
                        return
                    tr_input = dict(vertices = verts, segments = np.array(segs))
                    if not self.use_native_triangulation:
                        if len(hole_points)>0:
                            tr_input['holes']=hole_points
                        area_limit = (u_max-u_min)*(v_max-v_min)/(self.resolution**2)
                        tr_output = tr.triangulate(tr_input, 'pa'+str(area_limit))
                    else:
                        tr_output = {}
                        tr_output['vertices'], _, tr_output['triangles'], _,_,_ = geometry.delaunay_2d_cdt(tr_input['vertices'], tr_input['segments'], [], 2, 1e-9)
                        
                    # Generate vertices and triangle faces
                    for co in tr_output['vertices']:
                        bm.verts.new(xy0(co))               # Use 2D coordinates temporarily for now
                    bm.verts.ensure_lookup_table()
                    bm.verts.index_update()
                    for f in tr_output['triangles']:
                        v_list = (bm.verts[f[0]], bm.verts[f[1]], bm.verts[f[2]])
                        if self.use_native_triangulation:   # When using the native method, we need to manually remove faces inside holes
                            face_center = ((tr_output['vertices'][f[0]][0]+tr_output['vertices'][f[1]][0]+tr_output['vertices'][f[2]][0]) / 3.0,
                                        (tr_output['vertices'][f[0]][1]+tr_output['vertices'][f[1]][1]+tr_output['vertices'][f[2]][1]) / 3.0)
                            for poly in local_mask_polys:
                                if pyclipper.PointInPolygon(face_center, poly) != 0:
                                    break
                            else:
                                bm.faces.new(v_list)
                        else:
                            bm.faces.new(v_list)    

                # Attribute interpolation based on 2D distance
                fill_color = get_mixed_color(current_gp_obj, stroke_list[i])
                if hasattr(stroke_list[i], 'fill_opacity'):
                    fill_color[3] *= stroke_list[i].fill_opacity
                maxmin_dist = 0.
                depth_offset = np.cos(self.max_vertical_angle) / np.sqrt(1 + 
                                                                (self.vertical_scale**2 - 1) *
                                                                (np.sin(self.max_vertical_angle)**2))
                for j,vert in enumerate(bm.verts):
                    co_2d = vert.co.xy
                    co_key = (int(co_2d[0]), int(co_2d[1]))
                    norm = Vector((0,0,0))
                    # Contour vertex case
                    if co_key in contour_normal_map:
                        norm = contour_normal_map[co_key]
                        vert_color = contour_color_map[co_key]
                    # Inner vertex case
                    else:
                        dist_sq = (contour_co_array[:,0]-co_2d[0])**2 + (contour_co_array[:,1]-co_2d[1])**2
                        weights = 1.0 / dist_sq
                        weights /= np.sum(weights)
                        maxmin_dist = max( np.min(dist_sq), maxmin_dist)
                        norm_u = np.dot(contour_normal_array[:,0], weights)
                        norm_v = np.dot(contour_normal_array[:,1], weights)
                        norm = Vector([norm_u, norm_v, np.sqrt(max(0,math.sin(self.max_vertical_angle)**2-norm_u**2-norm_v**2)) + math.cos(self.max_vertical_angle)])
                        vert_color = weights @ contour_color_array
                    # Scale vertical components
                    norm = Vector((norm.x * self.vertical_scale, norm.y * self.vertical_scale, norm.z)).normalized()
                    vert[normal_map_layer] = [ 0.5 * (norm.x + 1) , 0.5 * (norm.y + 1), 0.5 * (norm.z+1)]
                    vert[vertex_color_layer] = vert_color if self.vertex_color_mode == 'LINE' else fill_color
                    vert[vertex_start_frame_layer] = frame_range[0]
                    vert[vertex_end_frame_layer] = frame_range[1]
                    if vert.is_boundary and self.postprocess_double_sided and self.closed_double_sided:
                        vert[depth_layer] = 0
                    else:
                        vert[depth_layer] = norm.z - depth_offset
                    # Fading effect
                    if self.fade_out:
                        co_excluded, _, dist = kdt_excluded.find(xy0(co_key))
                        if co_excluded:
                            vert[vertex_color_layer][3] *= smoothstep(dist/scale_factor/self.transition_length)
                maxmin_dist = np.sqrt(maxmin_dist) / scale_factor
                
                # Normalize the depth
                depth_scale = maxmin_dist * self.vertical_scale * np.sign(self.max_vertical_angle)
                for vert in bm.verts:
                    vert[depth_layer] *= depth_scale

                # UV projection, required for correct tangent direction
                for face in bm.faces:
                    for loop in face.loops:
                        co_2d = loop.vert.co.xy
                        loop[uv_layer].uv = ( (co_2d[0]-u_min)/(u_max-u_min),
                                            (co_2d[1]-v_min)/(v_max-v_min))

                # 2D operations finished; Transform coordinates for 3D operations
                bm3d = bm.copy()
                for vert in bm3d.verts:
                    vert.co = restore_3d_co(vert.co.xy, mean_depth, inv_mat, scale_factor)

                # Determine the depth coordinate by ray-casting to every mesh generated earlier
                ray_direction = inv_mat @ Vector([0,0,1])
                vertical_pos = 0
                hit_points = []
                if self.stacked:
                    percentile = 100 if self.mesh_type == 'NORMAL' else 100 - self.overlap_tolerance
                    for v in bm3d.verts:
                        max_hit = 0
                        for j,obj in enumerate(generated_objects):
                            ray_emitter = np.array(v.co)
                            ray_emitter += ray_direction * MAX_DEPTH
                            res, loc, norm, idx = obj.ray_cast(ray_emitter, -ray_direction)
                            ray_hitpoint = t_mat @ (loc - v.co)
                            if res:
                                max_hit = max(max_hit, ray_hitpoint[2])
                        hit_points.append(max_hit)
                    vertical_pos = np.percentile(hit_points, percentile)
                    vertical_pos += self.vertical_gap

                # Calling the advanced solver: must use the 2D data again
                if self.mesh_type == 'MESH' and self.advanced_solver:
                    solver = MeshDepthSolver()
                    solver.initialize_from_bmesh(bm, scale_factor, contour_normal_map, depth_scale < 0)
                    solver.solve(self.solver_max_iter)
                    solver.write_back(bm)

                # Convert attribute to depth value
                depth_scale = maxmin_dist * self.vertical_scale * np.sign(self.max_vertical_angle)
                for j,v in enumerate(bm3d.verts):
                    if self.mesh_type == 'MESH':
                        v.co += float(bm.verts[j][depth_layer]) * ray_direction

                # Object generation
                bm3d.to_mesh(new_mesh)
                bm3d.free()
                bm.free()
                new_object = bpy.data.objects.new(mesh_names[i], new_mesh)
                new_object['nijigp_mesh'] = 'planar' if self.mesh_type=='NORMAL' else '3d'
                new_object['nijigp_parent'] = current_gp_obj
                new_object.location = vertical_pos * ray_direction
                bpy.context.collection.objects.link(new_object)
                new_object.parent = current_gp_obj
                generated_objects.append(new_object)

                # Apply transform, necessary because of the movement in depth
                mb = new_object.matrix_basis
                new_object.data.transform(mb)
                applied_offset = Vector(new_object.location)
                gp_layer = current_gp_obj.data.layers[stroke_info[i][1]]
                new_object.location = gp_layer.matrix_layer.to_translation()
                new_object.rotation_euler = gp_layer.matrix_layer.to_euler()
                new_object.scale = gp_layer.matrix_layer.to_scale()
                
                # Assign material
                if mesh_material:
                    new_object.data.materials.append(mesh_material)

                # Post-processing
                bpy.ops.object.mode_set(mode='OBJECT')
                bpy.ops.object.select_all(action='DESELECT')
                new_object.select_set(True)
                context.view_layer.objects.active = new_object
                if bpy.app.version < (4, 1, 0):
                    bpy.ops.object.shade_smooth(use_auto_smooth=False)
                else:
                    bpy.ops.object.shade_smooth(keep_sharp_edges=False)
                mean_depth_offset = inv_mat @ Vector((0, 0, mean_depth))
                if self.mesh_type=='MESH' and self.postprocess_double_sided:
                    apply_mirror_in_depth(new_object, inv_mat, applied_offset + mean_depth_offset)
                if self.stop_motion_animation:
                    new_object.modifiers.new(name="nijigp_GeoNodes", type='NODES')
                    new_object.modifiers["nijigp_GeoNodes"].node_group = append_geometry_nodes(context, 'NijiGP Stop Motion')
            
            for i,co_list in enumerate(poly_list):
                # Identify the holes that should be considered: same layer, arranged beyond and inside
                mask_indices = []
                for mask_idx, info in enumerate(mask_info):
                    if (stroke_info[i][1] == mask_info[mask_idx][1] and
                        stroke_info[i][2] < mask_info[mask_idx][2] and
                        is_poly_in_poly(mask_poly_list[mask_idx], poly_list[i])):
                        mask_indices.append(mask_idx)
                process_single_stroke(i, co_list, mask_indices)

            # Delete old strokes
            if not self.keep_original:
                for info in (stroke_info + mask_info):
                    info[3].nijigp_strokes.remove(info[0])

            bpy.ops.object.select_all(action='DESELECT')
            current_gp_obj.select_set(True)
            context.view_layer.objects.active = current_gp_obj
            bpy.ops.object.mode_set(mode=get_obj_mode_str('EDIT'))
            weight_helper.commit(abort=True)
            
        bpy.ops.object.mode_set(mode='OBJECT') 
        context.scene.frame_set(current_frame_number)
        return {'FINISHED'}

class MeshGenerationByOffsetting(CommonMeshConfig, bpy.types.Operator):
    """Generate an embossed mesh by offsetting the selected strokes"""
    bl_idname = "gpencil.nijigp_mesh_generation_offset"
    bl_label = "Convert to Meshes by Offsetting"
    bl_category = 'View'
    bl_options = {'REGISTER', 'UNDO'}

    offset_amount: bpy.props.FloatProperty(
            name='Offset',
            default=0.1, soft_min=0, unit='LENGTH',
            description='Offset length'
    )
    resolution: bpy.props.IntProperty(
            name='Resolution',
            default=4, min=2, max=256,
            description='Number of offsets calculated'
    )
    corner_shape: bpy.props.EnumProperty(
            name='Corner Shape',
            items=[('JT_ROUND', 'Round', ''),
                    ('JT_SQUARE', 'Square', ''),
                    ('JT_MITER', 'Miter', '')],
            default='JT_ROUND',
            description='Shape of corners generated by offsetting'
    )
    slope_style: bpy.props.EnumProperty(
            name='Slope Style',
            items=[('LINEAR', 'Linear', ''),
                    ('SPHERE', 'Sphere', ''),
                    ('STEP', 'Step', ''),
                    ('RIDGE', 'Ridge', 'SciPy is required for this style')],
            default='SPHERE',
            description='Slope shape of the generated mesh'
    )
    postprocess_shade_smooth: bpy.props.BoolProperty(
            name='Shade Smooth',
            default=False,
            description='Enable face smooth shading and auto smooth normals'
    )
    postprocess_merge: bpy.props.BoolProperty(
            name='Merge',
            default=False,
            description='Merge vertices close to each other'   
    )
    merge_distance: bpy.props.FloatProperty(
            name='Distance',
            default=0.01,
            min=0.0001,
            unit='LENGTH',
            description='Distance used during merging'   
    )
    postprocess_remesh: bpy.props.BoolProperty(
            name='Remesh',
            default=False,
            description='Perform a voxel remesh'   
    )
    remesh_voxel_size: bpy.props.FloatProperty(
            name='Voxel Size',
            default=0.1,
            min=0.0001,
            unit='LENGTH',
            description='Voxel size used during remeshing'   
    )
    mesh_material: bpy.props.StringProperty(
        name='Material',
        description='The material applied to generated mesh. Principled BSDF by default',
        default='Principled BSDF',
        search=lambda self, context, edit_text: get_material_list('MESH', bpy.context.engine)
    )
    
    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self, width=300)

    def draw(self, context):
        layout = self.layout
        layout.label(text = "Multi-Object Alignment:")
        box1 = layout.box()
        row = box1.row()
        row.prop(self, "stacked")
        if self.stacked:
            row.prop(self, "vertical_gap")
            box1.prop(self, "overlap_tolerance")
        box1.prop(self, "ignore_mode")
        layout.label(text = "Geometry Options:")
        box2 = layout.box()
        row = box2.row()
        row.prop(self, "offset_amount", text = "Offset Amount")
        row = box2.row()
        row.prop(self, "resolution", text = "Resolution")
        if self.slope_style == 'RIDGE':
            row.enabled = False
        row = box2.row()
        row.label(text = "Corner Shape")
        row.prop(self, "corner_shape", text = "")
        row = box2.row()
        row.label(text = "Slope Style")
        row.prop(self, "slope_style", text = "")
        
        layout.label(text = "Post-Processing Options:")
        box3 = layout.box()
        box3.prop(self, "postprocess_double_sided", text = "Double-Sided")  
        box3.prop(self, "postprocess_shade_smooth", text = "Shade Smooth")
        row = box3.row()
        row.prop(self, "postprocess_merge", text='Merge By')
        row.prop(self, "merge_distance", text='Distance')
        row = box3.row()
        if not self.stop_motion_animation:
            row.prop(self, "postprocess_remesh", text='Remesh')
            row.prop(self, "remesh_voxel_size", text='Voxel Size')

        layout.prop(self, "mesh_material", text='Material', icon='MATERIAL')
        layout.prop(self, "reuse_material")
        layout.prop(self, "stop_motion_animation")
        layout.prop(self, "keep_original", text = "Keep Original")

    def execute(self, context):
        # Import and configure Clipper
        try:
            import pyclipper
            if self.slope_style == 'RIDGE':
                from scipy.spatial import Voronoi
        except ImportError:
            self.report({"ERROR"}, "Please install dependencies in the Preferences panel.")
            return {'FINISHED'}
        clipper = pyclipper.PyclipperOffset()
        clipper.MiterLimit = math.inf
        jt = pyclipper.JT_ROUND
        if self.corner_shape == "JT_SQUARE":
            jt = pyclipper.JT_SQUARE
        elif self.corner_shape == "JT_MITER":
            jt = pyclipper.JT_MITER
        et = pyclipper.ET_CLOSEDPOLYGON

        # Get input information & resources
        current_gp_obj = context.object
        current_frame_number = context.scene.frame_current
        mesh_material = append_material(context, 'MESH', self.mesh_material, self.reuse_material, self)
        frames_to_process = get_input_frames(current_gp_obj,
                                             multiframe = get_multiedit(current_gp_obj),
                                             return_map = True)
        
        # Process selected strokes frame by frame
        for frame_number, layer_frame_map in frames_to_process.items():
            stroke_info, stroke_list = [], []
            mask_info, mask_list = [], []
            mesh_names = []
            context.scene.frame_set(frame_number)
            
            for layer_idx, item in layer_frame_map.items():
                frame = item[0]
                if is_frame_valid(frame):
                    for j,stroke in enumerate(frame.nijigp_strokes):
                        if stroke.select:
                            if self.ignore_mode == 'LINE' and is_stroke_line(stroke, current_gp_obj):
                                continue
                            if self.ignore_mode == 'OPEN' and is_stroke_line(stroke, current_gp_obj) and not stroke.use_cyclic:
                                continue
                            if is_stroke_hole(stroke, current_gp_obj):
                                mask_info.append([stroke, layer_idx, j, frame])
                                mask_list.append(stroke)
                            else:
                                stroke_info.append([stroke, layer_idx, j, frame])
                                stroke_list.append(stroke)
                                mesh_names.append('Offset_' + current_gp_obj.data.layers[layer_idx].info + '_' + str(j))
            t_mat, inv_mat = get_transformation_mat(mode=context.scene.nijigp_working_plane,
                                                    gp_obj=current_gp_obj, strokes=stroke_list, operator=self,
                                                    requires_layer=False)
            poly_list, depth_list, scale_factor = get_2d_co_from_strokes(stroke_list, t_mat, 
                                                                scale=True, correct_orientation=True)
            mask_poly_list, mask_depth_list, _ = get_2d_co_from_strokes(mask_list, t_mat, 
                                                                scale=True, correct_orientation=True,
                                                                scale_factor=scale_factor)
            generated_objects = get_generated_meshes(current_gp_obj)
            if len(poly_list) < 1:
                continue
            # Holes should have an opposite direction
            for mask_co_list in mask_poly_list:
                mask_co_list.reverse()
                                        
            def process_single_stroke(i, co_list, mask_indices = []):
                '''
                Function that processes each stroke separately
                '''
                clipper.Clear()
                new_mesh = bpy.data.meshes.new(mesh_names[i])
                bm = bmesh.new()
                vertex_color_layer = bm.verts.layers.color.new('Color')
                vertex_start_frame_layer = bm.verts.layers.int.new('start_frame')
                vertex_end_frame_layer = bm.verts.layers.int.new('end_frame')
                frame_range = layer_frame_map[stroke_info[i][1]][1]
                mean_depth = np.mean(np.array(depth_list[i]))

                # Initialize the mask information
                local_mask_polys = []
                for idx in mask_indices:
                    local_mask_polys.append(np.array(mask_poly_list[idx]))
                        
                # Ridge style: the only extra vertices are the median axis
                if self.slope_style == 'RIDGE':
                    tr_input_verts = []
                    tr_input_edges = []
                    # Add the original contour (including holes)
                    full_contour = co_list + [co for mask in local_mask_polys for co in mask]
                    kdt = kdtree.KDTree(len(full_contour))
                    v_idx = 0
                    for j,poly in enumerate([co_list] + local_mask_polys):
                        first = None
                        for k,co in enumerate(poly):
                            kdt.insert(xy0(co), v_idx)
                            tr_input_verts.append(co.copy())
                            if first is None:
                                first = v_idx
                            if k > 0:
                                tr_input_edges.append([v_idx - 1, v_idx])
                            v_idx += 1
                        tr_input_edges.append([v_idx - 1, first])
                    kdt.balance()
                            
                    # Calculate Voronoi vertices and ridges
                    vor = Voronoi(full_contour)
                    ridge_idx_map = {}
                    for j,co in enumerate(vor.vertices):
                        # A valid vertex should be inside the contour but ouside any hole
                        if pyclipper.PointInPolygon(co, co_list) == 1:
                            for mask in local_mask_polys:
                                if pyclipper.PointInPolygon(co, mask) == 1:
                                    break
                            else:
                                ridge_idx_map[j] = v_idx
                                tr_input_verts.append(co.copy())
                                v_idx += 1
                    for j1,j2 in vor.ridge_vertices:
                        if j1 in ridge_idx_map and j2 in ridge_idx_map:
                            tr_input_edges.append([ridge_idx_map[j1], ridge_idx_map[j2]])
                    
                    # Generate triangle faces and determine the depth of vertices
                    tr_output = {}
                    tr_output['vertices'], _, tr_output['triangles'], _,_,_ = geometry.delaunay_2d_cdt(tr_input_verts, tr_input_edges, [], 2, 1e-9)
                    verts_by_level = [[]]
                    verts_depth = []
                    max_dist = 0
                    for j,co in enumerate(tr_output['vertices']):
                        if pyclipper.PointInPolygon(co, co_list) == 1:
                            _, _, dist = kdt.find(xy0(co))
                            verts_depth.append(dist)
                            max_dist = max(max_dist, dist)
                        else:
                            verts_depth.append(0)
                            verts_by_level[0].append(j)
                    # Fill data to BMesh
                    for j,co in enumerate(tr_output['vertices']):
                            v = bm.verts.new(restore_3d_co(co,
                                                        self.offset_amount * verts_depth[j] / max_dist + mean_depth,
                                                        inv_mat, scale_factor))
                    bm.verts.ensure_lookup_table()
                    bm.verts.index_update()
                    verts_by_level[0] = [bm.verts[j] for j in verts_by_level[0]]
                    for f in tr_output['triangles']:
                        center = (tr_output['vertices'][f[0]] + tr_output['vertices'][f[1]] + tr_output['vertices'][f[2]]) / 3
                        for mask in local_mask_polys:
                            if pyclipper.PointInPolygon(center, mask) == 1:
                                break
                        else:
                            bm.faces.new((bm.verts[f[0]], bm.verts[f[1]], bm.verts[f[2]]))
                                                            
                # Other styles: generate multiple levels of extra vertices by insetting
                else:
                    clipper.AddPaths([co_list] + local_mask_polys, jt, et)
                    contours = []       # 2D list: one inset level may contain multiple loops
                    vert_idx_list = []  # Start index and length of each loop
                    vert_counter = 0
                    offset_interval = self.offset_amount / self.resolution * scale_factor
                    for j in range(self.resolution):
                        clipper_res = clipper.Execute( -offset_interval * j)
                        # Step style requires duplicating each contour
                        for _ in range(1 + int(self.slope_style=='STEP')):
                            new_contour = []
                            new_idx_list = []
                            for poly in clipper_res:
                                # Merge points in advance for specific methods
                                true_poly = []
                                if not self.postprocess_merge:
                                    true_poly = poly
                                else:
                                    true_poly = merge_poly_points(poly, self.merge_distance * scale_factor)
                                    if len(true_poly)<3:
                                        true_poly = poly
                                num_vert = len(true_poly)
                                new_idx_list.append( (vert_counter, vert_counter + num_vert) )
                                vert_counter += num_vert
                                new_contour.append(true_poly)
                            contours.append(new_contour)
                            vert_idx_list.append(new_idx_list)   

                    # Step style: create the 3D mesh level-by-level
                    if self.slope_style == 'STEP':
                        edges_by_level = []
                        verts_by_level = []
                        for j,contour in enumerate(contours):
                            edges_by_level.append([])
                            verts_by_level.append([])
                            edge_extruded = []
                            height = abs( (j+1)//2 * offset_interval/scale_factor)
                            
                            # Convert each loop in a level to mesh
                            for k,poly in enumerate(contour):
                                for co in poly:
                                    co_3d = restore_3d_co(co, height + mean_depth, inv_mat, scale_factor)
                                    verts_by_level[-1].append( bm.verts.new(co_3d) )
                                bm.verts.ensure_lookup_table()                  
                                for v_idx in range(vert_idx_list[j][k][0],vert_idx_list[j][k][1] - 1):
                                    edges_by_level[-1].append( bm.edges.new([bm.verts[v_idx],bm.verts[v_idx + 1]]) )
                                edges_by_level[-1].append( bm.edges.new([ bm.verts[vert_idx_list[j][k][0]], bm.verts[vert_idx_list[j][k][1] - 1] ]))
                            bm.edges.ensure_lookup_table()
                            
                            # Connect two levels: alternatively extrude and inset to get a step shape
                            if j%2 > 0:
                                for v_idx,_ in enumerate(verts_by_level[-1]):
                                    edge_extruded.append( bm.edges.new([verts_by_level[-1][v_idx], verts_by_level[-2][v_idx]]) )
                                bm.edges.ensure_lookup_table()
                                bmesh.ops.edgenet_fill(bm, edges=edges_by_level[-1]+edges_by_level[-2]+edge_extruded)
                            elif j > 0:
                                bmesh.ops.triangle_fill(bm, use_beauty=True, edges=edges_by_level[-1]+edges_by_level[-2])
                        bmesh.ops.triangle_fill(bm, use_beauty=True, edges=edges_by_level[-1])
                        bm.faces.ensure_lookup_table()
                        
                    # Sphere and linear styles: create mesh of all levels in 2D first, then set the heights
                    else:                      
                        verts_by_level = []
                        edges_by_level = []
                        heights_by_level = []
                        kdt_by_level = []
                        
                        # Process the independent part of each level's geometry
                        for j,contour in enumerate(contours):
                            # Calculate the height of each level
                            height = abs(j * offset_interval/scale_factor)
                            if self.slope_style == 'SPHERE':
                                sphere_rad = abs(self.offset_amount)
                                height = math.sqrt(sphere_rad ** 2 - (sphere_rad - height) ** 2)
                            heights_by_level.append(height)
                            
                            # Add each polygon loop to mesh. Use 2D coordinates for now
                            verts_by_level.append([])
                            edges_by_level.append([])
                            for k,poly in enumerate(contour):
                                for co in poly:
                                    verts_by_level[-1].append(bm.verts.new(xy0(co)/scale_factor))
                                bm.verts.ensure_lookup_table()
                                for v_idx in range(vert_idx_list[j][k][0],vert_idx_list[j][k][1] - 1):
                                    edges_by_level[-1].append( bm.edges.new([bm.verts[v_idx],bm.verts[v_idx + 1]]) )
                                edges_by_level[-1].append( bm.edges.new([ bm.verts[vert_idx_list[j][k][0]], bm.verts[vert_idx_list[j][k][1] - 1] ]) )
                            bm.edges.ensure_lookup_table()
                               
                            # Prepare a KDTree for each level for later lookup
                            kdt_by_level.append(kdtree.KDTree(len(verts_by_level[-1])))
                            for v_idx, v in enumerate(verts_by_level[-1]):
                                kdt_by_level[-1].insert(v.co, v_idx)
                            kdt_by_level[-1].balance()
                        
                        # To connect adjacent levels, first manually create edges with best efforts
                        for j,contour in enumerate(contours):
                            if j == 0 or len(verts_by_level[j]) == 0:
                                continue
                            for v in verts_by_level[j-1]:
                                _, idx, _ = kdt_by_level[j].find(v.co)
                                new_e = (v, verts_by_level[j][idx])
                                # New edge should not intersect with existing edges
                                for e in edges_by_level[j-1]:
                                    if e.verts[1] != v and e.verts[0] != v and \
                                       geometry.intersect_line_line_2d(e.verts[0].co.xy, e.verts[1].co.xy, new_e[0].co.xy, new_e[1].co.xy):
                                        break
                                else:
                                    bm.edges.new(new_e)
                                
                        # Then do triangulation to ensure all faces are filled
                        bm.edges.ensure_lookup_table()
                        bmesh.ops.triangle_fill(bm, use_beauty=True, edges=bm.edges)
                        bm.faces.ensure_lookup_table()
                        # This method may generate faces outside the original shape, which should be removed
                        faces_to_remove = []
                        for f in bm.faces:
                            co = f.calc_center_median().xy * scale_factor
                            if pyclipper.PointInPolygon(co, co_list) == 0:
                                faces_to_remove.append(f)
                            else:
                                for mask in local_mask_polys:
                                    if pyclipper.PointInPolygon(co, mask) == 1:
                                        faces_to_remove.append(f)
                                        break
                        for f in faces_to_remove:
                            bm.faces.remove(f)
                        edges_to_remove = []
                        for e in bm.edges:
                            if len(e.link_faces) == 0:
                                edges_to_remove.append(e)
                        for e in edges_to_remove:
                            bm.edges.remove(e)
                            
                        # Set the height of each vertex
                        for j,verts in enumerate(verts_by_level):
                            for v in verts:
                                co_3d = restore_3d_co(v.co.xy, heights_by_level[j] + mean_depth, inv_mat, 1)
                                v.co = co_3d
                
                if self.postprocess_shade_smooth:
                    for face in bm.faces:
                        face.smooth = True
                
                # Cleanup
                if self.slope_style == 'STEP':
                    to_remove = []
                    for face in bm.faces:
                        if len(face.verts) > 4:
                            to_remove.append(face)
                    for face in to_remove:
                        bm.faces.remove(face)
                bmesh.ops.recalc_face_normals(bm, faces=bm.faces)

                # Post-processing: merge
                if self.postprocess_merge:
                    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=self.merge_distance)

                # Set vertex attributes
                fill_color = get_mixed_color(current_gp_obj, stroke_list[i])
                if hasattr(stroke_list[i], 'fill_opacity'):
                    fill_color[3] *= stroke_list[i].fill_opacity
                for v in bm.verts:
                    v[vertex_color_layer] = fill_color
                    v[vertex_start_frame_layer] = frame_range[0]
                    v[vertex_end_frame_layer] = frame_range[1]

                # Determine the depth coordinate by ray-casting to every mesh generated earlier
                ray_direction = inv_mat @ Vector([0,0,1])
                vertical_pos = 0
                hit_points = []
                if self.stacked:
                    percentile = 100 - self.overlap_tolerance
                    for v in bm.verts:
                        max_hit = 0
                        for j,obj in enumerate(generated_objects):
                            ray_emitter = np.array(v.co)
                            ray_emitter += ray_direction * MAX_DEPTH
                            res, loc, norm, idx = obj.ray_cast(ray_emitter, -ray_direction)
                            ray_hitpoint = t_mat @ (loc-v.co)
                            if res:
                                max_hit = max(max_hit, ray_hitpoint[2])
                        hit_points.append(max_hit)
                    vertical_pos = np.percentile(hit_points, percentile)
                    vertical_pos += self.vertical_gap

                bm.to_mesh(new_mesh)
                bm.free()

                # Object generation
                new_object = bpy.data.objects.new(mesh_names[i], new_mesh)
                new_object['nijigp_mesh'] = '3d'
                new_object['nijigp_parent'] = current_gp_obj
                new_object.location = inv_mat @ Vector([0,0,vertical_pos])
                bpy.context.collection.objects.link(new_object)
                new_object.parent = current_gp_obj
                generated_objects.append(new_object)

                # Apply transform, necessary because of the movement in depth
                mb = new_object.matrix_basis
                new_object.data.transform(mb)
                applied_offset = Vector(new_object.location)
                gp_layer = current_gp_obj.data.layers[stroke_info[i][1]]
                new_object.location = gp_layer.matrix_layer.to_translation()
                new_object.rotation_euler = gp_layer.matrix_layer.to_euler()
                new_object.scale = gp_layer.matrix_layer.to_scale()

                # Assign material
                if mesh_material:
                    new_object.data.materials.append(mesh_material)

                # Post-processing: mirror
                context.view_layer.objects.active = new_object
                mean_depth_offset = inv_mat @ Vector((0, 0, mean_depth))
                bpy.ops.object.mode_set(mode='OBJECT')
                if self.postprocess_double_sided:
                    apply_mirror_in_depth(new_object, inv_mat, applied_offset + mean_depth_offset)
                
                # Post-processing: remesh
                if self.postprocess_remesh:
                    new_object.data.remesh_voxel_size = self.remesh_voxel_size
                    new_object.data.use_remesh_preserve_volume = True
                    # API change from Blender 4.1
                    if hasattr(new_object.data, "use_remesh_preserve_vertex_attributes"):
                        new_object.data.use_remesh_preserve_vertex_attributes = True
                    elif hasattr(new_object.data, "use_remesh_preserve_vertex_colors"):
                        new_object.data.use_remesh_preserve_vertex_colors = True
                    bpy.ops.object.voxel_remesh("EXEC_DEFAULT")

                if bpy.app.version < (4, 1, 0):
                    new_object.data.use_auto_smooth = self.postprocess_shade_smooth
                elif self.postprocess_shade_smooth:
                    bpy.ops.object.shade_smooth_by_angle()

                if self.stop_motion_animation and not self.postprocess_remesh:
                    new_object.modifiers.new(name="nijigp_GeoNodes", type='NODES')
                    new_object.modifiers["nijigp_GeoNodes"].node_group = append_geometry_nodes(context, 'NijiGP Stop Motion')

            for i,co_list in enumerate(poly_list):
                mask_indices = []
                for mask_idx, info in enumerate(mask_info):
                    if (stroke_info[i][1] == mask_info[mask_idx][1] and
                        stroke_info[i][2] < mask_info[mask_idx][2] and
                        is_poly_in_poly(mask_poly_list[mask_idx], poly_list[i])):
                        mask_indices.append(mask_idx)
                process_single_stroke(i, co_list, mask_indices)

            # Delete old strokes
            if not self.keep_original:
                for info in stroke_info + mask_info:
                    info[3].nijigp_strokes.remove(info[0])
            context.view_layer.objects.active = current_gp_obj
            bpy.ops.object.mode_set(mode=get_obj_mode_str('EDIT'))
            
        bpy.ops.object.mode_set(mode='OBJECT')
        context.scene.frame_set(current_frame_number)
        return {'FINISHED'}
