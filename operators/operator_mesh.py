import bpy
import math
import bmesh
from ..utils import *
from ..resources import *
from mathutils import *

MAX_DEPTH = 4096

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

class MeshGenerationByNormal(bpy.types.Operator):
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
    postprocess_double_sided: bpy.props.BoolProperty(
            name='Double-Sided',
            default=True,
            description='Make the mesh symmetric to the working plane'
    )
    vertical_gap: bpy.props.FloatProperty(
            name='Vertical Gap',
            default=0.01, min=0,
            unit='LENGTH',
            description='Mininum vertical space between generated meshes'
    )    
    ignore_mode: bpy.props.EnumProperty(
            name='Ignore',
            items=[('NONE', 'None', ''),
                    ('LINE', 'All Lines', ''),
                    ('OPEN', 'All Open Lines', '')],
            default='NONE',
            description='Skip strokes without fill'
    )
    mesh_style: bpy.props.EnumProperty(
            name='Mesh Style',
            items=[('TRI', 'Delaunay Triangulation', ''),
                    ('QUAD', 'Grid', '')],
            default='TRI',
            description='Method of creating faces inside the stroke shape'
    )
    transition: bpy.props.BoolProperty(
            name='Transition',
            default=False,
            description='Add transition effects at vertices near the open ends, if there is another generated planar mesh below'
    )
    transition_length: bpy.props.FloatProperty(
            name='Transition Length',
            default=0.1, min=0.001,
            unit='LENGTH',
            description='The distance from a vertex to the open edge of the stroke, below which transparency will be applied and the normal vector will be adjusted'
    ) 
    contour_subdivision: bpy.props.IntProperty(
            name='Contour Subdivision',
            default=2, min=0, soft_max=5,
            description='Generate denser mesh near the contour for better quality shading'
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
    mesh_material: bpy.props.StringProperty(
        name='Material',
        description='The material applied to generated mesh. Principled BSDF by default',
        default='Principled BSDF',
        search=lambda self, context, edit_text: get_material_list(self.mesh_type, bpy.context.engine)
    )
    reuse_material: bpy.props.BoolProperty(
            name='Reuse Materials',
            default=True,
            description='Do not create a new material if it exists'
    )
    keep_original: bpy.props.BoolProperty(
            name='Keep Original',
            default=True,
            description='Do not delete the original stroke'
    )

    def draw(self, context):
        layout = self.layout
        layout.label(text = "Multi-Object Alignment:")
        box1 = layout.box()
        box1.prop(self, "mesh_type")
        box1.prop(self, "vertical_gap", text = "Vertical Gap")
        if self.mesh_type == 'NORMAL':
            row = box1.row()
            row.prop(self, "transition")
            row.prop(self, "transition_length", text = "Length")
        else:
            box1.prop(self, "postprocess_double_sided")
        box1.prop(self, "ignore_mode")

        layout.label(text = "Geometry Options:")
        box2 = layout.box()
        box2.label(text = "Mesh Style:")
        box2.prop(self, "mesh_style", text = "")   
        if self.mesh_style == 'TRI':
            box2.prop(self, "contour_subdivision", text = "Contour Subdivision")
        elif self.mesh_style == 'QUAD':
            box2.prop(self, "contour_trim", text = "Contour Trim")
        box2.prop(self, "resolution", text = "Resolution")
        box2.prop(self, "max_vertical_angle")
        box2.prop(self, "vertical_scale")

        layout.prop(self, "mesh_material", text='Material', icon='MATERIAL')
        layout.prop(self, "reuse_material")
        layout.prop(self, "keep_original", text = "Keep Original")

    def execute(self, context):
        import numpy as np
        try:
            import pyclipper
            import triangle as tr
        except ImportError:
            self.report({"ERROR"}, "Please install dependencies in the Preferences panel.")
            return {'FINISHED'}
        
        # Load related resources
        current_gp_obj = context.object
        generated_objects = []
        for obj in current_gp_obj.children:
            if 'nijigp_mesh' in obj:
                generated_objects.append(obj)
        mesh_material = append_material(context, self.mesh_type, self.mesh_material, self.reuse_material, self)

        # Convert selected strokes to 2D polygon point lists
        stroke_info, stroke_list = [], []
        mask_info, mask_list = [], []
        mesh_names = []
        for i,layer in enumerate(current_gp_obj.data.layers):
            if not layer.lock and not layer.hide and hasattr(layer.active_frame, "strokes"):
                for j,stroke in enumerate(layer.active_frame.strokes):
                    if stroke.select:
                        if self.ignore_mode == 'LINE' and is_stroke_line(stroke, current_gp_obj):
                            continue
                        if self.ignore_mode == 'OPEN' and is_stroke_line(stroke, current_gp_obj) and not stroke.use_cyclic:
                            continue
                        if is_stroke_hole(stroke, current_gp_obj):
                            mask_info.append([stroke, i, j])
                            mask_list.append(stroke)
                        else:
                            stroke_info.append([stroke, i, j])
                            stroke_list.append(stroke)
                            mesh_names.append('Planar_' + layer.info + '_' + str(j))
        poly_list, scale_factor = stroke_to_poly(stroke_list, scale=True, correct_orientation=True)
        mask_poly_list, _ = stroke_to_poly(mask_list, scale=True, correct_orientation=True, scale_factor=scale_factor)

        # Holes should have an opposite direction, and they need a coordinate for triangle input
        mask_hole_points = []
        for mask_co_list in mask_poly_list:
            mask_co_list.reverse()
            mask_hole_points.append(get_an_inside_co(mask_co_list))

        def process_single_stroke(i, co_list, mask_indices = []):
            """
            1. Calculate the normal vectors of the stroke's points
            2. Generate vertices and faces inside the stroke
            3. Interpolate normal vectors of inner vertices
            """

            # Initialize the mesh to be generated
            new_mesh = bpy.data.meshes.new(mesh_names[i])
            bm = bmesh.new()
            vertex_color_layer = bm.verts.layers.color.new('Color')
            normal_map_layer = bm.verts.layers.float_vector.new('NormalMap')
            depth_layer = bm.verts.layers.float.new('Depth')
            uv_layer = bm.loops.layers.uv.new()

            # Initialize the mask information
            local_mask_polys, local_mask_list, hole_points = [], [], []
            for idx in mask_indices:
                local_mask_polys.append(np.array(mask_poly_list[idx]))
                local_mask_list.append(mask_list[idx])
                if mask_hole_points[idx]:
                    hole_points.append(mask_hole_points[idx])

            # Calculate the normal vectors of the original stroke points
            # Map for fast lookup; arrays for inner-production of weighted sum
            contour_normal_map = {}
            contour_normal_array = []
            contour_co_array = []

            for poly in [co_list]+local_mask_polys:
                for j,co in enumerate(poly):
                    contour_co_array.append(co)
                    _co = poly[j-1]
                    norm = Vector([co[1]-_co[1], 0, co[0]-_co[0]]).normalized()
                    norm = norm * math.sin(self.max_vertical_angle) + Vector((0, math.cos(self.max_vertical_angle), 0))
                    contour_normal_array.append(norm)
                    contour_normal_map[(int(co[0]),int(co[1]))] = norm
            contour_normal_array = np.array(contour_normal_array)
            contour_co_array = np.array(contour_co_array)

            co_list = np.array(co_list)
            u_min, u_max = np.min(co_list[:,0]), np.max(co_list[:,0])
            v_min, v_max = np.min(co_list[:,1]), np.max(co_list[:,1])
            
            # Generate vertices and faces in BMesh
            # Method 1: use Knife Project with a 2D grid
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
                grid_loc = vec2_to_vec3([(u_max+u_min)/2, (v_max+v_min)/2],0,scale_factor)
                bpy.ops.mesh.primitive_grid_add(x_subdivisions=self.resolution,
                                                y_subdivisions=self.resolution,
                                                size= grid_size + margin_size,  
                                                align='WORLD',
                                                location=grid_loc, 
                                                rotation= [(bpy.context.scene.nijigp_working_plane == 'X-Z')*math.pi/2,
                                                            (bpy.context.scene.nijigp_working_plane == 'Y-Z')*math.pi/2,0],
                                                scale=(1, 1, 1))
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

                # Trim the boundary in BMesh
                if self.contour_trim:
                    to_trim = []
                    for vert in bm.verts:
                        if vert.is_boundary and len(vert.link_edges)==2:
                            to_trim.append(vert)
                    bmesh.ops.dissolve_verts(bm, verts=to_trim)

            # Method 2: use the triangle library
            elif self.mesh_style=='TRI':
                segs = []
                verts= []
                num_verts = 0
                offset_size = -min((u_max-u_min),(v_max-v_min))/self.resolution

                # Preprocessing the polygon using Clipper
                # The triangle library may crash in several cases, which should be avoided with every effort
                # https://www.cs.cmu.edu/~quake/triangle.trouble.html
                clipper = pyclipper.PyclipperOffset()
                clipper.AddPaths([co_list]+local_mask_polys, join_type = pyclipper.JT_ROUND, end_type = pyclipper.ET_CLOSEDPOLYGON)
                for k in range(self.contour_subdivision+1):
                    poly_results = clipper.Execute(offset_size * k/max(1, self.contour_subdivision))
                    for poly_result in poly_results:
                        for j,co in enumerate(poly_result):
                            verts.append(co)
                            segs.append( [j + num_verts, (j+1)%len(poly_result) + num_verts] )
                        num_verts = len(verts)

                # Refer to: https://rufat.be/triangle/API.html
                tr_input = dict(vertices = verts, segments = np.array(segs))
                if len(hole_points)>0:
                    tr_input['holes']=hole_points
                area_limit = (u_max-u_min)*(v_max-v_min)/(self.resolution**2)
                tr_output = tr.triangulate(tr_input, 'pa'+str(area_limit))
                # Generate vertices and triangle faces
                for co in tr_output['vertices']:
                    bm.verts.new(vec2_to_vec3(co,0,scale_factor)) 
                bm.verts.ensure_lookup_table()
                for f in tr_output['triangles']:
                    v_list = (bm.verts[f[2]], bm.verts[f[1]], bm.verts[f[0]])
                    bm.faces.new(v_list)    

            # Normal and height calculation
            maxmin_dist = 0.
            depth_offset = np.cos(self.max_vertical_angle) / np.sqrt(1 + 
                                                              (self.vertical_scale**2 - 1) *
                                                              (np.sin(self.max_vertical_angle)**2))
            for vert in bm.verts:
                co_2d = vec3_to_vec2(vert.co) * scale_factor
                co_key = (int(co_2d[0]), int(co_2d[1]))
                norm = Vector((0,0,0))
                # Contour vertex case
                if co_key in contour_normal_map:
                    norm = contour_normal_map[co_key]
                # Inner vertex case
                else:
                    dist_sq = (contour_co_array[:,0]-co_2d[0])**2 + (contour_co_array[:,1]-co_2d[1])**2
                    weights = 1.0 / dist_sq
                    weights /= np.sum(weights)
                    maxmin_dist = max( np.min(dist_sq), maxmin_dist)
                    norm_u = np.dot(contour_normal_array[:,0], weights)
                    norm_v = np.dot(contour_normal_array[:,2], weights)
                    norm = Vector([norm_u, np.sqrt(max(0,math.sin(self.max_vertical_angle)**2-norm_u**2-norm_v**2)) + math.cos(self.max_vertical_angle), norm_v])
                # Scale vertical components
                norm = Vector((norm.x * self.vertical_scale, norm.y, norm.z * self.vertical_scale)).normalized()
                vert[normal_map_layer] = [ 0.5 * (norm.x + 1) , 0.5 * (norm.z + 1), 0.5 * (norm.y+1)]
                if vert.is_boundary:
                    vert[depth_layer] = 0
                else:
                    vert[depth_layer] = norm.y - depth_offset
            maxmin_dist = np.sqrt(maxmin_dist) / scale_factor

            # UV projection, required for correct tangent direction
            for face in bm.faces:
                for loop in face.loops:
                    co = vec3_to_vec2(loop.vert.co)
                    loop[uv_layer].uv = ( (co[0]*scale_factor-u_min)/(u_max-u_min),
                                           1 -(co[1]*scale_factor-v_min)/(v_max-v_min))

            # Set vertex color from the stroke's both vertex and fill colors
            fill_base_color = [1,1,1,1]
            if current_gp_obj.data.materials[stroke_list[i].material_index].grease_pencil.show_fill:
                fill_base_color[0] = current_gp_obj.data.materials[stroke_list[i].material_index].grease_pencil.fill_color[0]
                fill_base_color[1] = current_gp_obj.data.materials[stroke_list[i].material_index].grease_pencil.fill_color[1]
                fill_base_color[2] = current_gp_obj.data.materials[stroke_list[i].material_index].grease_pencil.fill_color[2]
                fill_base_color[3] = current_gp_obj.data.materials[stroke_list[i].material_index].grease_pencil.fill_color[3]
            if hasattr(stroke_list[i],'vertex_color_fill'):
                alpha = stroke_list[i].vertex_color_fill[3]
                fill_base_color[0] = fill_base_color[0] * (1-alpha) + alpha * stroke_list[i].vertex_color_fill[0]
                fill_base_color[1] = fill_base_color[1] * (1-alpha) + alpha * stroke_list[i].vertex_color_fill[1]
                fill_base_color[2] = fill_base_color[2] * (1-alpha) + alpha * stroke_list[i].vertex_color_fill[2]
            for v in bm.verts:
                v[vertex_color_layer] = [linear_to_srgb(fill_base_color[0]), linear_to_srgb(fill_base_color[1]), linear_to_srgb(fill_base_color[2]), fill_base_color[3]]

            # Determine the depth coordinate by ray-casting to every mesh generated earlier
            vertical_pos = 0
            ray_receiver = {}
            for j,obj in enumerate(generated_objects):
                for v in bm.verts:
                    ray_emitter = Vector(v.co)
                    set_depth(ray_emitter, MAX_DEPTH)
                    res, loc, norm, idx = obj.ray_cast(ray_emitter, -get_depth_direction())
                    if res:
                        depth = vec3_to_depth(loc)
                        vertical_pos = max(depth, vertical_pos)
                        if self.transition and 'NormalMap' in obj.data.attributes:
                            if v not in ray_receiver or ray_receiver[v][2]<depth:
                                ray_receiver[v] = (obj, idx, depth)
            vertical_pos += self.vertical_gap

            # Adjust transparency and normal vector values if transition is enabled
            if self.mesh_type == 'NORMAL' and self.transition and not stroke_list[i].use_cyclic:
                for v in bm.verts:
                    if v in ray_receiver:
                        # Get the normal vector of the mesh below the current one
                        receiver_obj = ray_receiver[v][0]
                        receiver_face = receiver_obj.data.polygons[ray_receiver[v][1]]
                        receiver_norm = Vector((0,0,0))
                        for vid in receiver_face.vertices:
                            receiver_norm += receiver_obj.data.attributes['NormalMap'].data[vid].vector
                        receiver_norm /= len(receiver_face.vertices)

                        # Change values based on the vertex's distance to the open edge
                        nearest_point, portion = geometry.intersect_point_line(v.co, stroke_list[i].points[0].co, stroke_list[i].points[-1].co)
                        nearest_point = stroke_list[i].points[0].co if portion<0 else stroke_list[i].points[-1].co if portion > 1 else nearest_point
                        dist = (v.co - nearest_point).length
                        weight = smoothstep(dist/self.transition_length)
                        v[normal_map_layer] = weight * v[normal_map_layer] + (1-weight) *receiver_norm
                        v[vertex_color_layer][3] = weight

            # Update vertices locations and make a new BVHTree
            depth_scale = maxmin_dist * self.vertical_scale * np.sign(self.max_vertical_angle)
            for v in bm.verts:
                if self.mesh_type == 'MESH':
                    set_depth(v, v[depth_layer]*depth_scale)
            bm.to_mesh(new_mesh)
            bm.free()

            # Object generation
            new_object = bpy.data.objects.new(mesh_names[i], new_mesh)
            new_object['nijigp_mesh'] = 'planar' if self.mesh_type=='NORMAL' else '3d'
            set_depth(new_object.location, vertical_pos)
            bpy.context.collection.objects.link(new_object)
            new_object.parent = current_gp_obj
            generated_objects.append(new_object)

            # Assign material
            if mesh_material:
                new_object.data.materials.append(mesh_material)

            # Post-processing
            # TODO: Find a better way for post-processing, especially for mirror
            bpy.ops.object.mode_set(mode='OBJECT')
            bpy.ops.object.select_all(action='DESELECT')
            new_object.select_set(True)
            context.view_layer.objects.active = new_object
            bpy.ops.object.shade_smooth(use_auto_smooth=True)
            if self.mesh_type=='MESH' and self.postprocess_double_sided:
                new_object.modifiers.new(name="nijigp_Mirror", type='MIRROR')
                new_object.modifiers["nijigp_Mirror"].use_axis[0] = (bpy.context.scene.nijigp_working_plane == 'Y-Z')
                new_object.modifiers["nijigp_Mirror"].use_axis[1] = (bpy.context.scene.nijigp_working_plane == 'X-Z')
                new_object.modifiers["nijigp_Mirror"].use_axis[2] = (bpy.context.scene.nijigp_working_plane == 'X-Y')
                bpy.ops.object.modifier_apply("EXEC_DEFAULT", modifier = "nijigp_Mirror") 
            # Apply transform, necessary because of the movement in depth
            mb = new_object.matrix_basis
            if hasattr(new_object.data, "transform"):
                new_object.data.transform(mb)
            new_object.location = Vector((0, 0, 0))           

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
            for info in stroke_info:
                layer_index = info[1]
                current_gp_obj.data.layers[layer_index].active_frame.strokes.remove(info[0])

        bpy.ops.object.select_all(action='DESELECT')
        current_gp_obj.select_set(True)
        context.view_layer.objects.active = current_gp_obj
        return {'FINISHED'}

class MeshGenerationByOffsetting(bpy.types.Operator):
    """Generate an embossed mesh by offsetting the selected strokes"""
    bl_idname = "gpencil.nijigp_mesh_generation_offset"
    bl_label = "Convert to Meshes by Offsetting"
    bl_category = 'View'
    bl_options = {'REGISTER', 'UNDO'}

    vertical_gap: bpy.props.FloatProperty(
            name='Vertical Gap',
            default=0, min=0,
            unit='LENGTH',
            description='Mininum vertical space between generated meshes'
    ) 
    ignore_mode: bpy.props.EnumProperty(
            name='Ignore',
            items=[('NONE', 'None', ''),
                    ('LINE', 'All Lines', ''),
                    ('OPEN', 'All Open Lines', '')],
            default='NONE',
            description='Skip strokes without fill'
    )
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
                    ('STEP', 'Step', '')],
            default='SPHERE',
            description='Slope shape of the generated mesh'
    )
    extrude_method: bpy.props.EnumProperty(
            name='Extrude Method',
            items=[('DEFAULT', 'Default Method', ''),
                    ('ACUTE', 'Acute-Angle Optimized', '')],
            default='DEFAULT',
            description='Method of creating side faces'
    )
    keep_original: bpy.props.BoolProperty(
            name='Keep Original',
            default=True,
            description='Do not delete the original stroke'
    )
    postprocess_double_sided: bpy.props.BoolProperty(
            name='Double-Sided',
            default=True,
            description='Make the mesh symmetric to the working plane'
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
    reuse_material: bpy.props.BoolProperty(
            name='Reuse Materials',
            default=True,
            description='Do not create a new material if it exists'
    )

    def draw(self, context):
        layout = self.layout
        layout.label(text = "Multi-Object Alignment:")
        box1 = layout.box()
        box1.prop(self, "vertical_gap", text = "Vertical Gap")
        box1.prop(self, "ignore_mode")
        layout.label(text = "Geometry Options:")
        box2 = layout.box()
        box2.prop(self, "offset_amount", text = "Offset Amount")
        box2.prop(self, "resolution", text = "Resolution")
        row = box2.row()
        row.label(text = "Corner Shape")
        row.prop(self, "corner_shape", text = "")
        row = box2.row()
        row.label(text = "Slope Style")
        row.prop(self, "slope_style", text = "")
        row = box2.row()
        row.label(text = "Extrude Method")
        row.prop(self, "extrude_method", text = "")

        layout.label(text = "Post-Processing Options:")
        box3 = layout.box()
        box3.prop(self, "postprocess_double_sided", text = "Double-Sided")  
        box3.prop(self, "postprocess_shade_smooth", text = "Shade Smooth")
        row = box3.row()
        row.prop(self, "postprocess_merge", text='Merge By')
        row.prop(self, "merge_distance", text='Distance')
        row = box3.row()
        row.prop(self, "postprocess_remesh", text='Remesh')
        row.prop(self, "remesh_voxel_size", text='Voxel Size')

        layout.prop(self, "mesh_material", text='Material', icon='MATERIAL')
        layout.prop(self, "reuse_material")
        layout.prop(self, "keep_original", text = "Keep Original")


    def execute(self, context):

        # Import and configure Clipper
        try:
            import pyclipper
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


        # Convert selected strokes to 2D polygon point lists
        current_gp_obj = context.object
        mesh_material = append_material(context, 'MESH', self.mesh_material, self.reuse_material, self)
        stroke_info = []
        stroke_list = []
        mesh_names = []
        for i,layer in enumerate(current_gp_obj.data.layers):
            if not layer.lock and not layer.hide and hasattr(layer.active_frame, "strokes"):
                for j,stroke in enumerate(layer.active_frame.strokes):
                    if stroke.select:
                        if self.ignore_mode == 'LINE' and is_stroke_line(stroke, current_gp_obj):
                            continue
                        if self.ignore_mode == 'OPEN' and is_stroke_line(stroke, current_gp_obj) and not stroke.use_cyclic:
                            continue
                        stroke_info.append([stroke, i, j])
                        stroke_list.append(stroke)
                        mesh_names.append('Offset_' + layer.info + '_' + str(j))
        poly_list, scale_factor = stroke_to_poly(stroke_list, scale = True)

        generated_objects = []
        for obj in current_gp_obj.children:
            if 'nijigp_mesh' in obj:
                generated_objects.append(obj)
        
        def process_single_stroke(i, co_list):
            '''
            Function that processes each stroke separately
            '''
            # Calculate offsets
            clipper.Clear()
            clipper.AddPath(co_list, jt, et)
            contours = []
            vert_idx_list = []
            vert_counter = 0
            offset_interval = self.offset_amount / self.resolution * scale_factor
            for j in range(self.resolution):
                clipper_res = clipper.Execute( -offset_interval * j)
                # STEP style requires duplicating each contour
                for _ in range(1 + int(self.slope_style=='STEP')):
                    new_contour = []
                    new_idx_list = []
                    for poly in clipper_res:
                        # Merge points in advance for specific methods
                        true_poly = []
                        if not self.postprocess_merge or not self.extrude_method=='ACUTE':
                            true_poly = poly
                        else:
                            for k,point in enumerate(poly):
                                if k==0:
                                    true_poly.append(point)
                                    continue
                                p0 = vec2_to_vec3(point,0,scale_factor)
                                p1 = vec2_to_vec3(true_poly[-1],0,scale_factor)
                                if (p0-p1).length > self.merge_distance:
                                    true_poly.append(point)
                            if len(true_poly)<3:
                                true_poly = poly

                        num_vert = len(true_poly)
                        new_idx_list.append( (vert_counter, vert_counter + num_vert) )
                        vert_counter += num_vert
                        new_contour.append(true_poly)
                    contours.append(new_contour)
                    vert_idx_list.append(new_idx_list)
                    

            # Mesh generation
            new_mesh = bpy.data.meshes.new(mesh_names[i])
            bm = bmesh.new()
            vertex_color_layer = bm.verts.layers.color.new('Color')
            edges_by_level = []
            verts_by_level = []
            
            for j,contour in enumerate(contours):
                edges_by_level.append([])
                verts_by_level.append([])
                edge_extruded = []

                if len(contour)==0 and self.extrude_method=='ACUTE':
                    break

                # One contour may contain more than one closed loops
                for k,poly in enumerate(contour):
                    height = abs(j * offset_interval/scale_factor)
                    if self.slope_style == 'SPHERE':
                        sphere_rad = abs(self.offset_amount)
                        height = math.sqrt(sphere_rad ** 2 - (sphere_rad - height) ** 2)
                    elif self.slope_style == 'STEP':
                        height = abs( (j+1)//2 * offset_interval/scale_factor)

                    for co in poly:
                        verts_by_level[-1].append(
                            bm.verts.new(vec2_to_vec3(co, height, scale_factor))
                            )
                    bm.verts.ensure_lookup_table()

                    # Connect same-level vertices
                    for v_idx in range(vert_idx_list[j][k][0],vert_idx_list[j][k][1] - 1):
                        edges_by_level[-1].append( bm.edges.new([bm.verts[v_idx],bm.verts[v_idx + 1]]) )
                    edges_by_level[-1].append( 
                            bm.edges.new([ bm.verts[vert_idx_list[j][k][0]], bm.verts[vert_idx_list[j][k][1] - 1] ]) 
                            )


                connect_edge_manually = False
                # STEP style only: connect extruding edges
                if self.slope_style=='STEP' and j%2 > 0:
                    for v_idx,_ in enumerate(verts_by_level[-1]):
                        edge_extruded.append(
                                bm.edges.new([verts_by_level[-1][v_idx], verts_by_level[-2][v_idx]])
                            )
                # Connect vertices from two levels
                elif j>0 and self.extrude_method=='ACUTE':
                    connect_edge_manually = True

                if connect_edge_manually:
                    # From lower level to higher level
                    kdt = kdtree.KDTree(len(verts_by_level[-1]))
                    for v_idx, v in enumerate(verts_by_level[-1]):
                        kdt.insert(v.co, v_idx)
                    kdt.balance()

                    for v in verts_by_level[-2]:
                        num_edges = len(v.link_edges)
                        vec_, idx, dist_ = kdt.find(v.co)
                        edge_extruded.append(
                            bm.edges.new([v, verts_by_level[-1][idx]])
                        )
                bm.edges.ensure_lookup_table()

                if j>0 and not connect_edge_manually:
                    if self.slope_style=='STEP' and j%2==1:
                        bmesh.ops.edgenet_fill(bm, edges= edges_by_level[-1]+edges_by_level[-2]+edge_extruded)
                    else:
                        bmesh.ops.triangle_fill(bm, use_beauty=True, edges= edges_by_level[-1]+edges_by_level[-2])
            bmesh.ops.triangle_fill(bm, use_beauty=True, edges= edges_by_level[-1])
            bm.faces.ensure_lookup_table()

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

            # Bottom large face
            if not self.postprocess_double_sided:
                bm.faces.new(verts_by_level[0])

            bmesh.ops.recalc_face_normals(bm, faces= bm.faces)

            # Post-processing: merge
            if self.postprocess_merge:
                bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=self.merge_distance)

            # Set vertex color from the stroke's both vertex and fill colors
            fill_base_color = [1,1,1,1]
            if current_gp_obj.data.materials[stroke_list[i].material_index].grease_pencil.show_fill:
                fill_base_color[0] = current_gp_obj.data.materials[stroke_list[i].material_index].grease_pencil.fill_color[0]
                fill_base_color[1] = current_gp_obj.data.materials[stroke_list[i].material_index].grease_pencil.fill_color[1]
                fill_base_color[2] = current_gp_obj.data.materials[stroke_list[i].material_index].grease_pencil.fill_color[2]
                fill_base_color[3] = current_gp_obj.data.materials[stroke_list[i].material_index].grease_pencil.fill_color[3]
            if hasattr(stroke_list[i],'vertex_color_fill'):
                alpha = stroke_list[i].vertex_color_fill[3]
                fill_base_color[0] = fill_base_color[0] * (1-alpha) + alpha * stroke_list[i].vertex_color_fill[0]
                fill_base_color[1] = fill_base_color[1] * (1-alpha) + alpha * stroke_list[i].vertex_color_fill[1]
                fill_base_color[2] = fill_base_color[2] * (1-alpha) + alpha * stroke_list[i].vertex_color_fill[2]
            for v in bm.verts:
                v[vertex_color_layer] = [linear_to_srgb(fill_base_color[0]), linear_to_srgb(fill_base_color[1]), linear_to_srgb(fill_base_color[2]), fill_base_color[3]]

            # Determine the depth coordinate by ray-casting to every mesh generated earlier
            vertical_pos = 0
            for j,obj in enumerate(generated_objects):
                for v in bm.verts:
                    ray_emitter = Vector(v.co)
                    set_depth(ray_emitter, MAX_DEPTH)
                    res, loc, norm, idx = obj.ray_cast(ray_emitter, -get_depth_direction())
                    if res:
                        vertical_pos = max(vertical_pos, vec3_to_depth(loc))
                        if obj['nijigp_mesh'] == 'planar':
                            break
            vertical_pos += self.vertical_gap

            bm.to_mesh(new_mesh)
            bm.free()

            # Object generation
            new_object = bpy.data.objects.new(mesh_names[i], new_mesh)
            new_object['nijigp_mesh'] = '3d'
            set_depth(new_object.location, vertical_pos)
            bpy.context.collection.objects.link(new_object)
            new_object.parent = current_gp_obj
            generated_objects.append(new_object)

            # Assign material
            if mesh_material:
                new_object.data.materials.append(mesh_material)

            # Create faces for manually generated edges
            context.view_layer.objects.active = new_object
            if self.extrude_method=='ACUTE':
                bpy.ops.object.mode_set(mode='EDIT')
                bpy.ops.mesh.select_all(action='SELECT')
                bpy.ops.mesh.edge_face_add()
                if self.postprocess_shade_smooth:
                    bpy.ops.mesh.faces_shade_smooth()

            # Post-processing: mirror
            bpy.ops.object.mode_set(mode='OBJECT')
            if self.postprocess_double_sided:
                new_object.modifiers.new(name="nijigp_Mirror", type='MIRROR')
                new_object.modifiers["nijigp_Mirror"].use_axis[0] = (bpy.context.scene.nijigp_working_plane == 'Y-Z')
                new_object.modifiers["nijigp_Mirror"].use_axis[1] = (bpy.context.scene.nijigp_working_plane == 'X-Z')
                new_object.modifiers["nijigp_Mirror"].use_axis[2] = (bpy.context.scene.nijigp_working_plane == 'X-Y')
                bpy.ops.object.modifier_apply("EXEC_DEFAULT", modifier = "nijigp_Mirror")
            
            # Post-processing: remesh
            if self.postprocess_remesh:
                new_object.data.remesh_voxel_size = self.remesh_voxel_size
                new_object.data.use_remesh_preserve_volume = True
                new_object.data.use_remesh_preserve_vertex_colors = True
                bpy.ops.object.voxel_remesh("EXEC_DEFAULT")

            new_object.data.use_auto_smooth = self.postprocess_shade_smooth

            # Apply transform, necessary because of the movement in depth
            mb = new_object.matrix_basis
            if hasattr(new_object.data, "transform"):
                new_object.data.transform(mb)
            new_object.location = Vector((0, 0, 0))

        for i,co_list in enumerate(poly_list):
            process_single_stroke(i, co_list)

        # Delete old strokes
        if not self.keep_original:
            for info in stroke_info:
                layer_index = info[1]
                current_gp_obj.data.layers[layer_index].active_frame.strokes.remove(info[0])

        return {'FINISHED'}

