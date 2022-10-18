import bpy
import os
import math
import bmesh
from .utils import *
from mathutils import *

class MeshGenerationByNormal(bpy.types.Operator):
    """Generate a planar mesh with an interpolated normal map calculated from the selected strokes"""
    bl_idname = "nijigp.mesh_generation_normal"
    bl_label = "Convert to Meshes by Normal Interpolation"
    bl_category = 'View'
    bl_options = {'REGISTER', 'UNDO'}

    vertical_gap: bpy.props.FloatProperty(
            name='Resolution',
            default=0.05, min=0,
            unit='LENGTH',
            description='Mininum vertical space between generated meshes'
    )    
    mesh_style: bpy.props.EnumProperty(
            name='Mesh Style',
            items=[('TRI', 'Delaunay Triangulation', ''),
                    ('QUAD', 'Grid', '')],
            default='TRI',
            description='Method of creating faces inside the stroke shape'
    )
    resolution: bpy.props.IntProperty(
            name='Resolution',
            default=20, min=2, soft_max=100,
            description='Relative dimension of polygons of the generated mesh'
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
        box1.prop(self, "vertical_gap", text = "Vertical Gap")
        layout.label(text = "Geometry Options:")
        box2 = layout.box()
        box2.label(text = "Mesh Style:")
        box2.prop(self, "mesh_style", text = "")        
        box2.prop(self, "resolution", text = "Resolution")
        box2.prop(self, "keep_original", text = "Keep Original")

    def execute(self, context):
        import numpy as np
        try:
            import pyclipper
            import triangle as tr
        except ImportError:
            self.report({"ERROR"}, "Please install dependencies in the Preferences panel.")
        
        # Preprocess using Offset operator
        # The triangle library may crash in several cases, which should be avoided with every effort
        # https://www.cs.cmu.edu/~quake/triangle.trouble.html
        if self.mesh_style == 'TRI':
            if context.object.data.use_multiedit:
                context.object.data.use_multiedit = False
                bpy.ops.nijigp.offset_selected()
                context.object.data.use_multiedit = True
            else:
                bpy.ops.nijigp.offset_selected()

        # Convert selected strokes to 2D polygon point lists
        current_gp_obj = context.object
        stroke_info = []
        stroke_list = []
        mesh_names = []
        for i,layer in enumerate(current_gp_obj.data.layers):
            if not layer.lock and hasattr(layer.active_frame, "strokes"):
                for j,stroke in enumerate(layer.active_frame.strokes):
                    if stroke.select:
                        stroke_info.append([stroke, i, j])
                        stroke_list.append(stroke)
                        mesh_names.append('Planar_' + layer.info + '_' + str(j))
        poly_list, scale_factor = stroke_to_poly(stroke_list, scale = True)

        # Use Clipper to determine the orientation of strokes to ensure the consistency of normal vector calculation
        for co_list in poly_list:
            if not pyclipper.Orientation(co_list):
                co_list.reverse()

        def find_loop_seq(edges, src_vert, dst_vert):
            """
            Given an edge loop, return a list of vertices in the sequence of connection
            """
            # Initialization: search in two directions
            edge_set = set(edges)
            e1 = None
            e2 = None
            v1 = src_vert
            v2 = src_vert
            p1 = [v1]
            p2 = [v2]
            for e in src_vert.link_edges:
                if e in edge_set and not e1:
                    e1 = e
                elif e in edge_set and not e2:
                    e2 = e
                if e2:
                    break
            if not e1 or not e2:
                return [src_vert, dst_vert]
            def get_another_vert(e, v):
                return e.other_vert(v)
            def get_another_edge(e, v):
                if len(v.link_edges)==0:
                    return None
                for edge in v.link_edges:
                    if edge != e and edge in edge_set:
                        return edge 
            
            # BFS
            while True:
                v1 = get_another_vert(e1, v1)
                v2 = get_another_vert(e2, v2)
                if not v1 or not v2:
                    return [src_vert, dst_vert]
                p1.append(v1)
                p2.append(v2)
                if v1==dst_vert:
                    return p1
                if v2==dst_vert:
                    return p2
                if v1==src_vert or v2==src_vert:
                    return [src_vert, dst_vert]
                e1 = get_another_edge(e1, v1)
                e2 = get_another_edge(e2, v2)
                if not e1 or not e2:
                    return [src_vert, dst_vert]

        generated_objects = []
        vertical_pos_list = []
        def process_single_stroke(i, co_list):
            """
            Mesh is generated with following 3 types of vertices:
                Contour: the original points in the stroke
                Inner: a 2D grid filling the shape
                Border: a subset of inner vertices connecting directly to the contour
            """

            # Initialize the mesh to be generated
            new_mesh = bpy.data.meshes.new(mesh_names[i])
            bm = bmesh.new()
            vertex_color_layer = bm.verts.layers.color.new('Color')
            normal_map_layer = bm.verts.layers.float_vector.new('NormalMap')
            uv_layer = bm.loops.layers.uv.new()

            co_list = np.array(co_list)
            u_min, u_max = np.min(co_list[:,0]), np.max(co_list[:,0])
            v_min, v_max = np.min(co_list[:,1]), np.max(co_list[:,1])
            contour = []
            inner = []
            # Method 1: generate a grid inside the stroke
            if self.mesh_style == 'QUAD':
                # Generate a 2D grid filling the shape
                inner_verts_co = np.zeros((self.resolution-1, self.resolution-1, 2))
                inner_verts_co[:,:,0] =  np.linspace(u_min, u_max, num=self.resolution, endpoint=False)[1:]
                inner_verts_co[:,:,1] =  np.linspace(v_min, v_max, num=self.resolution, endpoint=False)[1:,None]

                # Exclude points outside the shape
                inner_verts = []
                for u in range(self.resolution-1):
                    verts_row = []
                    for v in range(self.resolution-1):
                        crossing_number = crossing_number_2d_up(stroke_list[i], inner_verts_co[u,v,0], inner_verts_co[u,v,1], scale_factor)
                        if crossing_number%2 == 1:
                            verts_row.append( bm.verts.new(vec2_to_vec3(inner_verts_co[u,v],0,scale_factor)) )
                        else:
                            verts_row.append(None)
                    inner_verts.append(verts_row)

                # Connect inner edges and faces       
                for u,row in enumerate(inner_verts):
                    for v,vert in enumerate(row):
                        if u>0 and inner_verts[u][v] and inner_verts[u-1][v]:
                            bm.edges.new([inner_verts[u][v], inner_verts[u-1][v]])
                        if v>0 and inner_verts[u][v] and inner_verts[u][v-1]:
                            bm.edges.new([inner_verts[u][v], inner_verts[u][v-1]])        
                bmesh.ops.holes_fill(bm, edges = bm.edges, sides=4)

                # Remove isolated vertices
                while True:
                    to_remove = []
                    for vert in bm.verts:
                        if len(vert.link_faces)<1:
                            to_remove.append(vert)
                        elif len(vert.link_faces)+1 < len(vert.link_edges):
                            to_remove.append(vert)
                    for vert in to_remove:
                        bm.verts.remove(vert)
                    if len(to_remove)==0:
                        break
                inner = list(bm.verts)   

                # Mark the border vertices and edges
                if len(inner)>0:
                    border_edges = []
                    border_verts = []
                    for vert in bm.verts:
                        if len(vert.link_faces)<4:
                            border_verts.append(vert)
                    for edge in bm.edges:
                        if len(edge.link_faces)<2:
                            border_edges.append(edge)

                    kd = kdtree.KDTree(len(border_verts))
                    for j,vert in enumerate(border_verts):
                        kd.insert(vert.co, j)
                    kd.balance()

                # Add coutour vertices and edges
                contour_edges = []
                for j,co in enumerate(co_list):
                    contour.append(bm.verts.new( vec2_to_vec3(co,0,scale_factor) ))
                for j,vert in enumerate(contour):
                    contour_edges.append(bm.edges.new([vert, contour[j-1]]) )
            
                # Interconnect contour and inner vertices
                if len(inner)>0:
                    inter_verts = []
                    for vert in contour:
                        loc, idx, dist = kd.find(vert.co)
                        inter_verts.append( border_verts[idx] )     
                    border_faces = []
                    for j,vert in enumerate(contour):
                        if inter_verts[j-1] == inter_verts[j]:
                            border_faces.append( bm.faces.new([contour[j], contour[j-1], inter_verts[j]]) )
                        else:
                            border_faces.append( bm.faces.new([contour[j], contour[j-1]] + 
                                            find_loop_seq(border_edges, inter_verts[j-1], inter_verts[j]) ) )
                    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)  
                else:
                    bm.faces.new(contour)
            
            # Method 2: use the triangle library
            elif self.mesh_style=='TRI':
                segs = []
                for j,co in enumerate(co_list):
                    segs.append( [j, (j+1)%len(co_list)] )

                # Refer to: https://rufat.be/triangle/API.html
                tr_input = dict(vertices = co_list, segments = np.array(segs))
                area_limit = (u_max-u_min)*(v_max-v_min)/(self.resolution**2)
                tr_output = tr.triangulate(tr_input, 'pqa'+str(area_limit))
                # Generate vertices and triangle faces
                for co in tr_output['vertices']:
                    bm.verts.new(vec2_to_vec3(co,0,scale_factor)) 
                bm.verts.ensure_lookup_table()
                for f in tr_output['triangles']:
                    v_list = (bm.verts[f[2]], bm.verts[f[1]], bm.verts[f[0]])
                    bm.faces.new(v_list)    

                # Identify contour vertices from inner vertices
                src_v = None
                contour_set = set()
                contour_edges = set()
                for vert in bm.verts:
                    if vert.is_boundary:
                        contour_set.add(vert)
                        if not src_v:
                            src_v = vert
                    else:
                        inner.append(vert)
                for edge in bm.edges:
                    if edge.is_boundary:
                        contour_edges.add(edge)
                # Sort the contour vertices
                contour = find_loop_seq(contour_edges, src_v, src_v) 
                # Keep the direction consistent
                tmp_co_list = []
                for vert in contour:
                    tmp_co_list.append(vec3_to_vec2(vert.co)*scale_factor)
                if not pyclipper.Orientation(tmp_co_list):
                    contour.reverse()

            # Normal map calculation from the stroke tangent
            norm_u_ref = []
            norm_v_ref = []
            u_ref = []
            v_ref = []
            for j,vert in enumerate(contour):
                norm = Vector([vec3_to_vec2(contour[j].co)[1] -  vec3_to_vec2(contour[j-1].co)[1]
                                , 0
                                , vec3_to_vec2(contour[j].co)[0] -  vec3_to_vec2(contour[j-1].co)[0]]).normalized()
                vert[normal_map_layer] = [ 0.5 * (norm.x + 1), 0.5 * (norm.z + 1), 0.5]
                norm_u_ref.append(norm.x)
                norm_v_ref.append(norm.z)
                u_ref.append(vec3_to_vec2(contour[j].co)[0])
                v_ref.append(vec3_to_vec2(contour[j].co)[1])
            norm_u_ref=np.array(norm_u_ref)
            norm_v_ref=np.array(norm_v_ref)
            u_ref=np.array(u_ref)
            v_ref=np.array(v_ref)

            # Normal map interpolation for inner points
            for vert in inner:
                weights = 1 / ((u_ref - vec3_to_vec2(vert.co)[0])**2 + (v_ref - vec3_to_vec2(vert.co)[1])**2)
                weights /= np.sum(weights)
                norm_u = np.dot(norm_u_ref, weights)
                norm_v = np.dot(norm_v_ref, weights)
                norm = Vector([norm_u, np.sqrt(1-norm_u**2-norm_v**2), norm_v])
                vert[normal_map_layer] = [ 0.5 * (norm.x + 1) , 0.5 * (norm.z + 1), 0.5 * (norm.y+1)]

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
                # Not supported until Blender 3.2; Currently, using a homemade function instead
                #vertex_color = Color([fill_base_color[0], fill_base_color[1], fill_base_color[2]])
                #vertex_color = vertex_color.from_scene_linear_to_srgb()
                #v[vertex_color_layer] = [vertex_color.r, vertex_color.g, vertex_color.b, fill_base_color[3]]
                v[vertex_color_layer] = [linear_to_srgb(fill_base_color[0]), linear_to_srgb(fill_base_color[1]), linear_to_srgb(fill_base_color[2]), fill_base_color[3]]

            # Determine the depth coordinate by ray-casting to every mesh generated earlier
            vertical_pos = 0
            for j,obj in enumerate(generated_objects):
                for v in bm.verts:
                    res, loc, norm, idx = obj.ray_cast(v.co, get_depth_direction())
                    if res:
                        vertical_pos = max(vertical_pos, vertical_pos_list[j])
                        break
            vertical_pos += self.vertical_gap

            # Update vertices locations and make a new BVHTree
            for v in bm.verts:
                set_depth(v, vertical_pos)
            vertical_pos_list.append(vertical_pos)

            bm.to_mesh(new_mesh)
            bm.free()

            # Object generation
            new_object = bpy.data.objects.new(mesh_names[i], new_mesh)
            bpy.context.collection.objects.link(new_object)
            new_object.parent = current_gp_obj
            generated_objects.append(new_object)

            # Assign material
            if "nijigp_mat_with_normal" not in bpy.data.materials:
                new_mat = bpy.data.materials.new("nijigp_mat_with_normal")
                new_mat.use_nodes = True
                attr_node = new_mat.node_tree.nodes.new("ShaderNodeAttribute")
                attr_node.attribute_name = 'Color'
                normal_attr_node = new_mat.node_tree.nodes.new("ShaderNodeAttribute")
                normal_attr_node.attribute_name = 'NormalMap'
                normal_map_node = new_mat.node_tree.nodes.new("ShaderNodeNormalMap")
                new_mat.node_tree.links.new(normal_attr_node.outputs['Vector'], normal_map_node.inputs['Color'])
                for node in new_mat.node_tree.nodes:
                    if node.type == "BSDF_PRINCIPLED":
                        new_mat.node_tree.links.new(node.inputs['Base Color'], attr_node.outputs['Color'])
                        new_mat.node_tree.links.new(node.inputs['Alpha'], attr_node.outputs['Alpha'])
                        new_mat.node_tree.links.new(node.inputs['Normal'], normal_map_node.outputs['Normal'])
            new_object.data.materials.append(bpy.data.materials["nijigp_mat_with_normal"])

        for i,co_list in enumerate(poly_list):
            process_single_stroke(i, co_list)

        # Delete old strokes
        if not self.keep_original:
            for info in stroke_info:
                layer_index = info[1]
                current_gp_obj.data.layers[layer_index].active_frame.strokes.remove(info[0])

        bpy.ops.object.mode_set(mode='OBJECT')
        return {'FINISHED'}

class MeshGenerationByOffsetting(bpy.types.Operator):
    """Generate an embossed mesh by offsetting the selected strokes"""
    bl_idname = "nijigp.mesh_generation_offset"
    bl_label = "Convert to Meshes by Offsetting"
    bl_category = 'View'
    bl_options = {'REGISTER', 'UNDO'}

    # Define properties
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


    def draw(self, context):
        layout = self.layout
        layout.label(text = "Geometry Options:")
        box1 = layout.box()
        box1.prop(self, "offset_amount", text = "Offset Amount")
        box1.prop(self, "resolution", text = "Resolution")
        box1.label(text = "Corner Shape")
        box1.prop(self, "corner_shape", text = "")
        box1.label(text = "Slope Style")
        box1.prop(self, "slope_style", text = "")
        box1.label(text = "Extrude Method")
        box1.prop(self, "extrude_method", text = "")
        box1.prop(self, "keep_original", text = "Keep Original")

        layout.label(text = "Post-Processing Options:")
        box2 = layout.box()
        box2.prop(self, "postprocess_double_sided", text = "Double-Sided")  
        box2.prop(self, "postprocess_shade_smooth", text = "Shade Smooth")
        row = box2.row()
        row.prop(self, "postprocess_merge", text='Merge By')
        row.prop(self, "merge_distance", text='Distance')
        row = box2.row()
        row.prop(self, "postprocess_remesh", text='Remesh')
        row.prop(self, "remesh_voxel_size", text='Voxel Size')

    def execute(self, context):

        # Import and configure Clipper
        try:
            import pyclipper
        except ImportError:
            self.report({"ERROR"}, "Please install dependencies in the Preferences panel.")
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
        stroke_info = []
        stroke_list = []
        mesh_names = []
        for i,layer in enumerate(current_gp_obj.data.layers):
            if not layer.lock and hasattr(layer.active_frame, "strokes"):
                for j,stroke in enumerate(layer.active_frame.strokes):
                    if stroke.select:
                        stroke_info.append([stroke, i, j])
                        stroke_list.append(stroke)
                        mesh_names.append('Offset_' + layer.info + '_' + str(j))
        poly_list, scale_factor = stroke_to_poly(stroke_list, scale = True)

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
                new_contour = clipper.Execute( -offset_interval * j)
                # STEP style requires duplicating each contour
                for _ in range(1 + int(self.slope_style=='STEP')):
                    contours.append( new_contour )
                    new_idx_list = []
                    for poly in new_contour:
                        num_vert = len(poly)
                        new_idx_list.append( (vert_counter, vert_counter + num_vert) )
                        vert_counter += num_vert
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

            bm.to_mesh(new_mesh)
            bm.free()

            # Object generation
            new_object = bpy.data.objects.new(mesh_names[i], new_mesh)
            bpy.context.collection.objects.link(new_object)
            new_object.parent = current_gp_obj

            # Assign material
            if "nijigp_mat" not in bpy.data.materials:
                new_mat = bpy.data.materials.new("nijigp_mat")
                new_mat.use_nodes = True
                attr_node = new_mat.node_tree.nodes.new("ShaderNodeAttribute")
                attr_node.attribute_name = 'Color'
                for node in new_mat.node_tree.nodes:
                    if node.type == "BSDF_PRINCIPLED":
                        new_mat.node_tree.links.new(node.inputs['Base Color'], attr_node.outputs['Color'])
                        new_mat.node_tree.links.new(node.inputs['Alpha'], attr_node.outputs['Alpha'])
            new_object.data.materials.append(bpy.data.materials["nijigp_mat"])

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

        for i,co_list in enumerate(poly_list):
            process_single_stroke(i, co_list)

        # Delete old strokes
        if not self.keep_original:
            for info in stroke_info:
                layer_index = info[1]
                current_gp_obj.data.layers[layer_index].active_frame.strokes.remove(info[0])

        return {'FINISHED'}

