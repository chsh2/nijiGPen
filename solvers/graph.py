import numpy as np
from mathutils import *
from scipy.sparse import csr_matrix, csgraph

class MstSolver:
    """
    Gets a minimum spanning tree (MST) from other types of graphs and calculates related metrics
    """
    tr_info: map
    mst: csgraph
    def mst_from_triangles(self, tr_output):
        def e_dist(i,j):
            src = tr_output['vertices'][i]
            dst = tr_output['vertices'][j]
            res = np.sqrt((dst[0]-src[0])**2 + (dst[1]-src[1])**2)
            return max(res, 1e-9)
        # Graph and tree construction: vertex -> node, edge length -> link weight
        # All input/output graphs are undirected
        self.tr_info = tr_output
        num_vert = len(tr_output['vertices'])
        row, col, data = [], [], []
        for f in tr_output['triangles']:
            row += [f[0], f[1], f[2]]
            col += [f[1], f[2], f[0]]
            data += [e_dist(f[0], f[1]), e_dist(f[2], f[1]), e_dist(f[0], f[2])]
        dist_graph = csr_matrix((data, (row, col)), shape=(num_vert, num_vert))
        self.mst = csgraph.minimum_spanning_tree(dist_graph)
        return self.mst
    
    def mst_from_voronoi(self, vor, poly=None):
        import pyclipper
        def e_dist(i,j):
            src = vor.vertices[i]
            dst = vor.vertices[j]
            res = np.sqrt((dst[0]-src[0])**2 + (dst[1]-src[1])**2)
            return max(res, 1e-9)
        # Consider Voronoi ridges with both vertices inside the polygon
        num_vert = len(vor.vertices)
        is_vertex_valid = [True] * num_vert
        if poly is not None:
            for i in range(num_vert):
                if pyclipper.PointInPolygon(vor.vertices[i], poly) < 1:
                    is_vertex_valid[i] = False
        row, col, data = [], [], []
        for i,j in vor.ridge_vertices:
            if i>=0 and j>=0 and is_vertex_valid[i] and is_vertex_valid[j]:
                row.append(i)
                col.append(j)
                data.append(e_dist(i,j))
        dist_graph = csr_matrix((data, (row, col)), shape=(num_vert, num_vert))
        self.mst = csgraph.minimum_spanning_tree(dist_graph)
        return self.mst

    def get_longest_path(self):
        """
        The longest path of MST is required by the line fitting operators
        """
        # Find two ends of the longest path in the tree by executing BFS twice
        def tree_bfs(graph, idx):
            dist_map = {idx: 0}
            predecessor_map = {idx: None}
            max_dist = 0
            farthest_idx = idx
            queue = [idx]
            pointer = 0
            while pointer < len(queue):
                node = queue[pointer]
                for next_node in graph.getcol(node).nonzero()[0]:
                        if next_node not in dist_map:
                            dist_value = dist_map[node] + graph[next_node,node]
                            if max_dist < dist_value:
                                max_dist = dist_value
                                farthest_idx = next_node
                            dist_map[next_node] = dist_value
                            predecessor_map[next_node] = node
                            queue.append(next_node)
                for next_node in graph.getrow(node).nonzero()[1]:
                        if next_node not in dist_map:
                            dist_value = dist_map[node] + graph[node,next_node]
                            if max_dist < dist_value:
                                max_dist = dist_value
                                farthest_idx = next_node
                            dist_map[next_node] = dist_value
                            predecessor_map[next_node] = node
                            queue.append(next_node)
                pointer += 1
            return predecessor_map, max_dist, farthest_idx
        
        _, _, src_idx = tree_bfs(self.mst, self.mst.nonzero()[0][0])
        predecessor_map, total_length, dst_idx = tree_bfs(self.mst, src_idx)
        
        # Trace the map back to get the whole path
        path_whole = [dst_idx]
        predecessor = predecessor_map[dst_idx]
        while predecessor != src_idx and predecessor in predecessor_map:
            path_whole.append(predecessor)
            predecessor = predecessor_map[predecessor]
        path_whole.append(src_idx)
        
        return total_length, path_whole

class SmartFillSolver:
    tr_map: dict
    graph: csr_matrix
    solid_edges: set                # Edges from original strokes are treated as "walls" that prevent nodes from connecting to each other
    labels: np.ndarray              # Labels of each triangle (node) to mark different colors
    tr_center_kdt: kdtree.KDTree    # KDTree based on the center coordinate of each triangle
    link_ends_map: dict             # Mapping a triangle pair to a vertex pair
        
    def build_graph(self, tr_map):
        """
        Build a graph data structure in the following way:
            triangle -> node, edge shared by two triangles -> node link, edge length -> link weight
        Also, label triangles with bounary edges as transparent
        """

        def e_key(e):
            return (min(e[0],e[1]), max(e[0],e[1]))
        def e_weight(e):
            src = tr_map['vertices'][e[0]]
            dst = tr_map['vertices'][e[1]]
            res = np.sqrt((dst[0]-src[0])**2 + (dst[1]-src[1])**2)
            return max(res, 1e-9)
    
        num_nodes = len(tr_map['triangles'])
        edge_map = {}           # Triangles that shares an edge
        self.solid_edges = set()     
        self.tr_center_kdt = kdtree.KDTree(num_nodes)
        self.link_ends_map = {}
        
        # Put edge and triangle information into maps
        for i,edge in enumerate(tr_map['segments']):
            if len(tr_map['orig_edges'][i]) > 0:
                self.solid_edges.add(e_key(edge))
        for i,tri in enumerate(tr_map['triangles']):
            for edge in (e_key((tri[0],tri[1])), e_key((tri[0],tri[2])), e_key((tri[2],tri[1]))):
                if edge not in edge_map:
                    edge_map[edge] = []
                edge_map[edge].append(i)
            self.tr_center_kdt.insert( ( (tr_map['vertices'][tri[0]][0]+tr_map['vertices'][tri[1]][0]+tr_map['vertices'][tri[2]][0])/3.0
                                      ,  (tr_map['vertices'][tri[0]][1]+tr_map['vertices'][tri[1]][1]+tr_map['vertices'][tri[2]][1])/3.0, 0)
                                      , i)
        self.tr_center_kdt.balance()
        
        # Prepare data for sparse graph (bidirectional)
        row, col, data = [], [], []
        for edge in edge_map:
            if len(edge_map[edge]) > 1:
                self.link_ends_map[e_key((edge_map[edge][0], edge_map[edge][1]))] = edge
                row += [edge_map[edge][0], edge_map[edge][1]]
                col += [edge_map[edge][1], edge_map[edge][0]]
                data += ([e_weight(edge)] * 2) if edge not in self.solid_edges else [1e-9, 1e-9]
        self.tr_map = tr_map
        self.graph = csr_matrix((data, (row, col)), shape=(num_nodes, num_nodes))
        
        # Label boundary nodes as zeros
        self.labels = -np.ones(num_nodes).astype('int32')
        for edge in edge_map:
            if len(edge_map[edge]) == 1:
                self.labels[edge_map[edge][0]] = 0
                
    def set_labels_from_points(self, points_co, points_label):
        """
        Label each node(triangle) according to input points inside it
        """
        import pyclipper
        def point_in_triangle(point_co, tr_idx):
            poly = [self.tr_map['vertices'][i] for i in self.tr_map['triangles'][tr_idx]]
            return (pyclipper.PointInPolygon(point_co, poly) != 0)
        # TODO: The higher bound of time complexity is linear. May need better searching strategy.
        # Approximate search using KDTree
        if len(points_co)>0:
            _, current_tr, _ = self.tr_center_kdt.find((points_co[0][0], points_co[0][1], 0))
        # Exact search using BFS
        for i, co in enumerate(points_co):
            search_queue = [current_tr]
            pointer = 0
            searched = {current_tr}
            while pointer < len(search_queue):
                current_tr = search_queue[pointer]
                # To speed up, label each triangle only once
                if point_in_triangle(co, current_tr):
                    if self.labels[current_tr] < 0:
                        self.labels[current_tr] = points_label[i]
                    break
                # Move to the next triangle
                for next_tr in self.graph.getrow(current_tr).nonzero()[1]:
                    if next_tr not in searched:
                        search_queue.append(next_tr)
                        searched.add(next_tr)
                pointer += 1

    def propagate_labels(self):
        """
        Fill unlabelled triangles with known labels, using max-flow min-cut
        """
        num_nodes = self.graph.shape[0]
        graph_links = self.graph.nonzero()
        max_capacity = 65535     # A constant larger than any link weight
        source = num_nodes
        sink = num_nodes + 1
        
        # Perform max-flow once for every color
        for current_label in range(max(self.labels), 0, -1):
            # Build a new graph excluding the labels that have been processed
            row, col, data = [], [], []
            for i in range(len(graph_links[0])):
                n0, n1 = graph_links[0][i], graph_links[1][i]
                if self.labels[n0] <= current_label and self.labels[n1] <= current_label:
                    row.append(n0)
                    col.append(n1)
                    data.append(int(self.graph[n0,n1]))
            # Add two more nodes, source (-2) and sink (-1)
            for i in range(num_nodes):
                if self.labels[i] == current_label:
                    row += [i, source]
                    col += [source, i]
                    data += [max_capacity, max_capacity]
                elif self.labels[i] > -1 and self.labels[i] < current_label:
                    row += [i, sink]
                    col += [sink, i]
                    data += [max_capacity, max_capacity]
                
            # Set labels based on the residual graph of max-flow
            tmp_graph = csr_matrix((data, (row, col)), shape=(num_nodes+2, num_nodes+2))
            res = csgraph.maximum_flow(tmp_graph, source, sink)
            if hasattr(res, 'flow'):    # depending on Scipy versions
                cut_graph = tmp_graph - res.flow
            else:
                cut_graph = tmp_graph - res.residual
            source_nodes = csgraph.depth_first_order(cut_graph, source)[0]
            for node in source_nodes:
                if node < len(self.labels) and self.labels[node] == -1:
                    self.labels[node] = current_label

    def complete_labels(self):
        """
        Assign labels of all unlabelled (-1) nodes according to their neighbors
        """
        num_nodes = len(self.tr_map['triangles'])
        graph_links = self.graph.nonzero()
        
        # Get all connected unlabelled components
        row, col = [], []
        for i in range(len(graph_links[0])):
            n0, n1 = graph_links[0][i], graph_links[1][i] 
            if self.labels[n0] == -1 and self.labels[n1] == -1:
                row.append(n0)
                col.append(n1)
        unlabelled_graph = csr_matrix((np.ones(len(row)), (row, col)), shape=(num_nodes, num_nodes))
        cnum, components = csgraph.connected_components(unlabelled_graph, directed=False)
        
        # Check neighbor labels of these unlabelled components
        neighbor_label_weights = np.zeros((cnum,max(self.labels)+1))
        for i in range(len(graph_links[0])):
            n0, n1 = graph_links[0][i], graph_links[1][i] 
            if self.labels[n0] > -1 and self.labels[n1] == -1:
                neighbor_label_weights[components[n1]][self.labels[n0]] += self.graph[n0,n1]
        components_new_label = np.argmax(neighbor_label_weights, axis=1)
        
        # Assign a new label to each component
        for n,c in enumerate(components):
            if self.labels[n] == -1:
                self.labels[n] = components_new_label[c]
            
    def get_contours(self): 
        """
        Get the outer contour of each connected component of the graph
        """
        import pyclipper
        clipper = pyclipper.PyclipperOffset()
        def e_key(e):
            return (min(e[0],e[1]), max(e[0],e[1]))

        # Find links connecting two nodes with different labels
        # Remove them from the original graph
        num_nodes = len(self.tr_map['triangles'])
        contour_links = set()
        graph_links = self.graph.nonzero()
        row, col = [], []
        for i in range(len(graph_links[0])):
            n0, n1 = graph_links[0][i], graph_links[1][i]
            if self.labels[n0] != self.labels[n1]:
                contour_links.add(e_key((n0,n1)))
            else:
                row.append(n0)
                col.append(n1)
        segmented_graph = csr_matrix((np.ones(len(row)), (row, col)), (num_nodes,num_nodes))
        cnum, components = csgraph.connected_components(segmented_graph, directed=False)
        
        # Identify the color label of each segmented component
        components_label = np.zeros(cnum).astype('int32')
        for n,c in enumerate(components):
            components_label[c] = self.labels[n]
        # Categorize contour links based on the two components they belong to
        contour_vertices = []
        for _ in range(cnum):
            contour_vertices.append({'row':[], 'col':[]})
        for link in contour_links:
            n0, n1 = link[0], link[1]
            for c in (components[n0], components[n1]):
                if components_label[c] > 0:
                    v0, v1 = self.link_ends_map[e_key((n0,n1))]
                    contour_vertices[c]['row'].append(v0)
                    contour_vertices[c]['col'].append(v1)
                    
        # Process the contour of each segmented component
        contours_co = []
        num_verts = len(self.tr_map['vertices'])
        for c in range(cnum):
            contours_co.append([])
            if components_label[c] < 1:
                continue
            
            contour_graph = csr_matrix((np.ones(len(contour_vertices[c]['row'])),
                                        (contour_vertices[c]['row'], contour_vertices[c]['col'])),
                                        shape = (num_verts,num_verts))
            sub_cnum, sub_components = csgraph.connected_components(contour_graph, directed=False)
            component_map = {}
            for v,i in enumerate(sub_components):
                if i not in component_map:
                    component_map[i] = [0, v]
                component_map[i][0] += 1
            
            # Each connected subgraph should have a ring topology. Therefore, the contour can be found through DFS.
            contour_candidates = []
            for i in range(sub_cnum):
                if component_map[i][0] > 2:      # A contour should have at least 3 points
                    vseq = csgraph.depth_first_order(contour_graph, component_map[i][1], 
                                              directed=False, return_predecessors=False)
                    co_seq = [self.tr_map['vertices'][v] for v in vseq]
                    # Eliminate self overlaps
                    clipper.Clear()
                    clipper.AddPath(co_seq, join_type = pyclipper.JT_ROUND, end_type = pyclipper.ET_CLOSEDPOLYGON)
                    contour_candidates += clipper.Execute(0)
            
            # Choose only the outer contours as the output
            for i,sc1 in enumerate(contour_candidates):
                for j,sc2 in enumerate(contour_candidates):
                    if (i!=j and
                        pyclipper.PointInPolygon(sc1[0], sc2) != 0 and
                        pyclipper.PointInPolygon(sc1[-1], sc2) != 0):
                            break
                else:
                    contours_co[c].append(sc1)
        
        return contours_co, components_label
