from scipy.sparse import csr_matrix, csgraph
import numpy as np

def get_mst_longest_path_from_triangles(tr_output):
    """
    Convert triangulated strokes to a minimum spanning stree (MST), then calcualte its longest path through BFS.
    This function is a step of the line fitting operators
    """
    def e_dist(i,j):
        src = tr_output['vertices'][i]
        dst = tr_output['vertices'][j]
        res = np.sqrt((dst[0]-src[0])**2 + (dst[1]-src[1])**2)
        return max(res, 1e-9)
    
    # Graph and tree construction: vertex -> node, edge length -> link weight
    # All input/output graphs are undirected
    num_vert = len(tr_output['vertices'])
    row, col, data = [], [], []
    for f in tr_output['triangles']:
        row += [f[0], f[1], f[2]]
        col += [f[1], f[2], f[0]]
        data += [e_dist(f[0], f[1]), e_dist(f[2], f[1]), e_dist(f[0], f[2])]
    dist_graph = csr_matrix((data, (row, col)), shape=(num_vert, num_vert))
    mst = csgraph.minimum_spanning_tree(dist_graph)
    
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
    
    _, _, src_idx = tree_bfs(mst, 0)
    predecessor_map, total_length, dst_idx = tree_bfs(mst, src_idx)
    
    # Trace the map back to get the whole path
    path_whole = [dst_idx]
    predecessor = predecessor_map[dst_idx]
    while predecessor != src_idx:
        path_whole.append(predecessor)
        predecessor = predecessor_map[predecessor]
    path_whole.append(src_idx)
    
    return total_length, path_whole