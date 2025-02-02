import numpy as np

"""
Variants based on skimage functions to better support the requirements of the add-on, mainly image vectorization operators.
"""

marching_map = {0: (0, 1), 1: (-1, 0), 2: (0, 1), 3: (0, 1),
                4: (0, -1), 5: (-1, 0), 6: (0, 1), 7: (0, 1),
                8: (1, 0), 9: (1, 0), 10: (1, 0), 11: (1, 0),
                12: (0, -1), 13: (-1, 0), 14: (0, -1), 15: (None, None)}

def marching_squares(padded_mat, start_pos, target_label):
    """
    Variant of marching squares algorithm that identifies the shared edges in a contour
    """
    def get_next_pos(pos, last_offset):
        key = (padded_mat[pos[0]  , pos[1]  ]==target_label) \
            + (padded_mat[pos[0]  , pos[1]+1]==target_label) * 2 \
            + (padded_mat[pos[0]+1, pos[1]  ]==target_label) * 4 \
            + (padded_mat[pos[0]+1, pos[1]+1]==target_label) * 8
        offset = marching_map[key]
        # Special cases of saddle points
        if key == 6 and last_offset == (1, 0):
            offset = (0, -1)
        if key == 9 and last_offset == (0, -1):
            offset = (-1, 0)
        return (pos[0] + offset[0], pos[1] + offset[1]), offset
        
    def is_critical_point(pos):
        """Check if a point is an end of a shared edge between two contours of different colors"""
        return len(np.unique(padded_mat[pos[0]:pos[0]+2, pos[1]:pos[1]+2])) > 2
    
    path = []
    visited = set()
    critical_idx = []
    pos = start_pos
    i = 0
    last_offset = None
    while pos != None and pos not in visited:
        visited.add(pos)
        path.append((pos[0] - 0.5, pos[1] - 0.5))
        if is_critical_point(pos):
            critical_idx.append(i)
        pos, last_offset = get_next_pos(pos, last_offset)
        i += 1
    return np.array(path), np.array(critical_idx, dtype='int')

def multicolor_contour_find(spatial_label_mat, color_label_mat):
    """
    Find all contours from a colored image, then do postprocessing on each contour path
    """
    processed_labels = set([0])
    contours_info = []  # (path, critical points, color label) of each contour
    
    # Perform marching squares on each connected component
    padded_mat = np.pad(spatial_label_mat, ((1,1),(1,1)), 'constant', constant_values=0)
    H, W = padded_mat.shape[0], padded_mat.shape[1]
    for i in range(H):
        for j in range(W):
            if padded_mat[i,j] not in processed_labels:
                path, critical_idx = marching_squares(padded_mat, (i-1,j-1), padded_mat[i,j])
                contours_info.append((path, critical_idx, color_label_mat[i-1,j-1]))
                processed_labels.add(padded_mat[i,j])
    return contours_info
    
def simplify_contour_path(path, segment_indices=[], smooth_level=0, sample_step=1):
    """
    Perform smoothing and simplification on a contour path, however keep specific points unchanged
    """
    def process_segment(vec, start, end):
        if vec[end][0] + vec[end][1] > vec[start][0] + vec[start][1]:
            res = np.concatenate((vec[end:start:-sample_step], [vec[start]]))[::-1]
        else:
            res = np.concatenate((vec[start:end:sample_step], [vec[end]]))
        for i in range(smooth_level):
            tmp = np.roll(res, 1, axis=0) * .25 + np.roll(res, -1, axis=0) * .25 + res * .5
            res[1:-1] = tmp[1:-1]
        return res[:-1]
        
    if len(segment_indices) == 0:
        path = np.append(path, [path[0]], axis=0)
        start, end = 0, len(path)-1
        new_path = list(process_segment(path, start, end))
    else:
        # Align the path so that the first point is the start of a segment
        new_path = []
        offset = segment_indices[0]
        path = np.roll(path, -offset, axis=0)
        path = np.append(path, [path[0]], axis=0)
        segment_indices -= offset
        segment_indices = np.append(segment_indices, [len(path)-1], axis=0)

        for i,_ in enumerate(segment_indices):
            if i == 0:
                continue
            start, end = segment_indices[i-1], segment_indices[i]
            new_path += list(process_segment(path, start, end))
    return np.array(new_path)

def merge_small_areas(spatial_label_mat, color_label_mat, min_area):
    """
    Even when min_size is specified, Felzenszwalb algorithm may generate small segments. Therefore, deploy another algorithm to eliminate such areas
    """
    label_neighbors = {}
    label_area = {}
    label_color = {}
    
    # Count area and neighbors for each distinct spatial label
    H, W = spatial_label_mat.shape[0], spatial_label_mat.shape[1]
    for i in range(H):
        for j in range(W):
            label = spatial_label_mat[i,j]
            if label <= 0:
                continue
            if label not in label_neighbors:
                label_neighbors[label] = set()
                label_area[label] = 0
                label_color[label] = color_label_mat[i,j]
            label_area[label] += 1
            
            for di,dj in [(0,1), (1,0), (0,-1), (-1,0)]:
                if i+di >= 0 and i+di < H and j+dj >= 0 and j+dj < W:
                    new_label = spatial_label_mat[i+di,j+dj]
                    if new_label != label and new_label > 0:
                        label_neighbors[label].add(new_label)
                        
    # For every label with small area, either merge it to a larger area or mark it as invalid 
    for label in label_area:
        if label_area[label] < min_area:
            mask = spatial_label_mat==label
            for neighbor in label_neighbors[label]:
                if label_area[neighbor] >= min_area:
                    spatial_label_mat[mask] = neighbor
                    color_label_mat[mask] = label_color[neighbor]
                    break
            else:
                spatial_label_mat[mask] = 0
                color_label_mat[mask] = 0