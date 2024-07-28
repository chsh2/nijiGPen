import numpy as np
from scipy.interpolate import splprep, splrep, splev, bisplrep, bisplev

def shoelace_polygon_area(poly):
    """
    Calculate the signed area of a 2D polygon. Clipper library is not used here because it prefers integers.
    TODO: Consider if there are other operators that should use this function
    """
    area = 0
    for i,p in enumerate(poly):
        p2 = poly[i-1]
        area += p2[0] * p[1] - p2[1] * p[0]
    return 0.5 * area

class CurveFitter:
    """
    Fit coordinates {(x, y)} and any 1D attribute of stroke points to a parametrized curve x(u), y(u).
    Furthermore, when multi-frame data is provided as x_t, y_t, perform fitting again to get x(t, u), y(t, u).
    """
    is_periodic: bool
    xy_input:    dict[int, np.ndarray]
    attr_input:  dict[str, dict[int, np.ndarray]]
    total_len:   dict[int, float]
    xy_tck:      dict                 # 1D fit parameters
    attr_tck:    dict[str, dict]
    input_u:     dict                 # 1D sampling positions
    u_reversed:  dict[int, bool]
    x_surf:      dict                 # 2D fit parameters: surfaces
    y_surf:      dict
    attr_surf:   dict[str, dict]
    
    def __init__(self, is_periodic=False):
        self.is_periodic = is_periodic
        self.xy_input = {}
        self.attr_input = {}
        self.total_len = {}
        self.xy_tck = {}
        self.attr_tck = {}
        self.input_u = {}
        self.u_reversed = {}
        self.attr_surf = {}
        
    def set_coordinates(self, frame_number, co_list, total_len):
        data = np.array(co_list)
        if self.is_periodic:
            data = np.append(data, [data[0]], axis=0)
        self.xy_input[frame_number] = data
        self.total_len[frame_number] = total_len
    
    def set_attribute_data(self, frame_number, attr_name, attr_data):
        if attr_name not in self.attr_input:
            self.attr_input[attr_name] = {}
            self.attr_tck[attr_name] = {}
            self.attr_surf[attr_name] = {}
        data = np.array(attr_data)
        if self.is_periodic:
            data = np.append(data, data[0])
        self.attr_input[attr_name][frame_number] = data

    def fit_spatial(self, co_smoothness, attr_smoothness):
        """
        Perform fitting independently on each frame's data.
        Using different smoothness for coordinates and attributes since the scale differs        
        """
        for frame in self.xy_input.keys():
            self.xy_tck[frame], u_list = splprep([self.xy_input[frame][:,0], self.xy_input[frame][:,1]], 
                                                        s=self.total_len[frame]**2 * co_smoothness, 
                                                        per=self.is_periodic)
            for name, data in self.attr_input.items():
                self.attr_tck[name][frame] = splrep(u_list, data[frame],
                                                        s=attr_smoothness, 
                                                        per=self.is_periodic)
            if frame not in self.input_u:
                self.input_u[frame] = u_list
        self.co_smoothness = co_smoothness
        self.attr_smoothness = attr_smoothness

    def eval_spatial(self, frame_number):
        """
        Get coordinate and attribute values from the fit curve given a list of sampling points
        """
        res = np.array(splev(self.input_u[frame_number], self.xy_tck[frame_number])).transpose()
        res_attr = {}
        for name, tck in self.attr_tck.items():
            res_attr[name] = np.array(splev(self.input_u[frame_number], tck[frame_number]))
        return res, res_attr

    def correct_direction(self):
        """
        Compare strokes of adjacent keyframes to determine whether to reverse the direction
        """
        sorted_frames = sorted(self.xy_input)
        last_frame = sorted_frames[0]
        for t,frame in enumerate(sorted_frames):
            if t == 0:
                self.u_reversed[frame] = False
                if self.is_periodic:
                    path = np.array(splev(self.input_u[frame], self.xy_tck[frame])).transpose()
                    last_area = shoelace_polygon_area(path)
            else:
                # For open stroke, compare the position of start/end points
                if not self.is_periodic:
                    start, end = np.array(splev([0, 1], self.xy_tck[frame])).transpose()
                    start0, end0 = np.array(splev([0, 1], self.xy_tck[last_frame])).transpose()
                    dist = np.linalg.norm(start - start0) + np.linalg.norm(end - end0)
                    dist_r = np.linalg.norm(start - end0) + np.linalg.norm(end - start0)
                    self.u_reversed[frame] = (dist > dist_r) ^ self.u_reversed[last_frame]
                # For closed stroke, check if the shape is clockwise
                else:
                    path = np.array(splev(self.input_u[frame], self.xy_tck[frame])).transpose()
                    area = shoelace_polygon_area(path)
                    self.u_reversed[frame] = (area * last_area < 0) ^ self.u_reversed[last_frame]
                    last_area = area
                last_frame = frame

    def fit_temporal(self):
        """
        Resample spatial fitting results of multiple frames and then perform spatio-temporal 2D fitting
        """
        if len(self.input_u) > 0:
            self.correct_direction()
        # Determine the parameters of surface approximation
        size_u = 5
        size_t = 2  # Pad two data points in the time domain
        min_frame = np.inf
        max_frame = -1
        for frame, u in self.input_u.items():
            size_u = max(size_u, u.shape[0])
            size_t += 1
            min_frame = int(min(min_frame, frame))
            max_frame = int(max(max_frame, frame))
        # Choose the frame with the most sampling points as size_u
        self.input_u[-1] = np.linspace(0, 1, size_u, endpoint=True)

        # Prepare the dataset
        dataset_x = np.zeros((size_t, size_u, 3))
        dataset_y = np.zeros((size_t, size_u, 3))
        i = 1
        for frame, tck in self.xy_tck.items():
            sample_points = np.flip(self.input_u[-1]) if self.u_reversed[frame] else self.input_u[-1]
            res = np.array(splev(sample_points, tck))
            dataset_x[i, :, 0], dataset_x[i, :, 1], dataset_x[i, :, 2] = self.input_u[-1], frame, res[0]
            dataset_y[i, :, 0], dataset_y[i, :, 1], dataset_y[i, :, 2] = self.input_u[-1], frame, res[1]
            if frame == min_frame:
                dataset_x[0, :, 0], dataset_x[0, :, 1], dataset_x[0, :, 2] = self.input_u[-1], frame-1, res[0]
                dataset_y[0, :, 0], dataset_y[0, :, 1], dataset_y[0, :, 2] = self.input_u[-1], frame-1, res[1]
            if frame == max_frame:
                dataset_x[-1, :, 0], dataset_x[-1, :, 1], dataset_x[-1, :, 2] = self.input_u[-1], frame+1, res[0]
                dataset_y[-1, :, 0], dataset_y[-1, :, 1], dataset_y[-1, :, 2] = self.input_u[-1], frame+1, res[1]                        
            i += 1
        dataset_x = np.reshape(dataset_x, (size_t * size_u, 3))
        dataset_y = np.reshape(dataset_y, (size_t * size_u, 3))

        dataset_attr = {}
        for name, tck_map in self.attr_tck.items():
            dataset_attr[name] = np.zeros((size_t, size_u, 3))
            i = 1
            for frame, tck in tck_map.items():
                sample_points = np.flip(self.input_u[-1]) if self.u_reversed[frame] else self.input_u[-1]
                dataset_attr[name][i, :, 0], dataset_attr[name][i, :, 1], dataset_attr[name][i, :, 2] = self.input_u[-1], frame, np.array(splev(sample_points, tck))
                if frame == min_frame:
                    dataset_attr[name][0, :, 0], dataset_attr[name][0, :, 1], dataset_attr[name][0, :, 2] = self.input_u[-1], frame-1, np.array(splev(sample_points, tck))
                if frame == max_frame:
                    dataset_attr[name][-1, :, 0], dataset_attr[name][-1, :, 1], dataset_attr[name][-1, :, 2] = self.input_u[-1], frame+1, np.array(splev(sample_points, tck))
                i += 1
            dataset_attr[name] = np.reshape(dataset_attr[name], (size_t * size_u, 3))
        
        # Perform fitting
        self.x_surf = bisplrep(dataset_x[:,0], dataset_x[:,1], dataset_x[:,2], s=self.co_smoothness)
        self.y_surf = bisplrep(dataset_y[:,0], dataset_y[:,1], dataset_y[:,2], s=self.co_smoothness)
        for name, dataset in dataset_attr.items():
            self.attr_surf[name] = bisplrep(dataset[:,0], dataset[:,1], dataset[:,2], s=self.attr_smoothness)

    def eval_temporal(self, frame_number):
        """
        Get coordinate and attribute values after the temporal fitting
        """        
        res = np.zeros((len(self.input_u[-1]), 2))
        res[:,0] = bisplev(self.input_u[-1], frame_number, self.x_surf)[:,0]
        res[:,1] = bisplev(self.input_u[-1], frame_number, self.y_surf)[:,0]
        res_attr = {}
        for name, tck in self.attr_surf.items():
            res_attr[name] = np.array(bisplev(self.input_u[-1], frame_number, tck))[:,0]
        return res, res_attr
        