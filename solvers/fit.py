import numpy as np
from scipy.interpolate import splprep, splrep, splev, bisplrep, bisplev
from skimage.transform import SimilarityTransform, estimate_transform

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
        self.delta_transforms = {}
        self.attr_surf = {}
        self.trajectory_length = .0
        
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
                    trajectory_delta = self.trajectory_length * t
                    start0, end0 = np.array(splev([0, 1], self.xy_tck[last_frame])).transpose()
                    # Non-reversed case
                    start, end = np.array(splev([-trajectory_delta, 1 - trajectory_delta], self.xy_tck[frame])).transpose()
                    dist = np.linalg.norm(end - end0)
                    if self.trajectory_length < 0.01:
                        dist += np.linalg.norm(start - start0)
                    # Reversed case
                    start, end = np.array(splev([trajectory_delta, 1 + trajectory_delta], self.xy_tck[frame])).transpose()
                    dist_r = np.linalg.norm(start - end0)
                    if self.trajectory_length < 0.01:
                        dist_r += np.linalg.norm(end - start0)
                    self.u_reversed[frame] = (dist > dist_r) ^ self.u_reversed[last_frame]
                # For closed stroke, check if the shape is clockwise
                else:
                    path = np.array(splev(self.input_u[frame], self.xy_tck[frame])).transpose()
                    area = shoelace_polygon_area(path)
                    self.u_reversed[frame] = (area * last_area < 0) ^ self.u_reversed[last_frame]
                    last_area = area
                last_frame = frame

    def fit_temporal(self, use_1d_fitting=False, as_rigid_body=False):
        """
        Resample spatial fitting results of multiple frames and then either:
          - perform spatio-temporal 2D fitting, or
          - mix space and time into one dimension to perform 1D fitting
        """
        if len(self.input_u) > 0:
            self.correct_direction()

        # Determine the resolution of surface approximation
        size_u = 5
        size_t = 2  # Pad two data points in the time domain
        min_frame = np.inf
        max_frame = -1
        for frame, u in self.input_u.items():
            size_u = max(size_u, u.shape[0])
            size_t += 1
            min_frame = int(min(min_frame, frame))
            max_frame = int(max(max_frame, frame))
        samples_u = np.linspace(0, 1, size_u, endpoint=True)
        self.input_u["INTERPOLATION"] = samples_u

        sorted_frames = sorted(self.xy_input)
        trajectory_offset = lambda f: samples_u + self.trajectory_length * sorted_frames.index(f)

        # Evaluate each keyframe's curve with the same resolution
        polys = {}
        for frame, tck in self.xy_tck.items():
            sample_points = np.flip(samples_u) if self.u_reversed[frame] else samples_u
            polys[frame] = np.array(splev(sample_points, tck))

        # Assuming the shape as a rigid body, get the optimal transform matrix between keyframes
        if as_rigid_body:
            last_frame = sorted_frames[0]
            accum_transforms = {}
            for t,frame in enumerate(sorted_frames):
                if t == 0:
                    self.delta_transforms[frame] = SimilarityTransform()
                    accum_transforms[frame] = SimilarityTransform()
                else:
                    self.delta_transforms[frame] = estimate_transform("similarity", polys[last_frame].transpose(), polys[frame].transpose())
                    accum_transforms[frame] = SimilarityTransform(
                        matrix= accum_transforms[last_frame].params @ self.delta_transforms[frame].inverse.params
                    )
                    last_frame = frame
            for frame in self.xy_tck:
                polys[frame] = accum_transforms[frame](polys[frame].transpose()).transpose()

        # Prepare the dataset
        dataset_x = np.zeros((size_t, size_u, 3))
        dataset_y = np.zeros((size_t, size_u, 3))
        i = 1
        for frame, tck in self.xy_tck.items():
            frame_u = trajectory_offset(frame)
            dataset_x[i, :, 0], dataset_x[i, :, 1], dataset_x[i, :, 2] = frame_u, frame, polys[frame][0]
            dataset_y[i, :, 0], dataset_y[i, :, 1], dataset_y[i, :, 2] = frame_u, frame, polys[frame][1]
            if frame == min_frame:
                dataset_x[0, :, 0], dataset_x[0, :, 1], dataset_x[0, :, 2] = frame_u, frame-1, polys[frame][0]
                dataset_y[0, :, 0], dataset_y[0, :, 1], dataset_y[0, :, 2] = frame_u, frame-1, polys[frame][1]
            if frame == max_frame:
                dataset_x[-1, :, 0], dataset_x[-1, :, 1], dataset_x[-1, :, 2] = frame_u, frame+1, polys[frame][0]
                dataset_y[-1, :, 0], dataset_y[-1, :, 1], dataset_y[-1, :, 2] = frame_u, frame+1, polys[frame][1]                        
            i += 1
        dataset_x = np.reshape(dataset_x, (size_t * size_u, 3))
        dataset_y = np.reshape(dataset_y, (size_t * size_u, 3))

        dataset_attr = {}
        for name, tck_map in self.attr_tck.items():
            dataset_attr[name] = np.zeros((size_t, size_u, 3))
            i = 1
            for frame, tck in tck_map.items():
                sample_points = np.flip(samples_u) if self.u_reversed[frame] else samples_u
                frame_u = trajectory_offset(frame)
                res = np.array(splev(sample_points, tck))
                dataset_attr[name][i, :, 0], dataset_attr[name][i, :, 1], dataset_attr[name][i, :, 2] = frame_u, frame, res
                if frame == min_frame:
                    dataset_attr[name][0, :, 0], dataset_attr[name][0, :, 1], dataset_attr[name][0, :, 2] = frame_u, frame-1, res
                if frame == max_frame:
                    dataset_attr[name][-1, :, 0], dataset_attr[name][-1, :, 1], dataset_attr[name][-1, :, 2] = frame_u, frame+1, res
                i += 1
            dataset_attr[name] = np.reshape(dataset_attr[name], (size_t * size_u, 3))
        
        # Perform fitting
        if not use_1d_fitting:
            self.x_surf = bisplrep(dataset_x[:,0], dataset_x[:,1], dataset_x[:,2], s=self.co_smoothness)
            self.y_surf = bisplrep(dataset_y[:,0], dataset_y[:,1], dataset_y[:,2], s=self.co_smoothness)
            for name, dataset in dataset_attr.items():
                self.attr_surf[name] = bisplrep(dataset[:,0], dataset[:,1], dataset[:,2], s=self.attr_smoothness)
        else:
            uniq_i = np.unique(dataset_x[:,0], return_index=True)[1]
            self.x_surf = splrep(dataset_x[uniq_i,0], dataset_x[uniq_i,2])
            self.y_surf = splrep(dataset_y[uniq_i,0], dataset_y[uniq_i,2])
            for name, dataset in dataset_attr.items():
                self.attr_surf[name] = splrep(dataset[uniq_i,0], dataset[uniq_i,2])

    def eval_temporal(self, frame_number, use_1d_fitting=False, as_rigid_body=False):
        """
        Get coordinate and attribute values after the temporal fitting
        """        
        sorted_frames = sorted(self.xy_input)
        offset = -1.0
        last_frame = sorted_frames[0]
        for i, f in enumerate(sorted_frames):
            if f > frame_number:
                offset += (frame_number - sorted_frames[i-1]) / (f - sorted_frames[i-1])
                break
            else:
                offset += 1
                last_frame = f
        offset *= self.trajectory_length
        samples_u = self.input_u["INTERPOLATION"] + offset

        res = np.zeros((len(samples_u), 2))
        if not use_1d_fitting:
            f = frame_number if not as_rigid_body else last_frame
            res[:,0] = bisplev(samples_u, f, self.x_surf)[:,0]
            res[:,1] = bisplev(samples_u, f, self.y_surf)[:,0]
            res_attr = {}
            for name, tck in self.attr_surf.items():
                res_attr[name] = np.array(bisplev(samples_u, f, tck))[:,0]
        else:
            res[:,0] = splev(samples_u, self.x_surf)
            res[:,1] = splev(samples_u, self.y_surf)
            res_attr = {}
            for name, tck in self.attr_surf.items():
                res_attr[name] = np.array(splev(samples_u, tck))

        if as_rigid_body:
            for i, f in enumerate(sorted_frames):
                if f > frame_number:
                    mat = self.delta_transforms[f]
                    factor = (frame_number - sorted_frames[i-1]) / (f - sorted_frames[i-1])
                    partial_mat = SimilarityTransform(
                        translation=mat.translation * factor,
                        rotation=mat.rotation * factor,
                        scale=mat.scale ** factor
                    )
                    res = partial_mat(res)
                    break
                else:
                    res = self.delta_transforms[f](res)

        return res, res_attr
        