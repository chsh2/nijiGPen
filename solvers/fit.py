import numpy as np
from scipy.interpolate import splprep, splrep, splev
from .geomdl import fitting, BSpline
# https://github.com/orbingol/geomdl-examples/blob/master/fitting/approximation/global_surface.py

class CurveFitter:
    """
    Fit coordinates {(x, y)} and any 1D attribute of stroke points to a parametrized curve x(u), y(u).
    Furthermore, when multi-frame data is provided as x_t, y_t, perform fitting again to get x(t, u), y(t, u).
    Use SciPy as the backend for spatial fitting, and use NURBS-Python for temporal fitting
    """
    is_periodic: bool
    xy_input:    dict[int, np.ndarray]
    attr_input:  dict[str, dict[int, np.ndarray]]
    total_len:   dict[int, float]
    xy_tck:      dict                 # 1D fit parameters
    attr_tck:    dict
    input_u:     dict                 # 1D sampling positions
    x_surf:      BSpline.Surface      # 2D fit parameters: surfaces
    y_surf:      BSpline.Surface
    attr_surf:   dict[str, BSpline.Surface]
    
    def __init__(self, is_periodic=False):
        self.is_periodic = is_periodic
        self.xy_input = {}
        self.attr_input = {}
        self.total_len = {}
        self.xy_tck = {}
        self.attr_tck = {}
        self.input_u = {}
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
            self.xy_tck[frame], self.input_u[frame] = splprep([self.xy_input[frame][:,0], self.xy_input[frame][:,1]], 
                                                        s=self.total_len[frame]**2 * co_smoothness, 
                                                        per=self.is_periodic)
            
            for name, data in self.attr_input.items():
                self.attr_tck[name][frame] = splrep(self.input_u[frame], data[frame],
                                                        s=attr_smoothness, 
                                                        per=self.is_periodic)

    def eval_spatial(self, frame_number):
        """
        Get coordinate and attribute values from the fit curve given a list of sampling points
        """
        res = np.array(splev(self.input_u[frame_number], self.xy_tck[frame_number])).transpose()
        res_attr = {}
        for name, tck in self.attr_tck.items():
            res_attr[name] = np.array(splev(self.input_u[frame_number], tck[frame_number]))
        return res, res_attr

    def fit_temporal(self, temporal_scale):
        pass

    def eval_temporal(self, frame_number, u_list):
        pass