import struct
import numpy as np

# Convert Blender identifiers to Photoshop ones
dict_color_mode = {'BW': 1, 'RGB': 3, 'RGBA': 3}
dict_blend_mode = {'REGULAR': b'norm',
                   'HARDLIGHT': b'hLit',
                   'ADD': b'lite',
                   'SUBTRACT': b'fsub',
                   'MULTIPLY': b'mul ',
                   'DIVIDE': b'fdiv'}

class PsdLayer:
    def __init__(self, 
                img_mat,
                top = None, left = None, bottom = None, right = None,
                num_channels = None,
                blend_mode_key = 'REGULAR',
                opacity = 1,
                hide = False,
                bit_depth = 8,
                name = 'new_layer'):
        assert bit_depth==8 or bit_depth==16
        self.img_mat = img_mat.astype('>u'+str(bit_depth//8))
        if top and left and bottom and right:
            self.top, self.left, self.bottom, self.right = top, left, bottom, right
        else:
            self.top, self.left, self.bottom, self.right = 0,0, img_mat.shape[0], img_mat.shape[1]
        if num_channels:
            self.num_channels = num_channels
        else:
            self.num_channels = img_mat.shape[2]
        self.blend_mode_key = blend_mode_key
        self.opacity = int(opacity * 255)
        self.hide = int(hide)
        self.bit_depth = bit_depth
        self.name = name
        
    def get_layer_record_bytes(self):
        section_bytes = struct.pack('>IIII H',
                                    self.top, self.left, self.bottom, self.right,
                                    self.num_channels)
        for i in range(self.num_channels):
            channel_id = i if i<3 else -1
            channel_data_length = ((self.bottom-self.top)
                                   *(self.right-self.left)
                                   *(self.bit_depth//8)
                                   +2)
            section_bytes += struct.pack('>hI', channel_id, channel_data_length)
            
        flags = 2 * self.hide
        layer_name = struct.pack('>B',len(self.name)) + self.name.encode()
        padding = len(layer_name)%4
        if padding:
            layer_name += bytes(4-padding)
        extra_data_length = 8+len(layer_name)
            
        section_bytes += struct.pack('>4s 4s BBBB I I I',
                                     b'8BIM',
                                     dict_blend_mode[self.blend_mode_key],
                                     self.opacity,
                                     0, flags, 0,
                                     extra_data_length, 0, 0
                                    )
        section_bytes += layer_name
        return section_bytes
    
    def get_channel_image_data_bytes(self):
        img_bytes = b''
        for i in range(self.num_channels):
            img_bytes += struct.pack('>H',0)
            img_bytes += self.img_mat[:,:,i].tobytes()
        return img_bytes
        

class PsdFileWriter:
    """
    Convert multi-layer image data to a binary PSD file,
    according to https://www.adobe.com/devnet-apps/photoshop/fileformatashtml/.
    Please notice that only a small subset of features are supported
    """
    def __init__(self, num_channels = 4, height = 256, width = 256, depth = 8, color_mode = 'RGBA'):
        self.num_channels = num_channels
        self.height = height
        self.width = width
        self.depth = depth
        self.color_mode = dict_color_mode[color_mode]
        self.layers = []
        self.merged_img_mat = np.zeros((height,width,depth), dtype=('>u'+str(depth//8)) )
        
    def append_layer(self, layer: PsdLayer):
        self.layers.append(layer)
        
    def set_merged_img(self, img_mat):
        self.merged_img_mat = img_mat.astype('>u'+str(self.depth//8))
        
    def get_file_bytes(self):
        # File Header Section
        section_format = '>4s H 6x H II H H'
        file_bytes = struct.pack(section_format, b'8BPS', 1, 
                                self.num_channels,
                                self.height, self.width, self.depth,
                                self.color_mode)
        
        # Color Mode Data Section: Skipped
        # Image Resources Section: Skipped
        file_bytes += struct.pack('>II', 0, 0)

        # Layer and Mask Information Section
        layer_record_bytes = b''
        channel_image_data_bytes = b''
        for layer in self.layers:
            layer_record_bytes += layer.get_layer_record_bytes()
            channel_image_data_bytes += layer.get_channel_image_data_bytes()
            
        layer_section_length = 2 + len(layer_record_bytes) + len(channel_image_data_bytes)
        layer_section_length += layer_section_length%2
        file_bytes += struct.pack('>IIH', 
                                 layer_section_length+8, 
                                 layer_section_length, 
                                 len(self.layers))
        file_bytes += layer_record_bytes
        file_bytes += channel_image_data_bytes
        file_bytes += struct.pack('>I', 0)
        
        # Image Data Section
        for i in range(self.num_channels):
            # TODO: Implement a compression algorithm
            file_bytes += struct.pack('>H',0) 
            file_bytes += self.merged_img_mat[:,:,i].tobytes()
        return file_bytes