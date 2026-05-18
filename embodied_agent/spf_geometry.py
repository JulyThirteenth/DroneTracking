import math

class SPFGeometry:
    def __init__(self,
                 img_width: int,
                 img_height: int,
                 hfov: float,
                 vfov: float):
        self.width = img_width
        self.height = img_height

        # Convert fov angles to half-fov radians
        self.hfov_rad = math.radians(hfov / 2)
        self.vfov_rad = math.radians(vfov / 2)

    def calculate_adjusted_depth(self, vlm_depth: int, d_min: float = 0.1, s: int = 8, L: int = 10, p: int = 2):
        '''
        Returns: d_adj

            d_adj = max(d_min, s*(d_VLM/L)^p)
        '''
        return max(d_min, s * (vlm_depth / L)**p)

    def reverse_project_point(self, pixel_x: float, pixel_y: float, d_adj: float):
        """
        Returns: (s_x, s_y, s_z) in camera frame

            \n s_x = horizontal offset (left-right)
            \n s_y = forward depth
            \n s_z = vertical offset (up-down)
        """
        center_x = self.width / 2
        center_y = self.height / 2
        # Tello-specific: Use a 35% vertical offset as the reference horizon
        ref_y = self.height * 0.35

        x_norm = (pixel_x - center_x) / center_x
        y_norm = (ref_y - pixel_y) / center_y

        depth_factor = 1.0 + (y_norm * 0.5)
        d_adj = d_adj * depth_factor

        s_x = x_norm * d_adj * math.tan(self.hfov_rad)
        s_y = d_adj
        s_z = y_norm * d_adj * math.tan(self.vfov_rad)
        return s_x, s_y, s_z
