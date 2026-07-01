from .trellis2_setup import Trellis2Setup
from .trellis2_load_model import Trellis2LoadModel
from .trellis2_image_to_3d import Trellis2ImageTo3D

NODE_CLASS_MAPPINGS = {
    "Trellis2Setup": Trellis2Setup,
    "Trellis2LoadModel": Trellis2LoadModel,
    "Trellis2ImageTo3D": Trellis2ImageTo3D,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Trellis2Setup": "TRELLIS.2 Setup (Install Worker)",
    "Trellis2LoadModel": "TRELLIS.2 Load Model",
    "Trellis2ImageTo3D": "TRELLIS.2 Image to 3D",
}
