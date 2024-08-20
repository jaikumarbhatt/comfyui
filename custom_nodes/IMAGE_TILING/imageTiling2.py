import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageSequence, ImageOps
import tensorflow as tf
import node_helpers
class Tiling:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image":("IMAGE",),
                "int_field_row":("INT", {
                    "default":1,
                    "min" : 1,
                    "max" : 5,
                     "step": 1,
                    "display" : "number",
                }),
                "int_field_column":("INT", {
                    "default":1,
                    "min" : 1,
                    "max" : 5,
                     "step": 1,
                    "display" : "number",
                }),
            },
        }

    RETURN_TYPES = ( "IMAGE",)
    # RETURN_NAMES = ("TxtO", "IntO", "FloatO", "Latent output. Really cool, huh?", "A condition" , "Our image." , "Mo mo modell!!!")

    FUNCTION = "test"

    #OUTPUT_NODE = False

    CATEGORY = "image/mynode2"
    # def test(self, image, int_field_row, int_field_column):
    #     tensor_to_pil = 
    #     batch_size, channels, height, width = image.shape
    #     print("image.shape_after", image.shape)
    #     image_tensor = image.permute(0, 3, 1, 2) 
    #     batch_size, channels, height, width = image_tensor.shape
    #     tiled_image_tensor = image_tensor.repeat( 1,1,int_field_column, int_field_row)
    #     tiled_image_tensor = tiled_image_tensor.view(1, channels, height * int_field_column, width * int_field_row)
    #     image_tensor = tiled_image_tensor.permute(0, 2, 3, 1)
    #     print("image.shape_after", image_tensor.shape)
    #     tiled_image = T.ToPILImage()(tiled_image_tensor[0])
    #     tiled_image.save("/workspace/comfyui/custom_nodes/IMAGE_TILING/tiled_image_torch.png")
    #     image = cv2.imread("tiled_image_torch.png")
    #     # image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #     image_tensor = T.ToTensor()(image).unsqueeze(0) 
    #     print("image_tensor.shape_after", image_tensor.shape)
    #     image_tensor = image_tensor.permute(0, 1, 2, 3)
    #     return (image_tensor)

    def test(self, image, int_field_row, int_field_column):
        
        # pattern_tensor = image.unsqueeze(0)  # Add batch dimension
        print("pattern_tensorbefore", image.shape)
        # 2. Tile on GPU
s
        pattern_tensor = image.to(device)
        tiled_tensor = torch.tile(pattern_tensor, (1, 1, int_field_column, int_field_row)) 
        print("pattern_tensorafter", tiled_tensor.shape)
        image_tensor = tiled_tensor.permute(0, 2, 3, 1)
        print("pattern_tensorafter", image_tensor.shape)
        return (image_tensor)

NODE_CLASS_MAPPINGS = {
    "Tiling of Image": Tiling
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "Tiling": "Tiling of Image"
}
