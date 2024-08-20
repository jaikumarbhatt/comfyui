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
        batchsize, height, width, channels = image.shape
        print("input image", image.shape)
        i = 255. * image.cpu().numpy()
        i =  i.squeeze(0) 
        pil_image = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        
        # Number of times to tile the image in both directions
        num_tiles_x = int_field_column
        num_tiles_y = int_field_row
        
        # Get the size of the original image
        width, height = pil_image.size
        
        # Create a new image with the appropriate size
        tiled_image = Image.new('RGB', (width * num_tiles_x, height * num_tiles_y))
        
        # Paste the original image into the new image multiple times
        for i in range(num_tiles_x):
            for j in range(num_tiles_y):
                tiled_image.paste(pil_image, (i * width, j * height))
        # Resize the image
        new_size = (width, height)  # New width and height in pixels
        resized_image = tiled_image.resize(new_size)
        image_tensor = self.load_image(resized_image)
        image_tensor = image_tensor.permute(0, 2, 3, 1)
        print("size_of_tensor", image_tensor.shape)
        return (image_tensor)

    def load_image(self, image):
        # Define transformations
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
    
        # Apply transformations
        tensor = transform(image)
    
        # Add batch dimension
        tensor = tensor.unsqueeze(0)    
        return tensor

NODE_CLASS_MAPPINGS = {
    "Tiling of Image": Tiling
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "Tiling": "Tiling of Image"
}
