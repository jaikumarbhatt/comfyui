import torch
import torchvision.transforms as transforms
from PIL import Image

def tile_pattern_gpu(pattern_path, num_tiles_x, num_tiles_y):
    # 1. Load and Transform the Pattern
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    pattern = Image.open(pattern_path)
    print("pattern", pattern)
    pattern_tensor = transform(pattern).unsqueeze(0)  # Add batch dimension

    # 2. Tile on GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pattern_tensor = pattern_tensor.to(device)
    tiled_tensor = torch.tile(pattern_tensor, (1, 1, num_tiles_y, num_tiles_x)) 

    # (Optional) Convert back to PIL Image for Visualization or Saving
    tiled_image = transforms.ToPILImage()(tiled_tensor.squeeze(0))
    return tiled_image


image = tile_pattern_gpu("colorfullpixer.png", 2,2)
image.save("tensorimage.png")