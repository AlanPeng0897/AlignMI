import os
import PIL

from typing import Any, Callable, List, Optional, Union, Tuple
from torchvision.datasets import VisionDataset
import torchvision.transforms as T

def preprocess_ffhq_fn(crop_center_size, output_resolution):
    if crop_center_size is not None:
        return T.Compose(
            [
                T.CenterCrop((crop_center_size, crop_center_size)),
                T.Resize((output_resolution, output_resolution), antialias=True),
                T.ToTensor(),  # Convert image to tensor
            ]
        )
    else:
        return T.Compose(
            [
                T.Resize((output_resolution, output_resolution), antialias=True),  # Resize the image
                T.ToTensor(),  # Convert image to tensor
            ]
        )

class FFHQ(VisionDataset):
    """ 
    Modified CelebA dataset to adapt for custom cropped images.
    """

    def __init__(
            self,
            root='data/ffhq',
            mode=None,
            crop_center_size=800,
            preprocess_resolution=224,
            transform: Optional[Callable] = None,
    ):
        super(FFHQ, self).__init__(root, transform=transform)
        self.preprocess_transform = preprocess_ffhq_fn(
            crop_center_size, preprocess_resolution
        )
        self.filename = os.listdir(root)[:20000]

    def __len__(self):
        return len(self.filename)
    
    
    def __getitem__(self, index):
        file_path = os.path.join(self.root, self.filename[index])

        if os.path.exists(file_path) == False:
            file_path = file_path.replace('.jpg', '.png')
        im = PIL.Image.open(file_path)

        if self.transform:
            return self.transform(im)
        else:
            return im


class FFHQ64(FFHQ):

    def __init__(
        self, root='/home/csxpeng/code/BiDO+_low/attack_datasets/FFHQ/thumbnails128x128', mode=None, output_transform=None
    ):

        super().__init__(root, mode, 88, 64, output_transform)