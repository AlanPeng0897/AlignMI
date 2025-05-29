import os

import numpy as np
from torch.utils.data import ConcatDataset, Dataset, Subset
from torchvision.datasets import ImageFolder
import torchvision.transforms as T

def preprocess_facescrub_fn(crop_center, output_resolution):
    if crop_center:
        crop_size = int(54 * output_resolution / 64)
        return T.Compose(
            [
                T.Resize((output_resolution, output_resolution), antialias=True),
                T.CenterCrop((crop_size, crop_size)),
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

class FaceScrub(Dataset):
    def __init__(self,
                 group,
                 mode=None,
                 cropped=True,
                 preprocess_resolution=224,
                 transform=None,
                 split_seed=42,
                 root='data/facescrub'):

        if group == 'actors':
            if cropped:
                root = os.path.join(root, 'actors/faces')
            else:
                root = os.path.join(root, 'actors/images')
            self.dataset = ImageFolder(root=root, transform=None)
            self.targets = self.dataset.targets
            self.name = 'facescrub_actors'

        elif group == 'actresses':
            if cropped:
                root = os.path.join(root, 'actresses/faces')
            else:
                root = os.path.join(root, 'actresses/images')
            self.dataset = ImageFolder(root=root, transform=None)
            self.targets = self.dataset.targets
            self.name = 'facescrub_actresses'

        elif group == 'all':
            if cropped:
                root_actors = os.path.join(root, 'actors/faces')
                root_actresses = os.path.join(root, 'actresses/faces')
            else:
                root_actors = os.path.join(root, 'actors/images')
                root_actresses = os.path.join(root, 'actresses/images')
            dataset_actors = ImageFolder(root=root_actors, transform=None)
            target_transform_actresses = lambda x: x + len(dataset_actors.classes)
            dataset_actresses = ImageFolder(
                root=root_actresses,
                transform=None,
                target_transform=target_transform_actresses)
            dataset_actresses.class_to_idx = {
                key: value + len(dataset_actors.classes)
                for key, value in dataset_actresses.class_to_idx.items()
            }
            self.dataset = ConcatDataset([dataset_actors, dataset_actresses])
            self.classes = dataset_actors.classes + dataset_actresses.classes
            self.class_to_idx = {
                **dataset_actors.class_to_idx,
                **dataset_actresses.class_to_idx
            }

            self.targets = dataset_actors.targets + [
                t + len(dataset_actors.classes)
                for t in dataset_actresses.targets
            ]
            self.name = 'facescrub_all'
            self.preprocess_transform = preprocess_facescrub_fn(
                cropped, preprocess_resolution
            )

        else:
            raise ValueError(
                f'Dataset group {group} not found. Valid arguments are \'all\', \'actors\' and \'actresses\'.'
            )

        self.transform = transform
        indices = list(range(len(self.dataset)))
        np.random.seed(split_seed)
        np.random.shuffle(indices)
        training_set_size = int(0.9 * len(self.dataset))
        train_idx = indices[:training_set_size] # 33411
        test_idx = indices[training_set_size:] # 3713

        if mode == "train":
            self.dataset = Subset(self.dataset, train_idx)
            self.targets = np.array(self.targets)[train_idx].tolist()
        else:
            self.dataset = Subset(self.dataset, test_idx)
            self.targets = np.array(self.targets)[test_idx].tolist()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        im, _ = self.dataset[idx]
        im = self.preprocess_transform(im)
        if self.transform:
            return self.transform(im), self.targets[idx]
        else:
            return im, self.targets[idx]


class FaceScrub64(FaceScrub):

    def __init__(
        self,
        root_path='/home/csxpeng/code/BiDO+_low/attack_datasets/FaceScrub',
        mode='test_ood',
        output_transform=None,
    ):
        super().__init__(root_path, mode, True, 64, output_transform)