from torchvision.datasets import CelebA
from torch.utils.data import Dataset, Subset
from collections import Counter
import torchvision.transforms as T
import numpy as np
from functools import partial
import torch
import os
import PIL
from typing import Any, Callable, List, Optional, Union, Tuple
import pandas
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import verify_str_arg


class CelebA1000(Dataset):
    def __init__(self,
                 mode,
                 split_seed=42,
                 transform=None,
                 root='data/celeba',
                 download: bool = False):
        # Load default CelebA dataset
        celeba = CustomCelebA(root=root,
                        split='all',
                        target_type="identity")
        celeba.targets = celeba.identity

        # Select the 1,000 most frequent celebrities from the dataset
        targets = np.array([t.item() for t in celeba.identity])
        ordered_dict = dict(
            sorted(Counter(targets).items(),
                   key=lambda item: item[1],
                   reverse=True))
        
        sorted_targets = list(ordered_dict.keys())[:1000]

        # Select the corresponding samples for train and test split
        indices = np.where(np.isin(targets, sorted_targets))[0]
        np.random.seed(split_seed)
        np.random.shuffle(indices)
        training_set_size = int(0.9 * len(indices))
        train_idx = indices[:training_set_size]
        test_idx = indices[training_set_size:]

        # Assert that there are no overlapping datasets
        assert len(set.intersection(set(train_idx), set(test_idx))) == 0

        # Set transformations
        self.transform = transform
        target_mapping = {
            sorted_targets[i]: i
            for i in range(len(sorted_targets))
        }

        self.target_transform = T.Lambda(lambda x: target_mapping[x])

        # Split dataset
        if mode == "train":
            self.dataset = Subset(celeba, train_idx)
            train_targets = np.array(targets)[train_idx]
            self.targets = [self.target_transform(t) for t in train_targets]
            self.name = 'CelebA1000_train'
        elif mode == "test":
            self.dataset = Subset(celeba, test_idx)
            test_targets = np.array(targets)[test_idx]
            self.targets = [self.target_transform(t) for t in test_targets]
            self.name = 'CelebA1000_test'

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        im, _ = self.dataset[idx]
        if self.transform:
            return self.transform(im), self.targets[idx]
        else:
            return im, self.targets[idx]


class CelebA_OOD(Dataset):
    def __init__(self,
                 mode,
                 split_seed=42,
                 transform=None,
                 root='data/celeba',
                 download: bool = False):
        # Load default CelebA dataset
        celeba = CustomCelebA(root=root,
                        split='all',
                        target_type="identity")
        celeba.targets = celeba.identity

        # Select the 1,000 most frequent celebrities from the dataset
        targets = np.array([t.item() for t in celeba.identity])
        ordered_dict = dict(
            sorted(Counter(targets).items(),
                   key=lambda item: item[1],
                   reverse=True))
        
        if mode == "aux_ood":
            sorted_targets = list(ordered_dict.keys())[1000:3000]

        elif mode == "test_ood":
            sorted_targets = list(ordered_dict.keys())[3000:5000]
        
        # Select the corresponding samples for train and test split
        indices = np.where(np.isin(targets, sorted_targets))[0]

        # Set transformations
        self.transform = transform
        target_mapping = {
            sorted_targets[i]: i
            for i in range(len(sorted_targets))
        }

        self.target_transform = T.Lambda(lambda x: target_mapping[x])

        # Split dataset
        if mode == "aux_ood":
            self.dataset = Subset(celeba, indices)
            # train_targets = np.array(targets)[indices]
            # self.targets = [self.target_transform(t) for t in train_targets]
            self.targets = [1000] * len(indices)
            self.name = 'CelebA_aux_ood'
        
        elif mode == "test_ood":
            np.random.seed(split_seed)
            np.random.shuffle(indices)
            training_set_size = int(0.9 * len(indices))
            train_idx = indices[:training_set_size]
            test_idx = indices[training_set_size:]

            self.dataset = Subset(celeba, test_idx)
            # train_targets = np.array(targets)[indices]
            # self.targets = [self.target_transform(t) for t in train_targets]
            self.targets = [1000] * len(indices)
            self.name = 'CelebA_test_ood'
        

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        im, _ = self.dataset[idx]
        if self.transform:
            return self.transform(im), self.targets[idx]
        else:
            return im, self.targets[idx]


class CustomCelebA(VisionDataset):
    """ 
    Modified CelebA dataset to adapt for custom cropped images.
    """

    def __init__(
            self,
            root: str,
            split: str = "all",
            target_type: Union[List[str], str] = "identity",
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
    ):
        super(CustomCelebA, self).__init__(root, transform=transform,
                                     target_transform=target_transform)
        self.split = split
        if isinstance(target_type, list):
            self.target_type = target_type
        else:
            self.target_type = [target_type]

        if not self.target_type and self.target_transform is not None:
            raise RuntimeError('target_transform is specified but target_type is empty')

        split_map = {
            "train": 0,
            "valid": 1,
            "test": 2,
            "all": None,
        }
        split_ = split_map[verify_str_arg(split.lower(), "split",
                                          ("train", "valid", "test", "all"))]

        fn = partial(os.path.join, self.root)
        splits = pandas.read_csv(fn("list_eval_partition.txt"), delim_whitespace=True, header=None, index_col=0)
        identity = pandas.read_csv(fn("identity_CelebA.txt"), delim_whitespace=True, header=None, index_col=0)
        bbox = pandas.read_csv(fn("list_bbox_celeba.txt"), delim_whitespace=True, header=1, index_col=0)
        landmarks_align = pandas.read_csv(fn("list_landmarks_align_celeba.txt"), delim_whitespace=True, header=1)
        attr = pandas.read_csv(fn("list_attr_celeba.txt"), delim_whitespace=True, header=1)
        mask = slice(None) if split_ is None else (splits[1] == split_)

        self.filename = splits[mask].index.values
        self.identity = torch.as_tensor(identity[mask].values)
        self.bbox = torch.as_tensor(bbox[mask].values)
        self.landmarks_align = torch.as_tensor(landmarks_align[mask].values)
        self.attr = torch.as_tensor(attr[mask].values)
        self.attr = torch.div(self.attr + 1, 2, rounding_mode='floor') # map from {-1, 1} to {0, 1}
        self.attr_names = list(attr.columns)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        file_path = os.path.join(self.root, "img_align_celeba", self.filename[index])

        if os.path.exists(file_path) == False:
            file_path = file_path.replace('.jpg', '.png')
        X = PIL.Image.open(file_path)

        target: Any = []
        for t in self.target_type:
            if t == "attr":
                target.append(self.attr[index, :])
            elif t == "identity":
                target.append(self.identity[index, 0])
            elif t == "bbox":
                target.append(self.bbox[index, :])
            elif t == "landmarks":
                target.append(self.landmarks_align[index, :])
            else:
                raise ValueError("Target type \"{}\" is not recognized.".format(t))

        if self.transform is not None:
            X = self.transform(X)

        if target:
            target = tuple(target) if len(target) > 1 else target[0]

            if self.target_transform is not None:
                target = self.target_transform(target)
        else:
            target = None

        return X, target

    def __len__(self) -> int:
        return len(self.attr)

    def extra_repr(self) -> str:
        lines = ["Target type: {target_type}", "Split: {split}"]
        return '\n'.join(lines).format(**self.__dict__)


class TargetCelebA(Dataset):
    def __init__(self, dataset_config, mode=None, transform=None):
        self._dataset_config = dataset_config
        self.mode = mode
        self.transform = transform
        self.id_dataset_name = self._dataset_config["name"]

        if self.mode == 'gan':
            self.img_path = self._dataset_config["img_path"]
            self.dataset_file_path = self._dataset_config["gan_file_path"]
        elif self.mode == 'train':
            self.img_path = self._dataset_config["img_path"]
            self.dataset_file_path = self._dataset_config["train_file_path"]
        elif self.mode == 'test':
            self.img_path = self._dataset_config["img_path"]
            self.dataset_file_path = self._dataset_config["test_file_path"]
        elif self.mode == 'private':
            self.img_path = self._dataset_config["img_path"]
            self.dataset_file_path = self._dataset_config["private_file_path"]

        # self.img_list = os.listdir(self.img_path)
        self.name_list, self.label_list = self.get_list(self.dataset_file_path)
        self.image_list = self.load_img()
        self.num_img = len(self.image_list)
        if self.mode != "gan":
            print("Load " + str(self.num_img) + " images")

    def get_list(self, file_path):
        name_list, label_list = [], []
        f = open(file_path, "r")
        for line in f.readlines():
            if self.mode == "gan":
                img_name = line.strip()
            else:
                img_name, iden = line.strip().split(' ')
                label_list.append(int(iden))
            name_list.append(img_name)

        return name_list, label_list

    def load_img(self):
        img_list = []
        for i, img_name in enumerate(self.name_list):
            if img_name.endswith(".png") or img_name.endswith(".jpg") or img_name.endswith(".jpeg"):
                path = os.path.join(self.img_path, img_name)
                img = PIL.Image.open(path)
                img = img.convert('RGB')
                # if self.transform != None:
                #     img = self.transform(img)
                img_list.append(img)
        return img_list

    def __getitem__(self, index):
        img = self.image_list[index]

        img = self.transform(img)

        if self.mode == "gan":
            return img
        label = self.label_list[index]

        return img, label

    def __len__(self):
        return self.num_img


class TangentBasisDataset(Dataset):
    """
    Efficient dataset storing only:
      - original images `x`
      - labels `y`
      - tangent-space basis `U`

    Expects a `data_dict` with keys:
      - 'x': torch.Tensor of shape [N, 3, H, W]
      - 'y': torch.Tensor of shape [N]
      - 'U': torch.Tensor of shape [N, latent_dim, C*H*W]
    """
    def __init__(self, data_dict):
        super().__init__()
        self.x = data_dict['x']    # [N, 3, H, W]
        self.y = data_dict['y']    # [N]
        self.U = data_dict['U']    # [N, latent_dim, C*H*W]

    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, idx):
        sample = {
            'image': self.x[idx],
            'tangent_basis': self.U[idx]
        }
        label = self.y[idx]
        return sample, label
