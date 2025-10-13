from types import SimpleNamespace
from typing import List

import torch, os
import numpy as np
import torch.optim as optim
import torchvision.transforms as T
import yaml
from models.classifier import Classifier
from torchvision.datasets import *

from datasets.celeba import CelebA1000, TargetCelebA
from datasets.custom_subset import Subset
from datasets.facescrub import FaceScrub
from datasets.ffhq import FFHQ
from utils.datasets import get_normalization


class TrainingConfigParser:
    def __init__(self, config_file):
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
        self._config = config

    def create_model(self):
        model_config = self._config['model']
        
        model = Classifier(**model_config)
        
        return model

    def save_config(self, save_dir):
        output_file = os.path.join(save_dir, 'config.yaml')
        
        # Save the configuration to the file
        with open(output_file, 'w') as file:
            yaml.safe_dump(self._config, file)
                
    def create_datasets(self):
        dataset_config = self._config['dataset']
        name = dataset_config['name'].lower()

        data_transformation_train = self.create_transformations(mode='train',
                                                                normalize=False)
        data_transformation_test = self.create_transformations(mode='test',
                                                               normalize=False)
        # Load datasets
        if name == 'facescrub':
            if 'facescrub_group' in dataset_config:
                group = dataset_config['facescrub_group']
            else:
                group = 'all'
            train_set = FaceScrub(group=group, train=True, cropped=True)
            test_set = FaceScrub(group=group,
                                 train=False,
                                 cropped=True,
                                 transform=data_transformation_test)

        elif name == 'celeba':
            # Note: 
            if self.align_train:
                train_set = None
            else:
                train_set = TargetCelebA(dataset_config, mode='train',
                                        transform=data_transformation_train)
                train_set.name = name + '_train'
            
            test_set  = TargetCelebA(dataset_config, mode='test',
                                    transform=data_transformation_test)
            test_set.name  = name + '_test'

        else:
            raise Exception(
                f'{name} is no valid dataset. Please use one of [\'facescrub\', \'celeba\', \'stanford_dogs_uncropped\'].'
            )

        if self.align_train:
            print(f'Created {name} datasets: '
                f'{len(test_set):,} test samples.\n')
        else:
            print(f'Created {name} datasets: '
                f'{len(train_set):,} train samples, '
                f'{len(test_set):,} test samples.\n')

        return train_set, test_set

    def create_transformations(self, mode, normalize=True):
        """
        mode: 'train' or 'test'
        """
        cfg = self._config['dataset']
        name = cfg['name'].lower()
        img_size   = cfg['transformations']['image_size']
        crop_size  = cfg['transformations']['crop_size']
        off_h      = cfg['transformations']['offset_height']
        off_w      = cfg['transformations']['offset_width']

        # crop function on a tensor
        crop_fn = lambda x: x[:, off_h:off_h+crop_size, off_w:off_w+crop_size]

        transformation_list = []

        # always convert to tensor first
        transformation_list.append(T.ToTensor())

        if 'celeba' in name:
            # 1) crop
            transformation_list.append(T.Lambda(crop_fn))
            # 2) resize to final image_size
            transformation_list.append(T.Resize(img_size, antialias=True))

        elif 'facescrub' in name:
            # only resize
            transformation_list.append(T.Resize(img_size, antialias=True))

        else:
            raise ValueError(f"Unsupported dataset '{name}'")

        # train‐only random flip
        if mode == 'train':
            transformation_list.append(T.RandomHorizontalFlip(p=0.5))

        # finally normalize, if requested
        if normalize:
            transformation_list.append(get_normalization())

        data_transformation = T.Compose(transformation_list)

        return data_transformation        

    def create_optimizer(self, model):
        optimizer_config = self._config['optimizer']
        for optimizer_type, args in optimizer_config.items():
            if not hasattr(optim, optimizer_type):
                raise Exception(
                    f'{optimizer_type} is no valid optimizer. Please write the type exactly as the PyTorch class'
                )

            optimizer_class = getattr(optim, optimizer_type)
            optimizer = optimizer_class(model.parameters(), **args)
            break

        return optimizer

    def create_lr_scheduler(self, optimizer):
        if not 'lr_scheduler' in self._config:
            return None

        scheduler_config = self._config['lr_scheduler']
        for scheduler_type, args in scheduler_config.items():
            if not hasattr(optim.lr_scheduler, scheduler_type):
                raise Exception(
                    f'{scheduler_type} is no valid learning rate scheduler. Please write the type exactly as the PyTorch class'
                )

            scheduler_class = getattr(optim.lr_scheduler, scheduler_type)
            scheduler = scheduler_class(optimizer, **args)
        return scheduler

    def load_state_dict(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                print(name)
                continue
            own_state[name].copy_(param.data)
    
    @property
    def experiment_name(self):
        return self._config['experiment_name']

    @property
    def adv_cfg(self):
        numerator, denominator = self._config['adv_cfg']['eps'].split('/')
        eps = float(numerator) / float(denominator)
        
        numerator, denominator = self._config['adv_cfg']['sts'].split('/')
        sts = float(numerator) / float(denominator)
        
        self._config['adv_cfg']['eps'] = eps
        self._config['adv_cfg']['sts'] = sts
        
        return self._config['adv_cfg']
    
    @property
    def model(self):
        return self._config['model']

    @property
    def dataset(self):
        return self._config['dataset']

    @property
    def optimizer(self):
        return self._config['optimizer']

    @property
    def lr_scheduler(self):
        return self._config['lr_scheduler']

    @property
    def training(self):
        return self._config['training']

    @property
    def align_train(self):
        if 'align_train' in self._config['training']:
            return True
        
        return False

    @property
    def rtpt(self):
        return self._config['rtpt']

    @property
    def seed(self):
        return self._config['seed']
