from copy import copy
from typing import List

import numpy as np
import torch
import torch.optim as optim
import torchvision.transforms as T
import yaml
from attacks.initial_selection import find_initial_z, find_initial_w
from matplotlib.pyplot import fill
from models.classifier import Classifier
from datasets.celeba import CelebA1000, TargetCelebA
from datasets.facescrub import FaceScrub
from datasets.ffhq import FFHQ
from utils.datasets import get_normalization
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class AttackConfigParser:

    def __init__(self, config_file):
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)

        self._config = config

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
        
        # Load datasets
        if name == 'facescrub':
            if 'facescrub_group' in dataset_config:
                group = dataset_config['facescrub_group']
            else:
                group = 'all'
            train_set = FaceScrub(group=group,
                                  train=True,
                                  cropped=True,
                                  transform=data_transformation_train)

            private_set = FaceScrub(group=group, train=True, cropped=True)

        elif name == 'celeba':
            train_set = TargetCelebA(dataset_config, mode='train',
                                     transform=data_transformation_train)
            train_set.name = name
            private_set = TargetCelebA(dataset_config, mode='private',
                                       transform=data_transformation_train)
            private_set.name = name
        else:
            raise Exception(
                f'{name} is no valid dataset. Please use one of [\'facescrub\', \'celeba\'].'
            )

        print(
            f'Created {name} datasets with {len(train_set):,} training, {len(private_set):,} private samples.\n')

        return train_set, private_set 


    def create_transformations(self, mode, normalize=True):
        """
        mode: 'train' or 'test'
        """
        dataset_config = self._config['dataset']
        dataset_name = self._config['dataset']['name'].lower()
        crop_size = dataset_config['crop_size']
        image_size = dataset_config['image_size']
        offset_height = dataset_config['offset_height']
        offset_width = dataset_config['offset_width']

        crop = lambda x: x[:, offset_height:offset_height + crop_size, offset_width:offset_width + crop_size]

        transformation_list = []
        if dataset_name == 'celeba':
            # resize images to the expected size
            transformation_list.append(T.ToTensor())
            transformation_list.append(T.Lambda(crop))
            if mode == 'train' and 'transformations' in self._config:
                transformations = self._config['transformations']
                if transformations != None:
                    for transform, args in transformations.items():
                        if not hasattr(T, transform):
                            raise Exception(
                                f'{transform} is no valid transformation. Please write the type exactly as the Torchvision class'
                            )
                        else:
                            transformation_class = getattr(T, transform)
                            transformation_list.append(transformation_class(**args))
            else:
                raise Exception(f'{mode} is no valid mode for augmentation')

        transformation_list.append(T.ToTensor())

        if normalize:
            transformation_list.append(get_normalization())
        data_transformation = T.Compose(transformation_list)

        return data_transformation

    def create_target_models(self):
        if 'target_model' in self._config:
            target_config = self._config['target_model']
            dataset_config = self._config['dataset']
            n_classes = target_config['n_classes']

            model_types_ = target_config['architecture'].split(',')
            checkpoints = target_config['weights'].split(',')

            dataset = self._config['dataset']['name']
            cid = self.classid.split(',')

            for i in range(len(cid)):
                id_ = int(cid[i])
                model_types_[id_] = model_types_[id_].strip()
                checkpoints[id_] = checkpoints[id_].strip()
                print('Load classifier {} at {}'.format(model_types_[id_], checkpoints[id_]))
                model = Classifier(n_classes, architecture=model_types_[id_], resume_path=checkpoints[id_])
                model = torch.nn.DataParallel(model).cuda()
                model.name = model_types_[id_]
                model = model.to(device)
                model = model.eval()
                if i == 0:
                    targetnets = [model]
                else:
                    targetnets.append(model)

                # p_reg
                if self._config['attack']['loss'] == 'logit_loss':
                    if model_types_[id_] == 'IR152' or model_types_[id_] == 'VGG16' or model_types_[id_] == 'FaceNet64':
                        # target model
                        p_reg = os.path.join(dataset_config['p_reg_path'],
                                             '{}_{}_p_reg.pt'.format(dataset, model_types_[id_])) 
                    else:
                        # aug model
                        p_reg = os.path.join(dataset_config['p_reg_path'],
                                             '{}_{}_{}_p_reg.pt'.format(dataset, model_types_[0], model_types_[id_]))
                    
                    if not os.path.exists(p_reg):
                        print('p_reg not exist')
                        exit()
                        gan_dataset = self.create_datasets(mode='gan')
                        dataloader_gan = torch.utils.data.DataLoader(gan_dataset,
                                                                  batch_size=64,
                                                                  shuffle=True,
                                                                  drop_last=True,
                                                                  num_workers=2,
                                                                  pin_memory=True)

                        fea_mean_, fea_logvar_ = self.get_act_reg(dataloader_gan, model, device)
                        torch.save({'fea_mean': fea_mean_, 'fea_logvar': fea_logvar_}, p_reg)
                    else:
                        fea_reg = torch.load(p_reg)
                        fea_mean_ = fea_reg['fea_mean']
                        fea_logvar_ = fea_reg['fea_logvar']
                    if i == 0:
                        fea_mean = [fea_mean_.to(device)]
                        fea_logvar = [fea_logvar_.to(device)]
                    else:
                        fea_mean.append(fea_mean_)
                        fea_logvar.append(fea_logvar_)

                else:
                    fea_mean, fea_logvar = 0, 0

            return targetnets, fea_mean, fea_logvar

        else:
            raise RuntimeError('No target model stated in the config file.')

    def get_act_reg(train_loader, T, device, Nsample=5000):
        all_fea = []
        with torch.no_grad():
            for batch_idx, data in enumerate(train_loader):  # batchsize =100
                # print(data.shape)
                if batch_idx * len(data) > Nsample:
                    break
                data = data.to(device)
                fea, _ = T(data)
                if batch_idx == 0:
                    all_fea = fea
                else:
                    all_fea = torch.cat((all_fea, fea))
        fea_mean = torch.mean(all_fea, dim=0)
        fea_logvar = torch.std(all_fea, dim=0)

        print(fea_mean.shape, fea_logvar.shape, all_fea.shape)
        return fea_mean, fea_logvar

    def create_evaluation_model(self):
        if 'evaluation_model' in self._config:
            config = self._config['evaluation_model']
            evaluation_model = Classifier(num_classes=config['num_classes'],
                                          architecture=config['architecture'],
                                          resume_path=config['weights'])
        else:
            raise RuntimeError(
                'No evaluation model stated in the config file.')

        evaluation_model.eval()
        self.evaluation_model = evaluation_model
        return evaluation_model

    def create_optimizer(self, params, config=None):
        if config is None:
            optimizer_config = self._config['optimizer']

        for optimizer_type, args in optimizer_config.items():
            if not hasattr(optim, optimizer_type):
                raise Exception(
                    f'{optimizer_type} is no valid optimizer. Please write the type exactly as the PyTorch class'
                )

            optimizer_class = getattr(optim, optimizer_type)
            optimizer = optimizer_class(params, **args)
            break
        return optimizer

    def create_lr_scheduler(self, optimizer):
        if not 'lr_scheduler' in self._config['attack']:
            return None

        scheduler_config = self._config['attack']['lr_scheduler']
        for scheduler_type, args in scheduler_config.items():
            if not hasattr(optim.lr_scheduler, scheduler_type):
                raise Exception(
                    f'{scheduler_type} is no valid learning rate scheduler. Please write the type exactly as the PyTorch class.'
                )

            scheduler_class = getattr(optim.lr_scheduler, scheduler_type)
            scheduler_instance = scheduler_class(optimizer, **args)
            break
        return scheduler_instance

    def create_candidates(self, stylegan, generator, targets, targetnets, z_dim, num_candidates, plg=False, distribution=None, seed=0):
        print("Init selection")
        candidate_config = self._config['candidates']
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if plg:
            all_candidates = [
                torch.empty(num_candidates, z_dim, dtype=torch.float32, device=device)
                .normal_() if distribution == 'normal' else torch.empty(num_candidates, z_dim, dtype=torch.float32, device=device).uniform_()
                for target in torch.unique(targets)
            ]
            
            return torch.cat(all_candidates, dim=0)

        if stylegan:
            candidate_config = {
                "candidate_search": {
                    "search_space_size": 2000,
                    "center_crop": 64,
                    "resize": 64,
                    "horizontal_flip": True,
                    "batch_size": 50,
                    "truncation_psi": 0.5,
                    "truncation_cutoff": 8
                }
            }  
            search_config = candidate_config['candidate_search']  
            w = find_initial_w(generator=generator,
                                target_model=targetnets[0],
                                targets=targets,
                                seed=seed,
                                **search_config)
            print(f'Created {w.shape[0]} candidates randomly in w space.')
            w = w.to(device)
            return w
        
        else:
            candidate_config = {
                "candidate_search": {
                    "search_space_size": 2000,
                    "center_crop": 64,
                    "resize": 64,
                    "horizontal_flip": True,
                    "batch_size": 200
                }
            }  
            search_config = candidate_config['candidate_search']  

            z = find_initial_z(
                generator=generator,
                target_model=targetnets[0],
                targets=targets,
                z_dim=z_dim,
                seed=seed,
                **search_config
            )

            print(f'Created {z.shape[0]} candidates based on highest confidence in z space.')

            z = z.to(device)
            return z

    def create_target_vector(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        attack_config = self._config['attack']
        targets = None
        target_classes = attack_config['targets']
        num_candidates = self._config['candidates']['num_candidates']
        
        if type(target_classes) is list:
            targets = torch.tensor(target_classes)
            targets = torch.repeat_interleave(targets, num_candidates)
        elif target_classes == 'all':
            targets = torch.tensor([i for i in range(self.model.num_classes)])
            targets = torch.repeat_interleave(targets, num_candidates)
        elif type(target_classes) == int:
            targets = torch.tensor([i for i in range(target_classes)])
            targets = torch.repeat_interleave(targets, num_candidates)
            # targets = torch.full(size=(num_candidates, ),
            #                      fill_value=target_classes)
        elif type(target_classes) is str:
            start, end = map(int, target_classes.split('-'))
            print(f"start {start} end {end}")
            targets = torch.tensor([i for i in range(start, end)])
            targets = torch.repeat_interleave(targets, num_candidates)
        else:
            raise Exception(
                f' Please specify a target class or state a target vector.')

        targets = targets.to(device)
        return targets
    

    @property
    def target_model_config(self):
        if 'target_model' in self._config:
            config = self._config['target_model']

        return config

    @property
    def candidates(self):
        return self._config['candidates']

    @property
    def wandb_target_run(self):
        if 'wandb_target_run' in self._config:
            return self._config['wandb_target_run']
        elif 'run_id' in self._config:
            return self._config['run_id']

    @property
    def logging(self):
        return self._config['wandb']['enable_logging']

    @property
    def wandb_init_args(self):
        return self._config['wandb']['wandb_init_args']

    @property
    def attack(self):
        return self._config['attack']

    @property
    def wandb(self):
        return self._config['wandb']

    @property
    def optimizer(self):
        return self._config['optimizer']

    @property
    def lr_scheduler(self):
        return self._config['lr_scheduler']

    @property
    def final_selection(self):
        if 'final_selection' in self._config:
            return self._config['final_selection']
        else:
            return None

    @property
    def gan(self):
        return self._config['gan']

    @property
    def seed(self):
        return self._config['seed']

    @property
    def cas_evaluation(self):
        return self._config['cas_evaluation']

    @property
    def dataset(self):
        return self._config['dataset']

    @property
    def fid_evaluation(self):
        return self._config['fid_evaluation']

    @property
    def attack_center_crop(self):
        if 'transformations' in self._config['attack']:
            if 'CenterCrop' in self._config['attack']['transformations']:
                return self._config['attack']['transformations']['CenterCrop'][
                    'size']
        else:
            return None

    @property
    def attack_resize(self):
        if 'transformations' in self._config['attack']:
            if 'Resize' in self._config['attack']['transformations']:
                return self._config['attack']['transformations']['Resize'][
                    'size']
        else:
            return None

    @property
    def num_classes(self):
        targets = self._config['attack']['targets']
        if isinstance(targets, int):
            return 1
        else:
            return len(targets)

    @property
    def improved_flag(self):
        attack_method = self._config['attack']['method']
        if attack_method == 'kedmi':
            return True
        else:
            return False

    @property
    def clipz(self):
        attack_method = self._config['attack']['method']
        if attack_method == 'kedmi':
            return True
        else:
            return False

    @property
    def root_path(self):
        return self._config['root_path']

    @property
    def classid(self):
        return self._config['attack']['classid']

    @property
    def log_progress(self):
        if 'log_progress' in self._config['attack']:
            return self._config['attack']['log_progress']
        else:
            return True
