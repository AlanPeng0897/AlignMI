from copy import copy
from typing import List

import numpy as np
import torch
import torch.optim as optim
import torchvision.transforms as T
import yaml, os
from attacks.initial_selection import find_initial_w
from matplotlib.pyplot import fill
from models.classifier import Classifier

import wandb
from utils.wandb import load_model


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

    def create_target_model(self):
        if 'wandb_target_run' in self._config:
            model = load_model(self._config['wandb_target_run'])
        elif 'target_model' in self._config:
            config = self._config['target_model']
            model = Classifier(num_classes=config['num_classes'],
                               architecture=config['architecture'])
            # model.load_state_dict(torch.load(config['weights']))

            try:
                state_dict = torch.load(config['weights'], map_location="cpu")['model_state_dict']
            except:
                state_dict = torch.load(config['weights'], map_location="cpu")['state_dict']

            new_state_dict = {}
            for key in state_dict:
                # new_key = key.replace('model._orig_mod.', 'model.')
                new_key = key.replace('module.', '')

                new_state_dict[new_key] = state_dict[key]

            tmp = model.load_state_dict(new_state_dict)
            print(tmp)

            model.wandb_name = None
        else:
            raise RuntimeError('No target model stated in the config file.')

        model.eval()
        self.model = model
        return model

    def get_target_dataset(self):
        try:
            api = wandb.Api(timeout=60)
            run = api.run(self._config['wandb_target_run'])
            return run.config['Dataset'].strip().lower()
        except:
            return self._config['dataset']

    def create_evaluation_model(self):
        if 'wandb_evaluation_run' in self._config:
            evaluation_model = load_model(self._config['wandb_evaluation_run'])
        elif 'evaluation_model' in self._config:
            config = self._config['evaluation_model']
            evaluation_model = Classifier(num_classes=config['num_classes'],
                                          architecture=config['architecture'])
            # evaluation_model.load_state_dict(torch.load(config['weights']))
            tmp = torch.load(config['weights'])['model_state_dict']
            new_state_dict = {}
            for key in tmp:
                new_key = key.replace('model._orig_mod.', 'model.')
                new_state_dict[new_key] = tmp[key]

            evaluation_model.load_state_dict(new_state_dict)

        else:
            raise RuntimeError(
                'No evaluation model stated in the config file.')

        evaluation_model.eval()
        self.evaluation_model = evaluation_model
        return evaluation_model

    def create_adv_trained_model(self):
        if 'adv_trained_model' in self._config:
            config = self._config['adv_trained_model']
            adv_model = Classifier(num_classes=config['num_classes'],
                                   architecture=config['architecture'])
            # 加载权重，并对 state_dict 做必要的处理
            tmp = torch.load(config['weights'])['model_state_dict']
            new_state_dict = {}
            for key in tmp:
                new_key = key.replace('model._orig_mod.', 'model.')
                new_state_dict[new_key] = tmp[key]
            adv_model.load_state_dict(new_state_dict)
        else:
            raise RuntimeError('No adv_trained_model stated in the config file.')

        adv_model.eval()
        self.adv_trained_model = adv_model
        return adv_model

    def create_optimizer(self, params, config=None):
        if config is None:
            config = self._config['attack']['optimizer']

        optimizer_config = self._config['attack']['optimizer']
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

    def create_candidates(self, generator, target_model, targets):
        candidate_config = self._config['candidates']
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if 'candidate_file' in candidate_config:

            candidate_file = candidate_config['candidate_file']

            if os.path.exists(candidate_file):
                w = torch.load(candidate_file)
                print(f'Loaded {w.shape[0]} candidates from {candidate_file}.')

            elif 'candidate_search' in candidate_config:
                search_config = candidate_config['candidate_search']
                w = find_initial_w(generator=generator,
                                   target_model=target_model,
                                   targets=targets,
                                   seed=self.seed,
                                   **search_config)
                dir_path = os.path.dirname(candidate_file)
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path, exist_ok=True)

                torch.save(w, candidate_file)
                print(f'Created {w.shape[0]} candidates randomly in w space.')
                print(f'Candidates have been saved to {candidate_file}')

        elif 'candidate_file' not in candidate_config:
            search_config = candidate_config['candidate_search']

            w = find_initial_w(generator=generator,
                               target_model=target_model,
                               targets=targets,
                               seed=self.seed,
                               **search_config)

            print(f'Created {w.shape[0]} candidates randomly in w space.')
        else:
            raise Exception(f'No valid candidate initialization stated.')

        w = w.to(device)
        return w

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
            targets = torch.full(size=(num_candidates,),
                                 fill_value=target_classes)
        elif isinstance(target_classes, str) and '-' in target_classes:
            start, end = map(int, target_classes.split('-'))
            targets = torch.tensor([i for i in range(start, end)])
            targets = torch.repeat_interleave(targets, num_candidates)
        else:
            raise Exception(
                f' Please specify a target class or state a target vector.')

        targets = targets.to(device)
        return targets

    def create_wandb_config(self):
        for _, args in self.optimizer.items():
            lr = args['lr']
            break

        config = {
            **self.attack, 'optimizer': self.optimizer,
            'lr': lr,
            'use_scheduler': 'lr_scheduler' in self._config,
            'target_architecture': self.model.architecture,
            'target_extended': self.model.wandb_name,
            'selection_method': self.final_selection['approach'],
            'final_samples': self.final_selection['samples_per_target']
        }
        if 'lr_scheduler' in self._config:
            config['lr_scheduler'] = self.lr_scheduler

        return config

    def create_attack_transformations(self):
        transformation_list = []
        if 'transformations' in self._config['attack']:
            transformations = self._config['attack']['transformations']
            for transform, args in transformations.items():
                if not hasattr(T, transform):
                    raise Exception(
                        f'{transform} is no valid transformation. Please write the type exactly as the Torchvision class'
                    )
                transformation_class = getattr(T, transform)
                transformation_list.append(transformation_class(**args))
        if len(transformation_list) > 0:
            attack_transformations = T.Compose(transformation_list)
            return attack_transformations

        return None

    @property
    def candidates(self):
        return self._config['candidates']

    @property
    def wandb_target_run(self):
        return self._config['wandb_target_run']

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
        return self._config['attack']['optimizer']

    @property
    def lr_scheduler(self):
        return self._config['attack']['lr_scheduler']

    @property
    def final_selection(self):
        if 'final_selection' in self._config:
            return self._config['final_selection']
        else:
            return None

    @property
    def stylegan_model(self):
        return self._config['stylegan_model']

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
    def target_classes(self):
        target_classes = self._config['attack']['targets']
        if type(target_classes) is list:
            targets = target_classes
        elif target_classes == 'all':
            targets = [i for i in range(self.model.num_classes)]
        elif type(target_classes) == int:
            targets = [target_classes]
        elif isinstance(target_classes, str) and '-' in target_classes:
            start, end = map(int, target_classes.split('-'))
            targets = [i for i in range(start, end)]

        return targets

    @property
    def log_progress(self):
        if 'log_progress' in self._config['attack']:
            return self._config['attack']['log_progress']
        else:
            return True
