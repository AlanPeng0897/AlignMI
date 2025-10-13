import numpy as np
import torch
from torch.utils.data import Dataset
import wandb
import yaml
import os


class AttackLatents(Dataset):

    def __init__(self, attack_run_path=None, latent_file=None, transform=None):

        assert bool(attack_run_path) != bool(
            latent_file
        ), 'Either attack_run_path or latent_file must be specified'

        self.attack_run_path = attack_run_path
        self.latent_file = latent_file
        self.transform = transform

        if attack_run_path:
            weights_file_name = 'results/optimized_w_selected_' + attack_run_path.split(
                '/')[-1] + '.pt'
            w_optimized = wandb.restore(weights_file_name,
                                        run_path=attack_run_path,
                                        replace=True,
                                        root='wandb/downloads')
            self.latents = torch.load(w_optimized.name)

        api = wandb.Api()
        run = api.run(attack_run_path)
        self.target_identities = run.config['targets']
        if self.target_identities == 'all':
            self.num_classes = len(self.latents) // run.config['final_samples']
        else:
            self.num_classes = len(self.target_identities)
        samples_per_class = len(self.latents) // self.num_classes
        targets = [[i for j in range(samples_per_class)]
                   for i in range(self.num_classes)]
        self.targets = [t for sublist in targets for t in sublist]

        assert len(self.latents) == len(self.targets)

    def __len__(self):
        return len(self.latents)

    def __getitem__(self, idx):
        latent = self.latents[idx]
        if self.transform:
            latent = self.transform(latent)
        return latent, self.targets[idx]

    
class AttackLatentsLocal(Dataset):
    """
    从类似下述结构中加载 final_w_unselected.pt 文件：
        res18_0_1000/
            000/
                final_w_unselected.pt
            001/
                final_w_unselected.pt
            ...
    文件夹名 (如 '000','001') 用作类别标签。
    如果 final_w_unselected.pt 中保存的是多个 latent，可在加载后遍历并分别存储。
    """

    def __init__(self, root_dir, transform=None):
        """
        参数：
            root_dir: 根目录，如 "res18_0_1000"
            transform: 对 latent 进行的预处理或转换（可选）
        """
        self.root_dir = root_dir
        self.transform = transform
        self.latents = []
        self.targets = []

        # 列出所有子文件夹 (类别文件夹)
        subfolders = sorted(
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        )

        for class_name in subfolders:
            class_path = os.path.join(root_dir, class_name)
            latent_path = os.path.join(class_path, 'final_w_unselected.pt')
            if os.path.isfile(latent_path):
                data = torch.load(latent_path)
                # 此处只处理形状为 (N, 1, 512) 的 tensor
                if isinstance(data, torch.Tensor) and data.dim() == 3:
                    # 遍历第一维，将每个 latent code 单独存储
                    for latent in data:
                        self.latents.append(latent)
                        self.targets.append(int(class_name))
                else:
                    print(f"Warning: Unexpected tensor shape {data.shape} in {latent_path}")
            else:
                print(f"Warning: {latent_path} not found, skipping.")
                pass

    def __len__(self):
        return len(self.latents)

    def __getitem__(self, idx):
        latent = self.latents[idx]
        label = self.targets[idx]
        if self.transform:
            latent = self.transform(latent)
        return latent, label
    
    