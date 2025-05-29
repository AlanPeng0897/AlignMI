import sys, math

import numpy as np
import torch

from datasets.custom_subset import SingleClassSubset
from pytorch_fid.inception import InceptionV3
import torch.nn.functional as F
import torchvision.transforms.functional as TF


def create_image(z,
                 generator,
                 crop_size=None,
                 resize=None,
                 batch_size=20,
                 device='cuda:0'):
    with torch.no_grad():
        if z.shape[1] == 1:
            z_expanded = torch.repeat_interleave(z,
                                                 repeats=generator.num_ws,
                                                 dim=1)
        else:
            z_expanded = z

        z_expanded = z_expanded.to(device)
        imgs = []
        for i in range(math.ceil(z_expanded.shape[0] / batch_size)):
            z_batch = z_expanded[i * batch_size:(i + 1) * batch_size]
            imgs_generated = generator(z_batch,
                                       noise_mode='const',
                                       force_fp32=True)
            imgs.append(imgs_generated.cpu())

        imgs = torch.cat(imgs, dim=0)
        if crop_size is not None:
            imgs = TF.center_crop(imgs, (crop_size, crop_size))
        if resize is not None:
            imgs = TF.resize(imgs, resize, antialias=True)
        return imgs
    

class PRCD:
    def __init__(self, dataset_real, dataset_fake, device, crop_size=None, generator=None, batch_size=128, dims=2048, num_workers=8, gpu_devices=[0]):
        self.dataset_real = dataset_real
        self.dataset_fake = dataset_fake
        self.batch_size = batch_size
        self.dims = dims
        self.num_workers = num_workers
        self.device = device
        self.generator = generator
        self.crop_size = crop_size
        
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[self.dims]
        inception_model = InceptionV3([block_idx])
        if len(gpu_devices) > 1:
            self.inception_model = torch.nn.DataParallel(inception_model, device_ids=gpu_devices)
        else:
            self.inception_model = inception_model
        self.inception_model.to(self.device)
        
    def compute_metric(self, num_classes, k=3, rtpt=None):
        precision_list = []
        recall_list = []
        density_list = []
        coverage_list = []
        for step, cls in enumerate(range(num_classes)):
            with torch.no_grad():
                embedding_fake = self.compute_embedding(self.dataset_fake, cls)
                embedding_real = self.compute_embedding(self.dataset_real, cls)
                pair_dist_real = torch.cdist(embedding_real, embedding_real, p=2)
                pair_dist_real = torch.sort(pair_dist_real, dim=1, descending=False)[0]
                pair_dist_fake = torch.cdist(embedding_fake, embedding_fake, p=2)
                pair_dist_fake = torch.sort(pair_dist_fake, dim=1, descending=False)[0]
                radius_real = pair_dist_real[:, k]
                radius_fake = pair_dist_fake[:, k]

                # Compute precision
                distances_fake_to_real = torch.cdist(embedding_fake, embedding_real, p=2)
                min_dist_fake_to_real, nn_real = distances_fake_to_real.min(dim=1)
                precision = (min_dist_fake_to_real <= radius_real[nn_real]).float().mean()
                precision_list.append(precision.cpu().item())

                # Compute recall
                distances_real_to_fake = torch.cdist(embedding_real, embedding_fake, p=2)
                min_dist_real_to_fake, nn_fake = distances_real_to_fake.min(dim=1)
                recall = (min_dist_real_to_fake <= radius_fake[nn_fake]).float().mean()
                recall_list.append(recall.cpu().item())

                # Compute density
                num_samples = distances_fake_to_real.shape[0]
                sphere_counter = (distances_fake_to_real <= radius_real.repeat(num_samples, 1)).float().sum(dim=0).mean()
                density = sphere_counter / k
                density_list.append(density.cpu().item())

                # Compute coverage
                num_neighbors = (distances_fake_to_real <= radius_real.repeat(num_samples, 1)).float().sum(dim=0)
                coverage = (num_neighbors > 0).float().mean()
                coverage_list.append(coverage.cpu().item())
                # Update rtpt
                if rtpt:
                    rtpt.step(
                        subtitle=f'PRCD Computation step {step} of {num_classes}')

        # Compute mean over targets
        precision = np.mean(precision_list)
        recall = np.mean(recall_list)
        density = np.mean(density_list)
        coverage = np.mean(coverage_list)
        return precision, recall, density, coverage

    def compute_embedding(self, dataset, labels, cls=None):
        self.inception_model.eval()
        if cls:
            dataset = SingleClassSubset(dataset, labels, cls)
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=self.batch_size,
                                                 shuffle=False,
                                                 drop_last=False,
                                                 pin_memory=True,
                                                 num_workers=self.num_workers)
        pred_arr = np.empty((len(dataset), self.dims))
        start_idx = 0
        for step, (x, y) in enumerate(dataloader):
            with torch.no_grad():
                if x.shape[1] != 3:
                    try:
                        x = self.generator(x)
                    except:
                        x = create_image(x, self.generator,
                                        crop_size=self.crop_size, resize=64, batch_size=int(self.batch_size / 2))

                x = x.to(self.device)
                pred = self.inception_model(x)[0]

            if pred.shape[2] != 1 or pred.shape[3] != 1:
                pred = F.adaptive_avg_pool2d(pred, output_size=(1, 1))

            pred = pred.squeeze(3).squeeze(2).cpu().numpy()
            pred_arr[start_idx:start_idx + pred.shape[0]] = pred
            start_idx = start_idx + pred.shape[0]

        return torch.from_numpy(pred_arr)