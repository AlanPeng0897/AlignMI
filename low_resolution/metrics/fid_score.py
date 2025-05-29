import numpy as np
import torch
from torch.utils.data import DataLoader
from pytorch_fid.inception import InceptionV3
from scipy.linalg import sqrtm
import torch.nn.functional as F
import torchvision.transforms.functional as TF

import math

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
    

class FID_Score:
    def __init__(self, dataset_1, dataset_2, device, crop_size=None, generator=None, batch_size=128, dims=512, num_workers=8, gpu_devices=[0]):
        self.dataset_1 = dataset_1
        self.dataset_2 = dataset_2
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

    def compute_fid(self, rtpt=None):
        m1, s1 = self.compute_statistics(self.dataset_1, rtpt)
        m2, s2 = self.compute_statistics(self.dataset_2, rtpt)
        fid_value = self.calculate_frechet_distance(m1, s1, m2, s2)
        return fid_value

    def compute_statistics(self, dataset, rtpt=None):
        self.inception_model.eval()
        dataloader = DataLoader(dataset,
                                batch_size=self.batch_size,
                                shuffle=False,
                                drop_last=False,
                                pin_memory=True,
                                num_workers=self.num_workers)
        pred_arr = np.empty((len(dataset), self.dims))
        start_idx = 0
        max_iter = int(len(dataset) / self.batch_size)
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
            
            # Apply global average pooling to match expected shape [B, C]
            if pred.shape[2] != 1 or pred.shape[3] != 1:
                pred = F.adaptive_avg_pool2d(pred, output_size=(1, 1))

            pred = pred.squeeze(3).squeeze(2).cpu().numpy()
            pred_arr[start_idx:start_idx + pred.shape[0]] = pred
            start_idx = start_idx + pred.shape[0]

            if rtpt:
                rtpt.step(
                    subtitle=f'FID Score Computation step {step} of {max_iter}')

        mu = np.mean(pred_arr, axis=0)
        sigma = np.cov(pred_arr, rowvar=False)
        return mu, sigma

    def calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2):
        """Calculate the Frechet Distance between two distributions."""
        diff = mu1 - mu2
        covmean = sqrtm(sigma1.dot(sigma2))
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2.0 * covmean)
        return fid
