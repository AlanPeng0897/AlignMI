from losses.poincare import poincare_loss
import math, os

import numpy as np
import torch, torchvision
import torch.fft
import torch.nn as nn
import PIL
from torch.autograd.functional import jacobian
from torch.autograd.functional import jvp

import torchvision.utils as vutils
import matplotlib.pyplot as plt
from skimage import exposure
from matplotlib import colors
from matplotlib.colors import TwoSlopeNorm
from PIL import Image

import torchvision.transforms as T
import torch.nn.functional as F
from torchvision.transforms import InterpolationMode

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def build_transformations(transformations_dict=None) -> T.Compose:
    transformations_dict = {
        "RandomResizedCrop": {
            "size": 224,
            "scale": [0.8, 1.0],
            "ratio": [0.9, 1.1],
            "antialias": True
        },
        "RandomHorizontalFlip": {
            "p": 0.5
        },
        "RandomRotation": {
            "degrees": 5,  # ±5°
            "interpolation": InterpolationMode.BILINEAR,
            "expand": False,
            "center": None
        },
    }

    transformation_list = []

    for transform, args in transformations_dict.items():
        if not hasattr(T, transform):
            raise Exception(
                f"{transform} is not a valid transformation. Please write the type exactly as the Torchvision class."
            )
        transformation_class = getattr(T, transform)
        transformation_list.append(transformation_class(**args))

    if transformation_list:
        return T.Compose(transformation_list)
    else:
        return None


def save_image_grid(img, fname, drange, grid_size):
    lo, hi = drange
    img = np.asarray(img, dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo))
    img = np.rint(img).clip(0, 255).astype(np.uint8)

    gw, gh = grid_size
    _N, C, H, W = img.shape
    img = img.reshape(gh, gw, C, H, W)
    img = img.transpose(0, 3, 1, 4, 2)
    img = img.reshape(gh * H, gw * W, C)

    assert C in [1, 3]
    if C == 1:
        PIL.Image.fromarray(img[:, :, 0], 'L').save(fname)
    if C == 3:
        PIL.Image.fromarray(img, 'RGB').save(fname)


def save_grid_with_cmap(tensor, save_path, cmap="seismic", nrow=5, padding=2, handle_imgs=False):
    # RdBu / viridis / seismic

    # Per-sample max-abs normalization to keep structure while normalizing
    max_vals = tensor.view(tensor.size(0), -1).abs().max(dim=1)[0]  # shape: (B,)
    max_vals = max_vals.view(-1, 1, 1, 1) + 1e-8  # reshape for broadcasting
    tensor = tensor / max_vals  # shape-preserving normalization

    if handle_imgs:
        grid = vutils.make_grid(tensor, nrow=nrow, normalize=True, scale_each=False, padding=padding, pad_value=0)
        grid_np = grid.cpu().numpy()
        grid_np = np.transpose(grid_np, (1, 2, 0))
        plt.imsave(save_path, grid_np)
    else:
        grid = vutils.make_grid(tensor, nrow=nrow, normalize=False, scale_each=False, padding=padding, pad_value=-1.0)
        grid_np = grid.cpu().numpy()
        if grid_np.shape[0] == 1:
            grid_np = grid_np[0]  # (H, W)
        else:
            grid_np = np.mean(grid_np, axis=0)  # (H, W)

        # Display and save
        plt.figure(figsize=(10, 10))

        # 1) Alternative: plt.imshow(grid_np, cmap=cmap, norm=colors.CenteredNorm())
        # 2) Use symmetric norm around zero
        vmax = np.max(np.abs(grid_np))
        # norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        norm = colors.CenteredNorm()
        plt.imshow(grid_np, cmap=cmap, norm=norm)

        plt.axis("off")
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0, dpi=300)
        plt.close()


def cosine_similarity_batch(a: torch.Tensor, b: torch.Tensor):
    """
    Compute per-sample cosine similarity between two tensors of shape (B, C, H, W)
    """
    a = a.view(a.size(0), -1)  # (B, D)
    b = b.view(b.size(0), -1)  # (B, D)

    a_norm = a / (a.norm(dim=1, keepdim=True) + 1e-8)
    b_norm = b / (b.norm(dim=1, keepdim=True) + 1e-8)

    return (a_norm * b_norm).sum(dim=1)  # shape (B,)


# Visualize dL/dx
class Optimization():
    def __init__(self, target_model, synthesis, discriminator, transformations, num_ws, config, save_dir=None):
        self.synthesis = synthesis
        self.target = target_model
        self.discriminator = discriminator
        self.config = config
        self.transformations = transformations
        self.discriminator_weight = self.config.attack['discriminator_loss_weight']
        self.num_ws = num_ws
        self.clip = config.attack['clip']
        self.save_dir = save_dir

        self.log_progress = 1
        self.visualize_intermediate_results = 1

    def G_forward(self, w):
        return self.synthesize(w, num_ws=self.num_ws)

    def optimize(self, batch_i, w_batch, targets_batch, num_epochs):
        # Initialize attack
        optimizer = self.config.create_optimizer(params=[w_batch.requires_grad_()])
        scheduler = self.config.create_lr_scheduler(optimizer)

        self.visualize_intermediate_results = 1

        # Start optimization
        for i in range(num_epochs):
            # synthesize imagesnd preprocess images
            imgs = self.synthesize(w_batch, num_ws=self.num_ws)

            # compute discriminator loss
            if self.discriminator_weight > 0:
                discriminator_loss = self.compute_discriminator_loss(imgs)
            else:
                discriminator_loss = torch.tensor(0.0, device=imgs.device)

            # perform image transformations
            if self.clip:
                imgs = self.clip_images(imgs)
            if self.transformations:
                imgs = self.transformations(imgs)

            # Compute target loss
            outputs = self.target(imgs)
            target_loss = poincare_loss(
                outputs, targets_batch).mean()

            # combine losses and compute gradients
            optimizer.zero_grad()
            loss = target_loss + discriminator_loss * self.discriminator_weight

            imgs.retain_grad()
            loss.backward()  # backward once

            grad_x = imgs.grad.detach().cpu()

            vis_condition = self.save_dir and ((i + 1) % 20 == 0 or i == 0 or (i + 1) == 70)

            optimizer.step()

            if scheduler:
                scheduler.step()

            if self.log_progress or self.visualize_intermediate_results:
                with torch.no_grad():
                    confidence_vector = outputs.softmax(dim=1)
                    confidences = torch.gather(confidence_vector, 1, targets_batch.unsqueeze(1))
                    mean_conf = confidences.mean().detach().cpu()
                    min_conf = confidences.min().detach().cpu()
                    max_conf = confidences.max().detach().cpu()

            # Log results
            if self.log_progress:
                log_condition = (i + 1) % 10 == 0 or i == 0
                if log_condition:
                    print(
                        f'iteration {i + 1}: \t total_loss={loss:.4f} \t target_loss={target_loss:.4f} \t',
                        f'discriminator_loss={discriminator_loss:.4f} \t mean_conf={mean_conf:.4f} ({min_conf:.4f}, {max_conf:.4f})'
                    )

            self.visualize_intermediate_results = 0
            if self.visualize_intermediate_results:
                if vis_condition:
                    torchvision.utils.save_image(
                        torchvision.utils.make_grid(imgs.detach().cpu(), nrow=5, normalize=True, scale_each=True),
                        os.path.join(self.save_dir, f'Batch_{batch_i + 1}_iter_{i + 1:02d}_Acc_{mean_conf:.4f}.png')
                    )

                    save_grid_with_cmap(
                        grad_x, os.path.join(self.save_dir,
                                             f'grad_x_Batch_{batch_i + 1}_iter_{i + 1:02d}_Acc_{mean_conf:.4f}.png'),
                        nrow=5
                    )

                    save_grid_with_cmap(
                        proj_x, os.path.join(self.save_dir,
                                             f'proj_x_Batch_{batch_i + 1}_iter_{i + 1:02d}_Acc_{mean_conf:.4f}.png'),
                        nrow=5
                    )

                    del grad_w, grad_x, proj_x

            torch.cuda.empty_cache()

        return w_batch.detach()

    def optimize_demo(self, batch_i, w_batch, targets_batch, num_epochs):
        # Initialize attack
        optimizer = self.config.create_optimizer(params=[w_batch.requires_grad_()])
        scheduler = self.config.create_lr_scheduler(optimizer)

        self.visualize_intermediate_results = 0

        # List to store (epoch_num, norm_ratio) pairs
        epoch_records = []

        # Start optimization
        for i in range(num_epochs):
            # synthesize imagesnd preprocess images
            imgs = self.synthesize(w_batch, num_ws=self.num_ws)

            # compute discriminator loss
            if self.discriminator_weight > 0:
                discriminator_loss = self.compute_discriminator_loss(imgs)
            else:
                discriminator_loss = torch.tensor(0.0, device=imgs.device)

            # perform image transformations
            if self.clip:
                imgs = self.clip_images(imgs)
            if self.transformations:
                imgs = self.transformations(imgs)

            # Compute target loss
            outputs = self.target(imgs)
            target_loss = poincare_loss(
                outputs, targets_batch).mean()

            # combine losses and compute gradients
            optimizer.zero_grad()
            loss = target_loss + discriminator_loss * self.discriminator_weight

            imgs.retain_grad()
            loss.backward()  # backward once

            grad_x = imgs.grad.detach()

            vis_condition = self.save_dir and ((i + 1) % 20 == 0 or i == 0 or (i + 1) == 70)
            # if self.visualize_intermediate_results and vis_condition:
            if (i + 1) % 10 == 0:
                #
                grad_w = w_batch.grad.detach()
                proj_x = self.compute_proj_x_orthogonalized(imgs, w_batch, grad_x, chunk_size=50)
                proj_x = proj_x.detach()

                B = grad_x.shape[0]

                norm_grad = torch.norm(grad_x.view(B, -1), dim=1)
                norm_proj = torch.norm(proj_x.view(B, -1), dim=1)
                norm_ratio = norm_proj / norm_grad

                mean_ratio = norm_ratio.mean().item()
                min_ratio = norm_ratio.min().item()
                max_ratio = norm_ratio.max().item()

                with torch.no_grad():
                    confidence_vector = outputs.softmax(dim=1)
                    confidences = torch.gather(confidence_vector, 1, targets_batch.unsqueeze(1))

                for idx in range(B):
                    epoch_records.append({
                        'epoch': i + 1,
                        'cosine': norm_ratio[idx].item(),
                        'conf': confidences[idx].item(),
                    })

                print(f"[Norm Ratio] mean: {mean_ratio:.4f}, min: {min_ratio:.4f}, max: {max_ratio:.4f}")

            optimizer.step()

            if scheduler:
                scheduler.step()

            if self.log_progress or self.visualize_intermediate_results:
                with torch.no_grad():
                    confidence_vector = outputs.softmax(dim=1)
                    confidences = torch.gather(confidence_vector, 1, targets_batch.unsqueeze(1))
                    mean_conf = confidences.mean().detach().cpu()
                    min_conf = confidences.min().detach().cpu()
                    max_conf = confidences.max().detach().cpu()

            # Log results
            if self.log_progress:
                log_condition = (i + 1) % 10 == 0 or i == 0
                if log_condition:
                    print(
                        f'iteration {i + 1}: \t total_loss={loss:.4f} \t target_loss={target_loss:.4f} \t',
                        f'discriminator_loss={discriminator_loss:.4f} \t mean_conf={mean_conf:.4f} ({min_conf:.4f}, {max_conf:.4f})'
                    )

            if self.visualize_intermediate_results:
                if vis_condition:
                    torchvision.utils.save_image(
                        torchvision.utils.make_grid(imgs.detach().cpu(), nrow=5, normalize=True, scale_each=True),
                        os.path.join(self.save_dir, f'Batch_{batch_i + 1}_iter_{i + 1:02d}_Acc_{mean_conf:.4f}.png')
                    )

                    grad_x = imgs.grad.detach().cpu()
                    save_grid_with_cmap(
                        grad_x, os.path.join(self.save_dir,
                                             f'grad_x_Batch_{batch_i + 1}_iter_{i + 1:02d}_Acc_{mean_conf:.4f}.png'),
                        nrow=5
                    )

                    proj_x = proj_x.detach().cpu()
                    save_grid_with_cmap(
                        proj_x, os.path.join(self.save_dir,
                                             f'proj_x_Batch_{batch_i + 1}_iter_{i + 1:02d}_Acc_{mean_conf:.4f}.png'),
                        nrow=5
                    )

                    # save_grid_with_cmap(
                    #     residual, os.path.join(self.save_dir, f'res_Batch_{batch_i + 1}_iter_{i + 1:02d}_Acc_{mean_conf:.4f}.png'),
                    #     nrow=5
                    # )

                    del grad_w, grad_x, proj_x

            torch.cuda.empty_cache()

        return w_batch.detach(), epoch_records

    def optimization_PAA(self, batch_i, w_batch, targets_batch, num_epochs):
        # Initialize attack
        optimizer = self.config.create_optimizer(params=[w_batch.requires_grad_()])
        scheduler = self.config.create_lr_scheduler(optimizer)

        num_samples = 50  # SmoothGrad sampling times
        stdev_spread = 0.06

        epoch_records = []

        # Start optimization
        for i in range(num_epochs):
            # synthesize imagesnd preprocess images
            imgs = self.synthesize(w_batch, num_ws=self.num_ws)

            # compute discriminator loss
            if self.discriminator_weight > 0:
                discriminator_loss = self.compute_discriminator_loss(imgs)
            else:
                discriminator_loss = torch.tensor(0.0, device=imgs.device)

            if self.transformations:
                imgs = self.transformations(imgs)

            # Compute target loss
            outputs = self.target(imgs)
            target_loss = poincare_loss(outputs, targets_batch).mean()

            # combine losses and compute gradients
            optimizer.zero_grad()
            loss = target_loss + discriminator_loss * self.discriminator_weight

            grad_x = torch.autograd.grad(loss, imgs, retain_graph=True)[0]
            sigma = stdev_spread * (imgs.max().item() - imgs.min().item())

            # Use SmoothGrad to compute smoothed gradients:
            smooth_grad = 0
            for j in range(num_samples):
                noisy_imgs = imgs + torch.randn_like(imgs) * sigma
                outputs_noise = self.target(noisy_imgs)
                loss_noise = poincare_loss(outputs_noise, targets_batch).mean()

                grad_noise = torch.autograd.grad(loss_noise, noisy_imgs, retain_graph=True)[0]
                smooth_grad += grad_noise

            smooth_grad /= num_samples

            w_grad = torch.autograd.grad(outputs=imgs, inputs=w_batch, grad_outputs=smooth_grad, retain_graph=False)[0]
            w_batch.grad = w_grad

            optimizer.step()

            if scheduler:
                scheduler.step()

            # Log results
            if self.log_progress or self.visualize_intermediate_results:
                with torch.no_grad():
                    confidence_vector = outputs.softmax(dim=1)
                    confidences = torch.gather(confidence_vector, 1, targets_batch.unsqueeze(1))
                    mean_conf = confidences.mean().detach().cpu()
                    min_conf = confidences.min().detach().cpu()
                    max_conf = confidences.max().detach().cpu()

            # Log results
            if self.log_progress:
                log_condition = (i + 1) % 10 == 0 or i == 0
                if log_condition:
                    print(
                        f'iteration {i + 1}: \t total_loss={loss:.4f} \t target_loss={target_loss:.4f} \t',
                        f'discriminator_loss={discriminator_loss:.4f} \t mean_conf={mean_conf:.4f} ({min_conf:.4f}, {max_conf:.4f})'
                    )

            self.visualize_intermediate_results = 0
            if self.visualize_intermediate_results:
                vis_condition = self.save_dir and (i + 1) % 10 == 0 or i == 0
                if vis_condition:
                    save_grid_with_cmap(
                        noisy_imgs.detach().cpu(),
                        os.path.join(self.save_dir, f'x_Batch_{batch_i + 1}_iter_{i + 1:02d}_conf_{mean_conf:.4f}.png'),
                        nrow=5, handle_imgs=1
                    )

                    grad_x = grad_x.detach().cpu()
                    save_grid_with_cmap(
                        grad_x, os.path.join(self.save_dir,
                                             f'grad_x_Batch_{batch_i + 1}_iter_{i + 1:02d}_conf_{mean_conf:.4f}.png'),
                        nrow=5
                    )

                    smooth_grad_x = smooth_grad.detach().cpu()
                    save_grid_with_cmap(
                        smooth_grad_x, os.path.join(self.save_dir,
                                                    f'smooth_grad_x_Batch_{batch_i + 1}_iter_{i + 1:02d}_conf_{mean_conf:.4f}.png'),
                        nrow=5
                    )

            torch.cuda.empty_cache()

        return w_batch.detach()

    def optimization_TAA(self, batch_i, w_batch, targets_batch, num_epochs):
        optimizer = self.config.create_optimizer(params=[w_batch.requires_grad_()])
        scheduler = self.config.create_lr_scheduler(optimizer)

        # original parameter
        num_samples = 50

        transformations = build_transformations()
        epoch_records = []

        for i in range(num_epochs):
            imgs = self.synthesize(w_batch, num_ws=self.num_ws)

            if self.discriminator_weight > 0:
                discriminator_loss = self.compute_discriminator_loss(imgs)
            else:
                discriminator_loss = torch.tensor(0.0, device=imgs.device)

            if self.transformations:
                imgs = self.transformations(imgs)

            outputs = self.target(imgs)
            target_loss = poincare_loss(outputs, targets_batch).mean()

            optimizer.zero_grad()
            loss = target_loss + discriminator_loss * self.discriminator_weight

            grad_x = torch.autograd.grad(loss, imgs, retain_graph=True)[0]
            grad_x = grad_x.detach()

            smooth_grad = 0
            for j in range(num_samples):
                trans_imgs = transformations(imgs)

                outputs_trans = self.target(trans_imgs)
                loss_trans = poincare_loss(outputs_trans, targets_batch).mean()

                grad_trans = torch.autograd.grad(loss_trans, imgs, retain_graph=True)[0]
                smooth_grad += grad_trans

            smooth_grad /= num_samples
            smooth_grad_x = smooth_grad.detach().cpu()

            w_grad = torch.autograd.grad(outputs=imgs, inputs=w_batch, grad_outputs=smooth_grad, retain_graph=False)[0]
            w_batch.grad = w_grad

            optimizer.step()

            if scheduler:
                scheduler.step()

            # Log results
            if self.log_progress or self.visualize_intermediate_results:
                with torch.no_grad():
                    confidence_vector = outputs.softmax(dim=1)
                    confidences = torch.gather(confidence_vector, 1, targets_batch.unsqueeze(1))
                    mean_conf = confidences.mean().detach().cpu()
                    min_conf = confidences.min().detach().cpu()
                    max_conf = confidences.max().detach().cpu()

            # Log results
            if self.log_progress:
                log_condition = (i + 1) % 10 == 0 or i == 0
                if log_condition:
                    print(
                        f'iteration {i + 1}: \t total_loss={loss:.4f} \t target_loss={target_loss:.4f} \t',
                        f'discriminator_loss={discriminator_loss:.4f} \t mean_conf={mean_conf:.4f} ({min_conf:.4f}, {max_conf:.4f})'
                    )

            self.visualize_intermediate_results = 0
            if self.visualize_intermediate_results:
                vis_condition = self.save_dir and (i + 1) % 10 == 0 or i == 0
                if vis_condition:
                    save_grid_with_cmap(
                        trans_imgs.detach().cpu(),
                        os.path.join(self.save_dir, f'x_Batch_{batch_i + 1}_iter_{i + 1:02d}_conf_{mean_conf:.4f}.png'),
                        nrow=5, handle_imgs=1
                    )
                    grad_x = grad_x.detach().cpu()
                    save_grid_with_cmap(
                        grad_x.detach().cpu(), os.path.join(self.save_dir,
                                                            f'grad_x_Batch_{batch_i + 1}_iter_{i + 1:02d}_Acc_{mean_conf:.4f}.png'),
                        nrow=5
                    )

                    smooth_grad_x = smooth_grad.detach().cpu()
                    save_grid_with_cmap(
                        smooth_grad_x, os.path.join(self.save_dir,
                                                    f'smooth_grad_x_Batch_{batch_i + 1}_iter_{i + 1:02d}_Acc_{mean_conf:.4f}.png'),
                        nrow=5
                    )

            torch.cuda.empty_cache()

        return w_batch.detach()

    def optimization_TAA_jvp(self, batch_i, w_batch, targets_batch, num_epochs):
        # Initialize attack
        optimizer = self.config.create_optimizer(params=[w_batch.requires_grad_()])
        scheduler = self.config.create_lr_scheduler(optimizer)

        num_samples = 50  # SmoothGrad sampling times
        transformations = build_transformations()
        epoch_records = []

        # Start optimization
        for i in range(num_epochs):
            # synthesize imagesnd preprocess images
            imgs = self.synthesize(w_batch, num_ws=self.num_ws)

            # compute discriminator loss
            if self.discriminator_weight > 0:
                discriminator_loss = self.compute_discriminator_loss(imgs)
            else:
                discriminator_loss = torch.tensor(0.0, device=imgs.device)

            if self.transformations:
                imgs = self.transformations(imgs)

            # Compute target loss
            outputs = self.target(imgs)
            target_loss = poincare_loss(outputs, targets_batch).mean()

            # combine losses and compute gradients
            optimizer.zero_grad()
            loss = target_loss + discriminator_loss * self.discriminator_weight

            grad_x = torch.autograd.grad(loss, imgs, retain_graph=True)[0]

            smooth_grad = 0
            for j in range(num_samples):
                trans_imgs = transformations(imgs)

                outputs_trans = self.target(trans_imgs)
                loss_trans = poincare_loss(outputs_trans, targets_batch).mean()

                grad_trans = torch.autograd.grad(loss_trans, imgs, retain_graph=True)[0]
                smooth_grad += grad_trans

            smooth_grad /= num_samples
            smooth_grad_x = smooth_grad.detach().cpu()

            w_grad = torch.autograd.grad(outputs=imgs, inputs=w_batch, grad_outputs=smooth_grad, retain_graph=False)[0]
            w_batch.grad = w_grad

            if (i + 1) % 10 == 0:
                #
                grad_w = w_batch.grad.detach()
                proj_x = self.compute_proj_x_orthogonalized(imgs, w_batch, smooth_grad, chunk_size=30)
                proj_x = proj_x.detach()

                B = grad_x.shape[0]

                norm_grad = torch.norm(smooth_grad.view(B, -1), dim=1)
                norm_proj = torch.norm(proj_x.view(B, -1), dim=1)
                norm_ratio = norm_proj / norm_grad

                mean_ratio = norm_ratio.mean().item()
                min_ratio = norm_ratio.min().item()
                max_ratio = norm_ratio.max().item()

                with torch.no_grad():
                    confidence_vector = outputs.softmax(dim=1)
                    confidences = torch.gather(confidence_vector, 1, targets_batch.unsqueeze(1))

                print(f"[Norm Ratio] mean: {mean_ratio:.4f}, min: {min_ratio:.4f}, max: {max_ratio:.4f}")

                for idx in range(B):
                    epoch_records.append({
                        'epoch': i + 1,
                        'cosine': norm_ratio[idx].item(),
                        'conf': confidences[idx].item(),
                    })

            optimizer.step()

            if scheduler:
                scheduler.step()

            self.visualize_intermediate_results = 0
            # Log results
            if self.log_progress or self.visualize_intermediate_results:
                with torch.no_grad():
                    confidence_vector = outputs.softmax(dim=1)
                    confidences = torch.gather(confidence_vector, 1, targets_batch.unsqueeze(1))
                    mean_conf = confidences.mean().detach().cpu()
                    min_conf = confidences.min().detach().cpu()
                    max_conf = confidences.max().detach().cpu()

            # Log results
            if self.log_progress:
                log_condition = (i + 1) % 10 == 0 or i == 0
                if log_condition:
                    print(
                        f'iteration {i + 1}: \t total_loss={loss:.4f} \t target_loss={target_loss:.4f} \t',
                        f'discriminator_loss={discriminator_loss:.4f} \t mean_conf={mean_conf:.4f} ({min_conf:.4f}, {max_conf:.4f})'
                    )

            if self.visualize_intermediate_results:
                vis_condition = self.save_dir and (i + 1) % 10 == 0 or i == 0
                if vis_condition:
                    save_grid_with_cmap(
                        noisy_imgs.detach().cpu(),
                        os.path.join(self.save_dir, f'x_Batch_{batch_i + 1}_iter_{i + 1:02d}_conf_{mean_conf:.4f}.png'),
                        nrow=5, handle_imgs=1
                    )

                    grad_x = grad_x.detach().cpu()
                    save_grid_with_cmap(
                        grad_x, os.path.join(self.save_dir,
                                             f'grad_x_Batch_{batch_i + 1}_iter_{i + 1:02d}_conf_{mean_conf:.4f}.png'),
                        nrow=5
                    )

                    smooth_grad_x = smooth_grad.detach().cpu()
                    save_grid_with_cmap(
                        smooth_grad_x, os.path.join(self.save_dir,
                                                    f'smooth_grad_x_Batch_{batch_i + 1}_iter_{i + 1:02d}_conf_{mean_conf:.4f}.png'),
                        nrow=5
                    )

            torch.cuda.empty_cache()

        return w_batch.detach(), epoch_records

    def optimization_PAA_jvp(self, batch_i, w_batch, targets_batch, num_epochs):
        # Initialize attack
        optimizer = self.config.create_optimizer(params=[w_batch.requires_grad_()])
        scheduler = self.config.create_lr_scheduler(optimizer)

        num_samples = 50  # SmoothGrad sampling times
        stdev_spread = 0.1

        epoch_records = []

        # Start optimization
        for i in range(num_epochs):
            # synthesize imagesnd preprocess images
            imgs = self.synthesize(w_batch, num_ws=self.num_ws)

            # compute discriminator loss
            if self.discriminator_weight > 0:
                discriminator_loss = self.compute_discriminator_loss(imgs)
            else:
                discriminator_loss = torch.tensor(0.0, device=imgs.device)

            if self.transformations:
                imgs = self.transformations(imgs)

            # Compute target loss
            outputs = self.target(imgs)
            target_loss = poincare_loss(outputs, targets_batch).mean()

            # combine losses and compute gradients
            optimizer.zero_grad()
            loss = target_loss + discriminator_loss * self.discriminator_weight

            grad_x = torch.autograd.grad(loss, imgs, retain_graph=True)[0]
            sigma = stdev_spread * (imgs.max().item() - imgs.min().item())

            smooth_grad = 0
            for j in range(num_samples):
                noisy_imgs = imgs + torch.randn_like(imgs) * sigma
                outputs_noise = self.target(noisy_imgs)
                loss_noise = poincare_loss(outputs_noise, targets_batch).mean()

                grad_noise = torch.autograd.grad(loss_noise, noisy_imgs, retain_graph=True)[0]
                smooth_grad += grad_noise

            smooth_grad /= num_samples

            w_grad = torch.autograd.grad(outputs=imgs, inputs=w_batch, grad_outputs=smooth_grad, retain_graph=False)[0]
            w_batch.grad = w_grad

            if (i + 1) % 10 == 0:
                #
                grad_w = w_batch.grad.detach()
                proj_x = self.compute_proj_x_orthogonalized(imgs, w_batch, smooth_grad, chunk_size=30)
                proj_x = proj_x.detach()

                B = grad_x.shape[0]

                norm_grad = torch.norm(smooth_grad.view(B, -1), dim=1)
                norm_proj = torch.norm(proj_x.view(B, -1), dim=1)
                norm_ratio = norm_proj / norm_grad

                mean_ratio = norm_ratio.mean().item()
                min_ratio = norm_ratio.min().item()
                max_ratio = norm_ratio.max().item()

                with torch.no_grad():
                    confidence_vector = outputs.softmax(dim=1)
                    confidences = torch.gather(confidence_vector, 1, targets_batch.unsqueeze(1))

                print(f"[Norm Ratio] mean: {mean_ratio:.4f}, min: {min_ratio:.4f}, max: {max_ratio:.4f}")

                for idx in range(B):
                    epoch_records.append({
                        'epoch': i + 1,
                        'cosine': norm_ratio[idx].item(),
                        'conf': confidences[idx].item(),
                    })

            optimizer.step()

            if scheduler:
                scheduler.step()

            self.visualize_intermediate_results = 0
            # Log results
            if self.log_progress or self.visualize_intermediate_results:
                with torch.no_grad():
                    confidence_vector = outputs.softmax(dim=1)
                    confidences = torch.gather(confidence_vector, 1, targets_batch.unsqueeze(1))
                    mean_conf = confidences.mean().detach().cpu()
                    min_conf = confidences.min().detach().cpu()
                    max_conf = confidences.max().detach().cpu()

            # Log results
            if self.log_progress:
                log_condition = (i + 1) % 10 == 0 or i == 0
                if log_condition:
                    print(
                        f'iteration {i + 1}: \t total_loss={loss:.4f} \t target_loss={target_loss:.4f} \t',
                        f'discriminator_loss={discriminator_loss:.4f} \t mean_conf={mean_conf:.4f} ({min_conf:.4f}, {max_conf:.4f})'
                    )

            if self.visualize_intermediate_results:
                vis_condition = self.save_dir and (i + 1) % 10 == 0 or i == 0
                if vis_condition:
                    save_grid_with_cmap(
                        noisy_imgs.detach().cpu(),
                        os.path.join(self.save_dir, f'x_Batch_{batch_i + 1}_iter_{i + 1:02d}_conf_{mean_conf:.4f}.png'),
                        nrow=5, handle_imgs=1
                    )

                    grad_x = grad_x.detach().cpu()
                    save_grid_with_cmap(
                        grad_x, os.path.join(self.save_dir,
                                             f'grad_x_Batch_{batch_i + 1}_iter_{i + 1:02d}_conf_{mean_conf:.4f}.png'),
                        nrow=5
                    )

                    smooth_grad_x = smooth_grad.detach().cpu()
                    save_grid_with_cmap(
                        smooth_grad_x, os.path.join(self.save_dir,
                                                    f'smooth_grad_x_Batch_{batch_i + 1}_iter_{i + 1:02d}_conf_{mean_conf:.4f}.png'),
                        nrow=5
                    )

            torch.cuda.empty_cache()

        return w_batch.detach(), epoch_records

    def optimize_(self, batch_i, w_batch, targets_batch, num_epochs):
        # Initialize optimizer and scheduler
        optimizer = self.config.create_optimizer(params=[w_batch.requires_grad_()])
        scheduler = self.config.create_lr_scheduler(optimizer)

        for i in range(num_epochs):
            # Synthesize images and preprocess
            imgs = self.synthesize(w_batch, num_ws=self.num_ws)
            if self.discriminator_weight > 0:
                discriminator_loss = self.compute_discriminator_loss(imgs)
            else:
                discriminator_loss = torch.tensor(0.0, device=imgs.device)
            if self.clip:
                imgs = self.clip_images(imgs)
            if self.transformations:
                imgs = self.transformations(imgs)

            # Compute target outputs and loss
            outputs = self.target(imgs)
            target_loss = poincare_loss(outputs, targets_batch).mean()
            loss = target_loss + discriminator_loss * self.discriminator_weight

            optimizer.zero_grad()
            # Backward to update parameters
            loss.backward(retain_graph=True)
            optimizer.step()
            if scheduler:
                scheduler.step()

            # --- compute gradients: dL/dx and dfi/dx ---
            grad_L = torch.autograd.grad(loss, imgs, retain_graph=True, allow_unused=True)[0]
            # fi = outputs.gather(1, targets_batch.unsqueeze(1)).squeeze(1)
            # grad_fi = torch.autograd.grad(fi, imgs, retain_graph=True, allow_unused=True)[0]

            fi = outputs.gather(1, targets_batch.unsqueeze(1)).squeeze(1)
            grad_fi = \
                torch.autograd.grad(fi, imgs, grad_outputs=torch.ones_like(fi), retain_graph=True, allow_unused=True)[0]

            # compute dL/dx related projection
            # Already obtained w_batch.grad via loss.backward()
            grad_w = w_batch.grad.detach()
            _, proj_grad_L = jvp(self.G_forward, (w_batch,), (grad_w,))
            proj_grad_L = proj_grad_L.detach().cpu()

            # Compute projection related to dfi/dx
            # Here assume f_i(x) is the target-class logit in the model output
            # Compute gradient of f_i w.r.t. latent code
            grad_w_fi = torch.autograd.grad(
                fi, w_batch, grad_outputs=torch.ones_like(fi), retain_graph=True, allow_unused=True
            )[0]
            _, proj_grad_fi = jvp(self.G_forward, (w_batch,), (grad_w_fi,))
            proj_grad_fi = proj_grad_fi.detach().cpu()

            if self.transformations:
                proj_grad_L = self.transformations(proj_grad_L)
                proj_grad_fi = self.transformations(proj_grad_fi)

            # Move to CPU for visualization
            grad_L_vis = grad_L.detach().cpu()
            grad_fi_vis = grad_fi.detach().cpu()
            proj_grad_L_vis = proj_grad_L.detach().cpu()
            proj_grad_fi_vis = proj_grad_fi.detach().cpu()

            # Compare alignment via cosine similarity after normalization
            cos_sim_L = cosine_similarity_batch(grad_L_vis, proj_grad_L_vis)  # shape: (B,)
            cos_sim_fi = cosine_similarity_batch(grad_fi_vis, proj_grad_fi_vis)
            mean_sim_L = cos_sim_L.mean().item()
            mean_sim_fi = cos_sim_fi.mean().item()
            print(f"[Cosine Similarity dL/dx] mean: {mean_sim_L:.4f}")
            print(f"[Cosine Similarity dfi/dx] mean: {mean_sim_fi:.4f}")

            # --- Visualization ---
            if self.visualize_intermediate_results:
                torchvision.utils.save_image(
                    torchvision.utils.make_grid(imgs.detach().cpu(), nrow=5, normalize=True, scale_each=True),
                    os.path.join(self.save_dir, f'Batch_{batch_i + 1}_iter_{i + 1:02d}_imgs.png')
                )
                save_grid_with_cmap(
                    grad_L_vis, os.path.join(self.save_dir, f'gradL_Batch_{batch_i + 1}_iter_{i + 1:02d}.png'),
                    nrow=5
                )
                save_grid_with_cmap(
                    grad_fi_vis, os.path.join(self.save_dir, f'gradfi_Batch_{batch_i + 1}_iter_{i + 1:02d}.png'),
                    nrow=5
                )
                save_grid_with_cmap(
                    proj_grad_L_vis,
                    os.path.join(self.save_dir, f'proj_gradL_Batch_{batch_i + 1}_iter_{i + 1:02d}.png'),
                    nrow=5
                )
                save_grid_with_cmap(
                    proj_grad_fi_vis,
                    os.path.join(self.save_dir, f'proj_gradfi_Batch_{batch_i + 1}_iter_{i + 1:02d}.png'),
                    nrow=5
                )

            if self.log_progress:
                with torch.no_grad():
                    confidence_vector = outputs.softmax(dim=1)
                    confidences = torch.gather(confidence_vector, 1, targets_batch.unsqueeze(1))
                    mean_conf = confidences.mean().detach().cpu()
                    min_conf = confidences.min().detach().cpu()
                    max_conf = confidences.max().detach().cpu()
                log_condition = (i + 1) % 10 == 0 or i == 0
                if log_condition:
                    print(
                        f'iteration {i + 1}: total_loss={loss:.4f} \t target_loss={target_loss:.4f} \t',
                        f'discriminator_loss={discriminator_loss:.4f} \t mean_conf={mean_conf:.4f} ({min_conf:.4f}, {max_conf:.4f})'
                    )
            torch.cuda.empty_cache()

        return w_batch.detach()

    def synthesize(self, w, num_ws):
        if w.shape[1] == 1:
            w_expanded = torch.repeat_interleave(w,
                                                 repeats=num_ws,
                                                 dim=1)
            imgs = self.synthesis(w_expanded,
                                  noise_mode='const',
                                  force_fp32=True)
        else:
            imgs = self.synthesis(w, noise_mode='const', force_fp32=True)
        return imgs

    def clip_images(self, imgs):
        lower_limit = torch.tensor(-1.0).float().to(imgs.device)
        upper_limit = torch.tensor(1.0).float().to(imgs.device)
        imgs = torch.where(imgs > upper_limit, upper_limit, imgs)
        imgs = torch.where(imgs < lower_limit, lower_limit, imgs)
        return imgs

    def compute_discriminator_loss(self, imgs):
        print(imgs.shape)
        discriminator_logits = self.discriminator(imgs, None)
        discriminator_loss = nn.functional.softplus(-discriminator_logits).mean()
        return discriminator_loss

    def compute_proj_x_orthogonalized(self, imgs, w_batch, grad, chunk_size=100):
        proj_x_list = []
        B, _, latent_dim = w_batch.shape
        eye = torch.eye(latent_dim, device=w_batch.device)

        def G_forward(w):
            gen = self.synthesis.module if hasattr(self.synthesis, 'module') else self.synthesis
            return gen(w)

        for i in range(B):
            # Fetch the i-th sample's w and the corresponding grad
            w_i = w_batch[i].detach().requires_grad_()  # [1, 1, latent_dim]
            w_i = w_i.unsqueeze(0).repeat(1, self.num_ws, 1)

            output_shape = grad[i].shape

            # Define a function that flattens the generator output
            def flattened_g(_w):
                out = G_forward(_w)

                if self.transformations:
                    out = self.transformations(out)

                return out.flatten(start_dim=1)

            v_chunks = torch.split(eye, chunk_size, dim=0)  # each chunk shape: [chunk_size, z_dim]
            tangent_chunks = []

            for v_chunk in v_chunks:
                v_chunk_expanded = v_chunk.unsqueeze(1).expand(-1, self.num_ws, -1)  # [chunk_size, self.num_ws, z_dim]
                w_i_exp = w_i.expand(v_chunk_expanded.shape[0], *w_i.shape[1:])

                _, tangent_chunk = jvp(flattened_g, (w_i_exp,), (v_chunk_expanded,))
                tangent_chunks.append(tangent_chunk)

            tangent_all = torch.cat(tangent_chunks, dim=0)
            J = tangent_all.transpose(0, 1)

            # J = J.view(grad[i].numel(), -1)
            J = J.contiguous().view(grad[i].numel(), -1)

            # SVD to get an orthonormal basis U
            U, _, _ = torch.linalg.svd(J, full_matrices=False)

            # u_check = U.T @ U
            # I = torch.eye(u_check.shape[0], device=u_check.device)
            # orthonorm_error = torch.norm(u_check - I)
            # print(f"[compute_proj_x_orthogonalized] Sample {i}, orthonorm error = {orthonorm_error.item():.6f}")

            # Extract the real image gradient (assume imgs.grad exists, shape [B, C, H, W])
            # real_grad = imgs.grad[i].view(-1)  # [output_dim]
            real_grad = grad[i].view(-1)

            # Project the real gradient onto the column space of U
            proj_x = U @ (U.T @ real_grad)
            proj_x = proj_x.view(output_shape)
            proj_x_list.append(proj_x.detach())

        return torch.stack(proj_x_list)

    def compute_proj_and_basis(self, imgs, w_batch, grad, chunk_size=100):
        """
        imgs:    [B, C, H, W]  (real images, only used to obtain shape)
        w_batch: [B, 1, d_latent]
        grad:    [B, C, H, W]  (precomputed real gradients)
        """
        proj_x_list = []
        U_list = []

        B, C, H, W = imgs.shape
        D_out = C * H * W
        _, _, d_latent = w_batch.shape
        eye = torch.eye(d_latent, device=w_batch.device)

        # Used to call synthesis network
        def G_forward(w):
            gen = self.synthesis.module if hasattr(self.synthesis, 'module') else self.synthesis
            return gen(w)

        for i in range(B):
            # 1) Prepare single-sample latent input
            w_i = w_batch[i].detach().requires_grad_()  # [1, 1, d_latent]
            w_i = w_i.unsqueeze(0).repeat(1, self.num_ws, 1)  # [1, num_ws, d_latent]

            # 2) Define flattened generator output
            def flattened_g(_w):
                out = G_forward(_w)  # e.g. [n, C, H, W]
                if self.transformations:
                    out = self.transformations(out)
                return out.flatten(start_dim=1)  # [n, D_out]

            # 3) Compute JVP in chunks, concatenate into Jacobian J: [D_out, d_latent]
            J_chunks = []
            for v_chunk in torch.split(eye, chunk_size, dim=0):  # [k, d_latent]
                # Expand to match w_i batch shape
                w_rep = w_i.expand(v_chunk.size(0), *w_i.shape[1:])  # [k, num_ws, d_latent]
                v_rep = v_chunk.view(v_chunk.size(0), 1, d_latent)  # [k,1,d_latent]
                v_rep = v_rep.expand(-1, self.num_ws, -1)  # [k,num_ws,d_latent]

                _, Jc = jvp(flattened_g, (w_rep,), (v_rep,))  # Jc: [k, D_out]
                J_chunks.append(Jc)

            J = torch.cat(J_chunks, dim=0)  # [d_latent, D_out] if chunk covers all latent
            # Transpose to [D_out, d_latent]
            J = J.transpose(0, 1).contiguous()

            # 4) SVD → U basis (orthonormal basis vectors in image space)
            #    J = U S V^T, U.shape = [D_out, d_latent]
            U, S, Vt = torch.linalg.svd(J, full_matrices=False)

            U_list.append(U.detach())

        U_tensor = torch.stack(U_list, dim=0)  # [B, D_out, d_latent]
        return U_tensor

    def smooth_grad_optimization_jvp(self, batch_i, w_batch, targets_batch, num_epochs):
        # Initialize attack
        optimizer = self.config.create_optimizer(params=[w_batch.requires_grad_()])
        scheduler = self.config.create_lr_scheduler(optimizer)

        self.visualize_intermediate_results = 1

        num_samples = 100  # SmoothGrad sampling count
        stdev_spread = 0.1

        epoch_records = []

        # Start optimization
        for i in range(num_epochs):
            # synthesize imagesnd preprocess images
            imgs = self.synthesize(w_batch, num_ws=self.num_ws)

            # compute discriminator loss
            if self.discriminator_weight > 0:
                discriminator_loss = self.compute_discriminator_loss(imgs)
            else:
                discriminator_loss = torch.tensor(0.0, device=imgs.device)

            if self.transformations:
                imgs = self.transformations(imgs)

            # Compute target loss
            outputs = self.target(imgs)
            target_loss = poincare_loss(outputs, targets_batch).mean()

            # combine losses and compute gradients
            optimizer.zero_grad()
            loss = target_loss + discriminator_loss * self.discriminator_weight

            grad_x = torch.autograd.grad(loss, imgs, retain_graph=True)[0]
            sigma = stdev_spread * (imgs.max().item() - imgs.min().item())

            # Use SmoothGrad for smoothed gradient computation:
            smooth_grad = 0
            for j in range(num_samples):
                noisy_imgs = imgs + torch.randn_like(imgs) * sigma
                outputs_noise = self.target(noisy_imgs)
                loss_noise = poincare_loss(outputs_noise, targets_batch).mean()

                grad_noise = torch.autograd.grad(loss_noise, noisy_imgs, retain_graph=True)[0]
                smooth_grad += grad_noise

            smooth_grad /= num_samples

            w_grad = torch.autograd.grad(outputs=imgs, inputs=w_batch, grad_outputs=smooth_grad, retain_graph=False)[0]
            w_batch.grad = w_grad

            if (i + 1) % 10 == 0:
                #
                grad_w = w_batch.grad.detach()
                proj_x = self.compute_proj_x_orthogonalized(imgs, w_batch, smooth_grad, chunk_size=60)
                proj_x = proj_x.detach()

                B = grad_x.shape[0]

                # Compute L2 norms after flattening per sample
                norm_grad = torch.norm(smooth_grad.view(B, -1), dim=1)
                norm_proj = torch.norm(proj_x.view(B, -1), dim=1)
                norm_ratio = norm_proj / norm_grad  # ratio per sample

                mean_ratio = norm_ratio.mean().item()
                min_ratio = norm_ratio.min().item()
                max_ratio = norm_ratio.max().item()

                with torch.no_grad():
                    confidence_vector = outputs.softmax(dim=1)
                    confidences = torch.gather(confidence_vector, 1, targets_batch.unsqueeze(1))

                print(f"[Norm Ratio] mean: {mean_ratio:.4f}, min: {min_ratio:.4f}, max: {max_ratio:.4f}")

                for idx in range(B):
                    epoch_records.append({
                        'epoch': i + 1,
                        'cosine': norm_ratio[idx].item(),
                        'conf': confidences[idx].item(),
                    })

            optimizer.step()

            if scheduler:
                scheduler.step()

            self.visualize_intermediate_results = 0
            # Log results
            if self.log_progress or self.visualize_intermediate_results:
                with torch.no_grad():
                    confidence_vector = outputs.softmax(dim=1)
                    confidences = torch.gather(confidence_vector, 1, targets_batch.unsqueeze(1))
                    mean_conf = confidences.mean().detach().cpu()
                    min_conf = confidences.min().detach().cpu()
                    max_conf = confidences.max().detach().cpu()

            # Log results
            if self.log_progress:
                log_condition = (i + 1) % 10 == 0 or i == 0
                if log_condition:
                    print(
                        f'iteration {i + 1}: \t total_loss={loss:.4f} \t target_loss={target_loss:.4f} \t',
                        f'discriminator_loss={discriminator_loss:.4f} \t mean_conf={mean_conf:.4f} ({min_conf:.4f}, {max_conf:.4f})'
                    )

            if self.visualize_intermediate_results:
                vis_condition = self.save_dir and (i + 1) % 10 == 0 or i == 0
                if vis_condition:
                    save_grid_with_cmap(
                        noisy_imgs.detach().cpu(),
                        os.path.join(self.save_dir, f'x_Batch_{batch_i + 1}_iter_{i + 1:02d}_conf_{mean_conf:.4f}.png'),
                        nrow=5, handle_imgs=1
                    )

                    grad_x = grad_x.detach().cpu()
                    save_grid_with_cmap(
                        grad_x, os.path.join(self.save_dir,
                                             f'grad_x_Batch_{batch_i + 1}_iter_{i + 1:02d}_conf_{mean_conf:.4f}.png'),
                        nrow=5
                    )

                    smooth_grad_x = smooth_grad.detach().cpu()
                    save_grid_with_cmap(
                        smooth_grad_x, os.path.join(self.save_dir,
                                                    f'smooth_grad_x_Batch_{batch_i + 1}_iter_{i + 1:02d}_conf_{mean_conf:.4f}.png'),
                        nrow=5
                    )

            torch.cuda.empty_cache()

        return w_batch.detach()

    def smooth_manifold_grad_optimization(self, batch_i, w_batch, targets_batch, num_epochs):
        """
        Manifold-aware smoothing based optimization (manifold tangent-space smoothing).
        """
        optimizer = self.config.create_optimizer(params=[w_batch.requires_grad_()])
        scheduler = self.config.create_lr_scheduler(optimizer)

        # Smoothing parameters
        num_samples = 100  # number of smoothing samples
        stdev_spread = 0.5  # noise scale relative to the latent-space range

        w_mean = w_batch.mean(dim=0)  # [w_dim]
        w_std = w_batch.std(dim=0)  # [w_dim]
        low_p, high_p = 0.5, 99.5
        q_low = low_p / 100.0
        q_high = high_p / 100.0
        w_low = torch.quantile(w_batch, q_low, dim=0)  # [w_dim]
        w_high = torch.quantile(w_batch, q_high, dim=0)
        # Compute clip_range
        d_low = (w_mean - w_low).abs().max().item()
        d_high = (w_high - w_mean).abs().max().item()
        clip_range = max(d_low, d_high)
        print(f"Recommend clip_range = {clip_range:.3f}")

        clip_range = 2.124
        sigma_z = stdev_spread * clip_range

        epoch_records = []

        for i in range(num_epochs):
            # 1) Forward synthesis
            imgs = self.synthesize(w_batch, num_ws=self.num_ws)  # [B, C, H, W]

            # compute discriminator loss
            if self.discriminator_weight > 0:
                discriminator_loss = self.compute_discriminator_loss(imgs)
            else:
                discriminator_loss = torch.tensor(0.0, device=imgs.device)

            if self.transformations:
                imgs = self.transformations(imgs)
            B, C, H, W = imgs.shape

            optimizer.zero_grad()
            outputs = self.target(imgs)
            target_loss = poincare_loss(outputs, targets_batch).mean()
            loss = target_loss  # + discriminator_weight * ...
            grad_x = torch.autograd.grad(loss, imgs, retain_graph=False)[0]  # [B, C, H, W]

            U_tensor = self.compute_proj_and_basis(imgs, w_batch, grad_x, chunk_size=50)
            D_out = C * H * W

            smooth_grad = torch.zeros_like(grad_x)
            for _ in range(num_samples):
                # 4.1) Sample noise in latent subspace [B, d_latent]
                eps_z = torch.randn(B, U_tensor.size(-1), device=imgs.device) * sigma_z
                # 4.2) Map back to image space: eps_x_flat[b] = U[b] @ eps_z[b]
                eps_x_flat = torch.bmm(U_tensor, eps_z.unsqueeze(-1)).squeeze(-1)  # [B, D_out]
                eps_x = eps_x_flat.view(B, C, H, W)
                # 4.3) Form noisy images and compute gradients
                noisy_imgs = imgs + eps_x
                outs_noise = self.target(noisy_imgs)
                loss_noise = poincare_loss(outs_noise, targets_batch).mean()
                grad_noise = torch.autograd.grad(loss_noise, noisy_imgs, retain_graph=True)[0]
                smooth_grad += grad_noise

            smooth_grad /= num_samples  # smoothed gradients [B, C, H, W]

            # 5) Backpropagate smoothed gradients to latent space
            #    dL/dw = ∂x/∂w^T @ smooth_grad_flat
            w_grad = torch.autograd.grad(
                outputs=imgs, inputs=w_batch,
                grad_outputs=smooth_grad,
                retain_graph=False
            )[0]
            w_batch.grad = w_grad

            if i == 0 or (i + 1) % 10 == 0:
                grad_w = w_batch.grad.detach()

                proj_x_list = []
                B = smooth_grad.shape[0]
                for i in range(B):
                    real_grad = smooth_grad[i].view(-1)
                    U = U_tensor[i]

                    # Project the real gradient onto the column space of U
                    proj_x = U @ (U.T @ real_grad)
                    proj_x = proj_x.view(smooth_grad[i].shape)
                    proj_x_list.append(proj_x.detach())

                proj_x = torch.stack(proj_x_list)

                # Compute L2 norms per sample after flattening
                norm_grad = torch.norm(smooth_grad.view(B, -1), dim=1)
                norm_proj = torch.norm(proj_x.view(B, -1), dim=1)
                norm_ratio = norm_proj / norm_grad  # ratio per sample

                mean_ratio = norm_ratio.mean().item()
                min_ratio = norm_ratio.min().item()
                max_ratio = norm_ratio.max().item()

                with torch.no_grad():
                    confidence_vector = outputs.softmax(dim=1)
                    confidences = torch.gather(confidence_vector, 1, targets_batch.unsqueeze(1))

                print(f"[Norm Ratio] mean: {mean_ratio:.4f}, min: {min_ratio:.4f}, max: {max_ratio:.4f}")

                for idx in range(B):
                    epoch_records.append({
                        'epoch': i + 1,
                        'cosine': norm_ratio[idx].item(),
                        'conf': confidences[idx].item(),
                    })

            optimizer.step()

            if scheduler:
                scheduler.step()

            # Log results
            if self.log_progress or self.visualize_intermediate_results:
                with torch.no_grad():
                    confidence_vector = outputs.softmax(dim=1)
                    confidences = torch.gather(confidence_vector, 1, targets_batch.unsqueeze(1))
                    mean_conf = confidences.mean().detach().cpu()
                    min_conf = confidences.min().detach().cpu()
                    max_conf = confidences.max().detach().cpu()

            # Log results
            if self.log_progress:
                log_condition = (i + 1) % 10 == 0 or i == 0
                if log_condition:
                    print(
                        f'iteration {i + 1}: \t total_loss={loss:.4f} \t target_loss={target_loss:.4f} \t',
                        f'discriminator_loss={discriminator_loss:.4f} \t mean_conf={mean_conf:.4f} ({min_conf:.4f}, {max_conf:.4f})'
                    )

            if self.visualize_intermediate_results:
                vis_condition = self.save_dir and (i + 1) % 10 == 0 or i == 0
                if vis_condition:
                    save_grid_with_cmap(
                        noisy_imgs.detach().cpu(),
                        os.path.join(self.save_dir, f'x_Batch_{batch_i + 1}_iter_{i + 1:02d}_conf_{mean_conf:.4f}.png'),
                        nrow=5, handle_imgs=1
                    )

                    grad_x = grad_x.detach().cpu()
                    save_grid_with_cmap(
                        grad_x, os.path.join(self.save_dir,
                                             f'grad_x_Batch_{batch_i + 1}_iter_{i + 1:02d}_conf_{mean_conf:.4f}.png'),
                        nrow=5
                    )

                    smooth_grad_x = smooth_grad.detach().cpu()
                    save_grid_with_cmap(
                        smooth_grad_x, os.path.join(self.save_dir,
                                                    f'smooth_grad_x_Batch_{batch_i + 1}_iter_{i + 1:02d}_conf_{mean_conf:.4f}.png'),
                        nrow=5
                    )

            torch.cuda.empty_cache()

        return w_batch.detach()


# Test SmoothGrad
class Optimization_():
    def __init__(self, target_model, synthesis, discriminator, transformations, num_ws, config, save_dir=None):
        self.synthesis = synthesis
        self.target = target_model
        self.discriminator = discriminator
        self.config = config
        self.transformations = transformations
        self.discriminator_weight = self.config.attack['discriminator_loss_weight']
        self.num_ws = num_ws
        self.clip = config.attack['clip']
        self.save_dir = save_dir
        self.log_progress = 1
        self.visualize_intermediate_results = 1

    def confidence_gain(self, batch_i, w_batch, targets_batch, total_epochs, inner_epochs=10):
        evaluation_model = self.config.create_evaluation_model()
        evaluation_model = torch.nn.DataParallel(evaluation_model)
        evaluation_model.to(device)
        evaluation_model.eval()

        self.adv_model = self.config.create_adv_trained_model().to(device)

        from torchvision import transforms
        self.transformations = transforms.Compose([
            transforms.CenterCrop(size=(800, 800)),
            transforms.Resize(size=299, interpolation=transforms.InterpolationMode.BILINEAR, max_size=None,
                              antialias=True)
        ])

        self.log_progress = False
        self.visualize_intermediate_results = 0

        gains_regular = []
        gains_smooth = []

        for epoch in range(0, total_epochs, inner_epochs):
            with torch.no_grad():
                imgs_pre = self.synthesize(w_batch, num_ws=self.num_ws)
                if self.clip:
                    imgs_pre = self.clip_images(imgs_pre)
                if self.transformations:
                    imgs_pre = self.transformations(imgs_pre)
                outputs_pre = evaluation_model(imgs_pre)
                confidence_vector_pre = outputs_pre.softmax(dim=1)
                conf_pre = torch.gather(confidence_vector_pre, 1, targets_batch.unsqueeze(1)).mean()

            w_batch_smooth = w_batch.detach().clone()
            w_batch = self.optimize(batch_i, w_batch, targets_batch, inner_epochs)
            w_batch_smooth = self.smooth_grad_optimization_with_transforms(batch_i, w_batch_smooth, targets_batch,
                                                                           inner_epochs)

            with torch.no_grad():
                imgs_regular = self.synthesize(w_batch, num_ws=self.num_ws)
                if self.clip:
                    imgs_regular = self.clip_images(imgs_regular)
                if self.transformations:
                    imgs_regular = self.transformations(imgs_regular)
                outputs_regular = evaluation_model(imgs_regular)
                conf_vec_regular = outputs_regular.softmax(dim=1)
                conf_regular = torch.gather(conf_vec_regular, 1, targets_batch.unsqueeze(1)).mean()

                imgs_smooth = self.synthesize(w_batch_smooth, num_ws=self.num_ws)
                if self.clip:
                    imgs_smooth = self.clip_images(imgs_smooth)
                if self.transformations:
                    imgs_smooth = self.transformations(imgs_smooth)
                outputs_smooth = evaluation_model(imgs_smooth)
                conf_vec_smooth = outputs_smooth.softmax(dim=1)
                conf_smooth = torch.gather(conf_vec_smooth, 1, targets_batch.unsqueeze(1)).mean()

            gain_regular = conf_regular - conf_pre
            gain_smooth = conf_smooth - conf_pre
            gains_regular.append(gain_regular.item())
            gains_smooth.append(gain_smooth.item())

            print(f"Batch_{batch_i + 1}, [Epoch {epoch + inner_epochs}/{total_epochs}] "
                  f"Confidence Gain (regular) = {gain_regular:.4f}, "
                  f"Confidence Gain (smooth) = {gain_smooth:.4f}")

        return

    def optimize_autograd(self, batch_i, w_batch, targets_batch, num_epochs):
        # Initialize attack
        optimizer = self.config.create_optimizer(params=[w_batch.requires_grad_()])
        scheduler = self.config.create_lr_scheduler(optimizer)

        # Start optimization
        for i in range(num_epochs):
            # synthesize images and preprocess
            imgs = self.synthesize(w_batch, num_ws=self.num_ws)

            # compute discriminator loss
            if self.discriminator_weight > 0:
                discriminator_loss = self.compute_discriminator_loss(imgs)
            else:
                discriminator_loss = self.compute_discriminator_logit(imgs)

            # perform image transformations
            if self.clip:
                imgs = self.clip_images(imgs)
            if self.transformations:
                imgs = self.transformations(imgs)

            # Compute target loss
            outputs = self.target(imgs)
            target_loss = poincare_loss(outputs, targets_batch).mean()
            loss = target_loss + self.discriminator_weight * discriminator_loss

            # Zero gradients
            optimizer.zero_grad()

            # (1) Compute image-space gradient dL/dx (via autograd, without relying on full BP chain)
            # Ensure imgs retains gradients
            imgs.retain_grad()
            g_x = torch.autograd.grad(loss, imgs, retain_graph=True, create_graph=False)[0]

            g_x_denoised = fourier_denoise_tensor(g_x, cutoff=0.05)
            w_grad = torch.autograd.grad(outputs=imgs, inputs=w_batch, grad_outputs=g_x_denoised, retain_graph=False)[0]

            w_batch.grad = w_grad

            optimizer.step()
            if scheduler:
                scheduler.step()

            # Logging / visualization
            if self.log_progress or self.visualize_intermediate_results:
                with torch.no_grad():
                    confidence_vector = outputs.softmax(dim=1)
                    confidences = torch.gather(confidence_vector, 1, targets_batch.unsqueeze(1))
                    mean_conf = confidences.mean().detach().cpu()
                    min_conf = confidences.min().detach().cpu()
                    max_conf = confidences.max().detach().cpu()

            if self.log_progress:
                log_condition = (i + 1) % 10 == 0 or i == 0
                if log_condition:
                    print(
                        f'iteration {i + 1}: \t total_loss={loss:.4f} \t target_loss={target_loss:.4f} \t',
                        f'discriminator_loss={discriminator_loss:.4f} \t mean_conf={mean_conf:.4f} ({min_conf:.4f}, {max_conf:.4f})'
                    )

            if self.visualize_intermediate_results:
                vis_condition = self.save_dir and ((i + 1) % 10 == 0 or i == 0)
                if vis_condition:
                    torchvision.utils.save_image(
                        torchvision.utils.make_grid(imgs.detach().cpu(), nrow=5, normalize=True, scale_each=True),
                        os.path.join(self.save_dir, f'Batch_{batch_i + 1}_iter_{i + 1:02d}_Acc_{mean_conf:.4f}.png')
                    )

                    grad_x_vis = g_x.detach().cpu()
                    # Normalize gradient for better visualization
                    grad_x_vis = (grad_x_vis - grad_x_vis.min()) / (grad_x_vis.max() - grad_x_vis.min() + 1e-8)
                    save_grid_with_cmap(
                        grad_x_vis,
                        os.path.join(self.save_dir,
                                     f'grad_x_Batch_{batch_i + 1}_iter_{i + 1:02d}_Acc_{mean_conf:.4f}.png'),
                        nrow=5
                    )

                    grad_x_vis = g_x_denoised.detach().cpu()
                    grad_x_vis = (grad_x_vis - grad_x_vis.min()) / (grad_x_vis.max() - grad_x_vis.min() + 1e-8)
                    save_grid_with_cmap(
                        grad_x_vis,
                        os.path.join(self.save_dir,
                                     f'grad_x_PCA_Batch_{batch_i + 1}_iter_{i + 1:02d}_Acc_{mean_conf:.4f}.png'),
                        nrow=5
                    )

            torch.cuda.empty_cache()

        return w_batch.detach()

    def optimize(self, batch_i, w_batch, targets_batch, num_epochs):
        # Initialize attack
        optimizer = self.config.create_optimizer(params=[w_batch.requires_grad_()])
        scheduler = self.config.create_lr_scheduler(optimizer)
        self.visualize_intermediate_results = 0

        # Start optimization
        for i in range(num_epochs):
            # synthesize imagesnd preprocess images
            imgs = self.synthesize(w_batch, num_ws=self.num_ws)

            # compute discriminator loss
            if self.discriminator_weight > 0:
                discriminator_loss = self.compute_discriminator_loss(imgs)
            else:
                discriminator_loss = torch.tensor(0.0)

            # perform image transformations
            if self.clip:
                imgs = self.clip_images(imgs)
            if self.transformations:
                imgs = self.transformations(imgs)

            # Compute target loss
            outputs = self.target(imgs)
            target_loss = poincare_loss(outputs, targets_batch).mean()

            # combine losses and compute gradients
            optimizer.zero_grad()
            loss = target_loss + discriminator_loss * self.discriminator_weight

            imgs.retain_grad()
            loss.backward()
            optimizer.step()

            if scheduler:
                scheduler.step()

            if self.log_progress or self.visualize_intermediate_results:
                with torch.no_grad():
                    confidence_vector = outputs.softmax(dim=1)
                    confidences = torch.gather(confidence_vector, 1, targets_batch.unsqueeze(1))
                    mean_conf = confidences.mean().detach().cpu()
                    min_conf = confidences.min().detach().cpu()
                    max_conf = confidences.max().detach().cpu()

            # Log results
            if self.log_progress:
                log_condition = (i + 1) % 10 == 0 or i == 0
                if log_condition:
                    print(
                        f'iteration {i + 1}: \t total_loss={loss:.4f} \t target_loss={target_loss:.4f} \t',
                        f'discriminator_loss={discriminator_loss:.4f} \t mean_conf={mean_conf:.4f} ({min_conf:.4f}, {max_conf:.4f})'
                    )

            if self.visualize_intermediate_results:
                vis_condition = self.save_dir and (i + 1) % 10 == 0 or i == 0
                if vis_condition:
                    torchvision.utils.save_image(
                        torchvision.utils.make_grid(imgs.detach().cpu(), nrow=5, normalize=True, scale_each=True),
                        os.path.join(self.save_dir, f'Batch_{batch_i + 1}_iter_{i + 1:02d}_Acc_{mean_conf:.4f}.png')
                    )

                    grad_x = imgs.grad.detach().cpu()
                    save_grid_with_cmap(
                        grad_x, os.path.join(self.save_dir,
                                             f'grad_x_Batch_{batch_i + 1}_iter_{i + 1:02d}_Acc_{mean_conf:.4f}.png'),
                        nrow=5
                    )

            torch.cuda.empty_cache()

        return w_batch.detach()

    def optimize_with_adv_model(self, batch_i, w_batch, targets_batch, num_epochs):
        # Initialize attack
        optimizer = self.config.create_optimizer(params=[w_batch.requires_grad_()])
        scheduler = self.config.create_lr_scheduler(optimizer)

        # Start optimization
        for i in range(num_epochs):
            # synthesize imagesnd preprocess images
            imgs = self.synthesize(w_batch, num_ws=self.num_ws)

            # compute discriminator loss
            if self.discriminator_weight > 0:
                discriminator_loss = self.compute_discriminator_loss(imgs)
            else:
                discriminator_loss = self.compute_discriminator_logit(imgs)

                # perform image transformations
            if self.clip:
                imgs = self.clip_images(imgs)
            if self.transformations:
                imgs = self.transformations(imgs)

            # Compute target loss
            outputs = self.adv_model(imgs)
            target_loss = poincare_loss(outputs, targets_batch).mean()

            # combine losses and compute gradients
            optimizer.zero_grad()
            loss = target_loss + discriminator_loss * self.discriminator_weight

            imgs.retain_grad()
            loss.backward()
            optimizer.step()

            if scheduler:
                scheduler.step()

            if self.log_progress or self.visualize_intermediate_results:
                with torch.no_grad():
                    confidence_vector = outputs.softmax(dim=1)
                    confidences = torch.gather(confidence_vector, 1, targets_batch.unsqueeze(1))
                    mean_conf = confidences.mean().detach().cpu()
                    min_conf = confidences.min().detach().cpu()
                    max_conf = confidences.max().detach().cpu()

            # Log results
            if self.log_progress:
                log_condition = (i + 1) % 10 == 0 or i == 0
                if log_condition:
                    print(
                        f'iteration {i + 1}: \t total_loss={loss:.4f} \t target_loss={target_loss:.4f} \t',
                        f'discriminator_loss={discriminator_loss:.4f} \t mean_conf={mean_conf:.4f} ({min_conf:.4f}, {max_conf:.4f})'
                    )

            if self.visualize_intermediate_results:
                vis_condition = self.save_dir and (i + 1) % 10 == 0 or i == 0
                if vis_condition:
                    torchvision.utils.save_image(
                        torchvision.utils.make_grid(imgs.detach().cpu(), nrow=5, normalize=True, scale_each=True),
                        os.path.join(self.save_dir, f'Batch_{batch_i + 1}_iter_{i + 1:02d}_Acc_{mean_conf:.4f}.png')
                    )

                    grad_x = imgs.grad.detach().cpu()
                    grad_x = (grad_x - grad_x.min()) / (grad_x.max() - grad_x.min() + 1e-8)
                    save_grid_with_cmap(
                        grad_x, os.path.join(self.save_dir,
                                             f'grad_x_Batch_{batch_i + 1}_iter_{i + 1:02d}_Acc_{mean_conf:.4f}.png'),
                        nrow=5
                    )

            torch.cuda.empty_cache()

        return w_batch.detach()

    def smooth_grad_optimization(self, batch_i, w_batch, targets_batch, num_epochs):
        # Initialize attack
        optimizer = self.config.create_optimizer(params=[w_batch.requires_grad_()])
        scheduler = self.config.create_lr_scheduler(optimizer)

        self.visualize_intermediate_results = 1

        num_samples = 30  # SmoothGrad sample count
        # sigma = 0.1           # SmoothGrad standard deviation
        stdev_spread = 0.05

        # Start optimization
        for i in range(num_epochs):
            # synthesize imagesnd preprocess images
            imgs = self.synthesize(w_batch, num_ws=self.num_ws)

            # compute discriminator loss
            if self.discriminator_weight > 0:
                discriminator_loss = self.compute_discriminator_loss(imgs)
            else:
                discriminator_loss = self.compute_discriminator_logit(imgs)

            # perform image transformations
            if self.clip:
                imgs = self.clip_images(imgs)
            if self.transformations:
                imgs = self.transformations(imgs)

            # Compute target loss
            outputs = self.target(imgs)
            target_loss = poincare_loss(outputs, targets_batch).mean()

            # combine losses and compute gradients
            optimizer.zero_grad()
            loss = target_loss + discriminator_loss * self.discriminator_weight

            grad_x = torch.autograd.grad(loss, imgs, retain_graph=True)[0]
            sigma = stdev_spread * (imgs.max().item() - imgs.min().item())
            # sigma = stdev_spread * (imgs.amax(dim=(1, 2, 3)) - imgs.amin(dim=(1, 2, 3)))
            # sigma = sigma.view(-1, 1, 1, 1)
            # print(sigma)

            # Use SmoothGrad to compute smoothed gradients:
            smooth_grad = 0
            for j in range(num_samples):
                noisy_imgs = imgs + torch.randn_like(imgs) * sigma
                outputs_noise = self.target(noisy_imgs)
                loss_noise = poincare_loss(outputs_noise, targets_batch).mean()

                grad_noise = torch.autograd.grad(loss_noise, imgs, retain_graph=True)[0]
                smooth_grad += grad_noise

            smooth_grad /= num_samples
            smooth_grad_x = smooth_grad.detach().cpu()

            w_grad = torch.autograd.grad(outputs=imgs, inputs=w_batch, grad_outputs=smooth_grad, retain_graph=False)[0]
            w_batch.grad = w_grad

            optimizer.step()

            if scheduler:
                scheduler.step()

            # Log results
            if self.log_progress or self.visualize_intermediate_results:
                with torch.no_grad():
                    confidence_vector = outputs.softmax(dim=1)
                    confidences = torch.gather(confidence_vector, 1, targets_batch.unsqueeze(1))
                    mean_conf = confidences.mean().detach().cpu()
                    min_conf = confidences.min().detach().cpu()
                    max_conf = confidences.max().detach().cpu()

            # Log results
            if self.log_progress:
                log_condition = (i + 1) % 10 == 0 or i == 0
                if log_condition:
                    print(
                        f'iteration {i + 1}: \t total_loss={loss:.4f} \t target_loss={target_loss:.4f} \t',
                        f'discriminator_loss={discriminator_loss:.4f} \t mean_conf={mean_conf:.4f} ({min_conf:.4f}, {max_conf:.4f})'
                    )

            if self.visualize_intermediate_results:
                vis_condition = self.save_dir and (i + 1) % 10 == 0 or i == 0
                if vis_condition:
                    # torchvision.utils.save_image(
                    #     torchvision.utils.make_grid(imgs.detach().cpu(), nrow=5, normalize=True, scale_each=True),
                    #     os.path.join(self.save_dir, f'Batch_{batch_i + 1}_iter_{i + 1:02d}_Acc_{mean_conf:.4f}.png')
                    # )

                    save_grid_with_cmap(
                        grad_x.detach().cpu(), os.path.join(self.save_dir,
                                                            f'grad_x_Batch_{batch_i + 1}_iter_{i + 1:02d}_Acc_{mean_conf:.4f}.png'),
                        nrow=5
                    )

                    save_grid_with_cmap(
                        smooth_grad_x, os.path.join(self.save_dir,
                                                    f'smooth_grad_x_Batch_{batch_i + 1}_iter_{i + 1:02d}_Acc_{mean_conf:.4f}.png'),
                        nrow=5
                    )

            torch.cuda.empty_cache()

        return w_batch.detach()

    def smooth_grad_optimization_with_transforms(self, batch_i, w_batch, targets_batch, num_epochs):
        optimizer = self.config.create_optimizer(params=[w_batch.requires_grad_()])
        scheduler = self.config.create_lr_scheduler(optimizer)

        num_samples = 30  # 50
        transformations = build_transformations()

        for i in range(num_epochs):
            imgs = self.synthesize(w_batch, num_ws=self.num_ws)

            if self.discriminator_weight > 0:
                discriminator_loss = self.compute_discriminator_loss(imgs)
            else:
                discriminator_loss = torch.tensor(0.0)

            # perform image transformations
            if self.clip:
                imgs = self.clip_images(imgs)
            if self.transformations:
                imgs = self.transformations(imgs)

            outputs = self.target(imgs)
            target_loss = poincare_loss(outputs, targets_batch).mean()

            loss = target_loss + discriminator_loss * self.discriminator_weight

            optimizer.zero_grad()

            grad_x = torch.autograd.grad(loss, imgs, retain_graph=True)[0]
            grad_x = grad_x.detach().cpu()

            smooth_grad = 0
            for j in range(num_samples):
                trans_imgs = transformations(imgs)  # here, transformations is T.Compose(...)

                outputs_noise = self.target(trans_imgs)
                loss_noise = poincare_loss(outputs_noise, targets_batch).mean()

                grad_noise = torch.autograd.grad(loss_noise, imgs, retain_graph=True)[0]
                smooth_grad += grad_noise

            smooth_grad /= num_samples
            smooth_grad_x = smooth_grad.detach().cpu()  # for visualization

            # 9) Project smoothed gradient back to w_batch
            #    dL/dw = J_G(w)^T * smooth_grad
            w_grad = torch.autograd.grad(
                outputs=imgs,
                inputs=w_batch,
                grad_outputs=smooth_grad,
                retain_graph=False)[0]

            w_batch.grad = w_grad
            optimizer.step()

            if scheduler:
                scheduler.step()

            # Log results
            if self.log_progress or self.visualize_intermediate_results:
                with torch.no_grad():
                    confidence_vector = outputs.softmax(dim=1)
                    confidences = torch.gather(confidence_vector, 1, targets_batch.unsqueeze(1))
                    mean_conf = confidences.mean().detach().cpu()
                    min_conf = confidences.min().detach().cpu()
                    max_conf = confidences.max().detach().cpu()

            # Log results
            if self.log_progress:
                log_condition = (i + 1) % 10 == 0 or i == 0
                if log_condition:
                    print(
                        f'iteration {i + 1}: \t total_loss={loss:.4f} \t target_loss={target_loss:.4f} \t',
                        f'discriminator_loss={discriminator_loss:.4f} \t mean_conf={mean_conf:.4f} ({min_conf:.4f}, {max_conf:.4f})'
                    )

            if self.visualize_intermediate_results:
                vis_condition = self.save_dir and (i + 1) % 10 == 0 or i == 0
                if vis_condition:
                    save_grid_with_cmap(
                        grad_x.detach().cpu(), os.path.join(self.save_dir,
                                                            f'grad_x_Batch_{batch_i + 1}_iter_{i + 1:02d}_Acc_{mean_conf:.4f}.png'),
                        nrow=5
                    )

                    save_grid_with_cmap(
                        smooth_grad_x, os.path.join(self.save_dir,
                                                    f'smooth_grad_x_Batch_{batch_i + 1}_iter_{i + 1:02d}_Acc_{mean_conf:.4f}.png'),
                        nrow=5
                    )

            torch.cuda.empty_cache()

        return w_batch.detach()

    def compute_proj_x_orthogonalized_2(self, imgs, w_batch):
        """
        Compute the orthogonal projection of the real gradient (imgs.grad) onto the column space
        of the generator Jacobian (i.e., the manifold tangent space).

        Math:
            1) J = dG/dz, computed via torch.autograd.functional.jacobian
            2) Perform SVD on J to get U
            3) proj_x = U @ (U.T @ real_grad)

        Args:
        - imgs: image tensor that contains generator outputs and has retain_grad() called,
                shape [B, C, H, W]; imgs.grad stores the real gradient
        - w_batch: [B, 1, latent_dim], a batch of latent variables

        Returns:
        - proj_x_batch: [B, C, H, W], projected gradient
        """
        from functorch import vmap, jvp

        proj_x_list = []
        B, _, latent_dim = w_batch.shape
        device = w_batch.device
        eye = torch.eye(latent_dim, device=device)

        # def G_forward(w):
        #     gen = self.synthesis.module if hasattr(self.synthesis, 'module') else self.synthesis
        #     return gen(w)  # here we do not pass num_ws

        def G_forward(w):
            gen = self.synthesis.module if hasattr(self.synthesis, 'module') else self.synthesis
            return gen(w)

        for i in range(B):
            # Take the i-th sample w_i with shape [1, latent_dim] and expand to [1, num_ws, latent_dim]
            w_i = w_batch[i].detach().requires_grad_()  # shape: [1, latent_dim]
            w_i = w_i.unsqueeze(0).repeat(1, self.num_ws, 1)  # [1, num_ws, latent_dim]

            output_shape = imgs.grad[i].shape

            def flattened_g(_w):
                out = G_forward(_w)

                if self.transformations:
                    out = self.transformations(out)

                return out.view(-1)

            # Use vmap to compute JVP for all latent directions in one pass
            # jvp_all = vmap(lambda v: jvp(flattened_g, (w_i,), (v.unsqueeze(0).unsqueeze(0).expand(1, self.num_ws, -1),))[1])(eye)
            jvp_all = vmap(
                lambda v: jvp(flattened_g, (w_i,), (v.unsqueeze(0).unsqueeze(0).expand(1, self.num_ws, -1),))[1],
                randomness="same"
            )(eye)

            # Transpose to get the J matrix with shape [output_dim, latent_dim]
            J = jvp_all.transpose(0, 1)

            # Perform SVD to obtain U
            U, _, _ = torch.linalg.svd(J, full_matrices=False)

            # Extract the real image gradient (assume imgs.grad exists) and flatten to [output_dim]
            real_grad = imgs.grad[i].view(-1)

            proj_x = U @ (U.T @ real_grad)
            proj_x = proj_x.view(output_shape)
            proj_x_list.append(proj_x.detach())

            norm_real = torch.norm(real_grad)
            norm_proj = torch.norm(proj_x)
            ratio = norm_proj / norm_real if norm_real > 0 else torch.tensor(0.0)
            print(f"Sample {i}: Projection ratio = {ratio.item():.6f}")

        return torch.stack(proj_x_list)

    def synthesize(self, w, num_ws):
        if w.shape[1] == 1:
            w_expanded = torch.repeat_interleave(w,
                                                 repeats=num_ws,
                                                 dim=1)
            imgs = self.synthesis(w_expanded,
                                  noise_mode='const',
                                  force_fp32=True)
        else:
            imgs = self.synthesis(w, noise_mode='const', force_fp32=True)
        return imgs

    def clip_images(self, imgs):
        lower_limit = torch.tensor(-1.0).float().to(imgs.device)
        upper_limit = torch.tensor(1.0).float().to(imgs.device)
        imgs = torch.where(imgs > upper_limit, upper_limit, imgs)
        imgs = torch.where(imgs < lower_limit, lower_limit, imgs)
        return imgs

    def compute_discriminator_loss(self, imgs):
        discriminator_logits = self.discriminator(imgs, None)
        discriminator_loss = nn.functional.softplus(
            -discriminator_logits).mean()
        return discriminator_loss

    def compute_discriminator_logit(self, imgs):
        discriminator_logits = self.discriminator(imgs, None).mean()
        return discriminator_logits

# '''
