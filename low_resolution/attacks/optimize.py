import os
import torch
import torch.nn as nn
import torchvision.transforms as T
import torch.nn.functional as F

from losses.poincare import poincare_loss
from losses.max_margin import max_margin_loss
from matplotlib import colors
from torchvision.transforms import InterpolationMode
from kornia import augmentation
from utils.logger import save_grid_with_cmap
from torch.autograd import Variable
from skimage import exposure
from losses.reg_loss import reg_loss
from utils.utils import low2high, log_sum_exp
from functorch import vmap, jvp

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def build_transformations(transformations_dict=None) -> T.Compose:
    transformations_dict = {
        "RandomResizedCrop": {
            "size": 64,
            "scale": [0.9, 1.0],  
            "ratio": [0.9, 1.1],  
            "antialias": True     
        },
        "RandomHorizontalFlip": {
            "p": 0.5
        },
        "RandomRotation": {
            "degrees": 3,               # ±3°
            "interpolation": InterpolationMode.BILINEAR,
            "expand": False,
            "center": None
        }
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


def ked_build_transformations(transformations_dict=None) -> T.Compose:
    transformations_dict = {
        "RandomResizedCrop": {
            "size": 64,
            "scale": [0.8, 1.0],  
            "ratio": [0.9, 1.1],  
            "antialias": True     
        },
        "RandomHorizontalFlip": {
            "p": 0.5
        },
        "RandomRotation": {
            "degrees": 5,               # ±5°
            "interpolation": InterpolationMode.BILINEAR,
            "expand": False,
            "center": None
        }
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
        


class Optimization():
    def __init__(self, targetnets, generator, discriminator, evaluation_model, fea_mean, fea_logvar, config, clip_range=1.0, 
                save_dir=None, stylegan=False):
        
        self.config = config
        self.attack_config = config.attack
        self.generator = generator
        self.targetnets = targetnets
        self.discriminator = discriminator
        self.evaluation_model = evaluation_model
        self.stylegan = stylegan
        self.attack_method = self.attack_config['method']
        self.identity_weight = self.attack_config['identity_weight']
        self.attack_loss = self.attack_config['loss']
        self.lam = self.attack_config['lam']
        
        self.lr = self.attack_config['lr']
        self.momentum = self.attack_config['momentum']
        self.batch_size = self.attack_config['batch_size']
        
        self.z_dim = self.attack_config['z_dim']
        self.fea_mean = fea_mean
        self.fea_logvar = fea_logvar
        self.clip_range = clip_range
        self.criterion = self.find_criterion(self.attack_config['loss'])

        self.save_dir = save_dir
        self.log_progress = 1
        self.visualize_intermediate_results = 1
        self.use_poincare_loss=False


    def gmi_optimize(self, batch_i, z_batch, targets_batch, num_iterations):
        targets_batch = targets_batch.view(-1).long().to(device)
        z_batch.requires_grad = True

        # stylegan version
        v = torch.zeros_like(z_batch).to(device).float()
        
        for i in range(num_iterations):
            # Synthesize images
            if self.stylegan:
                imgs = self.generator.synthesis(z_batch, noise_mode='const', force_fp32=True)
                min_val = imgs.min().item()
                max_val = imgs.max().item()
                imgs = (imgs - min_val) / (max_val - min_val)
                print_every = 10
                viz_every = 20

            else:
                imgs = self.generator(z_batch)
                print_every = 100
                viz_every = 200

            if z_batch.grad is not None:
                z_batch.grad.data.zero_()

            # Compute losses
            # label = self.discriminator(imgs)
            # prior_loss = self.compute_prior_loss(label)
            
            prior_loss = 0.0
            if self.use_poincare_loss == True:
                _, outputs = self.targetnets[0](imgs)
                iden_loss = poincare_loss(outputs, targets_batch).mean()
                
                loss = iden_loss
            else:
                iden_loss = self.iden_loss(self.targetnets, imgs, targets_batch, self.attack_loss, self.criterion, self.lam) if self.identity_weight > 0 else torch.tensor(0.0, device=device)

                loss = prior_loss + iden_loss * self.identity_weight

            imgs.retain_grad()
            loss.backward()

            v_prev = v.clone()
            gradient = z_batch.grad
            
            v = self.momentum * v - self.lr * gradient
            z_batch = z_batch + (-self.momentum * v_prev + (1 + self.momentum) * v)
            z_batch = torch.clamp(z_batch.detach(), -self.clip_range, self.clip_range).float()
            z_batch.requires_grad = True
            
            vis_condition = self.save_dir and (i + 1) % viz_every == 0 or i == 0
            log_condition = (i + 1) % print_every == 0 or i == 0

            if self.log_progress or self.visualize_intermediate_results:
                with torch.no_grad():
                    _, outputs = self.targetnets[0](imgs)
                    confidence_vector = outputs.softmax(dim=1)
                    confidences = torch.gather(confidence_vector, 1, targets_batch.unsqueeze(1))
                    mean_conf = confidences.mean().detach().cpu()
                    min_conf = confidences.min().detach().cpu().item()
                    max_conf = confidences.max().detach().cpu().item()
            
            if self.log_progress:
                if log_condition:
                    print(
                        f'iteration {i + 1}: \t total_loss={loss:.4f} \t target_loss={iden_loss:.4f} \t'
                        f' mean_conf={mean_conf:.4f} ({min_conf:.4f}, {max_conf:.4f})'
                    )
            
            self.visualize_intermediate_results = 0
            if self.visualize_intermediate_results:
                if vis_condition:
                    save_grid_with_cmap(
                        imgs[:25].detach().cpu(), os.path.join(self.save_dir, 
                                                              f'x_Batch_{batch_i + 1}_iter_{i + 1:02d}_conf_{mean_conf:.4f}.png'),
                        nrow=5, handle_imgs=1
                    )
                    
                    grad_x = imgs.grad.detach().cpu()
                    save_grid_with_cmap(
                        grad_x[:25], os.path.join(self.save_dir,
                                             f'grad_x_Batch_{batch_i + 1}_iter_{i + 1:02d}_conf_{mean_conf:.4f}.png'),
                        nrow=5
                    )
            
        return z_batch.detach()
    

    def gmi_optimize_records(self, batch_i, z_batch, targets_batch, num_iterations):
        targets_batch = targets_batch.view(-1).long().to(device)
        z_batch.requires_grad = True
        v = torch.zeros(z_batch.shape[0], self.z_dim).to(device).float()
        self.generator.eval()

        # List to store (epoch_num, norm_ratio) pairs
        epoch_records = []
        
        for i in range(num_iterations):
            # Synthesize images
            if self.stylegan:
                imgs = self.generator.synthesis(z_batch, noise_mode='const', force_fp32=True)
                min_val = imgs.min().item()
                max_val = imgs.max().item()
                imgs = (imgs - min_val) / (max_val - min_val)
                print_every = 10
                viz_every = 20
            else:
                imgs = self.generator(z_batch)
                print_every = 100
                viz_every = 200

            # Compute losses
            # label = self.discriminator(imgs)
            # prior_loss = self.compute_prior_loss(label)

            prior_loss = 0.0
            iden_loss = self.iden_loss(self.targetnets, imgs, targets_batch, self.attack_loss, self.criterion, self.lam) if self.identity_weight > 0 else torch.tensor(0.0, device=device)

            # Optimize
            if z_batch.grad is not None:
                z_batch.grad.zero_()

            loss = prior_loss + iden_loss * self.identity_weight

            imgs.retain_grad()
            loss.backward()
            
            grad_x = imgs.grad.detach()
            
            if (i + 1) % viz_every == 0 or i == 0:
                proj_grad = self.compute_proj_orthogonalized(z_batch, grad_x)
                proj_grad = proj_grad.detach()

                B = grad_x.shape[0]
                norm_grad = torch.norm(grad_x.view(B, -1), dim=1)
                norm_proj = torch.norm(proj_grad.view(B, -1), dim=1)
                norm_ratio = norm_proj / norm_grad
                
                mean_ratio = norm_ratio.mean().item()
                min_ratio = norm_ratio.min().item()
                max_ratio = norm_ratio.max().item()
                
                with torch.no_grad():
                    _, outputs = self.targetnets[0](imgs)
                    confidence_vector = outputs.softmax(dim=1)
                    confidences = torch.gather(confidence_vector, 1, targets_batch.unsqueeze(1))

                # Append epoch, ratio, and identity for each sample in the batch
                for idx in range(B):
                    epoch_records.append({
                        'epoch': i + 1,
                        'cosine': norm_ratio[idx].item(),
                        'conf': confidences[idx].item(),
                    })
                
                print(f"[Norm Ratio] mean: {mean_ratio:.4f}, min: {min_ratio:.4f}, max: {max_ratio:.4f}")

            v_prev = v.clone()
            gradient = z_batch.grad.data
            v = self.momentum * v - self.lr * gradient
            z_batch = z_batch + (-self.momentum * v_prev + (1 + self.momentum) * v)
            z_batch = torch.clamp(z_batch.detach(), -self.clip_range, self.clip_range).float()
            z_batch.requires_grad = True
            
            vis_condition = (i + 1) % viz_every == 0 or i == 0
            log_condition = (i + 1) % print_every == 0 or i == 0
            
            if self.log_progress or self.visualize_intermediate_results:
                _, outputs = self.targetnets[0](imgs)
                confidence_vector = outputs.softmax(dim=1)
                confidences = torch.gather(confidence_vector, 1, targets_batch.unsqueeze(1))
                mean_conf = confidences.mean().detach().cpu()
                min_conf = confidences.min().detach().cpu().item()
                max_conf = confidences.max().detach().cpu().item()

            if self.log_progress:
                if log_condition:
                    print(
                        f'iteration {i + 1}: \t total_loss={loss:.4f} \t target_loss={iden_loss:.4f} \t'
                        f' mean_conf={mean_conf:.4f} ({min_conf:.4f}, {max_conf:.4f})'
                    )
                    
            if self.visualize_intermediate_results:
                if vis_condition:
                    save_grid_with_cmap(
                        imgs[:25].detach().cpu(), os.path.join(self.save_dir, f'x_Batch_{batch_i + 1}_iter_{i + 1:02d}_conf_{mean_conf:.4f}.png'),
                        nrow=5, handle_imgs=1
                    )
                    
                    save_grid_with_cmap(
                        grad_x[:25], os.path.join(self.save_dir, f'grad_x_Batch_{batch_i + 1}_iter_{i + 1:02d}_conf_{mean_conf:.4f}.png'),
                        nrow=5
                    )
                        
                    save_grid_with_cmap(
                        proj_grad[:25], os.path.join(self.save_dir, f'proj_grad_Batch_{batch_i + 1}_iter_{i + 1:02d}_conf_{mean_conf:.4f}.png'),
                        nrow=5
                    )


        return z_batch.detach(), epoch_records
    
    
    def gmi_optimize_paa(self, batch_i, z_batch, targets_batch, num_iterations):
        
        num_samples = 50      
        stdev_spread = 0.003

        targets_batch = targets_batch.view(-1).long().to(device)
        z_batch.requires_grad = True
        v = torch.zeros_like(z_batch).to(device).float()
        self.generator.eval()

        for i in range(num_iterations):
            # Synthesize images
            if self.stylegan:
                imgs = self.generator.synthesis(z_batch, noise_mode='const', force_fp32=True)
                print_every = 10
                viz_every = 20
            else:
                imgs = self.generator(z_batch)
                print_every = 100
                viz_every = 200
                
            prior_loss = 0.0
            if self.use_poincare_loss == True:
                # Optimize
                if z_batch.grad is not None:
                    z_batch.grad.zero_()

                prior_loss = 0.0
                _, outputs = self.targetnets[0](imgs)
                target_loss = poincare_loss(outputs, targets_batch).mean()
                
                loss = target_loss + prior_loss * self.identity_weight
            else:
                # Optimize
                if z_batch.grad is not None:
                    z_batch.grad.zero_()

                iden_loss = self.iden_loss(self.targetnets, imgs, targets_batch, self.attack_loss, self.criterion, self.lam)

                loss = prior_loss + iden_loss * self.identity_weight

            imgs.retain_grad()
            loss.backward(retain_graph=True)
            grad_x = imgs.grad.detach().cpu()

            sigma = stdev_spread * (imgs.max().item() - imgs.min().item())

            smooth_grad = 0
            for j in range(num_samples):
                noisy_imgs = imgs + torch.randn_like(imgs) * sigma
                
                min_val = noisy_imgs.min().item()
                max_val = noisy_imgs.max().item()
                noisy_imgs = (noisy_imgs - min_val) / (max_val - min_val)

                loss_noise = self.iden_loss(self.targetnets, noisy_imgs, targets_batch, self.attack_loss, self.criterion, self.lam) * self.identity_weight

                grad_noise = torch.autograd.grad(loss_noise, imgs, retain_graph=True)[0]
                smooth_grad += grad_noise

            smooth_grad /= num_samples
            smooth_grad_x = smooth_grad.detach().cpu()

            z_grad = torch.autograd.grad(
                outputs=imgs, 
                inputs=z_batch, 
                grad_outputs=smooth_grad, 
                retain_graph=False
            )[0]
            
            z_batch.grad = z_grad
            
            v_prev = v.clone()
            gradient = z_batch.grad.data
            v = self.momentum * v - self.lr * gradient
            z_batch = z_batch + (-self.momentum * v_prev + (1 + self.momentum) * v)
            z_batch = torch.clamp(z_batch.detach(), -self.clip_range, self.clip_range).float()
            z_batch.requires_grad = True
            
            vis_condition = self.save_dir and (i + 1) % viz_every == 0 or i == 0
            log_condition = (i + 1) % print_every == 0 or i == 0

            if self.log_progress or self.visualize_intermediate_results:
                with torch.no_grad():
                    _, outputs = self.targetnets[0](imgs)
                    confidence_vector = outputs.softmax(dim=1)
                    confidences = torch.gather(confidence_vector, 1, targets_batch.unsqueeze(1))
                    mean_conf = confidences.mean().detach().cpu()
                    min_conf = confidences.min().detach().cpu().item()
                    max_conf = confidences.max().detach().cpu().item()
                
            if self.log_progress:
                if log_condition:
                    print(
                        f'iteration {i + 1}: \t total_loss={loss:.4f} \t target_loss={iden_loss:.4f} \t'
                        f' mean_conf={mean_conf:.4f} ({min_conf:.4f}, {max_conf:.4f})'
                    )

            # self.visualize_intermediate_results = 0
            if self.visualize_intermediate_results:
                if vis_condition:
                    save_grid_with_cmap(
                        imgs[:25].detach().cpu(), os.path.join(self.save_dir, 
                                                              f'x_Batch_{batch_i + 1}_iter_{i + 1:02d}_conf_{mean_conf:.4f}.png'),
                        nrow=5, handle_imgs=1
                    )
                    grad_x = grad_x.detach().cpu()
                    save_grid_with_cmap(
                        grad_x[:25], os.path.join(self.save_dir,
                                                f'grad_x_Batch_{batch_i + 1}_iter_{i + 1:02d}_conf_{mean_conf:.4f}.png'),
                        nrow=5
                    )
                    smooth_grad_x = smooth_grad.detach().cpu() 
                    save_grid_with_cmap(
                        smooth_grad_x[:25], os.path.join(self.save_dir, 
                                                        f'smooth_x_Batch_{batch_i + 1}_iter_{i + 1:02d}_conf_{mean_conf:.4f}.png'),
                        nrow=5
                    )
            


        return z_batch.detach()
    

    def gmi_optimize_taa(self, batch_i, z_batch, targets_batch, 
                                       num_iterations, 
                                       ):
        
        num_samples = 30
        transformations = build_transformations()
    
        targets_batch = targets_batch.view(-1).long().to(device)
        z_batch.requires_grad = True
        v = torch.zeros_like(z_batch).to(device).float()
        self.generator.eval()

        for i in range(num_iterations):
            # Synthesize images
            if self.stylegan:
                imgs = self.generator.synthesis(z_batch, noise_mode='const', force_fp32=True)
                print_every = 10
                viz_every = 20
            else:
                imgs = self.generator(z_batch)
                print_every = 100
                viz_every = 200

            prior_loss = 0.0
            if self.use_poincare_loss == True:
                # Optimize
                if z_batch.grad is not None:
                    z_batch.grad.zero_()

                prior_loss = 0.0
                _, outputs = self.targetnets[0](imgs)
                target_loss = poincare_loss(outputs, targets_batch).mean()
                
                loss = prior_loss + target_loss * self.identity_weight
            else:
                # Optimize
                if z_batch.grad is not None:
                    z_batch.grad.zero_()

                iden_loss = self.iden_loss(self.targetnets, imgs, targets_batch, self.attack_loss, self.criterion, self.lam)

                loss = prior_loss + iden_loss * self.identity_weight

            imgs.retain_grad()
            loss.backward(retain_graph=True)
            grad_x = imgs.grad.detach().cpu()

            trans_grad = 0
            for j in range(num_samples):
                trans_imgs = transformations(imgs)
                min_val = trans_imgs.min().item()
                max_val = trans_imgs.max().item()
                trans_imgs = (trans_imgs - min_val) / (max_val - min_val)

                loss_trans = self.iden_loss(self.targetnets, trans_imgs, targets_batch, self.attack_loss, self.criterion, self.lam) * self.identity_weight

                grad_trans = torch.autograd.grad(loss_trans, imgs, retain_graph=True)[0]
                trans_grad += grad_trans

            trans_grad /= num_samples
            trans_grad_x = trans_grad.detach().cpu() 

            z_grad = torch.autograd.grad(
                outputs=imgs, 
                inputs=z_batch, 
                grad_outputs=trans_grad,
                retain_graph=False
            )[0]
            
            z_batch.grad = z_grad 
            
            v_prev = v.clone()
            gradient = z_batch.grad.data
            v = self.momentum * v - self.lr * gradient
            z_batch = z_batch + (-self.momentum * v_prev + (1 + self.momentum) * v)
            z_batch = torch.clamp(z_batch.detach(), -self.clip_range, self.clip_range).float()
            z_batch.requires_grad = True

            vis_condition = self.save_dir and (i + 1) % viz_every == 0 or i == 0
            log_condition = (i + 1) % print_every == 0 or i == 0

            if self.log_progress or self.visualize_intermediate_results:
                with torch.no_grad():
                    _, outputs = self.targetnets[0](imgs)
                    confidence_vector = outputs.softmax(dim=1)
                    confidences = torch.gather(confidence_vector, 1, targets_batch.unsqueeze(1))
                    mean_conf = confidences.mean().detach().cpu()
                    min_conf = confidences.min().detach().cpu().item()
                    max_conf = confidences.max().detach().cpu().item()

            if self.log_progress and log_condition:
                print(
                    f'iteration {i + 1}: \t total_loss={loss:.4f} \t target_loss={iden_loss:.4f} \t'
                    f' mean_conf={mean_conf:.4f} ({min_conf:.4f}, {max_conf:.4f})'
                )

            self.visualize_intermediate_results = 0
            if self.visualize_intermediate_results and vis_condition:
                save_grid_with_cmap(
                    imgs[:25].detach().cpu(),
                    os.path.join(self.save_dir, f'x_Batch_{batch_i + 1}_iter_{i + 1:02d}_conf_{mean_conf:.4f}.png'),
                    nrow=5,
                    handle_imgs=True
                )
                save_grid_with_cmap(
                    grad_x[:25],
                    os.path.join(self.save_dir, f'grad_x_Batch_{batch_i + 1}_iter_{i + 1:02d}_conf_{mean_conf:.4f}.png'),
                    nrow=5
                )
                save_grid_with_cmap(
                    trans_grad_x[:25],
                    os.path.join(self.save_dir, f'smooth_x_Batch_{batch_i + 1}_iter_{i + 1:02d}_conf_{mean_conf:.4f}.png'),
                    nrow=5
                )

        return z_batch.detach()


    def kedmi_optimize(self, batch_i, targets_batch, num_iterations):
        # Initialize attack
        batch_size = targets_batch.shape[0]
        clipz = self.config.clipz
        mu = Variable(torch.zeros(1, self.z_dim), requires_grad=True)
        log_var = Variable(torch.ones(1, self.z_dim), requires_grad=True)

        params = [mu, log_var]
        # Initialize optimizer and scheduler
        optimizer = self.config.create_optimizer(params=params)
        scheduler = self.config.create_lr_scheduler(optimizer)

        # Start optimization
        for i in range(num_iterations):
            # synthesize images and preprocess images
            z = self.reparameterize_batch(mu, log_var, batch_size)
            if clipz:
                z = torch.clamp(z, -self.clip_range, self.clip_range).float()
            imgs = self.generator(z)

            if self.config.improved_flag:
                _, label = self.discriminator(imgs)
            else:
                label = self.discriminator(imgs)

            for p in params:
                if p.grad is not None:
                    p.grad.data.zero_()

            prior_loss = self.compute_prior_loss(label)
            # compute identity loss
            iden_loss = self.iden_loss(self.targetnets, imgs, targets_batch, self.attack_loss, self.criterion,
                                        self.lam)

            # combine losses and compute gradients
            optimizer.zero_grad()
            loss = prior_loss + iden_loss * self.identity_weight

            imgs.retain_grad()
            loss.backward()
            grad_x = imgs.grad.detach().cpu()

            optimizer.step()

            if scheduler:
                scheduler.step()

            print_every = 50
            viz_every = 100
            vis_condition = self.save_dir and (i + 1) % viz_every == 0 or i == 0
            log_condition = (i + 1) % print_every == 0 or i == 0

            if self.log_progress or self.visualize_intermediate_results:
                with torch.no_grad():
                    _, outputs = self.targetnets[0](imgs)
                    confidence_vector = outputs.softmax(dim=1)
                    confidences = torch.gather(confidence_vector, 1, targets_batch.unsqueeze(1))
                    mean_conf = confidences.mean().detach().cpu()
                    min_conf = confidences.min().detach().cpu().item()
                    max_conf = confidences.max().detach().cpu().item()

            if self.log_progress:
                if log_condition:
                    print(
                        f'iteration {i + 1}: \t total_loss={loss:.4f} \t target_loss={iden_loss:.4f} \t'
                        f' mean_conf={mean_conf:.4f} ({min_conf:.4f}, {max_conf:.4f})'
                    )
            
            # self.visualize_intermediate_results = 0
            if self.visualize_intermediate_results:
                if vis_condition:
                    save_grid_with_cmap(
                        imgs[:25].detach().cpu(), os.path.join(self.save_dir, 
                                                              f'x_Batch_{batch_i + 1}_iter_{i + 1:02d}_conf_{mean_conf:.4f}.png'),
                        nrow=5, handle_imgs=1
                    )
                    grad_x = imgs.grad.detach().cpu()
                    save_grid_with_cmap(
                        grad_x[:25], os.path.join(self.save_dir,
                                             f'grad_x_Batch_{batch_i + 1}_iter_{i + 1:02d}_conf_{mean_conf:.4f}.png'),
                        nrow=5
                    )

        final_z = self.reparameterize_batch(mu, log_var, batch_size)
        final_z = torch.clamp(final_z, -self.clip_range, self.clip_range).float()
        return final_z.detach()
    

    def kedmi_optimize_paa(self, batch_i, targets_batch, num_iterations):
        num_samples = 50     
        stdev_spread = 0.05  

        mu = torch.zeros(1, self.z_dim, device=device, requires_grad=True)
        log_var = torch.ones(1, self.z_dim, device=device, requires_grad=True)
        batch_size = targets_batch.shape[0]
        clipz = self.config.clipz

        params = [mu, log_var]
        optimizer = self.config.create_optimizer(params=params)
        scheduler = self.config.create_lr_scheduler(optimizer)

        for i in range(num_iterations):
            z = self.reparameterize_batch(mu, log_var, batch_size)
            if clipz:
                z = torch.clamp(z, -self.clip_range, self.clip_range).float()            
            
            imgs = self.generator(z)

            if self.config.improved_flag == True:
                _, label = self.discriminator(imgs)
            else:
                label = self.discriminator(imgs)

            for p in params:
                if p.grad is not None:
                    p.grad.data.zero_()

            prior_loss = self.compute_prior_loss(label)
            iden_loss = self.iden_loss(self.targetnets, imgs, targets_batch, 
                                    self.attack_loss, self.criterion, self.lam)
            
            optimizer.zero_grad()
            loss = prior_loss + iden_loss * self.identity_weight

            imgs.retain_grad()
            loss.backward(retain_graph=True)
            grad_x = imgs.grad.detach().cpu()
            
            sigma = stdev_spread * (imgs.max().item() - imgs.min().item())
            
            smooth_grad = 0
            for _ in range(num_samples):
                noisy_imgs = imgs + torch.randn_like(imgs) * sigma
                
                _, label = self.discriminator(noisy_imgs)
                loss_prior_noise = self.compute_prior_loss(label)
                loss_iden_noise = self.iden_loss(self.targetnets, noisy_imgs, targets_batch,
                                        self.attack_loss, self.criterion, self.lam)
                loss_noise = loss_prior_noise + loss_iden_noise * self.identity_weight
                smooth_grad += torch.autograd.grad(loss_noise, imgs, retain_graph=True)[0]
                
            smooth_grad /= num_samples

            # backpropagate through imgs -> z -> [mu, log_var]
            grads = torch.autograd.grad(imgs, [mu, log_var], grad_outputs=smooth_grad, retain_graph=True)
            mu.grad = grads[0]
            log_var.grad = grads[1]
            optimizer.step()

            optimizer.step()
            if scheduler:
                scheduler.step()

            print_every = 25
            viz_every = 50
            vis_condition = self.save_dir and (i + 1) % viz_every == 0 or i == 0
            log_condition = (i + 1) % print_every == 0 or i == 0

            if self.log_progress or self.visualize_intermediate_results:
                with torch.no_grad():
                    _, outputs = self.targetnets[0](imgs)
                    confidence_vector = outputs.softmax(dim=1)
                    confidences = torch.gather(confidence_vector, 1, targets_batch.unsqueeze(1))
                    mean_conf = confidences.mean().detach().cpu()
                    min_conf = confidences.min().detach().cpu().item()
                    max_conf = confidences.max().detach().cpu().item()

            if self.log_progress:
                if log_condition:
                    print(
                        f'iteration {i + 1}: \t total_loss={loss:.4f} \t target_loss={iden_loss:.4f} \t'
                        f' mean_conf={mean_conf:.4f} ({min_conf:.4f}, {max_conf:.4f})'
                    )
            
            self.visualize_intermediate_results = 0
            if self.visualize_intermediate_results:
                if vis_condition:
                    save_grid_with_cmap(
                        imgs[:25].detach().cpu(), os.path.join(self.save_dir, 
                                                              f'x_Batch_{batch_i + 1}_iter_{i + 1:02d}_conf_{mean_conf:.4f}.png'),
                        nrow=5, handle_imgs=1
                    )
                    grad_x = grad_x.detach().cpu()
                    save_grid_with_cmap(
                        grad_x[:25], os.path.join(self.save_dir,
                                                f'grad_x_Batch_{batch_i + 1}_iter_{i + 1:02d}_conf_{mean_conf:.4f}.png'),
                        nrow=5
                    )
                    smooth_grad_x = smooth_grad.detach().cpu() 
                    save_grid_with_cmap(
                        smooth_grad_x[:25], os.path.join(self.save_dir, 
                                                        f'smooth_x_Batch_{batch_i + 1}_iter_{i + 1:02d}_conf_{mean_conf:.4f}.png'),
                        nrow=5
                    )

        final_z = self.reparameterize_batch(mu, log_var, batch_size)
        final_z = torch.clamp(final_z, -self.clip_range, self.clip_range).float()
        return final_z.detach()


    def kedmi_optimize_taa(self, batch_i, targets_batch, num_iterations):
        num_samples = 50      
        transformations = ked_build_transformations()

        mu = torch.zeros(1, self.z_dim, device=device, requires_grad=True)
        log_var = torch.ones(1, self.z_dim, device=device, requires_grad=True)
        batch_size = targets_batch.shape[0]
        clipz = self.config.clipz

        params = [mu, log_var]
        optimizer = self.config.create_optimizer(params=params)
        scheduler = self.config.create_lr_scheduler(optimizer)

        for i in range(num_iterations):
            z = self.reparameterize_batch(mu, log_var, batch_size)
            if clipz:
                z = torch.clamp(z, -self.clip_range, self.clip_range).float()            
            
            imgs = self.generator(z)

            if self.config.improved_flag == True:
                _, label = self.discriminator(imgs)
            else:
                label = self.discriminator(imgs)

            for p in params:
                if p.grad is not None:
                    p.grad.data.zero_()

            prior_loss = self.compute_prior_loss(label)
            iden_loss = self.iden_loss(self.targetnets, imgs, targets_batch, 
                                    self.attack_loss, self.criterion, self.lam)
            
            optimizer.zero_grad()
            loss = prior_loss + iden_loss * self.identity_weight

            imgs.retain_grad()
            loss.backward(retain_graph=True)
            grad_x = imgs.grad.detach().cpu()
            
            trans_grad = 0
            for _ in range(num_samples):
                trans_imgs = transformations(imgs)
                _, trans_label = self.discriminator(trans_imgs)
                loss_prior_trans = self.compute_prior_loss(trans_label)
                loss_iden_trans = self.iden_loss(self.targetnets, trans_imgs, targets_batch,
                                        self.attack_loss, self.criterion, self.lam)
                loss_trans = loss_prior_trans + loss_iden_trans * self.identity_weight
                trans_grad += torch.autograd.grad(loss_trans, imgs, retain_graph=True)[0]
            
            trans_grad /= num_samples

            # backpropagate through imgs -> z -> [mu, log_var]
            grads = torch.autograd.grad(imgs, [mu, log_var], grad_outputs=trans_grad, retain_graph=True)
            mu.grad = grads[0]
            log_var.grad = grads[1]
            optimizer.step()
            
            if scheduler:
                scheduler.step()

            print_every = 25
            viz_every = 50
            vis_condition = self.save_dir and (i + 1) % viz_every == 0 or i == 0
            log_condition = (i + 1) % print_every == 0 or i == 0

            if self.log_progress or self.visualize_intermediate_results:
                with torch.no_grad():
                    _, outputs = self.targetnets[0](imgs)
                    confidence_vector = outputs.softmax(dim=1)
                    confidences = torch.gather(confidence_vector, 1, targets_batch.unsqueeze(1))
                    mean_conf = confidences.mean().detach().cpu()
                    min_conf = confidences.min().detach().cpu().item()
                    max_conf = confidences.max().detach().cpu().item()

            if self.log_progress:
                if log_condition:
                    print(
                        f'iteration {i + 1}: \t total_loss={loss:.4f} \t target_loss={iden_loss:.4f} \t'
                        f' mean_conf={mean_conf:.4f} ({min_conf:.4f}, {max_conf:.4f})'
                    )
            
            # self.visualize_intermediate_results = 0
            if self.visualize_intermediate_results:
                if vis_condition:
                    save_grid_with_cmap(
                        imgs[:25].detach().cpu(), os.path.join(self.save_dir, 
                                                              f'x_Batch_{batch_i + 1}_iter_{i + 1:02d}_conf_{mean_conf:.4f}.png'),
                        nrow=5, handle_imgs=1
                    )
                    grad_x = grad_x.detach().cpu()
                    save_grid_with_cmap(
                        grad_x[:25], os.path.join(self.save_dir,
                                                f'grad_x_Batch_{batch_i + 1}_iter_{i + 1:02d}_conf_{mean_conf:.4f}.png'),
                        nrow=5
                    )
                    trans_grad_x = trans_grad.detach().cpu() 
                    save_grid_with_cmap(
                        trans_grad_x[:25], os.path.join(self.save_dir, 
                                                        f'smooth_x_Batch_{batch_i + 1}_iter_{i + 1:02d}_conf_{mean_conf:.4f}.png'),
                        nrow=5
                    )

        final_z = self.reparameterize_batch(mu, log_var, batch_size)
        final_z = torch.clamp(final_z, -self.clip_range, self.clip_range).float()
        return final_z.detach()
    

    def compute_prior_loss(self, label):
        if self.config.improved_flag:
            Prior_Loss = torch.mean(F.softplus(log_sum_exp(label))) - torch.mean(log_sum_exp(label))
        else:
            Prior_Loss = -label.mean()
        return Prior_Loss


    def find_criterion(self, used_loss):
        criterion = None
        if used_loss == 'logit_loss':
            criterion = nn.NLLLoss().to(device)
        elif used_loss == 'ce_loss':
            criterion = nn.CrossEntropyLoss().to(device)
        else:
            criterion = nn.CrossEntropyLoss().to(device)
        print('criterion:{}'.format(used_loss))
        return criterion


    def iden_loss(self, targetnets, imgs, iden, used_loss, criterion, lam=0.1):
        Iden_Loss = 0
        loss_reg = 0
        for target_model in targetnets:
            feat, out = target_model(imgs)
            if used_loss == 'logit_loss':  # reg only with the target classifier, reg is randomly from distribution
                if Iden_Loss == 0:
                    loss_sdt = criterion(out, iden)
                    loss_reg = lam * reg_loss(feat, self.fea_mean[0], self.fea_logvar[0])  # reg only with the target classifier

                    Iden_Loss = Iden_Loss + loss_sdt
                else:
                    loss_sdt = criterion(out, iden)
                    Iden_Loss = Iden_Loss + loss_sdt
            else:
                loss_sdt = criterion(out, iden)
                Iden_Loss = Iden_Loss + loss_sdt

        Iden_Loss = Iden_Loss / len(targetnets) + loss_reg
        return Iden_Loss


    def reparameterize_batch(self, mu, logvar, batch_size):
        std = torch.exp(0.5 * logvar) + 5
        eps = torch.randn((batch_size, std.shape[1]), device=std.device)

        return eps * std + mu
