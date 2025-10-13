from losses.poincare import poincare_loss
import math

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

input_features_of_last_layer = None

def forward_hook(module, inputs, output):
    global input_features_of_last_layer
    input_features_of_last_layer = inputs[0]


def toogle_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


class Optimization():
    def __init__(self, target_model, synthesis, discriminator, transformations, num_ws, config):
        # self.synthesis = synthesis
        self.target = target_model
        self.discriminator = discriminator
        self.config = config
        self.transformations = transformations
        self.discriminator_weight = self.config.attack['discriminator_loss_weight']
        self.num_ws = num_ws
        self.clip = config.attack['clip']

        self.target.module.model.fc.register_forward_hook(forward_hook)

        self.original_synthesis = copy.deepcopy(synthesis)

    def diversity_regularizer(self, feats):
        pairwise_distance = torch.norm(feats.unsqueeze(1) - feats.unsqueeze(0), p=2, dim=2)
        diversity_loss = pairwise_distance.mean()

        return diversity_loss

    def optimize(self, w_batch, targets_batch, num_epochs):
        # Initialize attack
        optimizer = self.config.create_optimizer(params=[w_batch.requires_grad_()])
        scheduler = self.config.create_lr_scheduler(optimizer)

        # self.optimizer_w = self.config.selective_create_optimizer(mode="latent_code",
        #                                                           params=[w_batch.requires_grad_()],
        #                                                           config=self.config)

        # self.scheduler_w = self.config.create_lr_scheduler(self.optimizer_w)
        self.synthesis = copy.deepcopy(self.original_synthesis)

        # Start optimization
        for i in range(num_epochs):
            # synthesize images and preprocess images
            imgs = self.synthesize(w_batch, num_ws=self.num_ws)

            # compute discriminator loss
            if self.discriminator_weight > 0:
                discriminator_loss = self.compute_discriminator_loss(imgs)
            else:
                # discriminator_loss = torch.tensor(0.0)
                discriminator_loss = self.compute_discriminator_loss(imgs)

            # perform image transformations
            if self.clip:
                imgs = self.clip_images(imgs)
            if self.transformations:
                imgs = self.transformations(imgs)

            # Compute target loss
            # print(len(input_features_of_last_layer))  # 0
            outputs = self.target(imgs)  # [40, 3, 224, 224]
            # print(len(input_features_of_last_layer))  # 4

            # target_loss = poincare_loss(outputs, targets_batch).mean()

            class_weights = self.target.module.model.fc.weight[targets_batch]
            input_features = input_features_of_last_layer  # 获取输入特征
            cosine_similarity = F.cosine_similarity(input_features, class_weights).mean()
            target_loss = -1 * cosine_similarity

            loss = target_loss + discriminator_loss * self.discriminator_weight

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Log results
            if self.config.log_progress and ((i + 1) % 10 == 0):
                with torch.no_grad():
                    confidence_vector = outputs.softmax(dim=1)
                    confidences = torch.gather(
                        confidence_vector, 1, targets_batch.unsqueeze(1))
                    mean_conf = confidences.mean().detach().cpu()

                    min_conf = confidences.min().detach().cpu().item()  # Get minimum confidence
                    max_conf = confidences.max().detach().cpu().item()  # Get maximum confidence

                if torch.cuda.current_device() == 0:
                    print(
                        f'iteration {i + 1}: \t total_loss={loss:.4f} \t cs_similarity={cosine_similarity.item():.4f} \t',
                        f'discriminator_loss={discriminator_loss:.4f} \t '
                        f'mean_conf={mean_conf:.4f} ({min_conf:.4f}, {max_conf:.4f})'
                    )

        return w_batch.detach()

    def print_synthesis_gradients(self):
        for name, param in self.synthesis.named_parameters():
            if param.grad is not None:
                print(f"Gradients of {name}: \n{param.grad}")
            else:
                print(f"No gradients for {name}")

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
