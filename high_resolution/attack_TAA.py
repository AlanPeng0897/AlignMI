import argparse
import copy
import csv
import dnnlib
import math
import numpy as np
import os
import random
import time
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms as transforms
import torchvision.utils as vutils
import traceback
# import wandb

from attacks.final_selection import perform_final_selection
from attacks.optimize import Optimization
from collections import Counter
from copy import deepcopy
from datasets.custom_subset import ClassSubset
from datetime import datetime
from facenet_pytorch import InceptionResnetV1
from metrics.classification_acc import ClassificationAccuracy
from metrics.distance_metrics import DistanceEvaluation
from metrics.fid_score import FID_Score
from metrics.prcd import PRCD
from pathlib import Path
from PIL import Image
from random import choice
from rtpt import RTPT
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from utils.attack_config_parser import AttackConfigParser
from utils.datasets import (
    create_target_dataset,
    get_facescrub_idx_to_class,
    get_stanford_dogs_idx_to_class,
)
from utils.models_utils import write_list
from utils.stylegan import create_image, load_discrimator, load_generator
from utils.wandb import Tee
# from utils.wandb import *


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def create_parser():
    parser = argparse.ArgumentParser(
        description='Performing model inversion attack')
    parser.add_argument('-c',
                        '--config',
                        default=None,
                        type=str,
                        dest="config",
                        help='Config .json file path (default: None)')
    parser.add_argument('--no_rtpt',
                        action='store_false',
                        dest="rtpt",
                        help='Disable RTPT')
    parser.add_argument('--exp_name',
                        default=None,
                        type=str,
                        help='Directory to save output files (default: None)')
    return parser


def parse_arguments(parser):
    args = parser.parse_args()

    if not args.config:
        print(
            "Configuration file is missing. Please check the provided path. Execution is stopped."
        )
        exit()

    # Load attack config
    config = AttackConfigParser(args.config)

    return config, args


def create_initial_vectors(config, G, target_model, targets):
    with torch.no_grad():
        w = config.create_candidates(G, target_model, targets).cpu()
        if config.attack['single_w']:
            w = w[:, 0].unsqueeze(1)

        w_init = deepcopy(w)

    return w, w_init


def log_nearest_neighbors_local(imgs, targets, eval_model, dataset,
                                img_size, seed, save_dir, nrow):
    # Find closest training samples to final results
    evaluater = DistanceEvaluation(eval_model, None, img_size, None, dataset, seed)
    closest_samples, distances = evaluater.find_closest_training_sample(
        imgs, targets)

    grid = vutils.make_grid(closest_samples, nrow=nrow, padding=2, normalize=True)
    img = Image.fromarray(grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy())

    img.save(os.path.join(save_dir, f'nearest_neighbor.png'))

    for i, (img, d) in enumerate(zip(closest_samples, distances)):
        img = (img - img.min()) / (img.max() - img.min())
        img = Image.fromarray(img.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy())
        img.save(os.path.join(save_dir, f'{i:02d}_target={target_id:03d}_distance_{d:.2f}.png'))

    return


def log_final_images_local(imgs, predictions, max_confidences, target_confidences,
                           idx2cls, save_dir, nrow):
    grid = vutils.make_grid(imgs, nrow=nrow, padding=2, normalize=True)
    img = Image.fromarray(grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy())

    img.save(os.path.join(save_dir, f'final_images.png'))

    for i, (img, pred, max_conf, target_conf) in enumerate(
            zip(imgs.cpu(), predictions, max_confidences, target_confidences)):
        img = (img - img.min()) / (img.max() - img.min())
        img = Image.fromarray(img.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy())
        img.save(os.path.join(save_dir,
                              f'{i:02d}_target={target_id:03d} ({target_conf:.2f})_pred={pred.item():03d} ({max_conf:.2f}).png'))
    return


def attack_single_id(target_model, w, synthesis, discriminator, targets_single_id):
    if config.logging:
        id_save_dir = os.path.join(save_dir, f'{target_id:03d}')
        Path(id_save_dir).mkdir(parents=True, exist_ok=True)

    ####################################
    #        Attack Preparation        #
    ####################################
    
    start_time = time.time()
    # Initialize RTPT
    rtpt = None
    
    final_w_path = f"{id_save_dir}/final_w_selected.pt"
    final_w_unselected_path = f"{id_save_dir}/final_w_unselected.pt"
    if os.path.exists(final_w_unselected_path):
        w_optimized_unselected = torch.load(final_w_unselected_path)
        print(f"Loaded z from {final_w_unselected_path}")

    else:
        # Print attack configuration
        print(
            f'Start attack against {target_model.name} optimizing w with shape {list(w.shape)} ',
            f'and targets {dict(Counter(targets_single_id.cpu().numpy()))}.')

        print(
            f'Performing attack on {torch.cuda.device_count()} gpus and an effective batch size of {batch_size} images.'
        )

        if args.rtpt:
            max_iterations = math.ceil(w.shape[0] / batch_size) \
                             + int(math.ceil(w.shape[0] / (batch_size * 3))) \
                             + 2 * int(math.ceil(config.candidates['num_candidates']
                                                 * len(set(targets_single_id.cpu().tolist())) / (batch_size * 3))) \
                             + 2 * len(set(targets_single_id.cpu().tolist()))
            rtpt = RTPT(name_initials='Unlearn',
                        experiment_name='Jailbreak',
                        max_iterations=max_iterations)
            rtpt.start()

        # Create attack transformations
        attack_transformations = config.create_attack_transformations()

        ####################################
        #         Attack Iteration         #
        ####################################
        optimization = Optimization(target_model, synthesis, discriminator,
                                    attack_transformations, num_ws, config, save_dir=id_save_dir)

        # Collect results
        w_optimized = []

        # Prepare batches for attack
        for i in range(math.ceil(w.shape[0] / batch_size)):
            w_batch = w[i * batch_size:(i + 1) * batch_size].cuda()
            targets_batch = targets_single_id[i * batch_size:(i + 1) * batch_size].cuda()
            print(
                f'\nOptimizing batch {i + 1} of {math.ceil(w.shape[0] / batch_size)} targeting classes {set(targets_batch.cpu().tolist())}.'
            )

            # Run attack iteration
            torch.cuda.empty_cache()
            w_batch_optimized = optimization.optimization_TAA(i, w_batch, targets_batch,
                                                      num_epochs).detach().cpu()

            if rtpt:
                num_batches = math.ceil(w.shape[0] / batch_size)
                rtpt.step(subtitle=f'batch {i + 1} of {num_batches}')

            # Collect optimized style vectors
            w_optimized.append(w_batch_optimized)

        # Concatenate optimized style vectors
        w_optimized_unselected = torch.cat(w_optimized, dim=0)
        torch.cuda.empty_cache()
        del discriminator

    mi_time = time.time() - start_time
    print(f"Inversion execution time: {mi_time:.4f} seconds")

    ####################################
    #          Filter Results          #
    ####################################

    start_time = time.time()
    # Filter results
    # final_w, final_targets = w_optimized_unselected, targets_single_id
    if config.final_selection:
        print(
            f'\nSelect final set of max. {config.final_selection["samples_per_target"]} ',
            f'images per target using {config.final_selection["approach"]} approach.'
        )
        final_w, final_targets = perform_final_selection(
            w_optimized_unselected,
            synthesis,
            config,
            targets_single_id,
            target_model,
            device=device,
            batch_size=batch_size,
            **config.final_selection,
            rtpt=rtpt)
        print(f'Selected a total of {final_w.shape[0]} final images ',
              f'of target classes {set(final_targets.cpu().tolist())}.')
    else:
        final_w, final_targets = w_optimized_unselected, targets_single_id
    del target_model

    selection_time = time.time() - start_time
    print(f"Selection execution time: {selection_time:.4f} seconds")

    time_consumption_list = [['mi_time', 'selection_time'], [mi_time, selection_time]]
    if config.logging:
        _ = write_list(
            f'{save_dir}/time_consumption',
            time_consumption_list)

    # Log selected vectors
    if config.logging:
        # torch.save(final_w.detach(), final_w_path)
        torch.save(w_optimized_unselected.detach(), final_w_unselected_path)

    ####################################
    #         Attack Accuracy          #
    ####################################
    try:
        evaluation_model = config.create_evaluation_model()
        evaluation_model = torch.nn.DataParallel(evaluation_model)
        evaluation_model.to(device)
        evaluation_model.eval()
        class_acc_evaluator = ClassificationAccuracy(evaluation_model,
                                                     device=device)

        acc_top1, acc_top5, predictions, avg_correct_conf, avg_total_conf, target_confidences, maximum_confidences, precision_list = class_acc_evaluator.compute_acc(
            w_optimized_unselected,
            targets_single_id,
            synthesis,
            config,
            batch_size=batch_size,
            resize=299,
            rtpt=rtpt)

        # Compute attack accuracy on filtered samples
        if config.final_selection:
            acc_top1, acc_top5, predictions, avg_correct_conf, avg_total_conf, target_confidences, maximum_confidences, precision_list = class_acc_evaluator.compute_acc(
                final_w,
                final_targets,
                synthesis,
                config,
                batch_size=batch_size,
                resize=299,
                rtpt=rtpt)

            if config.logging:
                _ = write_list(
                    f'{save_dir}/precision_list',
                    precision_list
                )

            print(
                f'Filtered Evaluation of {final_w.shape[0]} images on Inception-v3: \taccuracy@1={acc_top1:4f}, ',
                f'accuracy@5={acc_top5:4f}, correct_confidence={avg_correct_conf:4f}, total_confidence={avg_total_conf:4f}'
            )
        
        del evaluation_model

    except Exception:
        print(traceback.format_exc())

    ####################################
    #    FID Score and GAN Metrics     #
    ####################################

    paths = [
        './data/facescrub/actors',
        './data/facescrub/actresses',
    ]

    for p in paths:
        path = Path(p)
        path.touch()

    ####################################
    #         Feature Distance         #
    ####################################
    avg_dist_facenet = None
    try:
        if target_dataset in [
            'facescrub', 'celeba_identities', 'celeba_attributes'
        ]:
            # Load FaceNet model for face recognition
            facenet = InceptionResnetV1(pretrained='vggface2')
            facenet = torch.nn.DataParallel(facenet, device_ids=gpu_devices)
            facenet.to(device)
            facenet.eval()

            # Compute average feature distance on facenet
            # os.system("touch ./data/facescrub")
            evaluater_facenet = DistanceEvaluation(facenet, synthesis, 160,
                                                   config.attack_center_crop,
                                                   target_dataset, config.seed)
            
            avg_dist_facenet, mean_distances_list = evaluater_facenet.compute_dist(
                final_w,
                final_targets,
                batch_size=batch_size_single,
                rtpt=rtpt)
            if config.logging:
                _ = write_list(
                    f'{save_dir}/distance_facenet',
                    mean_distances_list)
                # wandb.save(filename_distance)

            print('Mean Distance on FaceNet: ', avg_dist_facenet.cpu().item())
    
    except Exception:
        print(traceback.format_exc())

    ####################################
    #          Finish Logging          #
    ####################################

    if rtpt:
        rtpt.step(subtitle=f'Finishing up')

    # Logging of final results
    if config.logging:
        print('Finishing attack, logging results and creating sample images.')

        num_target_classes = len(set(targets_single_id.tolist()))
        num_imgs = 10
        # Sample final images from the first and last classes
        label_subset = set(
            list(set(targets_single_id.tolist()))[:int(num_target_classes / 2)] +
            list(set(targets_single_id.tolist()))[-int(num_target_classes / 2):])

        log_imgs = []
        log_targets = []
        log_predictions = []
        log_max_confidences = []
        log_target_confidences = []
        # Log images with smallest feature distance
        for label in label_subset:
            mask = torch.where(final_targets == label, True, False)
            w_masked = final_w[mask][:num_imgs]
            imgs = create_image(w_masked,
                                synthesis,
                                crop_size=config.attack_center_crop,
                                resize=config.attack_resize
                                )
            log_imgs.append(imgs)
            log_targets += [label for i in range(num_imgs)]
            log_predictions.append(torch.tensor(predictions)[mask][:num_imgs])
            log_max_confidences.append(
                torch.tensor(maximum_confidences)[mask][:num_imgs])
            log_target_confidences.append(
                torch.tensor(target_confidences)[mask][:num_imgs])

        log_imgs = torch.cat(log_imgs, dim=0)
        log_predictions = torch.cat(log_predictions, dim=0)
        log_max_confidences = torch.cat(log_max_confidences, dim=0)
        log_target_confidences = torch.cat(log_target_confidences, dim=0)

        log_final_images_local(log_imgs, log_predictions, log_max_confidences,
                               log_target_confidences, idx_to_class,
                               save_dir=id_save_dir, nrow=num_imgs)

        # Use FaceNet only for facial images
        facenet = InceptionResnetV1(pretrained='vggface2')
        facenet = torch.nn.DataParallel(facenet, device_ids=gpu_devices)
        facenet.to(device)
        facenet.eval()
        if target_dataset in [
            'facescrub', 'celeba_identities', 'celeba_attributes'
        ]:
            log_nearest_neighbors_local(log_imgs,
                                        log_targets,
                                        facenet,
                                        target_dataset,
                                        img_size=160,
                                        seed=config.seed,
                                        save_dir=id_save_dir,
                                        nrow=num_imgs)

    return final_w, final_targets, [mi_time, selection_time]


if __name__ == '__main__':
    # Define and parse attack arguments
    parser = create_parser()
    config, args = parse_arguments(parser)

    torch.set_num_threads(24)
    gpu_devices = [i for i in range(torch.cuda.device_count())]

    # Set seeds
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)

    # Load idx to class mappings
    idx_to_class = None
    if config.dataset.lower() == 'facescrub':
        idx_to_class = get_facescrub_idx_to_class()
    elif config.dataset.lower() == 'stanford_dogs':
        idx_to_class = get_stanford_dogs_idx_to_class()
    else:
        class KeyDict(dict):

            def __missing__(self, key):
                return key

        idx_to_class = KeyDict()

    # Load target model and set dataset
    target_model = config.create_target_model()
    target_model_name = target_model.name
    target_dataset = config.get_target_dataset()

    # Load basic attack parameters
    num_epochs = config.attack['num_epochs'] 
    batch_size_single = config.attack['batch_size']

    batch_size = config.attack['batch_size'] * torch.cuda.device_count()
    targets = config.create_target_vector()

    # Load pre-trained StyleGan2 components
    G = load_generator(config.stylegan_model)
    D = load_discrimator(config.stylegan_model)
    num_ws = G.num_ws

    # Distribute models
    target_model = torch.nn.DataParallel(target_model, device_ids=gpu_devices)
    target_model.name = target_model_name
    G.synthesis = torch.nn.DataParallel(G.synthesis, device_ids=gpu_devices)
    G.synthesis.num_ws = num_ws
    discriminator = torch.nn.DataParallel(D, device_ids=gpu_devices)

    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = args.exp_name + '_' + current_time if args.exp_name is not None else current_time

    if config.logging:
        save_dir = os.path.join('attack_results', save_dir)
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        config.save_config(save_dir)
        Tee(os.path.join(save_dir, 'log.txt'))

    all_final_w = []
    all_final_targets = []
    w, w_init = create_initial_vectors(config, G, target_model, targets)

    target_id_list = sorted(list(set(targets.tolist())))
    for target_id in target_id_list:
        indices = torch.where(targets == target_id)[0]
        w_subset = w[indices.to(w.device)]
        targets_single_id = targets[indices].cpu()
        final_w, final_targets, time_list = attack_single_id(target_model, w_subset, G.synthesis, discriminator,
                                                             targets_single_id)
        all_final_w.append(final_w)
        all_final_targets.append(final_targets)

    all_final_w = torch.cat(all_final_w, dim=0)
    all_final_targets = torch.cat(all_final_targets, dim=0)
    rtpt = None

    ####################################
    #    FID Score and GAN Metrics     #
    ####################################

    fid_score = None
    precision, recall = None, None
    density, coverage = None, None
    try:
        # set transformations
        crop_size = config.attack_center_crop
        target_transform = T.Compose([
            T.ToTensor(),
            T.Resize((299, 299), antialias=True),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        # create datasets
        attack_dataset = TensorDataset(all_final_w, all_final_targets)
        attack_dataset.targets = all_final_targets
        training_dataset = create_target_dataset(target_dataset,
                                                 target_transform)

        training_dataset = ClassSubset(
            training_dataset,
            target_classes=torch.unique(all_final_targets).cpu().tolist())

        print(f"The attack dataset length is {len(attack_dataset)}")
        print(f"The private dataset length is {len(training_dataset)}")

        # compute FID score
        fid_evaluation = FID_Score(training_dataset,
                                   attack_dataset,
                                   device=device,
                                   crop_size=crop_size,
                                   generator=G.synthesis,
                                   batch_size=batch_size,
                                   dims=2048,
                                   num_workers=8,
                                   gpu_devices=gpu_devices)
        fid_score = fid_evaluation.compute_fid(rtpt)
        print(
            f'FID score computed on {final_w.shape[0]} attack samples and {config.dataset}: {fid_score:.4f}'
        )

        # compute precision, recall, density, coverage
        prdc = PRCD(training_dataset,
                    attack_dataset,
                    device=device,
                    crop_size=crop_size,
                    generator=G.synthesis,
                    batch_size=batch_size,
                    dims=2048,
                    num_workers=8,
                    gpu_devices=gpu_devices)

        precision, recall, density, coverage = prdc.compute_metric(
            target_id_list=target_id_list, k=3, rtpt=rtpt)

        print(
            f'Precision: {precision:.4f}, Recall: {recall:.4f}, Density: {density:.4f}, Coverage: {coverage:.4f}'
        )

        if config.logging:
            prdc_list = [['fid', 'precision', 'recall', 'density', 'coverage'],
                         [fid_score, precision, recall, density, coverage]]
            _ = write_list(
                f'{save_dir}/prdc_list', prdc_list)


    except Exception:
        print(traceback.format_exc())
