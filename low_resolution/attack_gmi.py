import argparse
import math
import random
import csv
import os
from datetime import datetime
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import TensorDataset
from utils import utils
from attacks.optimize import Optimization
from metrics.fid_score import FID_Score
from metrics.prcd import PRCD
from utils.evaluation import evaluate_results
from datasets.custom_subset import ClassSubset
from utils.attack_config_parser import AttackConfigParser
from utils.datasets import get_facescrub_idx_to_class
from utils.load_gan import get_DCGAN, get_stylegan
from utils.utils import write_list


def create_parser():
    parser = argparse.ArgumentParser(description='Performing model inversion attack')
    parser.add_argument('-c',
                        '--config',
                        default='./gmi_stylegan-celeba_vgg16-celeba.yaml',
                        type=str,
                        dest="config",
                        help='Config .json file path (default: None)')
    parser.add_argument('--exp_name', '-exp', default='stylegan_celeba_vgg16', help='')
    parser.add_argument('--stylegan', '-sg', action='store_true', help='use styleGAN')
    
    
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

def write_list(filename, precision_list):
    filename = f"{filename}.csv"
    file_exists = os.path.isfile(filename)

    with open(filename, 'a', newline='') as csv_file:
        wr = csv.writer(csv_file, quoting=csv.QUOTE_ALL)
        if file_exists:
            for row in precision_list[1:]:
                wr.writerow(row)
        else:
            for row in precision_list:
                wr.writerow(row)
    return filename


def attack_single_id(targetnets, z, G, D, evaluation_model, targets_single_id):
    id_save_dir = os.path.join(save_dir, f'{target_id:03d}')
    Path(id_save_dir).mkdir(parents=True, exist_ok=True)

    ####################################
    #        Attack Preparation        #
    ####################################

    start_time = time.time()
    # Initialize RTPT

    final_z_path = f"{id_save_dir}/final_z_selected.pt"
    final_z_unselected_path = f"{id_save_dir}/final_z_unselected.pt"
    if os.path.exists(final_z_unselected_path):
        z_optimized_unselected = torch.load(final_z_unselected_path)
        print(f"Loaded z from {final_z_unselected_path}")

    else:
        # Print attack configuration
        print(
            f'Start attack against {targetnets_name} optimizing z with shape {list(z.shape)} ',
            f'and target {target_id}.')

        ####################################
        #         Attack Iteration         #
        ####################################
        optimization = Optimization(targetnets, G, D, evaluation_model, fea_mean, fea_logvar, config, 
                                    stylegan=args.stylegan, save_dir=id_save_dir)

        # Collect results
        z_optimized = []

        # Prepare batches for attack
        for i in range(math.ceil(z.shape[0] / batch_size)):
            z_batch = z[i * batch_size:(i + 1) * batch_size].cuda()
            targets_batch = targets_single_id[i * batch_size:(i + 1) * batch_size].cuda()
            print(
                f'\nOptimizing batch {i + 1} of {math.ceil(z.shape[0] / batch_size)} targeting classes {set(targets_batch.cpu().tolist())}.'
            )

            # Run attack iteration
            torch.cuda.empty_cache()

            z_batch_optimized = optimization.gmi_optimize(i ,z_batch, targets_batch, num_iterations)
            
            z_batch_optimized = z_batch_optimized.detach().cpu()

            # Collect optimized style vectors
            z_optimized.append(z_batch_optimized)

        # Concatenate optimized style vectors
        z_optimized_unselected = torch.cat(z_optimized, dim=0)

        torch.cuda.empty_cache()

    mi_time = time.time() - start_time
    print(f"Inversion execution time: {mi_time:.4f} seconds")
    time_consumption_list = [['mi_time'], [mi_time]]
    _ = write_list(
        f'{save_dir}/time_consumption',
        time_consumption_list)
    
    ####################################
    #          Filter Results          #
    ####################################

    start_time = time.time()
    # Filter results
    final_z, final_targets = z_optimized_unselected, targets_single_id

    torch.save(final_z.detach(), final_z_path)
    torch.save(z_optimized_unselected.detach(), final_z_unselected_path)

    evaluate_results(evaluation_model,
                    G, False,
                    batch_size, idx_to_class,
                    final_z, final_targets,
                    training_dataset,
                    targets_single_id,
                    id_save_dir,
                    save_dir)

    return final_z, final_targets


if __name__ == '__main__':
    ####################################
    #        Attack Preparation        #
    ####################################

    # Set devices
    torch.set_num_threads(24)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    gpu_devices = [i for i in range(torch.cuda.device_count())]

    # Define and parse attack arguments
    parser = create_parser()
    config, args = parse_arguments(parser)

    # Set seeds
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)

    # Load idx to class mappings
    idx_to_class = None
    if config.dataset['name'].lower() == 'facescrub':
        idx_to_class = get_facescrub_idx_to_class()
    else:
        class KeyDict(dict):
            def __missing__(self, key):
                return key
        idx_to_class = KeyDict()

    # Load target model and set dataset
    targetnets, fea_mean, fea_logvar = config.create_target_models()
    targetnets_name = ",".join(net.name for net in targetnets)
    print(f"Target model name: {targetnets_name}")

    training_dataset, private_dataset = config.create_datasets()

    # Load evaluation model
    evaluation_model = config.create_evaluation_model()
    evaluation_model = torch.nn.DataParallel(evaluation_model)
    evaluation_model.to(device)
    evaluation_model.eval()

    # Create target vectors
    targets = config.create_target_vector()

    # Load pre-trained GANs
    if args.stylegan:
        G, D = get_stylegan(config.gan['prior'],
                            config.gan['stylegan_model_dir'])
        num_ws = G.num_ws
        print(f"num_ws: {num_ws}")
    else:
        G, D = get_DCGAN(config.gan["prior"], gan_type=config.improved_flag,
                gan_model_dir=config.gan["dcgan_model_dir"],
                n_classes=config.gan["n_classes"], z_dim=config.attack["z_dim"], target_model=targetnets[0].name)
    
    # Load basic attack parameters
    attack_method = config.attack['method']
    dataset_name = config.dataset['name']
    attack_loss = config.attack['loss']
    num_iterations = config.attack['num_iterations']
    batch_size = config.attack['batch_size'] 
    z_dim = config.attack["z_dim"]
    num_candidates = config.candidates["num_candidates"]

    prefix = os.path.join(config.root_path, attack_method)
    save_folder = os.path.join("{}_{}".format(dataset_name, targetnets[0].name),
                               attack_loss)
    prefix = os.path.join(prefix, save_folder)

    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    save_dir = f"{prefix}/{args.exp_name}_{current_time}"

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    config.save_config(save_dir)
    utils.Tee(os.path.join(save_dir, 'log.txt'))

    # Print attack configuration
    print(f'\nAttack parameters')
    for key in config.attack:
        print(f'\t{key}: {config.attack[key]}')
    print(
        f'Performing attack on {torch.cuda.device_count()} gpus and an effective batch size of {batch_size} images.')

    # Collect results
    all_final_z = []
    all_final_targets = []
    
    search_config = config.candidates
    z = config.create_candidates(args.stylegan, G, targets, targetnets, z_dim, num_candidates, seed=config.seed)

    for target_id in sorted(list(set(targets.tolist()))):
        print(f"\nAttack target class: [{target_id}]")

        indices = torch.where(targets == target_id)[0]
        z_subset = z[indices].to(device)
        targets_single_id = targets[indices].to(device)
        
        final_z, final_targets = attack_single_id(targetnets, z_subset, G, D, evaluation_model, targets_single_id)

        all_final_z.append(final_z)
        all_final_targets.append(final_targets)

    # Concatenate optimized style vectors
    all_final_z = torch.cat(all_final_z, dim=0)
    all_final_targets = torch.cat(all_final_targets, dim=0)

    torch.cuda.empty_cache()

    ####################################
    #    FID Score and GAN Metrics     #
    ####################################

    # create datasets
    attack_dataset = TensorDataset(all_final_z.cpu(), all_final_targets.cpu())

    attack_dataset.targets = all_final_targets
    
    private_dataset.targets = targets.cpu().numpy()
    private_dataset = ClassSubset(
        private_dataset, target_classes=torch.unique(all_final_targets).cpu().tolist())

    # compute FID score
    fid_evaluation = FID_Score(
        private_dataset, attack_dataset, device=device, crop_size=64, generator=G.synthesis, batch_size=batch_size * 3, dims=2048, num_workers=8, gpu_devices=gpu_devices)
    fid_score = fid_evaluation.compute_fid()
    print(
        f'FID score computed on {all_final_z.shape[0]} attack samples and {dataset_name}: {fid_score:.4f}'
    )

    # compute precision, recall, density, coverage
    prdc = PRCD(private_dataset, attack_dataset, device=device, crop_size=64, generator=G.synthesis, batch_size=batch_size * 3, dims=2048, num_workers=8, gpu_devices=gpu_devices)
    precision, recall, density, coverage = prdc.compute_metric(num_classes=config.num_classes, k=3)
    print(
        f' Precision: {precision:.4f}, Recall: {recall:.4f}, Density: {density:.4f}, Coverage: {coverage:.4f}'
    )