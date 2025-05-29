import argparse
import os
import time
import torch

from metrics.accuracy import Accuracy
from utils.training_config_parser import TrainingConfigParser
from utils.utils import Tee

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def main():
    # Define and parse arguments
    parser = argparse.ArgumentParser(
        description='Training a target classifier')
    parser.add_argument('-c',
                        '--config',
                        default='./configs/training/targets/vgg16_1000cls.yaml',
                        type=str,
                        dest="config",
                        help='Config .json file path (default: None)')
    
    parser.add_argument('--adversarial', '-adv', action='store_true', help='enable adversarial training')

    args = parser.parse_args()

    if not args.config:
        print(
            "Configuration file is missing. Please check the provided path. Execution is stopped."
        )
        exit()

    # Load json config file
    config = TrainingConfigParser(args.config.strip())

    # Set seeds and make deterministic
    seed = config.seed
    torch.manual_seed(seed)

    # Create the target model architecture
    target_model = config.create_model()

    # Build the datasets
    train_set, test_set = config.create_datasets()
    criterion = torch.nn.CrossEntropyLoss()
    metric = Accuracy

    # Set up optimizer and scheduler
    optimizer = config.create_optimizer(target_model)

    lr_scheduler = config.create_lr_scheduler(optimizer)
    # lr_scheduler = None

    target_model.to(device)

    # modify the save_path such that subfolders with a timestamp and the name of the run are created
    time_stamp = time.strftime("%Y%m%d_%H%M%S")

    if args.adversarial == True:
        save_dir = os.path.join(
            config.training['save_path'],
            f"adv_{config.dataset['name']}_{config.model['architecture']}_{time_stamp}"
        )
    else:
        save_dir = os.path.join(
            config.training['save_path'],
            f"adv_{config.dataset['name']}_{config.model['architecture']}_{time_stamp}"
        )
    os.makedirs(save_dir, exist_ok=True)
    config.save_config(save_dir)
    Tee(os.path.join(save_dir, 'log.txt'), 'w')

    # Start training
    if args.adversarial:
        print("\nAdversarial Training.")
        target_model.adv_fit(
            training_data=train_set,
            validation_data=test_set,
            test_data=test_set,
            criterion=criterion,
            metric=metric,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            adv_cfg=config.adv_cfg,
            batch_size=config.training['batch_size'],
            num_epochs=config.training['num_epochs'],
            dataloader_num_workers=config.training['dataloader_num_workers'],
            save_base_path=save_dir)
    else:
        print("\nRegular Training.")
        target_model.fit(
            training_data=train_set,
            validation_data=test_set,
            test_data=test_set,
            criterion=criterion,
            metric=metric,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            batch_size=config.training['batch_size'],
            num_epochs=config.training['num_epochs'],
            dataloader_num_workers=config.training['dataloader_num_workers'],
            save_base_path=save_dir)

if __name__ == '__main__':
    main()