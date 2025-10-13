import argparse
import os
import time
import torch
from metrics.accuracy import Accuracy
from utils.training_config_parser import TrainingConfigParser
from utils.utils import Tee
from datasets.celeba import TangentBasisDataset


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
    # Define and parse arguments
    parser = argparse.ArgumentParser(description='Training a target classifier')
    parser.add_argument('-c',
                        '--config',
                        default='./configs/training/targets/vgg16_100cls.yaml',
                        type=str,
                        dest="config",
                        help='Config .json file path (default: None)')
    
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
    # train_set, test_set = config.create_datasets()
    _, test_set = config.create_datasets()

    criterion = torch.nn.CrossEntropyLoss()
    metric = Accuracy

    # Set up optimizer and scheduler
    optimizer = config.create_optimizer(target_model)

    lr_scheduler = config.create_lr_scheduler(optimizer)
    # lr_scheduler = None

    target_model.to(device)

    # Modify the save_path such that subfolders with a timestamp and the name of the run are created
    time_stamp = time.strftime("%Y%m%d_%H%M%S")
    # time_stamp = args.exp_name + '_' + time_stamp if args.exp_name is not None else time_stamp
    time_stamp = f'align_{config.training["num_random_label"]}cls_{config.training["align_coeff"]:.1f}' + '_' + time_stamp
    save_dir = os.path.join(config.training['save_path'],
                            f"{config.model['architecture']}_{time_stamp}")
    os.makedirs(save_dir, exist_ok=True)
    config.save_config(save_dir)
    Tee(os.path.join(save_dir, 'log.txt'), 'w')
    
    # Load data
    data_dict = torch.load('./tangent_space/x_y_U_list_subset0.pt')

    train_set = TangentBasisDataset(data_dict)

    # Start training
    print("Align Training!")
    target_model.align_fit(
        training_data=train_set,
        validation_data=test_set,
        criterion=criterion,
        metric=metric,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        batch_size=config.training['batch_size'],
        num_epochs=config.training['num_epochs'],
        dataloader_num_workers=config.training['dataloader_num_workers'],
        save_base_path=save_dir,
        align_coeff=config.training['align_coeff'])

if __name__ == '__main__':
    main()
