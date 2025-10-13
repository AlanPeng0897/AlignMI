import pickle, os, csv
import functools
import torch
from configs import paths_config, global_config


def toogle_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def load_tuned_G(run_id, type):
    new_G_path = f'{paths_config.checkpoints_dir}/model_{run_id}_{type}.pt'
    with open(new_G_path, 'rb') as f:
        new_G = torch.load(f).to(global_config.device).eval()
    new_G = new_G.float()
    toogle_grad(new_G, False)
    return new_G


def load_old_G():
    with open(paths_config.stylegan2_ada_ffhq, 'rb') as f:
        old_G = pickle.load(f)['G_ema'].to(global_config.device).eval()
        old_G = old_G.float()
    return old_G



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
