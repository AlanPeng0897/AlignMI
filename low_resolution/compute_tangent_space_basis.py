import argparse
import os
import torch
import torch.nn.functional as F
from torch.utils.data import Subset, DataLoader
from torch.autograd.functional import jvp
from diffusers import AutoencoderKL
from matplotlib.colors import TwoSlopeNorm
import matplotlib.pyplot as plt
import numpy as np
import torchvision.utils as vutils


def save_grid_with_cmap(tensor, save_path, cmap="seismic", nrow=5, padding=2, handle_imgs=False):
    # normalize each sample by its max abs value
    max_vals = tensor.view(tensor.size(0), -1).abs().max(dim=1)[0].view(-1,1,1,1) + 1e-8
    tensor = tensor / max_vals

    if handle_imgs:
        grid = vutils.make_grid(tensor, nrow=nrow, normalize=True, padding=padding)
        arr = grid.cpu().permute(1,2,0).numpy()
        plt.imsave(save_path, arr)
    else:
        grid = vutils.make_grid(tensor, nrow=nrow, normalize=False, padding=padding, pad_value=-1.0)
        arr = grid.cpu().numpy()
        arr = arr[0] if arr.shape[0]==1 else arr.mean(0)
        vmax = np.abs(arr).max()
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        plt.figure(figsize=(5,5))
        plt.imshow(arr, cmap=cmap, norm=norm)
        plt.axis("off")
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
        plt.close()


def compute_tangent_space_basis(imgs, vae, latent_scale, chunk_size):
    device = imgs.device
    B, C, H, W = imgs.shape

    # encode → posterior mean
    q = vae.encode(imgs).latent_dist
    z_mean = q.mean * latent_scale
    latent_dim = z_mean[0].numel()

    # one decode pass (just to infer output dim)
    with torch.no_grad():
        dec = vae.decode((z_mean / latent_scale).float())
        recon = dec.sample if hasattr(dec, "sample") else dec[0]
    N_out = recon.flatten(1).shape[1]

    eye = torch.eye(latent_dim, device=device)
    U_list = []

    for i in range(B):
        z_i = z_mean[i:i+1].detach().requires_grad_()

        def flat_dec(z):
            out = vae.decode((z / latent_scale).float())
            img = out.sample if hasattr(out, "sample") else out[0]
            return img.flatten(start_dim=1)

        # build Jacobian in chunks
        J_parts = []
        for v_chunk in torch.split(eye, chunk_size, dim=0):
            z_rep = z_i.expand(v_chunk.size(0), *z_i.shape[1:])
            v_z   = v_chunk.view(v_chunk.size(0), *z_i.shape[1:])
            _, Jc = jvp(flat_dec, (z_rep,), (v_z,))
            J_parts.append(Jc)
        J = torch.cat(J_parts, dim=0)  # [latent_dim, N_out]

        # SVD → take Vᵀ so columns span tangent
        U, S, Vt = torch.linalg.svd(J, full_matrices=False)
        U_list.append(Vt)

    return U_list

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',    type=str, default='./configs/training/targets/compute_tangent_space_basis.yaml')
    parser.add_argument('--output_dir',type=str, default='./tangent_space')
    parser.add_argument('--batch_size',type=int, default=100)
    parser.add_argument('--chunk_size',type=int, default=8)
    parser.add_argument('--world_size',type=int, default=10)
    parser.add_argument('--rank',      type=int, default=0)
    args = parser.parse_args()

    from utils.training_config_parser import TrainingConfigParser
    config = TrainingConfigParser(args.config)
    torch.manual_seed(config.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # split dataset among ranks
    train_set, _ = config.create_datasets()
    N = len(train_set)
    per = N // args.world_size
    start = args.rank * per
    end   = N if args.rank==args.world_size-1 else (args.rank+1)*per
    subset = Subset(train_set, list(range(start, end)))
    loader = DataLoader(subset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    os.makedirs(args.output_dir, exist_ok=True)

    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device).eval()
    latent_scale = 0.18215

    all_x, all_y, all_U = [], [], []

    for b, (x,y) in enumerate(loader):
        x = x.to(device)
        y = y.to(device)

        U_list = compute_tangent_space_basis(x, vae, latent_scale, args.chunk_size)
        for i in range(x.size(0)):
            all_x.append(x[i].cpu())
            all_y.append(y[i].cpu())
            all_U.append(U_list[i].cpu())

        print(f"Rank {args.rank}: batch {b} → collected {len(all_x)} samples")

    out = {
        'x': torch.stack(all_x),
        'y': torch.stack(all_y),
        'U': torch.stack(all_U)
    }
    torch.save(out, os.path.join(args.output_dir, f"x_y_U_list_rank{args.rank}.pt"))
    print("Done.")

if __name__ == '__main__':
    main()
