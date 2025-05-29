import numpy as np
import torch
import torchvision.transforms.functional as F
from tqdm import tqdm
import torchvision.transforms as T


def find_initial_z(generator,
                   target_model,
                   targets,
                   z_dim,
                   search_space_size,
                   clip=True,
                   resize=64,
                   horizontal_flip=False,
                   center_crop=None,
                   batch_size=200,
                   seed=42):
    """Find initial latent vectors with highest confidence for each target class.

    Args:
        generator (nn.Module): GAN generator (x = generator(z)).
        target_model (nn.Module): Target classifier.
        targets (List[int] or torch.Tensor): Target class indices to search over.
        z_dim (int): Latent dimension of generator.
        search_space_size (int): Number of latent samples to evaluate.
        clip (bool): Clip images to [-1, 1] after generation.
        resize (int): Resize images before classification.
        horizontal_flip (bool): Apply horizontal flip augmentation.
        batch_size (int): Batch size for evaluation.
        seed (int): Random seed.

    Returns:
        torch.Tensor: [len(targets), z_dim] latent vectors for each target class.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(seed)
    np.random.seed(seed)

    z_all = torch.randn(search_space_size, z_dim, device=device)  # [N, z_dim]
    target_model.eval()

    # Run entire batch through generator and classifier
    with torch.no_grad():
        confidence_matrix = []

        dataset = torch.utils.data.TensorDataset(z_all)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

        for batch in tqdm(loader, desc='Evaluating latent candidates'):
            z_batch = batch[0].to(device)  # [B, z_dim]
            imgs = generator(z_batch)      # [B, C, H, W]

            if clip:
                lower_bound = torch.tensor(-1.0).float().to(imgs.device)
                upper_bound = torch.tensor(1.0).float().to(imgs.device)
                imgs = torch.where(imgs > upper_bound, upper_bound, imgs)
                imgs = torch.where(imgs < lower_bound, lower_bound, imgs)

            if center_crop is not None:
                imgs = F.center_crop(imgs, (center_crop, center_crop))

            img_variants = []
            imgs_resized = [F.resize(imgs, resize, antialias=True)] if resize is not None else [imgs]
            img_variants.extend(imgs_resized)

            if horizontal_flip:
                img_variants.append(F.hflip(imgs_resized[0]))

            avg_conf = None
            for im in img_variants:
                logits = target_model(im)
                if isinstance(logits, tuple):  # Some models return (features, logits)
                    logits = logits[-1]
                probs = logits.softmax(dim=1)

                if avg_conf is None:
                    avg_conf = probs
                else:
                    avg_conf += probs

            avg_conf /= len(img_variants)  # [B, num_classes]
            confidence_matrix.append(avg_conf.cpu())

        confidence_matrix = torch.cat(confidence_matrix, dim=0)  # [search_space_size, num_classes]

    # Select the best latent vector for each target class
    final_z_list = []
    targets = torch.tensor(targets) if not isinstance(targets, torch.Tensor) else targets

    for cls in targets:
        cls_conf = confidence_matrix[:, cls]               # [search_space_size]
        best_idx = torch.argmax(cls_conf).item()           # scalar index
        final_z_list.append(z_all[best_idx].unsqueeze(0))  # [1, z_dim]
        confidence_matrix[best_idx, cls] = -1.0            # suppress reuse

    final_candidates = torch.cat(final_z_list, dim=0)      # [len(targets), z_dim]
    print(f'Found {final_candidates.shape[0]} initial latent vectors.')

    return final_candidates


def find_initial_w(generator,
                   target_model,
                   targets,
                   search_space_size,
                   clip=True,
                   center_crop=None,
                   resize=None,
                   horizontal_flip=True,
                   truncation_psi=0.7,
                   truncation_cutoff=18,
                   batch_size=12,
                   seed=0):
    """Find good initial starting points in the style space.

    Args:
        generator (torch.nn.Module): StyleGAN2 model
        target_model (torch.nn.Module): [description]
        target_cls (int): index of target class.
        search_space_size (int): number of potential style vectors.
        clip (boolean, optional): clip images to [-1, 1]. Defaults to True.
        center_crop (int, optional): size of the center crop. Defaults 768.
        resize (int, optional): size for the resizing operation. Defaults to 224.
        horizontal_flip (boolean, optional): apply horizontal flipping to images. Defaults to true.
        truncation_psi (float, optional): truncation factor. Defaults to 0.7.
        truncation_cutoff (int, optional): truncation cutoff. Defaults to 18.
        batch_size (int, optional): batch size. Defaults to 25.

    Returns:
        torch.tensor: style vectors with highest confidences on the target model and target class.
    """
    print(f"seed {seed}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    z = torch.from_numpy(
        np.random.RandomState(seed).randn(search_space_size,
                                          generator.z_dim)).to(device)
    c = None
    target_model.eval()

    with torch.no_grad():
        confidences = []
        final_candidates = []
        final_confidences = []
        candidates = generator.mapping(z,
                                       c,
                                       truncation_psi=truncation_psi,
                                       truncation_cutoff=truncation_cutoff)

        candidate_dataset = torch.utils.data.TensorDataset(candidates)

        for w in tqdm(torch.utils.data.DataLoader(candidate_dataset,
                                                  batch_size=batch_size,
                                                  num_workers=0,
                                                  ),
                      desc='Find initial style vector w'):
            imgs = generator.synthesis(w[0],
                                       noise_mode='const',
                                       force_fp32=True)
            # Adjust images and perform augmentation
            if clip:
                lower_bound = torch.tensor(-1.0).float().to(imgs.device)
                upper_bound = torch.tensor(1.0).float().to(imgs.device)
                imgs = torch.where(imgs > upper_bound, upper_bound, imgs)
                imgs = torch.where(imgs < lower_bound, lower_bound, imgs)
            if center_crop is not None:
                imgs = F.center_crop(imgs, (center_crop, center_crop))
            if resize is not None:
                imgs = [F.resize(imgs, resize, antialias=True)]
            if horizontal_flip:
                imgs.append(F.hflip(imgs[0]))

            target_conf = None
            for im in imgs:
                if target_conf is not None:
                    target_conf += target_model(im)[-1].softmax(dim=1) / len(imgs)
                else:
                    target_conf = target_model(im)[-1].softmax(dim=1) / len(imgs)
            confidences.append(target_conf)

        confidences = torch.cat(confidences, dim=0)
        for target in targets:
            sorted_conf, sorted_idx = confidences[:, target].sort(descending=True)
            final_candidates.append(candidates[sorted_idx[0]].unsqueeze(0))
            final_confidences.append(sorted_conf[0].cpu().item())
            # Avoid identical candidates for the same target
            confidences[sorted_idx[0], target] = -1.0

    final_candidates = torch.cat(final_candidates, dim=0).to(device)
    final_confidences = [np.round(c, 2) for c in final_confidences]
    print(f'Found {final_candidates.shape[0]} initial style vectors.')

    return final_candidates


def get_deprocessor():
    # resize 112,112
    proc = []
    proc.append(T.Resize((112, 112)))
    proc.append(T.ToTensor())
    return T.Compose(proc)
