import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import TensorDataset, DataLoader


def scores_by_transform(imgs,
                        targets,
                        target_model,
                        transforms,
                        iterations=100):
    score = torch.zeros_like(targets, dtype=torch.float32).to(imgs.device)

    with torch.no_grad():
        for i in range(iterations):
            imgs_transformed = transforms(imgs)
            output = target_model(imgs_transformed)
            if type(output) is tuple:
                output = target_model(imgs_transformed)[1]
            prediction_vector = output.softmax(dim=1)
            score += torch.gather(prediction_vector, 1,
                                  targets.unsqueeze(1)).squeeze()
        score = score / iterations
    return score


def perform_final_selection(z,
                            G,
                            targets,
                            target_model,
                            samples_per_target,
                            batch_size,
                            device,
                            rtpt=None):
    target_values = set(targets.cpu().tolist())
    final_targets = []
    final_z = []
    target_model.eval()

    transformation = T.Compose([
        T.RandomResizedCrop(size=(64, 64), scale=(0.8, 1.0), ratio=(1.0, 1.0)),
        T.ColorJitter(brightness=0.2, contrast=0.2),
        T.RandomHorizontalFlip(0.5),
        T.RandomRotation(5)
    ])

    for step, target in enumerate(target_values):
        mask = torch.where(targets == target, True, False).cpu()
        z_masked = z[mask]
        candidates = G(z_masked).cpu()
        targets_masked = targets[mask].cpu()
        scores = []
        dataset = TensorDataset(candidates, targets_masked)
        for imgs, t in DataLoader(dataset, batch_size=batch_size):
            imgs, t = imgs.to(device), t.to(device)

            scores.append(
                scores_by_transform(imgs, t, target_model, transformation))
        scores = torch.cat(scores, dim=0).cpu()
        indices = torch.sort(scores, descending=True).indices
        selected_indices = indices[:samples_per_target]
        final_targets.append(targets_masked[selected_indices].cpu())
        final_z.append(z_masked[selected_indices].cpu())

        if rtpt:
            rtpt.step(
                subtitle=f'Sample Selection step {step} of {len(target_values)}'
            )
    # print(scores[selected_indices])
    final_targets = torch.cat(final_targets, dim=0)
    final_z = torch.cat(final_z, dim=0)
    return final_z, final_targets