import torch
import os
import csv
from PIL import Image
from metrics.classification_acc import ClassificationAccuracy
from metrics.distance_metrics import DistanceEvaluation
import torchvision.utils as vutils
from utils.utils import write_list


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def evaluate_results(evaluation_model, G, stylegan, batch_size, idx_to_class, final_z, final_targets, trainset, targets_single_id, id_save_dir, save_dir, seed=42):

    target_id = targets_single_id[0]

    ####################################
    #         Attack Accuracy          #
    ####################################

    class_acc_evaluator = ClassificationAccuracy(
        evaluation_model, stylegan, device=device)

    acc_top1, acc_top5, predictions, avg_correct_conf, avg_total_conf, target_confidences, maximum_confidences, precision_list = class_acc_evaluator.compute_acc(
        final_z, final_targets, G, batch_size=batch_size * 2)

    _ = write_list(
        f'{save_dir}/precision_list',
        precision_list
    )

    print(
        f'\nEvaluation of {final_z.shape[0]} images on FaceNet: \taccuracy@1={acc_top1:4f}',
        f', accuracy@5={acc_top5:4f}, correct_confidence={avg_correct_conf:4f}, total_confidence={avg_total_conf:4f}'
    )

    ####################################
    #         Feature Distance         #
    ####################################
    avg_dist_facenet = None

    # Compute average feature distance on FaceNet
    evaluate_facenet = DistanceEvaluation(
        evaluation_model, G, stylegan, trainset, seed)
    avg_dist_facenet, mean_distances_list = evaluate_facenet.compute_dist(
        final_z, final_targets, batch_size=batch_size * 5)

    _ = write_list(
        f'{save_dir}/distance_facenet',
        mean_distances_list)

    print(f'Mean Distance on FaceNet: {avg_dist_facenet.cpu().item():4f}')

    ####################################
    #          Finish Logging          #
    ####################################

    # Logging of final results
    print('Finishing attack, logging results and creating sample images.')
    num_classes = len(set(targets_single_id.tolist()))
    num_imgs = 10
    # Sample final images from the first and last classes
    label_subset = set(list(set(targets_single_id.tolist()))[
                       :int(num_classes / 2)] + list(set(targets_single_id.tolist()))[-int(num_classes / 2):])
    log_imgs = []
    log_targets = []
    log_predictions = []
    log_max_confidences = []
    log_target_confidences = []
    # Log images with smallest feature distance
    for label in label_subset:
        mask = torch.where(final_targets == label, True, False).to(final_z.device)
        z_masked = final_z[mask][:num_imgs].to(device)
        try:
            if stylegan:
                imgs = G(z_masked, noise_mode='const', force_fp32=True)
                min_val = imgs.min().item()
                max_val = imgs.max().item()
                imgs = (imgs - min_val) / (max_val - min_val)
            else:
                imgs = G(z_masked)
        except:
            # cGAN
            current_targets = torch.full((len(z_masked),), label, dtype=torch.long, device=device)
            imgs = G(z_masked, current_targets)

        log_imgs.append(imgs)
        log_targets += [label for i in range(num_imgs)]
        log_predictions.append(torch.tensor(predictions).to(device)[mask][:num_imgs])
        log_max_confidences.append(
            torch.tensor(maximum_confidences).to(device)[mask][:num_imgs])
        log_target_confidences.append(
            torch.tensor(target_confidences).to(device)[mask][:num_imgs])

    log_imgs = torch.cat(log_imgs, dim=0)
    log_predictions = torch.cat(log_predictions, dim=0)
    log_max_confidences = torch.cat(log_max_confidences, dim=0)
    log_target_confidences = torch.cat(log_target_confidences, dim=0)

    log_final_images_local(target_id, 
                           log_imgs, 
                           log_predictions, 
                           log_max_confidences,
                           log_target_confidences, 
                           idx_to_class, 
                           id_save_dir, nrow=num_imgs)

    log_nearest_neighbors_local(target_id,
                                log_imgs,
                                log_targets,
                                evaluate_facenet,
                                save_dir=id_save_dir,
                                nrow=num_imgs)


def log_nearest_neighbors_local(target_id, imgs, targets, evaluater,
                                save_dir, nrow):
    # Find closest training samples to final results
    closest_samples, distances = evaluater.find_closest_training_sample(
        imgs, targets)

    grid = vutils.make_grid(closest_samples, nrow=nrow, padding=2, normalize=True)
    img = Image.fromarray(grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy())
    img.save(os.path.join(save_dir, f'nearest_neighbor.png'))

    for i, (img, d) in enumerate(zip(closest_samples, distances)):
        img = img.permute(1, 2, 0).cpu().numpy()
        img = Image.fromarray((img * 255).astype('uint8'))
        img.save(os.path.join(save_dir, f'{i:02d}_target={target_id:03d}_distance_{d:.2f}.png'))

    return


def log_final_images_local(target_id, imgs, predictions, max_confidences, target_confidences,
                           idx2cls, save_dir, nrow):
    grid = vutils.make_grid(imgs, nrow=nrow, padding=2, normalize=True)
    img = Image.fromarray(grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy())
    img.save(os.path.join(save_dir, f'final_images.png'))

    for i, (img, pred, max_conf, target_conf) in enumerate(
            zip(imgs.cpu(), predictions, max_confidences, target_confidences)):
        img = img.permute(1, 2, 0).detach().numpy()
        img = Image.fromarray((img * 255).astype('uint8'))
        img.save(os.path.join(save_dir,
                              f'{i:02d}_target={target_id:03d} ({target_conf:.2f})_pred={idx2cls[pred.item()]:03d} ({max_conf:.2f}).png'))

    return
