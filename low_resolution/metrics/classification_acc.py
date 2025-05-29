import math

import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset
from metrics.accuracy import Accuracy, AccuracyTopK
import torchvision.transforms as T


class ClassificationAccuracy():
    def __init__(self, evaluation_network, device='cuda:0'):
        self.evaluation_network = evaluation_network
        self.device = device

    def compute_acc(self, z, targets, generator, batch_size=64, rtpt=None):
        # self.evaluation_network.eval()
        self.evaluation_network.to(self.device)
        dataset = TensorDataset(z, targets)
        acc_top1 = Accuracy()
        acc_top5 = AccuracyTopK(k=5)
        predictions = []
        top5_predictions = []
        correct_confidences = []
        total_confidences = []
        maximum_confidences = []

        max_iter = math.ceil(len(dataset) / batch_size)

        with torch.no_grad():
            for step, (z_batch, target_batch) in enumerate(
                    DataLoader(dataset, batch_size=batch_size, shuffle=False)):
                z_batch, target_batch = z_batch.to(
                    self.device), target_batch.to(self.device)
                try:
                    imgs = generator(z_batch)
                except:
                    imgs = generator(z_batch, target_batch)
                imgs = imgs.to(self.device)
                # _, output = self.evaluation_network(self.low2high(imgs))[-1]

                output = self.evaluation_network(self.low2high(imgs))[-1]

                acc_top1.update(output, target_batch)
                acc_top5.update(output, target_batch)

                pred = torch.argmax(output, dim=1)
                predictions.append(pred)

                _, top5_pred = output.topk(5, dim=1)
                top5_predictions.append(top5_pred)

                confidences = output.softmax(1)
                target_confidences = torch.gather(confidences, 1,
                                                  target_batch.unsqueeze(1))
                correct_confidences.append(
                    target_confidences[pred == target_batch])
                total_confidences.append(target_confidences)
                maximum_confidences.append(torch.max(confidences, dim=1)[0])

            acc_top1 = acc_top1.compute_metric()
            acc_top5 = acc_top5.compute_metric()
            correct_confidences = torch.cat(correct_confidences, dim=0)
            avg_correct_conf = correct_confidences.mean().cpu().item()
            confidences = torch.cat(total_confidences, dim=0).cpu()
            confidences = torch.flatten(confidences)
            maximum_confidences = torch.cat(maximum_confidences,
                                            dim=0).cpu().tolist()
            avg_total_conf = torch.cat(total_confidences,
                                       dim=0).mean().cpu().item()

            predictions = torch.cat(predictions, dim=0).cpu()
            top5_predictions = torch.cat(top5_predictions, dim=0)  # 合并所有批次的 top-5 预测

            # Compute class-wise precision
            target_list = targets.cpu().tolist()
            precision_list = [['target', 'mean_conf', 'precision', 'precision5']]
            for t in set(target_list):
                mask = torch.where(targets == t, True, False).cpu()
                conf_masked = confidences[mask]
                precision = torch.sum(predictions[mask] == t) / torch.sum(targets == t)

                top5_correct = torch.sum(top5_predictions[mask] == t, dim=1) > 0  #
                top5_precision = torch.sum(top5_correct).float() / torch.sum(targets == t)  # 计算 top-5 precision

                precision_list.append(
                    [t, conf_masked.mean().item(),
                     precision.cpu().item(), top5_precision.cpu().item()])
            confidences = confidences.tolist()
            predictions = predictions.tolist()

        if rtpt:
            rtpt.step(
                subtitle=f'Classification Evaluation step {step} of {max_iter}')

        return acc_top1, acc_top5, predictions, avg_correct_conf, avg_total_conf, \
            confidences, maximum_confidences, precision_list

    def get_deprocessor(self):
        # resize 112,112
        proc = []
        proc.append(T.Resize((112, 112)))
        proc.append(T.ToTensor())
        return T.Compose(proc)


    def low2high(self, img):
        # 0 and 1, 64 to 112
        bs = img.size(0)
        proc = self.get_deprocessor()
        img_tensor = img.detach().cpu().float()
        img = torch.zeros(bs, 3, 112, 112)
        for i in range(bs):
            img_i = T.ToPILImage()(img_tensor[i, :, :, :]).convert('RGB')
            img_i = proc(img_i)
            img[i, :, :, :] = img_i[:, :, :]

        img = img.cuda()
        return img