import torch
import torchvision.transforms as T
from datasets.celeba import CelebA1000
from datasets.custom_subset import SingleClassSubset
from datasets.facescrub import FaceScrub
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset
from torchvision.transforms.transforms import Resize


class DistanceEvaluation():
    def __init__(self, model, generator, dataset, seed):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.train_set = dataset
        self.model = model
        self.seed = seed
        self.generator = generator

    def compute_dist(self, z, targets, batch_size=64):
        self.model.eval()
        self.model.to(self.device)
        target_values = set(targets.cpu().tolist())
        smallest_distances = []
        mean_distances_list = [['target', 'mean_dist']]
        for step, target in enumerate(target_values):
            mask = torch.where(targets == target, True, False)
            z_masked = z[mask.cpu()]
            target_subset = SingleClassSubset(self.train_set,
                                              target_class=target)

            target_embeddings = []
            for x, y in DataLoader(target_subset, batch_size):
                with torch.no_grad():
                    if self.train_set.name == 'celeba':
                        x = self.low2high(x)
                    x = x.to(self.device)

                    outputs, _ = self.model(x)
                        
                    target_embeddings.append(outputs.cpu())

            attack_embeddings = []
            for z_batch in DataLoader(TensorDataset(z_masked), batch_size, shuffle=False):
                with torch.no_grad():
                    z_batch = z_batch[0].to(self.device)
                    current_targets = torch.full((batch_size,), target, dtype=torch.long, device=self.device)
                    try:
                        imgs = self.generator(z_batch)
                    except:
                        imgs = self.generator(z_batch, current_targets)
                    if self.train_set.name == 'celeba':
                        imgs = self.low2high(imgs)
                    imgs = imgs.to(self.device)
                    outputs, _ = self.model(imgs)

                    attack_embeddings.append(outputs.cpu())

            target_embeddings = torch.cat(target_embeddings, dim=0)
            attack_embeddings = torch.cat(attack_embeddings, dim=0)
            distances = torch.cdist(
                attack_embeddings, target_embeddings, p=2).cpu()
            distances = distances**2
            distances, _ = torch.min(distances, dim=1)
            smallest_distances.append(distances.cpu())
            mean_distances_list.append([target, distances.cpu().mean().item()])

        smallest_distances = torch.cat(smallest_distances, dim=0)
        return smallest_distances.mean(), mean_distances_list

    def find_closest_training_sample(self, imgs, targets, batch_size=100):
        self.model.eval()
        self.model.to(self.device)
        closest_imgs = []
        smallest_distances = []
        for img, target in zip(imgs, targets):
            img = img.to(self.device)
            if len(img) == 3:
                img = img.unsqueeze(0)
            
            if self.train_set.name == 'celeba':
                img = self.low2high(img)

            if torch.is_tensor(target):
                target = target.cpu().item()
            target_subset = SingleClassSubset(self.train_set,
                                              target_class=target)

            target_embeddings = []
            with torch.no_grad():
                # Compute embedding for generated image
                img = self.low2high(img)
                output_img, _ = self.model(img)
                output_img = output_img.cpu()

                # Compute embeddings for training samples from same class
                for x, y in DataLoader(target_subset, batch_size):
                    x = x.to(self.device)
                    if self.train_set.name == 'celeba':
                        x = self.low2high(x)
                    outputs, _ = self.model(x)
                    if isinstance(outputs, tuple) or isinstance(outputs, list):
                        outputs = outputs[0].cpu()
                    target_embeddings.append(outputs.cpu())

            # Compute squared L2 distance
            target_embeddings = torch.cat(target_embeddings, dim=0)
            distances = torch.cdist(output_img, target_embeddings, p=2)
            distances = distances ** 2
            # Take samples with smallest distances
            distance, idx = torch.min(distances, dim=1)
            smallest_distances.append(distance.item())
            closest_imgs.append(target_subset[idx.item()][0])

        return closest_imgs, smallest_distances

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