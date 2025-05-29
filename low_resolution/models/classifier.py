import os, sys
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from metrics.accuracy import Accuracy
from torch.utils.data import DataLoader

from models.base_model import BaseModel
from models.efficient_net import EfficientNet_b0, EfficientNet_b1, EfficientNet_b2
from models.vgg import VGG16
from models.resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from models.ViT import ViT
from models.facenet import FaceNet, FaceNet64, IR152

import utils.adv_attack as attack
from utils.logger import CSVLogger, plot_csv, save_grid_with_cmap
from tqdm import tqdm


class Classifier(BaseModel):
    def __init__(self,
                 num_classes,
                 in_channels=3,
                 architecture='VGG16',
                 pretrained=True,
                 resume_path=None,
                 name='Classifier',
                 *args,
                 **kwargs):
        super().__init__(name, *args, **kwargs)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.pretrained = pretrained
        self.resume_path = resume_path
        self.model = self._build_model(architecture)
        self.model.to(self.device)
        self.architecture = architecture

        self.to(self.device)

    def get_representation(self, x):
        return self.model.embedding(x)

    def _build_model(self, architecture):
        architecture = architecture.lower().replace('-', '').strip()
        if 'vgg16' in architecture:
            model = VGG16(n_classes=self.num_classes, pretrained=self.pretrained)

        elif 'facenet64' in architecture:
            model = FaceNet64(num_classes=self.num_classes)

        elif 'facenet' in architecture:
            model = FaceNet(num_classes=self.num_classes)

        elif 'ir152' in architecture:
            model = IR152(num_classes=self.num_classes)
            
        elif 'resnet' in architecture:
            if architecture == 'resnet18':
                model = ResNet18(num_classes=self.num_classes)
            elif architecture == 'resnet34':
                model = ResNet34(num_classes=self.num_classes)
            elif architecture == 'resnet50':
                model = ResNet50(num_classes=self.num_classes)
            elif architecture == 'resnet101':
                model = ResNet101(num_classes=self.num_classes)
            elif architecture == 'resnet152':
                model = ResNet152(num_classes=self.num_classes)

        elif 'vit' in architecture:
            model = ViT(num_classes=self.num_classes)
        
        elif 'efficientnet_b0' in architecture:
            model = EfficientNet_b0(n_classes=self.num_classes)
        elif 'efficientnet_b1' in architecture:
            model = EfficientNet_b1(n_classes=self.num_classes)
        elif 'efficientnet_b2' in architecture:
            model = EfficientNet_b2(n_classes=self.num_classes)

        else:
            raise ValueError(f"Unsupported architecture: {architecture}")

        model = model.to(self.device).eval()

        # Load pretrained resume_path if provided
        if self.resume_path is not None:
            # Load checkpoint
            try:
                checkpoint = torch.load(self.resume_path, map_location=self.device)
                state_dict = checkpoint.get('state_dict', checkpoint)

                # Strip unwanted prefixes from keys
                cleaned = {}
                for key, value in state_dict.items():
                    if key.startswith('model.'):
                        new_key = key[len('model.'):]
                    elif key.startswith('module.'):
                        new_key = key[len('module.'):]
                    else:
                        new_key = key
                    cleaned[new_key] = value

                # Load into model (strict=True requires exact match)
                info = model.load_state_dict(cleaned, strict=True)
                print(info)
            except:
                print("Bugs found in resuming models")
                exit()
            
        return model

    def forward(self, x):
        if type(x) is np.ndarray:
            x = torch.tensor(x, dtype=torch.float).to(self.device)
        
        feat, out = self.model(x)
        return feat, out

    def fit(self,
            training_data,
            validation_data,
            test_data,
            criterion=nn.CrossEntropyLoss(),
            metric=Accuracy,
            optimizer=None,
            lr_scheduler=None,
            batch_size=64,
            num_epochs=100,
            dataloader_num_workers=8,
            save_base_path="./results"):

        """
        Train and evaluate the model.

        Args:
            training_data:    PyTorch Dataset for training.
            validation_data:  PyTorch Dataset for validation (can be None to skip).
            test_data:        PyTorch Dataset for final testing (can be None to skip).
            criterion:        Loss function.
            metric:           Metric class (must implement update() and compute_metric()).
            optimizer:        Torch optimizer (already constructed).
            lr_scheduler:     Learning‐rate scheduler (optional).
            batch_size:       Batch size for all loaders.
            num_epochs:       Total number of epochs to run.
            dataloader_num_workers: Number of workers for data loading.
            save_base_path:   Directory in which to write logs and checkpoints.

        Returns:
            None (saves best model to disk and prints final metrics).
        """
        # --- prepare directories & logging -----------------------------------
        epoch_logger = CSVLogger(
            every=1,
            fieldnames=["global_iteration", "train_acc", "val_acc"],
            filename=os.path.join(save_base_path, "epoch_log.csv"),
            resume=0
        )

        # --- data loaders ----------------------------------------------------
        train_loader = DataLoader(
            training_data, batch_size=batch_size, shuffle=True,
            num_workers=dataloader_num_workers, pin_memory=True, drop_last=True
        )
        val_loader = (
            DataLoader(validation_data, batch_size=batch_size, shuffle=False,
                       num_workers=dataloader_num_workers, pin_memory=True, drop_last=False)
            if validation_data is not None else None
        )
        test_loader = (
            DataLoader(test_data, batch_size=batch_size, shuffle=False,
                       num_workers=dataloader_num_workers, pin_memory=True, drop_last=False)
            if test_data is not None else None
        )

        # Training cycle
        best = {
            "epoch": -1,
            "val_acc": 0.0,
            "val_loss": float("inf"),
            "state_dict": None,
            "optim_state": None
        }

         # --- main training loop ----------------------------------------------
        for epoch in range(1, num_epochs + 1):
            print(f"\nEpoch {epoch}/{num_epochs}")
            # 1) TRAIN
            self.train()
            train_metric = metric()
            total_loss = 0.0

            for inputs, labels in train_loader:
                inputs = inputs.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True).view(-1)

                optimizer.zero_grad()
                _, logits = self.forward(inputs)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * inputs.size(0)
                train_metric.update(logits, labels)

            train_acc = train_metric.compute_metric()
            avg_train_loss = total_loss / len(training_data)
            print(f"Train loss: {avg_train_loss:.4f}, Train {train_metric.name}: {train_acc:.2%}")

            # 2) VALIDATION
            val_acc, val_loss = 0.0, 0.0
            if val_loader:
                self.eval()
                val_metric = metric()
                total_val_loss = 0.0

                with torch.no_grad():
                    for inputs, labels in val_loader:
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device).view(-1)
                        _, logits = self.forward(inputs)
                        loss = criterion(logits, labels)
                        total_val_loss += loss.item() * inputs.size(0)
                        val_metric.update(logits, labels)

                val_acc = val_metric.compute_metric()
                val_loss = total_val_loss / len(validation_data)
                print(f"Val   loss: {val_loss:.4f}, Val   {val_metric.name}: {val_acc:.2%}")

            # 3) LOG & CHECKPOINT
            epoch_logger.writerow({
                "global_iteration": epoch,
                "train_acc": train_acc,
                "val_acc": val_acc
            })
            plot_csv(epoch_logger.filename,
                     os.path.join(save_base_path, "epoch_plot.png"))

            if val_loader and val_acc > best["val_acc"]:
                print(" ** New best model! Saving checkpoint.")
                best.update({
                    "epoch": epoch,
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                    "state_dict": deepcopy(self.state_dict()),
                    "optim_state": deepcopy(optimizer.state_dict())
                })
                torch.save({
                    "epoch": epoch,
                    "state_dict": best["state_dict"],
                    "optimizer": best["optim_state"]
                }, os.path.join(
                    save_base_path,
                    f"{self.architecture}_best.pth"
                ))

            if lr_scheduler:
                lr_scheduler.step()

        # --- FINAL TEST ------------------------------------------------------
        if best["state_dict"] is not None:
            self.load_state_dict(best["state_dict"])
        if test_loader:
            self.eval()
            test_metric = metric()
            total_test_loss = 0.0

            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device).view(-1)
                    _, logits = self.forward(inputs)
                    loss = criterion(logits, labels)
                    total_test_loss += loss.item() * inputs.size(0)
                    test_metric.update(logits, labels)

            test_acc = test_metric.compute_metric()
            test_loss = total_test_loss / len(test_data)
            print(f"\nTest  loss: {test_loss:.4f}, Test {test_metric.name}: {test_acc:.2%}")

        print("\nTraining finished.")

    def align_fit(
        self,
        training_data,
        validation_data,
        criterion: nn.Module = nn.CrossEntropyLoss(),
        metric=Accuracy,
        optimizer=None,
        lr_scheduler=None,
        batch_size=64,
        num_epochs=100,
        dataloader_num_workers=8,
        save_base_path="",
        save_every_epoch=1,
        align_coeff=0.5
    ):
        """
        Perform one-stage alignment training:
        - CE loss + geometry-alignment loss ("geo_loss")
        - every `save_every_epoch` epochs, dump images+grads and the model state_dict

        Args:
        training_data, validation_data: PyTorch Datasets
        criterion: classification loss
        metric: accuracy metric class
        optimizer, lr_scheduler: PyTorch optimizer & scheduler
        batch_size, num_epochs, dataloader_num_workers
        save_base_path: where to save visuals & model checkpoints
        save_every_epoch: how often (in epochs) to save
        """
        # data loaders
        train_loader = DataLoader(
            training_data, batch_size=batch_size, shuffle=True,
            num_workers=dataloader_num_workers, pin_memory=True, drop_last=True
        )
        val_loader = DataLoader(
            validation_data, batch_size=batch_size, shuffle=True,
            num_workers=dataloader_num_workers, pin_memory=True, drop_last=False
        )

        # metric tracker
        metric_train = metric()

        os.makedirs(save_base_path, exist_ok=True)
        n_vis = 5  # how many of the batch to visualize

        print("=== START ALIGNMENT TRAINING ===")
        for epoch in range(1, num_epochs + 1):
            metric_train.reset()
            self.train()

            running_ce = 0.0
            running_geo = 0.0
            running_total = 0.0

            for batch_idx, (data_batch, y) in enumerate(train_loader):
                # unpack
                x = data_batch['image'].to(self.device)            # [B,C,H,W]
                y = y.to(self.device)                              # [B]
                U = data_batch['tangent_basis'].to(self.device)    # [B, k, D]

                B, C, H, W = x.shape
                D = C*H*W

                # allow gradient w.r.t. x
                x = x.requires_grad_(True)

                # ---- CE LOSS ----
                _, logits = self.forward(x)
                loss_ce = criterion(logits, y)

                # ---- GEO LOSS (geometry alignment) ----
                # compute sum_i f_i(x), then its gradient wrt x
                scalar_sum = logits.sum()
                grads = torch.autograd.grad(
                    scalar_sum, x, create_graph=True
                )[0]                                   # [B,C,H,W]
                grads_flat = grads.view(B, D)          # [B, D]

                # project onto tangent basis U: g_proj = (U·g)·Uᵀ
                # U: [B, k, D], grads_flat: [B, D]
                coeff = (U * grads_flat.unsqueeze(1)).sum(dim=2)   # [B, k]
                g_proj = torch.einsum('bkd,bk->bd', U, coeff)     # [B, D]

                nom = g_proj.norm(dim=1)      # [B]
                den = grads_flat.norm(dim=1)       # [B]
                geo_loss = (nom / (den + 1e-8)).mean()

                # ---- TOTAL LOSS ----
                loss = loss_ce - align_coeff * geo_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_ce   += loss_ce.item() * B
                running_geo  += geo_loss.item() * B
                running_total+= loss.item()  * B
                metric_train.update(logits, y)

                # capture first batch for visualization
                if batch_idx == 0 and (epoch % save_every_epoch == 0):
                    # 1) original images
                    save_grid_with_cmap(
                        x[:n_vis].detach().cpu(),
                        os.path.join(save_base_path, f"imgs_epoch{epoch}.png"),
                        handle_imgs=True, nrow=n_vis
                    )
                    # 2) geometry‐aligned grads
                    save_grid_with_cmap(
                        grads[:n_vis].detach().cpu(),
                        os.path.join(save_base_path, f"grads_epoch{epoch}.png"),
                        cmap="seismic", handle_imgs=False, nrow=n_vis
                    )

            train_acc = metric_train.compute_metric()
            
            if val_loader:
                self.eval()
                val_metric, _, _ = self.evaluate(
                                        evaluation_data=validation_data,
                                        batch_size=batch_size,
                                        metric=metric,
                                        criterion=criterion,
                                        dataloader_num_workers=dataloader_num_workers)

            print(f"Epoch {epoch}: "
            f"Train Loss: {running_total/len(training_data):.4f} "
            f"(CE: {running_ce/len(training_data):.4f} "
            f"GEO: {running_geo/len(training_data):.4f}) "
            f"Train Acc: {train_acc:.2%} "
            f"Val Acc: {val_metric:.2%}")

            # ---- SAVE MODEL ----
            if epoch % save_every_epoch == 0:
                ckpt_path = os.path.join(save_base_path, f"model_epoch{epoch}.pth")
                torch.save(self.state_dict(), ckpt_path)
                # print(f"--> Saved checkpoint: {ckpt_path}")

            # LR step
            if lr_scheduler is not None:
                lr_scheduler.step()

        print("=== FINISHED ALIGNMENT TRAINING ===")

    def adv_fit(self,
                training_data,
                validation_data,
                test_data,
                optimizer=None,
                lr_scheduler=None,
                criterion=nn.CrossEntropyLoss(),
                metric=Accuracy,
                rtpt=None,
                adv_cfg=None,
                batch_size=64,
                num_epochs=30,
                dataloader_num_workers=8,
                save_base_path=""):
        """
        Adversarial training (AT, TRADES, or MART).

        Args:
            training_data:      training Dataset
            validation_data:    validation Dataset (or None)
            test_data:          test Dataset (or None)
            optimizer:          torch Optimizer
            lr_scheduler:       LR scheduler (optional)
            criterion:          classification loss
            metric:             metric class
            adv_cfg:            dict with keys:
                                - 'train_method': "AT" | "TRADES" | "MART"
                                - 'eps', 'sts', 'num_steps', 'loss_fn', 'category', 'beta'
            rtpt:               Real-Time-Plot-Tracker (optional)
            batch_size:         per-GPU batch size
            num_epochs:         number of epochs
            dataloader_num_workers: DataLoader workers
            save_base_path:     where to write logs & checkpoints
            ls_scheduler:       another scheduler (optional)

        Returns:
            None
        """
        epoch_logger = CSVLogger(
            every=1,
            fieldnames=["global_iteration", "train_acc", "val_acc"],
            filename=os.path.join(save_base_path, "adv_epoch_log.csv"),
            resume=0
        )

        # DataLoaders
        train_loader = DataLoader(
            training_data, batch_size=batch_size, shuffle=True,
            num_workers=dataloader_num_workers, pin_memory=True, drop_last=True
        )
        val_loader = (
            DataLoader(validation_data, batch_size=batch_size, shuffle=False,
                       num_workers=dataloader_num_workers, pin_memory=True, drop_last=False)
            if validation_data is not None else None
        )
        test_loader = (
            DataLoader(test_data, batch_size=batch_size, shuffle=False,
                       num_workers=dataloader_num_workers, pin_memory=True, drop_last=False)
            if test_data is not None else None
        )

        best = {
            "epoch": -1,
            "val_acc": 0.0,
            "val_loss": float("inf"),
            "state_dict": None,
            "optim_state": None
        }
        
        print("\n--- START ADV TRAINING ---")
        for epoch in range(1, num_epochs + 1):
            print(f"\nEpoch {epoch}/{num_epochs}")
            self.train()
            train_metric = metric()
            total_loss = 0.0

            for inputs, labels in train_loader:
                inputs = inputs.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True).view(-1)

                # 1) Generate adversarial examples
                x_adv, _ = attack.PGD(
                    model=self.model,
                    data=inputs,
                    target=labels,
                    epsilon=adv_cfg["eps"],
                    step_size=adv_cfg["sts"],
                    num_steps=adv_cfg["num_steps"],
                    loss_fn=adv_cfg["loss_fn"],
                    category=adv_cfg["category"],
                    rand_init=True
                )

                optimizer.zero_grad()
                _, adv_logits = self.forward(x_adv)
                _, nat_logits = self.forward(inputs)

                # 2) Compute adversarial training loss
                if adv_cfg["train_method"] == "AT":
                    loss_main = criterion(adv_logits, labels)
                elif adv_cfg["train_method"] == "TRADES":
                    loss_main = attack.TRADES_loss(
                        adv_logits, nat_logits, labels, beta=adv_cfg.get("beta", 6.0)
                    )
                elif adv_cfg["train_method"] == "MART":
                    loss_main = attack.MART_loss(
                        adv_logits, nat_logits, labels, beta=adv_cfg.get("beta", 6.0)
                    )
                else:
                    raise ValueError(f"Unknown train_method: {adv_cfg['train_method']}")

                loss = loss_main
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * inputs.size(0)
                train_metric.update(nat_logits, labels)

            avg_train_loss = total_loss / len(training_data)
            train_acc = train_metric.compute_metric()
            print(f"Train loss: {avg_train_loss:.4f}, Train {train_metric.name}: {train_acc:.2%}")

            # 3) Validation
            val_acc, val_loss = 0.0, 0.0
            if val_loader:
                self.eval()
                val_metric = metric()
                total_val_loss = 0.0

                with torch.no_grad():
                    for inputs, labels in val_loader:
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device).view(-1)
                        _, logits = self.forward(inputs)
                        loss = criterion(logits, labels)
                        total_val_loss += loss.item() * inputs.size(0)
                        val_metric.update(logits, labels)

                val_loss = total_val_loss / len(validation_data)
                val_acc  = val_metric.compute_metric()
                print(f"Val   loss: {val_loss:.4f}, Val   {val_metric.name}: {val_acc:.2%}")

            # 4) Logging & checkpoint
            epoch_logger.writerow({
                "global_iteration": epoch,
                "train_acc": train_acc,
                "val_acc": val_acc
            })
            plot_csv(epoch_logger.filename,
                     os.path.join(save_base_path, "adv_epoch_plot.png"))

            if val_loader and val_acc > best["val_acc"]:
                print(" ** New best adversarial model, saving checkpoint.")
                best.update({
                    "epoch": epoch,
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                    "state_dict": deepcopy(self.state_dict()),
                    "optim_state": deepcopy(optimizer.state_dict())
                })
                torch.save({
                    "epoch": epoch,
                    "state_dict": best["state_dict"],
                    "optimizer": best["optim_state"]
                }, os.path.join(save_base_path, f"{self.architecture}_adv_best.pth"))

            if rtpt:
                rtpt.step(subtitle=f"loss={avg_train_loss:.4f}")

            if lr_scheduler:
                lr_scheduler.step()
                
        # 5) Final test
        if best["state_dict"] is not None:
            self.load_state_dict(best["state_dict"])
        if test_loader:
            self.eval()
            test_metric = metric()
            total_test_loss = 0.0

            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device).view(-1)
                    _, logits = self.forward(inputs)
                    loss = criterion(logits, labels)
                    total_test_loss += loss.item() * inputs.size(0)
                    test_metric.update(logits, labels)

            test_loss = total_test_loss / len(test_data)
            test_acc  = test_metric.compute_metric()
            print(f"\nTest  loss: {test_loss:.4f}, Test  {test_metric.name}: {test_acc:.2%}")

        self.evaluate_pgd(test_data)
        print("\n--- ADV TRAINING COMPLETE ---")
            
    def evaluate(self,
                 evaluation_data,
                 batch_size=64,
                 metric=Accuracy,
                 criterion=nn.CrossEntropyLoss(),
                 dataloader_num_workers=4
                 ):

        evalloader = torch.utils.data.DataLoader(evaluation_data,
                                                shuffle=True,
                                                batch_size=batch_size,
                                                num_workers=2,
                                                pin_memory=True,
                                                drop_last=True
                                                )
                    
        num_val_data = len(evaluation_data)
        metric = metric()
        self.eval()
        with torch.no_grad():
            running_id_conf = 0.0
            running_loss = torch.tensor(0.0, device=self.device)
            for i, (inputs, labels) in enumerate(evalloader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                _, model_output = self.forward(inputs)

                id_bs = len(inputs)
                id_conf = F.softmax(model_output, dim=1)
                mean_id_conf = torch.gather(id_conf, dim=1, index=labels.unsqueeze(1)).mean()

                running_id_conf += mean_id_conf * id_bs
                
                labels = labels.view(-1)
                
                metric.update(model_output, labels)
                running_loss += criterion(model_output,
                                          labels).cpu() * inputs.shape[0]

            metric_result = metric.compute_metric()

            return metric_result, running_loss.item() / num_val_data, running_id_conf

    def dry_evaluate(self,
                     evalloader,
                     metric=Accuracy,
                     criterion=nn.CrossEntropyLoss()):

        num_val_data = len(evalloader.dataset)
        metric = metric()
        self.eval()
        with torch.no_grad():
            running_id_conf = 0.0
            running_loss = torch.tensor(0.0, device=self.device)
            for inputs, labels in tqdm(evalloader,
                                       desc='Evaluating',
                                       leave=False,
                                       file=sys.stdout, ncols=50):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                model_output = self.forward(inputs)
                if isinstance(model_output, tuple):
                    model_output = model_output[-1]

                id_conf = F.softmax(model_output, dim=1)
                mean_id_conf = torch.gather(id_conf, dim=1, index=labels.unsqueeze(1)).mean()

                id_bs = len(inputs)
                running_id_conf += mean_id_conf * id_bs

                metric.update(model_output, labels)
                running_loss += criterion(model_output,
                                          labels).cpu() * inputs.shape[0]

            metric_result = metric.compute_metric()

            print(
                f'Validation {metric.name}: {metric_result:.2%}',
                f'\t Validation Loss:  {running_loss.item() / num_val_data:.4f}',
                f'\t id conf: {running_id_conf / num_val_data:.4f}',
            )

        return metric_result

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.eval()

    def unfreeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.train()

    def evaluate_pgd(self,
                 evaluation_data,
                 batch_size=128,
                 dataloader_num_workers=4):
        
        evalloader = DataLoader(evaluation_data,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=dataloader_num_workers,
                                pin_memory=True)
        
        loss, pgd5_acc = attack.eval_robust(self.model, evalloader, perturb_steps=5, epsilon=8/255, step_size=1/255,loss_fn="cent", category="trades", random=True)
        print('PGD5 Test Accuracy: {:.2f}%'.format(100. * pgd5_acc))

