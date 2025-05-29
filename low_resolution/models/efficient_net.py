# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torchvision.models
import torch.nn.functional as F


class EfficientNet_b0(nn.Module):
    def __init__(self, n_classes):
        super(EfficientNet_b0, self).__init__()
        model = torchvision.models.efficientnet.efficientnet_b0(pretrained=True)
        self.feature = nn.Sequential(*list(model.children())[:-1])
        self.n_classes = n_classes
        self.feat_dim = 1280
        self.fc_layer = nn.Linear(self.feat_dim, self.n_classes)

    def forward(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        res = self.fc_layer(feature)
        return feature, res

    def predict(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        res = self.fc_layer(feature)
        out = F.softmax(res, dim=1)

        return feature, out


class EfficientNet_b1(nn.Module):
    def __init__(self, n_classes):
        super(EfficientNet_b1, self).__init__()
        model = torchvision.models.efficientnet.efficientnet_b1(pretrained=True)
        self.feature = nn.Sequential(*list(model.children())[:-1])
        self.n_classes = n_classes
        self.feat_dim = 1280
        self.fc_layer = nn.Linear(self.feat_dim, self.n_classes)

    def forward(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        res = self.fc_layer(feature)
        return feature, res

    def predict(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        res = self.fc_layer(feature)
        out = F.softmax(res, dim=1)

        return feature, out


class EfficientNet_b2(nn.Module):
    def __init__(self, n_classes):
        super(EfficientNet_b2, self).__init__()
        model = torchvision.models.efficientnet.efficientnet_b2(pretrained=True)
        self.feature = nn.Sequential(*list(model.children())[:-1])
        self.n_classes = n_classes
        self.feat_dim = 1408
        self.fc_layer = nn.Linear(self.feat_dim, self.n_classes)

    def forward(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        res = self.fc_layer(feature)
        return feature, res

    def predict(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        res = self.fc_layer(feature)
        out = F.softmax(res, dim=1)

        return feature, out


class EfficientNet_v2_s2(nn.Module):
    def __init__(self, n_classes, dataset='celeba'):
        super(EfficientNet_v2_s2, self).__init__()
        model = torchvision.models.efficientnet.efficientnet_v2_s(pretrained=True)
        self.feature = nn.Sequential(*list(model.children())[:-1])
        self.n_classes = n_classes
        self.feat_dim = 1028
        self.fc_layer = nn.Linear(self.feat_dim, self.n_classes)

    def forward(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        res = self.fc_layer(feature)
        return feature, res

    def predict(self, x):
        feature = self.feature(x)

        feature = feature.view(feature.size(0), -1)
        res = self.fc_layer(feature)
        out = F.softmax(res, dim=1)

        return feature, out


class EfficientNet_v2_m2(nn.Module):
    def __init__(self, n_classes, dataset='celeba'):
        super(EfficientNet_v2_m2, self).__init__()
        model = torchvision.models.efficientnet.efficientnet_v2_m(pretrained=True)

        self.feature = nn.Sequential(*list(model.children())[:-1])
        self.n_classes = n_classes
        self.feat_dim = 1028
        self.fc_layer = nn.Linear(self.feat_dim, self.n_classes)

    def forward(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        res = self.fc_layer(feature)
        return feature, res

    def predict(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        res = self.fc_layer(feature)
        out = F.softmax(res, dim=1)

        return feature, out


class EfficientNet_v2_l2(nn.Module):
    def __init__(self, n_classes, dataset='celeba'):
        super(EfficientNet_v2_l2, self).__init__()
        model = torchvision.models.efficientnet.efficientnet_v2_l(pretrained=True)
        self.feature = nn.Sequential(*list(model.children())[:-1])
        self.n_classes = n_classes
        self.feat_dim = 1028
        self.fc_layer = nn.Linear(self.feat_dim, self.n_classes)

    def forward(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        res = self.fc_layer(feature)
        return feature, res

    def predict(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        res = self.fc_layer(feature)
        out = F.softmax(res, dim=1)

        return feature, out


class EfficientNet_v2_s(nn.Module):
    def __init__(self, n_classes, dataset='celeba'):
        super(EfficientNet_v2_s, self).__init__()
        model = torchvision.models.efficientnet.efficientnet_v2_s(pretrained=True)
        self.feature = nn.Sequential(*list(model.children())[:-2])
        self.n_classes = n_classes
        self.feat_dim = 5120
        self.fc_layer = nn.Linear(self.feat_dim, self.n_classes)

    def forward(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        res = self.fc_layer(feature)
        return feature, res

    def predict(self, x):
        feature = self.feature(x)

        feature = feature.view(feature.size(0), -1)
        res = self.fc_layer(feature)
        out = F.softmax(res, dim=1)

        return feature, out


class EfficientNet_v2_m(nn.Module):
    def __init__(self, n_classes, dataset='celeba'):
        super(EfficientNet_v2_m, self).__init__()
        model = torchvision.models.efficientnet.efficientnet_v2_m(pretrained=True)

        self.feature = nn.Sequential(*list(model.children())[:-2])
        self.n_classes = n_classes
        self.feat_dim = 5120
        self.fc_layer = nn.Linear(self.feat_dim, self.n_classes)

    def forward(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        res = self.fc_layer(feature)
        return feature, res

    def predict(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        res = self.fc_layer(feature)
        out = F.softmax(res, dim=1)

        return feature, out


class EfficientNet_v2_l(nn.Module):
    def __init__(self, n_classes, dataset='celeba'):
        super(EfficientNet_v2_l, self).__init__()
        model = torchvision.models.efficientnet.efficientnet_v2_l(pretrained=True)
        self.feature = nn.Sequential(*list(model.children())[:-2])
        self.n_classes = n_classes
        self.feat_dim = 5120
        self.fc_layer = nn.Linear(self.feat_dim, self.n_classes)

    def forward(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        res = self.fc_layer(feature)
        return feature, res

    def predict(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        res = self.fc_layer(feature)
        out = F.softmax(res, dim=1)

        return feature, out