'''
- perhaps consider looking at distribution of class similarity rather than just the average
'''

import torch
from itertools import combinations
from pyhessian import hessian
from pyhessian.utils import *
from density_plot import get_esd_plot
import sklearn
import os
import torchvision
import torchvision.transforms as transforms
import models
import matplotlib.pyplot as plt
from config import *
import sys
import traceback

features = {}

def absolute_cosine_similarity(v1, v2):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    v1 = torch.tensor(v1).to(device)
    v2 = torch.tensor(v2).to(device)

    cos_sim_func = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    cos_sim_gpu = cos_sim_func(v1, v2)
    cos_sim = cos_sim_gpu.detach().cpu().numpy()

    del v1, v2, cos_sim_gpu

    return cos_sim


def compute_similarity(model, data_loader, num_true_classes):
    # info on hooks: https://kozodoi.me/python/deep%20learning/pytorch/tutorial/2021/05/27/extracting-features.html

    if model.module.__class__.__name__ == 'VisionTransformer':
        model.module.encoder.register_forward_hook(get_features('feats'))
    elif model.module.__class__.__name__ == 'ResNet_cifar':
        # hardcoded layer3, may be different for other resnet configs with different number of layers
        model.module.layer3.register_forward_hook(get_features('feats'))
    elif model.module.__class__.__name__ == 'VGG':
        model.module.features.register_forward_hook(get_features('feats'))

    model.eval()

    Y = []
    FEATS = []
    for inputs, targets in data_loader:
        Y.extend(targets.numpy())
        inputs, targets = inputs.cuda(non_blocking=True), targets.cuda(non_blocking=True)
        outputs = model(inputs)
        FEATS.extend(features['feats'].cpu().numpy())
        print(len(Y), len(FEATS))

    mapping = []
    for index, item in enumerate(FEATS):
        mapping.append([Y[index], item])

    mapping = np.asarray(mapping, dtype=object)

    num_sample_classes = np.max(mapping.T[0])

    class_container = [[] for i in range(num_true_classes)]

    for index, item in enumerate(mapping):
        class_container[item[0]].append(item[1])

    class_averages = []
    for c, classes in enumerate(class_container):
        if classes:
            av = np.average(classes, axis=0)
            if np.isnan(np.sum(av)):
                print(classes)
                print(av)
                print(c)
                print()

            class_averages.append([c, av])

    class_averages = np.asarray(class_averages, dtype=object)

    class_combinations = combinations(class_averages.T[1], 2)

    counter = 1
    current_class = 0
    cosine_similarity_list = []
    for combo in list(class_combinations):
        cosine_similarity_list.append(absolute_cosine_similarity(combo[0].reshape(1, -1), combo[1].reshape(1, -1)))
        if num_sample_classes - current_class - 1 == counter:
            current_class += 1
            counter = 1
        else:
            counter += 1

    return np.average(cosine_similarity_list)


def get_features(name):
    def hook(model, input, output):
        features[name] = output.detach()
    return hook