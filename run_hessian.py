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
import argparse
import torch.backends.cudnn as cudnn


def hessian_calcs(model, directory, data, cuda=False):
    # https://github.com/amirgholami/PyHessian/blob/master/Hessian_Tutorial.ipynb
    model.eval()

    hessian_comp = hessian(model, torch.nn.CrossEntropyLoss(), dataloader=data, cuda=cuda)
    trace = np.sum(np.abs(hessian_comp.diag()))
    density_eigen, density_weight = hessian_comp.density()
    get_esd_plot(density_eigen, density_weight, directory)

    return trace, density_eigen, density_weight


def hessian_stuff(model_directory, trained_model_directory):
    d_path = "./"

    setup_file = os.path.join('trained', model_directory, trained_model_directory, 'setup.txt')
    with open(setup_file, 'r') as f:
        setup = f.read()
    setup = setup.split('\n')
    dataset = setup[-2].split(' ')[-1]
    bsize = int(setup[7].split(' ')[-1])
    arch = setup[0].split(' ')[-1]
    norm = setup[1].split(': ')[-1]
    cfg_use = cfg_dict['cfg_10']
    conv_type = setup[2].split(' ')[-1]
    p_grouping = float(setup[3].split(' ')[-1])
    probe_layers = setup[4].split(' ')[-1]
    if probe_layers == 'True':
        probe_layers = True
    else:
        probe_layers = False
    skipinit = setup[5].split(' ')[-1]
    if skipinit == 'True':
        skipinit = True
    else:
        skipinit = False
    preact = setup[6].split(' ')[-1]
    if preact == 'True':
        preact = True
    else:
        preact = False

    if arch == 'ViT':
        transform = transforms.Compose(
            [transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
             transforms.Resize((224, 224))
             ])
        transform_test = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
             transforms.Resize((224, 224))
             ])
    else:
        transform = transforms.Compose(
            [transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
             ])
        transform_test = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
             ])

    if dataset == "CIFAR-10":
        n_classes = 10
        trainset = torchvision.datasets.CIFAR10(root=d_path + 'datasets/cifar10/', train=True, transform=transform)
        trainset, _ = torch.utils.data.random_split(trainset, [10000, len(trainset) - 10000])
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=bsize, shuffle=True)

    elif dataset == "CIFAR-100":
        n_classes = 100
        trainset = torchvision.datasets.CIFAR100(root=d_path + 'datasets/cifar100/', train=True, transform=transform)
        trainset, _ = torch.utils.data.random_split(trainset, [10000, len(trainset) - 10000])
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=bsize, shuffle=True)

    model = models.get_model(arch, cfg_use, conv_type=conv_type, norm_type=norm, p_grouping=p_grouping,
                             n_classes=n_classes, probe=probe_layers, skipinit=skipinit, preact=preact)
    model = torch.nn.DataParallel(model).cuda()

    if bsize > 32:
        end_prefix = '59.pth'
    else:
        end_prefix = '9.pth'

    for file in os.listdir(os.path.join('trained', model_directory, trained_model_directory)):
        if file.endswith(end_prefix):
            try:
                model.load_state_dict(torch.load(os.path.join('trained', model_directory, trained_model_directory,
                                                              file)))
                trace, density_eigen, density_weight = hessian_calcs(model, os.path.join('trained', model_directory,
                                                                                         trained_model_directory),
                                                                     trainloader, cuda=True)

                plt.clf()
                with open(os.path.join('trained', model_directory, trained_model_directory, 'hessian_stuff.txt'),
                          'w') as f:
                    f.write(str(trace) + '\n')
                    f.write(str(density_eigen) + '\n')
                    f.write(str(density_weight) + '\n')
            except:
                print(sys.exc_info())
                print(traceback.format_exc())

            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir")
    parser.add_argument("--trained_model_dir")

    args = parser.parse_args()
    model_dir = args.model_dir
    trained_model_dir = args.trained_model_dir

    hessian_stuff(model_dir, trained_model_dir)