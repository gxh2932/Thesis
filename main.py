'''
todo:
- look at connection between flatness and learning rate sensitivity
'''

import os
import numpy as np
import torch
import sklearn.datasets
import matplotlib.pyplot as plt
import sklearn.model_selection
import pandas as pd
import torch.backends.cudnn as cudnn
import argparse
from config import *
import torchvision
from torchvision import transforms
import models
import copy
from utils import *
from datetime import datetime
import subprocess
import ast
import torch.multiprocessing


def get_gpu_memory_map():
    """Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory_map


def train(model, optimizer, loss_function, scheduler, train_loader, epoch, model_name=None, current_model_time=None):

    model.train()

    train_loss_list_epoch = []
    correct = 0
    total = 0
    train_accuracy = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):

        inputs, targets = inputs.cuda(non_blocking=True), targets.cuda(non_blocking=True)

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = loss_function(outputs, targets)
        with torch.no_grad():
            train_loss_list_epoch.append(loss.cpu().detach().numpy())

        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        train_accuracy = 100 * correct / total

        loss.backward()
        optimizer.step()
        scheduler.step()

    if rank == 0:
        torch.save(model.state_dict(), current_model_time + '/' + str(model_name) + '_' + str(epoch) + '.pth')

    return np.average(train_loss_list_epoch), train_accuracy


def test(model, test_loader, loss_function, validation, model_name=None):

    if not validation:
        model.load_state_dict(torch.load(model_name))

    model.eval()

    correct = 0
    total = 0
    loss = []
    with torch.no_grad():
        for data in test_loader:
            inputs, targets = data
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            loss.append(loss_function(outputs, targets).cpu().detach().numpy())

            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    accuracy = (100 * correct / total)

    if validation:
        print('Accuracy of the network on the validation samples:', accuracy, '(rank %d)' % rank)
    else:
        print('Accuracy of the network on the test samples:', accuracy)

    return np.average(loss), accuracy


def job_file_copy(norm, model_dir):
    norm_list = ['Plain', 'BatchNorm', 'LayerNorm', 'Instance Normalization', 'GroupNorm',
    'Filter Response Normalization', 'Weight Normalization',
    'Scaled Weight Standardization', 'EvoNormSO', 'EvoNormBO', 'Variance Normalization']

    norm_abbrevs = ['Plain', 'BN', 'LN', 'IN', 'GN', 'FRN', 'WN', 'SWS', 'ENSO', 'ENBO', 'VN']

    if norm is 'GroupNorm':
        p_groupings = ['1.0', '0.5', '0.25', '0.125', '0.0625', '0.03125', '1e-07', '8.0', '16.0', '32.0', '64.0']
        p_grouping_abbrev = ['1', '5', '25', '125', '625', '3125', '01', '8', '16', '32', '64']
        p_grouping_dict = dict(zip(p_groupings, p_grouping_abbrev))

        abbrev = norm_abbrevs[norm_list.index(norm)]
        dataset_num = use_data.split('-')[1]
        run = abbrev + '_' + p_grouping_dict[str(p_grouping)] + '_' + dataset_num
        job_dir = 'jobs/' + arch + '/' + run + '/'

    else:
        abbrev = norm_abbrevs[norm_list.index(norm)]
        dataset_num = use_data.split('-')[1]
        run = abbrev + '_' + dataset_num
        job_dir = 'jobs/' + arch + '/' + run + '/'

    # copy files in job directory to model directory
    os.system('cp -r ' + job_dir + ' ' + model_dir)


def main():
    base_lr = init_lr * base_sched[0]
    final_lr = init_lr * base_sched[-1]

    if lr_warmup:
        warmup_lr = 0
        warmup_epochs = 1 if bsize == 16 else 5
    else:
        warmup_lr = base_lr
        warmup_epochs = 0

    model = models.get_model(arch, cfg_use, conv_type=conv_type, norm_type=norm_type, p_grouping=p_grouping,
                             n_classes=n_classes, probe=probe_layers, skipinit=skipinit, preact=preact)
    model_name = arch + '_' + norm_type
    model_name = model_name.replace(' ', '_')

    if c_dir:
        current_model_time = c_dir

        file_list = os.listdir(c_dir)
        epoch_list = []
        for file in file_list:
            if file.endswith('.pth'):
                epoch_list.append(int(file.split('_')[-1].split('.')[0]))
        epoch_list.sort()
        last_epoch = epoch_list[-1] + 1
        if last_epoch < base_epochs[0]:
            lr_ind = 0
        else:
            lr_ind = 1
        epoch = last_epoch
        model.load_state_dict(torch.load(c_dir + '/' + model_name + '_' + str(epoch-1) + '.pth'))

    else:
        current_model_time = 'trained/' + str(model_name) + '/' + str(model_name) + '_' + datetime.utcnow().strftime(
            '%Y%m%d-%H:%M:%S.%f')[:-3]
        last_epoch = 0
        lr_ind = 0
        epoch = 0

    if rank == 0:
        if not os.path.exists('trained/' + str(model_name)):
            os.makedirs('trained/' + str(model_name))

        if not os.path.exists(current_model_time):
            os.makedirs(current_model_time)

            # write the setup to a file
            with open(current_model_time + '/setup.txt', 'w') as f:
                f.write('architecture: ' + arch + '\n')
                f.write('norm: ' + norm_type + '\n')
                f.write('conv type: ' + conv_type + '\n')
                f.write('p_grouping: ' + str(p_grouping) + '\n')
                f.write('probe: ' + str(probe_layers) + '\n')
                f.write('skipinit: ' + str(skipinit) + '\n')
                f.write('preact: ' + str(preact) + '\n')
                f.write('batch size: ' + str(bsize) + '\n')
                f.write('total epochs: ' + str(total_epochs) + '\n')
                f.write('initial learning rate: ' + str(base_lr) + '\n')
                f.write('final learning rate: ' + str(final_lr) + '\n')
                f.write('warmup epochs: ' + str(warmup_epochs) + '\n')
                f.write('warmup learning rate: ' + str(warmup_lr) + '\n')
                f.write('optimizer: ' + str(opt_type) + '\n')
                f.write('dataset: ' + str(use_data) + '\n')
                f.write('mom & wd: ' + str(mom + wd != 0))

            job_file_copy(norm_type, current_model_time)

    torch.cuda.set_device(rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier()

    torch.manual_seed(int(args.seed))
    cudnn.deterministic = True
    cudnn.benchmark = False

    model.to(rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank], output_device=rank,
                                                      find_unused_parameters=True)

    optimizer = torch.optim.SGD(model.parameters(), lr=warmup_lr, momentum=mom, weight_decay=wd)
    loss_function = torch.nn.CrossEntropyLoss()
    scheduler = LR_Scheduler(optimizer, warmup_epochs=warmup_epochs, warmup_lr=warmup_lr, num_epochs=total_epochs,
                             base_lr=base_lr, final_lr=final_lr, iter_per_epoch=len(trainloader))

    train_loss_list = []
    train_accuracy_list = []
    val_loss_list = []
    val_accuracy_list = []
    lr_list = []
    while lr_ind < len(base_sched):

        if lr_ind == 0:
            iters = base_epochs[0] - last_epoch
        else:
            if last_epoch != 0:
                iters = base_epochs[0] + base_epochs[1] - last_epoch
            else:
                iters = base_epochs[1]

        for n in range(iters):
            # note that learning rate here isnt the real learning rate since its updated with each epoch
            trainloader.sampler.set_epoch(epoch)
            train_loss_epoch, train_accuracy_epoch = train(model, optimizer,loss_function, scheduler,
                                                                       trainloader, epoch, model_name, current_model_time)

            with torch.no_grad():
                train_loss_list.append(train_loss_epoch)
                train_accuracy_list.append(train_accuracy_epoch)
                val_loss_epoch, val_accuracy_epoch = test(model, valloader, loss_function, validation=True)
                val_loss_list.append(val_loss_epoch)
                val_accuracy_list.append(val_accuracy_epoch)
                lr_list.append(optimizer.param_groups[0]['lr'])

                if rank == 0:
                    if c_dir:
                        results_file = current_model_time + '/results.txt'
                        # load results.txt
                        with open(results_file, 'r') as f:
                            results = f.readlines()
                        # load first line
                        r_train_loss = ast.literal_eval(results[0].split(':')[1].strip())
                        r_train_accuracy = ast.literal_eval(results[1].split(':')[1].strip())
                        r_val_loss = ast.literal_eval(results[2].split(':')[1].strip())
                        r_val_accuracy = ast.literal_eval(results[3].split(':')[1].strip())
                        r_lr = ast.literal_eval(results[4].split(':')[1].strip())

                        r_train_loss.extend(train_loss_list)
                        r_train_accuracy.extend(train_accuracy_list)
                        r_val_loss.extend(val_loss_list)
                        r_val_accuracy.extend(val_accuracy_list)
                        r_lr.extend(lr_list)

                        # write to results.txt
                        with open(results_file, 'w') as f:
                            f.write('train loss: ' + str(r_train_loss) + '\n')
                            f.write('train accuracy: ' + str(r_train_accuracy) + '\n')
                            f.write('validation loss: ' + str(r_val_loss) + '\n')
                            f.write('validation accuracy: ' + str(r_val_accuracy) + '\n')
                            f.write('learning rate: ' + str(r_lr) + '\n')

                    else:
                        # write the results to a file
                        with open(current_model_time + '/results.txt', 'w') as f:
                            f.write('train loss: ' + str(train_loss_list) + '\n')
                            f.write('train accuracy: ' + str(train_accuracy_list) + '\n')
                            f.write('validation loss: ' + str(val_loss_list) + '\n')
                            f.write('validation accuracy: ' + str(val_accuracy_list) + '\n')
                            f.write('learning rate: ' + str(lr_list) + '\n')

            epoch += 1

        lr_ind += 1

    torch.distributed.barrier()
    if rank == 0:
        model_name_test = current_model_time + '/' + str(model_name) + '_' + str(epoch-1) + '.pth'
        test_loss, test_accuracy = test(model, testloader, loss_function, validation=False, model_name=model_name_test)

        with open(current_model_time + '/results.txt', 'a') as f:
            f.write('test loss: ' + str(test_loss) + '\n')
            f.write('test accuracy: ' + str(test_accuracy) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--arch", help="model architecture", default='VGG', choices=['VGG', 'ResNet-56', 'ViT'])
    parser.add_argument("--norm_type", help="Normalization layer to be used", default='BatchNorm',
                        choices=['Plain', 'BatchNorm', 'LayerNorm', 'InstanceNorm', 'GroupNorm',
                                 'FilterResponseNorm', 'WeightNorm',
                                 'ScaledWeightStandardization', 'EvoNormSO', 'EvoNormBO', 'VarianceNorm'])
    parser.add_argument("--p_grouping", help="Number of channels per group for GroupNorm", default='32',
                        choices=['1', '0.5', '0.25', '0.125', '0.0625', '0.03125', '0.0000001', '8', '16', '32', '64'])
    parser.add_argument("--conv_type", help="Convolutional layer to be used", default='Plain',
                        choices=['Plain', 'sWS', 'WeightNormalized', 'WeightCentered'])
    parser.add_argument("--probe_layers", help="Probe activations/gradients?", default='True',
                        choices=['True', 'False'])
    parser.add_argument("--cfg", help="Model configuration", default='cfg_10')
    parser.add_argument("--skipinit", help="Use skipinit initialization?", default='False', choices=['True', 'False'])
    parser.add_argument("--preact", help="Use preactivation variants for ResNet?", default='False',
                        choices=['True', 'False'])
    parser.add_argument("--dataset", help="CIFAR-10 or CIFAR-100 or Synthetic", default='CIFAR-100',
                        choices=['CIFAR-10', 'CIFAR-100'])
    parser.add_argument("--batch_size", help="Batch size for DataLoader", default='256')
    parser.add_argument("--init_lr", help="Initial learning rate", default='1')
    parser.add_argument("--lr_warmup", help="Use a learning rate warmup?", default='False', choices=['True', 'False'])
    parser.add_argument("--opt_type", help="Optimizer", default='SGD', choices=['SGD'])
    parser.add_argument("--seed", help="set random generator seed", default='0')
    parser.add_argument("--download", help="download CIFAR-10/-100?", default='False')
    parser.add_argument("--local_rank", help="local rank", default='0')
    parser.add_argument("--c_dir", help="checkpoint directory", default=None)
    args = parser.parse_args()

    ######### Setup #########
    local_rank = int(args.local_rank)
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1

    d_path = "./"

    if not os.path.exists('test_metrics'):
        os.mkdir('test_metrics')
    trained_root = 'trained/'

    arch = args.arch  # Model architecture
    norm_type = args.norm_type  # Normalization layer
    probe_layers = (args.probe_layers == 'True')
    p_grouping = float(
        args.p_grouping)  # Amount of grouping for GroupNorm (<1 will define a group size, e.g., 0.5 = group size of 2; >1 defines number of groups)
    conv_type = args.conv_type  # Convolutional layer
    cfg_use = cfg_dict[args.cfg]  # Model configuration
    skipinit = (args.skipinit == 'True')  # Use skipinit initialization
    preact = (args.preact == 'True')  # Use pre-activation ResNet architecture
    use_data = args.dataset  # Dataset
    bsize = int(args.batch_size)  # BatchSize
    init_lr = float(args.init_lr)  # Initial learning rate
    lr_warmup = (args.lr_warmup == 'True')  # Use learning rate warmup (used for stabilizing training in Filter Response Normalization by Singh and Krishnan, 2019)
    opt_type = args.opt_type  # Optimizer
    c_dir = args.c_dir  # Checkpoint directory
    if args.c_dir == 'None':
        c_dir = None  # Checkpoint directory

    # find position of norm_type in norm_type choices
    norm_choices = ['Plain', 'BatchNorm', 'LayerNorm', 'InstanceNorm', 'GroupNorm',
                    'FilterResponseNorm', 'WeightNorm',
                    'ScaledWeightStandardization', 'EvoNormSO', 'EvoNormBO', 'VarianceNorm']
    norm_type_pos = norm_choices.index(norm_type)
    norm_type = ['Plain', 'BatchNorm', 'LayerNorm', 'Instance Normalization', 'GroupNorm',
                 'Filter Response Normalization', 'Weight Normalization',
                 'Scaled Weight Standardization', 'EvoNormSO', 'EvoNormBO', 'Variance Normalization'][norm_type_pos]


    base_epochs, wd, mom = base_epochs_iter, wd_base, mom_base  # Training configuration
    base_sched = [1e-1 / (256 / bsize), 1e-2 / (256 / bsize)]  # Learning rate is linearly scaled according to
    # batch-size
    if bsize < 32:
        base_epochs = [8, 2]  # 10 epochs at batch-size of 16 have 2x number of iterations as 60 epochs of batch-size
        # 256 training
    total_epochs = np.sum(base_epochs)

    ######### Data #########
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

    if use_data == "CIFAR-10":
        n_classes = 10
        trainset = torchvision.datasets.CIFAR10(root=d_path + 'datasets/cifar10/', train=True,
                                                download=(args.download == 'True'), transform=transform)
        trainsampler = torch.utils.data.distributed.DistributedSampler(trainset, num_replicas=world_size, rank=rank,
                                                                       shuffle=False, drop_last=False)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=bsize, shuffle=False, num_workers=0,
                                                  sampler=trainsampler, pin_memory=False, drop_last=True)

        testset = torchvision.datasets.CIFAR10(root='datasets/cifar10/', train=False,
                                               download=(args.download == 'True'), transform=transform_test)
        val_size = int(0.1 * len(testset))
        val_set, test_set = torch.utils.data.random_split(testset, [val_size, len(testset) - val_size])
        testloader = torch.utils.data.DataLoader(test_set, batch_size=bsize, shuffle=False, num_workers=0, pin_memory=False, drop_last=False)
        valloader = torch.utils.data.DataLoader(val_set, batch_size=bsize, shuffle=False, num_workers=0, pin_memory=False, drop_last=False)

    elif use_data == "CIFAR-100":
        n_classes = 100
        trainset = torchvision.datasets.CIFAR100(root=d_path + 'datasets/cifar100/', train=True,
                                                 download=(args.download == 'True'), transform=transform)
        trainsampler = torch.utils.data.distributed.DistributedSampler(trainset, num_replicas=world_size, rank=rank,
                                                                       shuffle=False, drop_last=False)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=bsize, shuffle=False, num_workers=0,
                                                  sampler=trainsampler, pin_memory=False, drop_last=True)
        testset = torchvision.datasets.CIFAR100(root=d_path + 'datasets/cifar100/', train=False,
                                                download=(args.download == 'True'), transform=transform_test)
        val_size = int(0.1 * len(testset))
        val_set, test_set = torch.utils.data.random_split(testset, [val_size, len(testset) - val_size])
        testloader = torch.utils.data.DataLoader(test_set, batch_size=bsize, shuffle=False, num_workers=0, pin_memory=False, drop_last=False)
        valloader = torch.utils.data.DataLoader(val_set, batch_size=bsize, shuffle=False, num_workers=0, pin_memory=False, drop_last=False)

    else:
        raise Exception("No data selected.")

    ######### Print Setup #########
    print("Architecture:", arch)
    print("Normalization layer:", norm_type)
    print("Probing On:", probe_layers)
    if norm_type == "GroupNorm":
        print("Grouping amount:", p_grouping)
    print("Convolutional layer:", conv_type)
    if arch == "VGG":
        print("Model config:", args.cfg)
    if arch == "ResNet-56":
        print("Skipinit:", skipinit)
    print("Dataset:", use_data)
    print("Batch Size:", bsize)
    print("LR Warmup:", lr_warmup)
    print("Optimizer:", opt_type)

    main()