'''Train CIFAR10/CIFAR100 with PyTorch.'''
import argparse
import os
from optimizers import (KFACOptimizer, EKFACOptimizer)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR

import time
from tqdm import tqdm
from tensorboardX import SummaryWriter

from collections import OrderedDict
import torch
from torch.utils.data import DataLoader, RandomSampler
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
import numpy as np
import copy
import models_anp as models
import torch.nn.functional as F
# from torch.autograd.gradcheck import zero_gradients
import data.poison_cifar as poison
from torch.autograd import Variable
from autoaugment import CIFAR10Policy, ImageNetPolicy
from PIL import Image

from data_loader import *

# fetch args
parser = argparse.ArgumentParser()


# parser.add_argument('--network', default='vgg16_bn', type=str)

parser.add_argument('--log_dir', default='runs/pretrain', type=str)
parser.add_argument('--optimizer', default='ekfac', type=str)
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--epoch', default=100, type=int)
parser.add_argument('--milestone', default=None, type=str)
parser.add_argument('--learning_rate', default=0.01, type=float)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--stat_decay', default=0.95, type=float)
parser.add_argument('--damping', default=1e-3, type=float)
parser.add_argument('--kl_clip', default=1e-2, type=float)
parser.add_argument('--weight_decay', default=1e-4, type=float)
parser.add_argument('--TCov', default=10, type=int)
parser.add_argument('--TScal', default=10, type=int)
parser.add_argument('--TInv', default=100, type=int)

# Basic model parameters.
parser.add_argument('--arch', type=str, default='resnet18',
                    choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'MobileNetV2', 'vgg19_bn'])
parser.add_argument('--checkpoint', type=str, required=True, help='The checkpoint to be pruned')
parser.add_argument('--widen-factor', type=int, default=1, help='widen_factor for WideResNet')
parser.add_argument('--batch-size', type=int, default=128, help='the batch-size for dataloader')
parser.add_argument('--nb-iter', type=int, default=2000, help='the number of iterations for training')
parser.add_argument('--print-every', type=int, default=100, help='Print results every few iterations')
parser.add_argument('--data-dir', type=str, default='../data', help='dir to the dataset')
parser.add_argument('--val-frac', type=float, default=0.01, help='The fraction of the validate set')
parser.add_argument('--output-dir', type=str, default='logs/models/')
parser.add_argument('--gpuid', type=int, default=1, help='Gpu-Id')


parser.add_argument('--trigger-info', type=str, default='', help='The information of backdoor trigger')
parser.add_argument('--poison-type', type=str, default='badnets', choices=['badnets', 'Feature', 'FC',  'SIG', 'Dynamic', 'TrojanNet', 'blend', 'CLB', 'benign'],
                    help='type of backdoor attacks used during training')
parser.add_argument('--poison-target', type=int, default=0, help='target class of backdoor attack')
parser.add_argument('--trigger-alpha', type=float, default=0.2, help='the transparency of the trigger pattern.')


parser.add_argument('--isolation_model_root', type=str, default='./weight/ABL_results',
                    help='isolation model weights are saved here')
parser.add_argument('--log_root', type=str, default='./logs', help='logs are saved here')
parser.add_argument('--load_fixed_data', type=int, default=1, help='load the local poisoned dataest')
parser.add_argument('--poisoned_data_test_all2one', type=str, default='../Final_Aug/data/dynamic/poisoned_data/cifar10-test-inject0.1-target0-dynamic-all2one.npy', help='random seed')
parser.add_argument('--poisoned_data_test_all2all', type=str, default='../Final_Aug/data/dynamic/poisoned_data/cifar10-test-inject0.1-target0-dynamic-all2all_mask.npy', help='random seed')


# Backdoor Attacks
parser.add_argument('--inject_portion', type=float, default=0.1, help='ratio of backdoor samples')
parser.add_argument('--target_label', type=int, default=0, help='class of target label')
parser.add_argument('--trigger_type', type=str, default='squareTrigger', choices=['squareTrigger', 'gridTrigger', 'fourCornerTrigger', 'randomPixelTrigger',
                               'signalTrigger', 'trojanTrigger'], help='type of backdoor trigger')
parser.add_argument('--target_type', type=str, default='all2one', help='type of backdoor label')
parser.add_argument('--trig_w', type=int, default=3, help='width of trigger pattern')
parser.add_argument('--trig_h', type=int, default=3, help='height of trigger pattern')

args = parser.parse_args()
args_dict = vars(args)
print(args_dict)
os.makedirs(args.output_dir, exist_ok=True)
device = 'cuda:3' if torch.cuda.is_available() else 'cpu'
torch.cuda.set_device(3)


## Linear Transformation
MEAN_CIFAR10 = (0.4914, 0.4822, 0.4465)
STD_CIFAR10 = (0.2023, 0.1994, 0.2010)

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    # ImageNetPolicy(),                       ## For Strong Augmentation'
    # CIFAR10Policy(),
    transforms.ToTensor(),
    transforms.Normalize(MEAN_CIFAR10, STD_CIFAR10)
])

transform_none = transforms.ToTensor()
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MEAN_CIFAR10, STD_CIFAR10)
])

## Clean Test Loader 
clean_test = CIFAR10(root=args.data_dir, train=False, download=True, transform=transform_test)
clean_test_loader = DataLoader(clean_test, batch_size=args.batch_size, num_workers=0)

## Triggers 
triggers = {'badnets': 'checkerboard_1corner',
            'CLB': 'fourCornerTrigger',
            'blend': 'gaussian_noise',
            'SIG': 'signalTrigger',
            'TrojanNet': 'trojanTrigger',
            'FC': 'gridTrigger',
            'benign': None}

if args.poison_type == 'badnets':
    args.trigger_alpha = 1

# args.inject_portion = args.poison_rate

## Step 1: create dataset -- clean val set, poisoned test set, and clean test set.
if args.trigger_info:
    trigger_info = torch.load(args.trigger_info, map_location=device)


elif args.poison_type in ['badnets', 'blend']:
    trigger_type  = triggers[args.poison_type]
    pattern, mask = poison.generate_trigger(trigger_type=trigger_type)
    trigger_info  = {'trigger_pattern': pattern[np.newaxis, :, :, :], 'trigger_mask': mask[np.newaxis, :, :, :],
                    'trigger_alpha': args.trigger_alpha, 'poison_target': np.array([args.poison_target])}

    poison_test = poison.add_predefined_trigger_cifar(data_set=clean_test, trigger_info=trigger_info)                   ## To check how many of the poisonous sample is correctly classified to their "target labels"
    # poison_test = poison.add_predefined_trigger_cifar_true_label(data_set=clean_test, trigger_info=trigger_info)          ## To check how many of the poisonous sample is correctly classified to their "true labels"
    poison_test_loader = DataLoader(poison_test, batch_size=args.batch_size, num_workers=0)

elif args.poison_type in ['Dynamic', 'Feature']:
    if args.dataset == 'cifar10':
      args.dataset = 'CIFAR10'
    transform_test = transforms.Compose([
        # transforms.ToTensor(),
        transforms.Normalize(MEAN_CIFAR10, STD_CIFAR10)
    ])
    if args.target_type =='all2one':
        poisoned_data = Dataset_npy(np.load(args.poisoned_data_test_all2one, allow_pickle=True), transform = None)
    else:
        poisoned_data = Dataset_npy(np.load(args.poisoned_data_test_all2all, allow_pickle=True), transform = None)

    poison_test_loader = DataLoader(dataset=poisoned_data,
                                    batch_size=args.batch_size,
                                    shuffle=True)
    clean_test_loader = DataLoader(clean_test, batch_size=args.batch_size, num_workers=4)
    trigger_info = None

elif args.poison_type in ['SIG', 'TrojanNet', 'CLB', 'FC']:
    trigger_type      = triggers[args.poison_type]
    args.trigger_type = trigger_type        

    ## SIG and CLB are Clean-label Attacks 
    if args.poison_type in ['SIG', 'CLB', 'FC']:
        args.target_type = 'cleanLabel'

    if args.dataset == 'cifar10':
      args.dataset = 'CIFAR10'
    
    _, poison_test_loader = get_test_loader(args)
    clean_test_loader = DataLoader(clean_test, batch_size=args.batch_size, num_workers=4)

    trigger_info = None

elif args.poison_type == 'benign':
    trigger_info = None

## Get the dataloader for finetuning 
orig_train        = CIFAR10(root=args.data_dir, train=True, download=True, transform=transform_train)
_, clean_val_none = poison.split_dataset(dataset=orig_train, val_frac=args.val_frac,
                                    perm=np.loadtxt('./data/cifar_shuffle.txt', dtype=int))
clean_val = clean_val_none                                                                                                   ## Removal using Clean val data
# clean_val = poison.add_predefined_trigger_cifar_true_label(data_set=clean_val_none, trigger_info=trigger_info)             ## Removal using Triggered val data

random_sampler    = RandomSampler(data_source=clean_val, replacement=True,
                               num_samples =args.print_every * args.batch_size)
clean_val_loader  = DataLoader(clean_val, batch_size=args.batch_size,
                              shuffle=False, sampler=random_sampler, num_workers=0)


## Step 2: Load Model Checkpoints and Trigger Info
state_dict = torch.load(args.checkpoint, map_location=device)
if args.poison_type in ['Dynamic', 'Feature']:
    state_dict = torch.load(args.checkpoint, map_location=device)['netC']
    print("dynamic attack")
else:
    state_dict = torch.load(args.checkpoint, map_location=device)

## Init criterion
criterion = nn.CrossEntropyLoss()
start_epoch = 0
best_acc = 0

## Step 2: Load model checkpoints and trigger info
net = getattr(models, args.arch)(num_classes=10)
net.load_state_dict(state_dict)
net = net.cuda()

## Select which parameters to finetune (Determine from the architecture)
params_finetune = [param for name, param in net.named_parameters()]   
# params_finetune.remove(params_finetune[0])
# params_finetune.remove(params_finetune[1])
# params_finetune = [param for name, param in net.named_parameters() if "linear" not in name]

## Init summary writter
log_dir = os.path.join(args.log_dir, args.dataset, args.arch, args.optimizer,
                       'lr%.3f_wd%.4f_damping%.4f' %
                       (args.learning_rate, args.weight_decay, args.damping))
if not os.path.isdir(log_dir):
    os.makedirs(log_dir)
writer = SummaryWriter(log_dir)
 
## Step 2.1: FIM based optimizer 
optim_name = args.optimizer.lower()
tag = optim_name
if optim_name == 'sgd':
    optimizer = optim.SGD(params_finetune,
                          lr=args.learning_rate,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)

elif optim_name == 'adam':
    optimizer = optim.Adam(params_finetune,
                          lr=args.learning_rate,
                          weight_decay=args.weight_decay)  

elif optim_name == 'adagrad':
    optimizer = optim.Adagrad(params_finetune,
                          lr=args.learning_rate,
                          weight_decay=args.weight_decay)  

elif optim_name == 'adagmax':
    optimizer = optim.Adagrad(params_finetune,
                          lr=args.learning_rate,
                          weight_decay=args.weight_decay)  

elif optim_name == 'rmsprop':
    optimizer = optim.RMSprop(params_finetune,
                          lr=args.learning_rate,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)  

elif optim_name == 'kfac':
    optimizer = KFACOptimizer(net,
                              lr=args.learning_rate,
                              momentum=args.momentum,
                              stat_decay=args.stat_decay,
                              damping=args.damping,
                              kl_clip=args.kl_clip,
                              weight_decay=args.weight_decay,
                              TCov=args.TCov,
                              TInv=args.TInv)
elif optim_name == 'ekfac':
    optimizer = EKFACOptimizer(net,
                               lr=args.learning_rate,
                               momentum=args.momentum,
                               stat_decay=args.stat_decay,
                               damping=args.damping,
                               kl_clip=args.kl_clip,
                               weight_decay=args.weight_decay,
                               TCov=args.TCov,
                               TScal=args.TScal,
                               TInv=args.TInv)
else:
    raise NotImplementedError


nb_repeat = int(np.ceil(args.nb_iter / args.print_every))
if args.milestone is None:
    lr_scheduler = MultiStepLR(optimizer, milestones=[int(nb_repeat*0.35), int(nb_repeat*0.75)], gamma=0.1)
else:
    milestone    = [int(_) for _ in args.milestone.split(',')]
    lr_scheduler = MultiStepLR(optimizer, milestones=milestone, gamma=0.1)



def main():

  ## Validate the model 
  cl_test_loss, cl_test_acc = test(model=net, criterion=criterion, data_loader=clean_test_loader)
  print("Validation Accuracy of the Given Model:", cl_test_acc)

  if args.poison_type != 'benign':
      po_test_loss, po_test_acc = test(model=net, criterion=criterion, data_loader=poison_test_loader)
      print("Attack Success Rate of the Given Model:", po_test_acc)
  

  #     ## Losses and Accuracy ##
  # clean_losses  = np.zeros(nb_repeat)
  # poison_losses = np.zeros(nb_repeat)
  # clean_accs    = np.zeros(nb_repeat)
  # poison_accs   = np.zeros(nb_repeat)

  ## Step 3: Finetune backdoored Models
  print('Iter \t lr \t Time \t TrainLoss \t TrainACC \t PoisonLoss \t PoisonACC \t CleanLoss \t CleanACC')
  po_test_acc =  0
  po_test_loss = 0
  for i in range(nb_repeat):
    start = time.time()
    lr = optimizer.param_groups[0]['lr']

    train_loss, train_acc = finetune_train(i, net)
    cl_test_loss, cl_test_acc = test(model=net, criterion=criterion, data_loader=clean_test_loader)
    if args.poison_type != 'benign':
        po_test_loss, po_test_acc = test(model=net, criterion=criterion, data_loader=poison_test_loader)

    model_save = args.poison_type + '_' + str(i) + '_' + args.optimizer + '_' + str(args.inject_portion) + '_fisher.pth'
    torch.save(net.state_dict(), os.path.join(args.output_dir, model_save))
    
    end = time.time()
    
    # print(cl_test_acc)
    print('{} \t {:.3f} \t {:.1f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}'.format(
        (i + 1) * args.print_every, lr, end - start, train_loss, train_acc, po_test_loss, po_test_acc,
        cl_test_loss, cl_test_acc))

    # clean_losses[i]  = cl_test_loss
    # poison_losses[i] = po_test_loss
    # clean_accs[i]    = cl_test_acc
    # poison_accs[i]   = po_test_acc

    # ## Save numpy file
    # np.savez(os.path.join(args.output_dir,'remove_loss_accuracy_'+ args.poison_type + '_' + str(args.dataset) + str(args.print_every) + '_weak_strong_.npz'), cl_loss = clean_losses, cl_test = clean_accs, po_loss = poison_losses, po_acc = poison_accs)
    # # model_save = args.poison_type + '_' + str(i) + '_' + str(args.dataset) + '_True_Label.pth'
    # # torch.save(net.state_dict(), os.path.join(args.output_dir, model_save))
        




## Training Scheme
def finetune_train(epoch, net):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    lr_scheduler.step()
    desc = ('[%s][LR=%s] Loss: %.3f | Acc: %.3f%% (%d/%d)' %
            (tag, lr_scheduler.get_lr()[0], 0, 0, correct, total))

    writer.add_scalar('train/lr', lr_scheduler.get_lr()[0], epoch)

    prog_bar = tqdm(enumerate(clean_val_loader), total=len(clean_val_loader), desc=desc, leave=True)
    for batch_idx, (inputs, targets) in prog_bar:
        inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        if optim_name in ['kfac', 'ekfac'] and optimizer.steps % optimizer.TCov == 0:
            ## Compute True Fisher
            optimizer.acc_stats = True
            with torch.no_grad():
                sampled_y = torch.multinomial(torch.nn.functional.softmax(outputs.cpu().data, dim=1),
                                              1).squeeze().cuda()
            loss_sample = criterion(outputs, sampled_y)
            loss_sample.backward(retain_graph=True)
            optimizer.acc_stats = False
            optimizer.zero_grad()           # Clear the gradient for computing true-fisher.
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        desc = ('[%s][LR=%s] Loss: %.3f | Acc: %.3f%% (%d/%d)' %
                (tag, lr_scheduler.get_lr()[0], train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        prog_bar.set_description(desc, refresh=True)

    writer.add_scalar('train/loss', train_loss/(batch_idx + 1), epoch)
    writer.add_scalar('train/acc', 100. * correct / total, epoch)

    return train_loss/(batch_idx + 1), 100. * correct / total

def test(model, criterion, data_loader):
    model.eval()
    total_correct = 0
    total_loss = 0.0
    with torch.no_grad():
        for i, (images, labels) in enumerate(data_loader):
            images, labels = images.cuda(), labels.cuda()
            output = model(images)
            total_loss += criterion(output, labels).item()
            pred = torch.max(output,1)[1]
            total_correct += pred.eq(labels.data.view_as(pred)).sum()
    loss = total_loss / len(data_loader)
    acc = float(total_correct) / len(data_loader.dataset)
    return loss, acc

if __name__ == '__main__':
    main()


