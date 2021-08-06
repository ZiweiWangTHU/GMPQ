import os
import time
import math
import random
import shutil
import argparse
import torchvision
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import models as models
import random
import logging

from utils import Logger, AverageMeter, accuracy
from models.quantize_utils import QConv2d, calibrate

from math import ceil
from tensorboardX import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('-d', '--data', default='data/imagenet', type=str)
parser.add_argument('--data_name', default='imagenet', type=str)
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')

parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--warmup_epoch', default=0, type=int, metavar='N',
                    help='manual warmup epoch number (useful on restarts)')
parser.add_argument('--train_batch_per_gpu', default=256, type=int, metavar='N',
                    help='train batchsize (default: 256)')
parser.add_argument('--test_batch', default=512, type=int, metavar='N',
                    help='test batchsize (default: 512)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lr_type', default='cos', type=str,
                    help='lr scheduler (exp/cos/step3/fixed)')
parser.add_argument('--schedule', type=int, nargs='+', default=[31, 61, 91],
                    help='Decrease learning rate at these epochs.')
parser.add_argument('--arch-cfg', '--ac', default='', type=str, metavar='PATH',
                    help='path to architecture configuration')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-5, type=float,
                    metavar='W', help='weight decay (default: 1e-5)')
parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', default='', type=str, metavar='PATH',
                    help='use pretrained model')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet50', choices=model_names,
                    help='model architecture:' + ' | '.join(model_names) + ' (default: resnet50)')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pin_memory', default=2, type=int,
                    help='pin_memory of Dataloader')

parser.add_argument('--gpu_id', default='0,1', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}
lr_current = state['lr']
args.batch_size = args.train_batch_per_gpu * ceil(len(args.gpu_id) / 2)
print('batch size:', args.batch_size)
print('gpu:', args.gpu_id)
use_cuda = torch.cuda.is_available()

if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

best_acc = 0


def load_my_state_dict(model, state_dict):
    model_state = model.state_dict()
    for name, param in state_dict.items():
        if name not in model_state:
            continue
        param_data = param.data
        if model_state[name].shape == param_data.shape:
            model_state[name].copy_(param_data)


def train(train_loader, model, criterion, optimizer, epoch, use_cuda):
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()
    start = time.time()

    for batch_idx, (inputs, targets) in enumerate(train_loader):

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        outputs = model(inputs)

        loss = criterion(outputs, targets)

        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        if batch_idx % 50 == 0:
            print_logger.info("The train loss of epoch{}-batch-{}:{},top1 acc:{},top5 acc:{}".format(epoch,
                                                                                                     batch_idx,
                                                                                                     losses.avg,
                                                                                                     top1.avg,
                                                                                                     top5.avg))

    print_logger.info("The overall train loss of epoch{}:{},top1 acc:{},top5 acc:{},used_time:{}".format(epoch,
                                                                                                         losses.avg,
                                                                                                         top1.avg,
                                                                                                         top5.avg,
                                                                                                         time.time() - start))
    return losses.avg, top1.avg


def test(val_loader, model, criterion, epoch, use_cuda):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    with torch.no_grad():

        model.eval()

        start = time.time()

        for batch_idx, (inputs, targets) in enumerate(val_loader):

            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

    print_logger.info(
        'Epoch:{}, test loss:{}, top1 acc:{}, top5 acc:{}, used time:{}'.format(epoch, losses.avg, top1.avg, top5.avg,
                                                                                time.time() - start))
    return losses.avg, top1.avg


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar', epoch=0):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))


def adjust_learning_rate(optimizer, epoch):
    global lr_current

    if epoch < args.warmup_epoch:
        lr_current = state['lr'] * args.gamma
    elif args.lr_type == 'cos':
        lr_current = 0.5 * args.lr * (1 + math.cos(math.pi * epoch / args.epochs))
    elif args.lr_type == 'exp':
        step = 1
        decay = args.gamma
        lr_current = args.lr * (decay ** (epoch // step))
    elif epoch in args.schedule:
        lr_current *= args.gamma
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_current


def get_dataset(dataset_name, batch_size, n_worker, data_root='data/imagenet', for_inception=False, pin_memory=True):
    print('==> Preparing data..')
    if dataset_name == 'imagenet':
        traindir = os.path.join(data_root, 'train')
        valdir = os.path.join(data_root, 'val')
        assert os.path.exists(traindir), traindir + ' not found'
        assert os.path.exists(valdir), valdir + ' not found'
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        input_size = 299 if for_inception else 224

        train_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(
                traindir, transforms.Compose([
                    transforms.RandomResizedCrop(input_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ])),
            batch_size=batch_size, shuffle=True,
            num_workers=n_worker, pin_memory=pin_memory)

        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(int(input_size / 0.875)),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=batch_size, shuffle=False,
            num_workers=n_worker, pin_memory=pin_memory)

        n_class = 1000
    else:
        # Add customized data here
        raise NotImplementedError
    return train_loader, val_loader, n_class


def finetune(quant_strategy):
    lr_current = args.lr

    best_acc = 0
    start_epoch = args.start_epoch

    if not os.path.isdir(args.checkpoint):
        os.makedirs(args.checkpoint)
    run_name = os.path.join(args.checkpoint, 'visualization')
    if not os.path.isdir(run_name):
        os.makedirs(run_name)
    writer = SummaryWriter(log_dir=run_name)

    train_loader, val_loader, n_class = get_dataset(dataset_name=args.data_name, batch_size=args.batch_size,
                                                    n_worker=args.workers, data_root=args.data,
                                                    pin_memory=args.pin_memory)

    model = models.__dict__[args.arch](pretrained=args.pretrained, num_classes=n_class)
    cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    quantizable_idx = []
    for i, m in enumerate(model.modules()):
        if type(m) in [QConv2d]:
            quantizable_idx.append(i)

    print('quantizing:', (quantizable_idx))
    print((quant_strategy))
    quantize_layer_bit_dict = {n: b for n, b in zip(quantizable_idx, quant_strategy)}

    for i, layer in enumerate(model.modules()):
        if i not in quantizable_idx:
            continue
        else:

            layer.w_bit = quantize_layer_bit_dict[i][0]
            layer.a_bit = quantize_layer_bit_dict[i][1]
    model = model.cuda()
    model = calibrate(model, train_loader)

    model = torch.nn.DataParallel(model, device_ids=range(ceil(len(args.gpu_id) / 2)))
    title = 'ImageNet-' + args.arch
    logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
    logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])

    if args.resume:
        checkpoint = torch.load(args.resume)
        best_acc_resume = checkpoint['best_acc']
        print('resuming best accuracy:', best_acc_resume)
        model.load_state_dict(checkpoint['state_dict'], strict=False)

    if args.evaluate:
        print('\nEvaluation only')
        test_loss, test_acc = test(val_loader, model, criterion, start_epoch, use_cuda)
        print(' Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))
        return

    for epoch in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)
        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, lr_current))

        train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch, use_cuda)
        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('train_acc', train_acc, epoch)
        test_loss, test_acc = test(val_loader, model, criterion, epoch, use_cuda)
        writer.add_scalar('test_loss', test_loss, epoch)
        writer.add_scalar('test_acc', test_acc, epoch)

        logger.append([lr_current, train_loss, test_loss, train_acc, test_acc])

        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        writer.add_scalar('best_acc', best_acc, epoch)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'acc': test_acc,
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
        }, is_best, checkpoint=args.checkpoint, epoch=epoch)

    logger.close()
    writer.close()

    print('Best acc:')
    print(best_acc)
    return best_acc


def _load_arch(arch_path, names_nbits):
    checkpoint = torch.load(arch_path)
    state_dict = checkpoint['state_dict']
    best_arch, worst_arch = {}, {}
    for name in names_nbits.keys():
        best_arch[name], worst_arch[name] = [], []
    for name, params in state_dict.items():
        name = name.split('.')[-1]
        if name in names_nbits.keys():
            alpha = params.cpu().numpy()
            # print('name nbits', names_nbits[name], alpha.shape[0])
            assert names_nbits[name] == alpha.shape[0]

            best_arch[name].append(alpha.argmax())
            worst_arch[name].append(alpha.argmin())

    return best_arch, worst_arch


if __name__ == '__main__':

    if not os.path.isdir(args.checkpoint):
        os.makedirs(args.checkpoint)

    print_logger = logging.getLogger()
    print_logger.setLevel(logging.INFO)
    fh = logging.FileHandler(os.path.basename(args.checkpoint) + '_log_' + '.txt')
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    print_logger.addHandler(fh)
    print_logger.addHandler(ch)
    if len(args.arch_cfg) > 0:
        if os.path.isfile(args.arch_cfg):
            print("=> loading architecture config from '{}'".format(args.arch_cfg))
        else:
            print("=> no architecture found at '{}'".format(args.arch_cfg))
    wbits, abits = [2, 3, 4, 6], [2, 3, 4, 6]
    name_nbits = {'alpha_activ': len(abits), 'alpha_weight': len(wbits)}
    best_arch, worst_arch = _load_arch(args.arch_cfg, name_nbits)
    archas = [abits[a] for a in best_arch['alpha_activ']]
    archws = [wbits[w] for w in best_arch['alpha_weight']]
    quant_policy = []
    for i in range(len(archas)):
        quant_policy.append([archws[i], archas[i]])
    print_logger.info(quant_policy)

    best_acc = finetune(quant_policy)
    print_logger.info([quant_policy, best_acc])





