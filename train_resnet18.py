
import os
import time
import math
import random
import shutil
import argparse

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.models as models
import models as customized_models
import random
import logging
import numpy as np
import torch.nn.utils.prune as prune


from lib.utils.utils import Logger, AverageMeter, accuracy
from lib.utils.data_utils import get_dataset
from progress.bar import Bar
from lib.utils.quantize_utils import quantize_model, kmeans_update_model, QConv2d, QLinear, calibrate
# from lib.utils.quant_layer import QuantConv2d
from math import ceil
from tensorboardX import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets
# torch.cuda.set_device(0)
# torch.distributed.init_process_group(backend='nccl', world_size=4, init_method='...')

# Models
device = torch.device("cuda")
default_model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

customized_models_names = sorted(name for name in customized_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(customized_models.__dict__[name]))

for name in customized_models.__dict__:
    if name.islower() and not name.startswith("__") and callable(customized_models.__dict__[name]):
        models.__dict__[name] = customized_models.__dict__[name]

model_names = default_model_names + customized_models_names

# Parse arguments
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# Datasets
parser.add_argument('-d', '--data', default='data/imagenet', type=str)
parser.add_argument('--data_name', default='imagenet', type=str)
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options
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
parser.add_argument('--schedule', type=int, nargs='+', default=[51,71,86],#[31, 61, 91],48
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--first_schedule', type=int, nargs='+', default=[31],#[31, 61, 91],48
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-5, type=float,
                    metavar='W', help='weight decay (default: 1e-5)')
# Checkpoints
parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', action='store_true',
                    help='use pretrained model')
parser.add_argument('--sample', action='store_true',
                    help='mode of sample')
# Quantization
parser.add_argument('--linear_quantization', dest='linear_quantization', action='store_true',
                    help='quantize both weight and activation)')
parser.add_argument('--free_high_bit', default=True, type=bool,
                    help='free the high bit (>6)')
parser.add_argument('--half', action='store_true',
                    help='half')
parser.add_argument('--half_type', default='O1', type=str,
                    help='half type: O0/O1/O2/O3')
# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet50', choices=model_names,
                    help='model architecture:' + ' | '.join(model_names) + ' (default: resnet50)')
# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pin_memory', default=2, type=int,
                        help='pin_memory of Dataloader')
# Device options
parser.add_argument('--gpu_id', default='0,1', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}
lr_current = state['lr']
args.batch_size=args.train_batch_per_gpu* ceil(len(args.gpu_id) / 2)
print('batch size:',args.batch_size)

# Use CUDA
# os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
print('gpu:',args.gpu_id)
use_cuda = torch.cuda.is_available()
# torch.cuda.set_device(0)

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)


best_acc = 0  # best test accuracy


def load_my_state_dict(model, state_dict):
    model_state = model.state_dict()
    for name, param in state_dict.items():
        if name not in model_state:
            continue
        param_data = param.data
        if model_state[name].shape == param_data.shape:
            # print("load%s"%name)
            model_state[name].copy_(param_data)


def train(train_loader, model, criterion, optimizer, epoch, use_cuda):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()
    start=time.time()

    # bar = Bar('Processing', max=len(train_loader))
    for batch_idx, (inputs, targets) in enumerate(train_loader):

        # measure data loading time
        # data_time.update(time.time() - end)
        # print('getting input')
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
        # print('got input')

        # compute output
        # print('forwarding------')
        outputs = model(inputs)
        # print('forward Done')
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # compute gradient
        optimizer.zero_grad()

        loss.backward()
        # do SGD step
        optimizer.step()


        # measure elapsed time
        # batch_time.update(time.time() - end)
        # end = time.time()

        if batch_idx % 50 == 0:
            print_logger.info("The train loss of epoch{}-batch-{}:{},top1 acc:{},top5 acc:{}".format(epoch,
                                                                       batch_idx, losses.avg,top1.avg,top5.avg))

        # plot progress
    #     if batch_idx % 1 == 0:
    #         bar.suffix = \
    #             '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | ' \
    #             'Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
    #                 batch=batch_idx + 1,
    #                 size=len(train_loader),
    #                 data=data_time.val,
    #                 bt=batch_time.val,
    #                 total=bar.elapsed_td,
    #                 eta=bar.eta_td,
    #                 loss=losses.avg,
    #                 top1=top1.avg,
    #                 top5=top5.avg,
    #             )
    #
    #         print(
    #             '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | ' \
    #             'Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
    #                 batch=batch_idx + 1,
    #                 size=len(train_loader),
    #                 data=data_time.val,
    #                 bt=batch_time.val,
    #                 total=bar.elapsed_td,
    #                 eta=bar.eta_td,
    #                 loss=losses.avg,
    #                 top1=top1.avg,
    #                 top5=top5.avg,
    #             ))
    #
    #         bar.next()
    # bar.finish()
    print_logger.info("The overall train loss of epoch{}:{},top1 acc:{},top5 acc:{},used_time:{}".format(epoch,
                                                                                              losses.avg,
                                                                                             top1.avg, top5.avg,time.time()-start))
    return losses.avg, top1.avg


def test(val_loader, model, criterion, epoch, use_cuda):
    # global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    with torch.no_grad():
        # switch to evaluate mode
        model.eval()

        start = time.time()
        # bar = Bar('Processing', max=len(val_loader))
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            print(inputs.size(),targets.size())
            # measure data loading time
            # data_time.update(time.time() - end)

            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)

            # compute output
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

            # measure elapsed time
            # batch_time.update(time.time() - end)
            # end = time.time()
            # if batch_idx %50==0:


            # plot progress
            # if batch_idx % 1 == 0:
            #     bar.suffix  = \
            #         '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | ' \
            #         'Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
            #             batch=batch_idx + 1,
            #             size=len(val_loader),
            #             data=data_time.avg,
            #             bt=batch_time.avg,
            #             total=bar.elapsed_td,
            #             eta=bar.eta_td,
            #             loss=losses.avg,
            #             top1=top1.avg,
            #             top5=top5.avg,
            #             )
            #
            #     print(
            #         '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | ' \
            #         'Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
            #             batch=batch_idx + 1,
            #             size=len(val_loader),
            #             data=data_time.avg,
            #             bt=batch_time.avg,
            #             total=bar.elapsed_td,
            #             eta=bar.eta_td,
            #             loss=losses.avg,
            #             top1=top1.avg,
            #             top5=top5.avg,
            #             ))
            #
            #     bar.next()
        # bar.finish()
    print_logger.info('Epoch:{}, test loss:{}, top1 acc:{}, top5 acc:{}, used time:{}'.format(epoch, losses.avg,top1.avg,top5.avg,time.time()-start))
    return losses.avg, top1.avg


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar',epoch=0):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))
    # if epoch %10 ==0:
    #     shutil.copyfile(filepath,os.path.join(checkpoint, str(epoch)+'_model.pth.tar'))


def adjust_learning_rate(optimizer, epoch):
    global lr_current
    # global best_acc
    if epoch < args.warmup_epoch:
        lr_current = state['lr']*args.gamma
    elif args.lr_type == 'cos':
        # cos
        lr_current = 0.5 * args.lr * (1 + math.cos(math.pi * epoch / args.epochs))
    elif args.lr_type == 'exp':
        step = 1
        decay = args.gamma
        lr_current = args.lr * (decay ** (epoch // step))
    elif epoch in args.schedule:
        lr_current *= args.gamma
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_current


def finetune(quant_strategy,exp_id=0,prune_policy=None):
    # global best_acc
    # global lr_current

    lr_current = args.lr


    best_acc=0
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch


    if not os.path.isdir(args.checkpoint):
        os.makedirs(args.checkpoint)
    sub_checkpoint=os.path.join(args.checkpoint,str(exp_id))
    if not os.path.isdir(sub_checkpoint):
        os.makedirs(sub_checkpoint)
    run_name=os.path.join(args.checkpoint,'vis')
    if not os.path.isdir(run_name):
        os.makedirs(run_name)
    writer = SummaryWriter(log_dir=run_name)

    dataloader = datasets.CIFAR10
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = dataloader(root=args.data, train=True, download=True, transform=transform_train)
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
    else:
        train_sampler = None
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers,
                                               pin_memory=True)

    testset = dataloader(root=args.data, train=False, download=True, transform=transform_test)
    val_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.workers,
                                             pin_memory=True)

    model = models.__dict__[args.arch](pretrained=args.pretrained,num_classes=10)
    # print("=> creating model '{}'".format(args.arch), ' pretrained is ', args.pretrained)
    # print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
    cudnn.benchmark = True

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)#momentum

    # use HalfTensor
    if args.half:
        try:
            import apex
        except ImportError:
            raise ImportError("Please install apex from https://github.com/NVIDIA/apex")
        model.cuda()
        model, optimizer = apex.amp.initialize(model, optimizer, opt_level=args.half_type)



    # quantizable_idx = []
    # for i, m in enumerate(model.modules()):
    #     if type(m) in [QConv2d, QLinear]:
    #         quantizable_idx.append(i)
    #
    # print('quantizing:', (quantizable_idx))
    # print((quant_strategy))
    # quantize_layer_bit_dict = {n: b for n, b in zip(quantizable_idx, quant_strategy)}
    #
    # for i, layer in enumerate(model.modules()):
    #     if i not in quantizable_idx:
    #         continue
    #     else:
    #         # print(type(layer))
    #         layer.w_bit = quantize_layer_bit_dict[i][0]
    #         layer.a_bit = quantize_layer_bit_dict[i][1]
    model = model.cuda()
    # model = calibrate(model, train_loader)
    # if 'mobilenetv2' in args.arch:
    # new_strategy = [[8, -1], [7, 7], [5, 6], [4, 6], [5, 6],
    #             [5, 7], [5, 6], [7, 4], [4, 6], [4, 6],
    #             [7, 7], [5, 6], [4, 6], [7, 3], [5, 7], [4, 7],
    #             [7, 3], [5, 7], [4, 7], [7, 7], [4, 7], [4, 7],
    #             [6, 4], [6, 7], [4, 7], [7, 4], [6, 7], [5, 7],
    #             [7, 4], [6, 7], [5, 7], [7, 4], [6, 7], [6, 7],
    #             [6, 4], [5, 7], [6, 7], [6, 4], [5, 7], [6, 7],
    #             [7, 7], [4, 7], [7, 7], [7, 7], [4, 7], [7, 7],
    #             [7, 7], [4, 7], [7, 7], [7, 7], [4, 7], [7, 7],
    #             [8, 8]]
    #     strategy=[[4,4] for i in range(53)]
    # else:
    #     raise NotImplementedError

    # print(len(strategy))


    # for i, layer in enumerate(model.modules()):
    #     if i not in quantizable_idx:
    #         continue
    #     else:
    #         layer.w_bit = quantize_layer_bit_dict[i][0]
    #         layer.a_bit = quantize_layer_bit_dict[i][1]
    # model = model.cuda()
    # model=model.to(device)
    ##calibrate

    # if len(args.gpu_id)>1:

    model = torch.nn.DataParallel(model,device_ids=range(ceil(len(args.gpu_id)/2)))
    # model=torch.nn.parallel.DistributedDataParallel(model, device_ids=[0], output_device=0)

    # Resume
    title = 'ImageNet-' + args.arch

    # prune_model(model, prune_policy)

    logger = Logger(os.path.join(sub_checkpoint, 'log.txt'), title=title)
    logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])
    # print(args.resume)
    # exit()

    # checkpoint = torch.load(args.resume)
    # best_acc_resume = checkpoint['best_acc']
    # print('resuming best acc:', best_acc_resume)
    # model.load_state_dict(checkpoint['state_dict'],strict=False)
    # test_loss, test_acc = test(val_loader, model, criterion, start_epoch, use_cuda)
    # print(' Resuming Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))
    # exit()

    # if args.resume:
    #
    #
    #
    #     # if os.path.isdir(sub_checkpoint) and os.path.isfile(os.path.join(sub_checkpoint,'checkpoint.pth.tar')):
    #     checkpoint = torch.load(args.resume)
    #     # best_acc_resume = checkpoint['best_acc']
    #     # print(best_acc_resume)
    #
    #     model.load_state_dict(checkpoint['state_dict'])
    #     prune_model(model,prune_policy)
    #
    #
    #     # else:
    #     #     ch = {n.replace('module.', ''): v for n, v in checkpoint['state_dict'].items()}
    #     #     model.load_state_dict(ch)
    #     print('==> Resuming from checkpoint..')
    #     # assert os.path.isfile(os.path.join(sub_checkpoint,'checkpoint.pth.tar')), 'Error: no checkpoint directory found!'
    #     # args.checkpoint = os.path.dirname(args.resume)
    #     checkpoint = torch.load(os.path.join(args.resume))
    #     best_acc = checkpoint['best_acc']
    #     print('resuming best acc:',best_acc)
    #     start_epoch = checkpoint['epoch']
    #     model.load_state_dict(checkpoint['state_dict'],strict=False)
    #     optimizer.load_state_dict(checkpoint['optimizer'])
    #     test_loss, test_acc = test(val_loader, model, criterion, start_epoch, use_cuda)
    #     print(' Resuming Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))
    #     exit()
    #
    #     if os.path.isfile(os.path.join(sub_checkpoint, 'log.txt')):
    #         logger = Logger(os.path.join(sub_checkpoint, 'log.txt'), title=title, resume=True)
    #     else:
    #         logger = Logger(os.path.join(sub_checkpoint, 'log.txt'), title=title)
    #         logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])
    # exit()




    if args.evaluate:
        print('\nEvaluation only')
        test_loss, test_acc = test(val_loader, model, criterion, start_epoch, use_cuda)
        print(' Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))


    # exit()
    # test_loss, test_acc = test(val_loader, model, criterion, start_epoch, use_cuda)
    # print(' after pruning Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))




    # if start_epoch>9:
    #
    #     quantizable_idx = []
    #     for i, m in enumerate(model.modules()):
    #         if type(m) in [QConv2d, QLinear]:
    #             quantizable_idx.append(i)
    #
    #     print('quantizing:', (quantizable_idx))
    #     quantize_layer_bit_dict = {n: b for n, b in zip(quantizable_idx, strategy)}
    #
    #     for i, layer in enumerate(model.modules()):
    #         if i not in quantizable_idx:
    #             continue
    #         else:
    #             # print(type(layer))
    #             layer.w_bit = quantize_layer_bit_dict[i][0]
    #             layer.a_bit = quantize_layer_bit_dict[i][1]
    #     model = calibrate(model, train_loader)

    # if start_epoch > 9:
    #
    #     # best_acc = best_acc_resume
    #     # best_acc=0
    #     quantizable_idx = []
    #     for i, m in enumerate(model.modules()):
    #         if type(m) in [QConv2d, QLinear]:
    #             quantizable_idx.append(i)
    #
    #     print('quantizing:', (quantizable_idx))
    #     print((quant_strategy))
    #     quantize_layer_bit_dict = {n: b for n, b in zip(quantizable_idx, quant_strategy)}
    #
    #     for i, layer in enumerate(model.modules()):
    #         if i not in quantizable_idx:
    #             continue
    #         else:
    #             # print(type(layer))
    #             layer.w_bit = quantize_layer_bit_dict[i][0]
    #             layer.a_bit = quantize_layer_bit_dict[i][1]
    #     model = calibrate(model, train_loader)

    # Train and val
    for epoch in range(start_epoch, args.epochs):


        # adjust_learning_rate(optimizer, epoch)
        if epoch in args.schedule:
            lr_current *= args.gamma
        elif epoch in args.first_schedule:
            lr_current *= 0.5
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_current

        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, lr_current))

        train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch, use_cuda)
        writer.add_scalar('train_loss'+'_'+str(exp_id),train_loss,epoch)
        writer.add_scalar('train_acc'+'_'+str(exp_id),train_acc,epoch)
        test_loss, test_acc = test(val_loader, model, criterion, epoch, use_cuda)
        writer.add_scalar('test_loss'+'_'+str(exp_id),test_loss,epoch)
        writer.add_scalar('test_acc'+'_'+str(exp_id),test_acc,epoch)

        # append logger file
        logger.append([lr_current, train_loss, test_loss, train_acc, test_acc])

        # save model
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        writer.add_scalar('best_acc' + '_' + str(exp_id), best_acc, epoch)
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict(),
            }, is_best, checkpoint=sub_checkpoint, epoch=epoch)

    '''
    best_acc = 0
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    quantizable_idx = []
    for i, m in enumerate(model.modules()):
        if type(m) in [QConv2d, QLinear]:
            quantizable_idx.append(i)

    print('quantizing:', (quantizable_idx))
    print((quant_strategy))
    quantize_layer_bit_dict = {n: b for n, b in zip(quantizable_idx, quant_strategy)}

    for i, layer in enumerate(model.modules()):
        if i not in quantizable_idx:
            continue
        else:
            # print(type(layer))
            layer.w_bit = quantize_layer_bit_dict[i][0]
            layer.a_bit = quantize_layer_bit_dict[i][1]
    # model = calibrate(model, train_loader)
    # start_epoch
    quant_epoch=25
    if start_epoch>25:
        quant_epoch=start_epoch


    for epoch in range(quant_epoch,args.epochs+25):

        if epoch in args.schedule:
            lr_current *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_current

        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs+25, lr_current))

        train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch, use_cuda)
        writer.add_scalar('train_loss'+'_'+str(exp_id),train_loss,epoch)
        writer.add_scalar('train_acc'+'_'+str(exp_id),train_acc,epoch)
        test_loss, test_acc = test(val_loader, model, criterion, epoch, use_cuda)
        writer.add_scalar('test_loss'+'_'+str(exp_id),test_loss,epoch)
        writer.add_scalar('test_acc'+'_'+str(exp_id),test_acc,epoch)

        # append logger file
        logger.append([lr_current, train_loss, test_loss, train_acc, test_acc])

        # save model
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        writer.add_scalar('best_acc' + '_' + str(exp_id), best_acc, epoch)
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict(),
            }, is_best, checkpoint=sub_checkpoint, epoch=epoch)

    '''


        # if best_acc>65 and epoch>=19:
        #     return best_acc

    logger.close()
    writer.close()

    print('Best acc:')
    print(best_acc)
    return best_acc


# def pretrain():
#     # global best_acc
#     # global lr_current
#
#     lr_current = args.lr
#
#     best_acc = 0
#     start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch
#
#     if not os.path.isdir(args.checkpoint):
#         os.makedirs(args.checkpoint)
#     sub_checkpoint = os.path.join(args.checkpoint, 'pretrain')
#     if not os.path.isdir(sub_checkpoint):
#         os.makedirs(sub_checkpoint)
#     run_name = os.path.join(args.checkpoint, 'vis')
#     if not os.path.isdir(run_name):
#         os.makedirs(run_name)
#     writer = SummaryWriter(log_dir=run_name)
#
#     train_loader, val_loader, n_class = get_dataset(dataset_name=args.data_name, batch_size=args.batch_size,
#                                                     n_worker=args.workers, data_root=args.data,
#                                                     pin_memory=args.pin_memory)
#     model = models.__dict__[args.arch](pretrained=True,num_classes=100)
#     # print("=> creating model '{}'".format(args.arch), ' pretrained is ', args.pretrained)
#     # print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
#     cudnn.benchmark = True
#
#     # define loss function (criterion) and optimizer
#     criterion = nn.CrossEntropyLoss().cuda()
#     optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
#                           weight_decay=args.weight_decay)  # momentum
#
#     # use HalfTensor
#     if args.half:
#         try:
#             import apex
#         except ImportError:
#             raise ImportError("Please install apex from https://github.com/NVIDIA/apex")
#         model.cuda()
#         model, optimizer = apex.amp.initialize(model, optimizer, opt_level=args.half_type)
#
#     # if 'mobilenetv2' in args.arch:
#     # new_strategy = [[8, -1], [7, 7], [5, 6], [4, 6], [5, 6],
#     #             [5, 7], [5, 6], [7, 4], [4, 6], [4, 6],
#     #             [7, 7], [5, 6], [4, 6], [7, 3], [5, 7], [4, 7],
#     #             [7, 3], [5, 7], [4, 7], [7, 7], [4, 7], [4, 7],
#     #             [6, 4], [6, 7], [4, 7], [7, 4], [6, 7], [5, 7],
#     #             [7, 4], [6, 7], [5, 7], [7, 4], [6, 7], [6, 7],
#     #             [6, 4], [5, 7], [6, 7], [6, 4], [5, 7], [6, 7],
#     #             [7, 7], [4, 7], [7, 7], [7, 7], [4, 7], [7, 7],
#     #             [7, 7], [4, 7], [7, 7], [7, 7], [4, 7], [7, 7],
#     #             [8, 8]]
#     #     strategy=[[4,4] for i in range(53)]
#     # else:
#     #     raise NotImplementedError
#
#     # print(len(strategy))
#
#     # for i, layer in enumerate(model.modules()):
#     #     if i not in quantizable_idx:
#     #         continue
#     #     else:
#     #         layer.w_bit = quantize_layer_bit_dict[i][0]
#     #         layer.a_bit = quantize_layer_bit_dict[i][1]
#     # model = model.cuda()
#     model = model.to(device)
#     ##calibrate
#
#     if len(args.gpu_id) > 1:
#         model = torch.nn.DataParallel(model)  # .cuda()#,device_ids=range(ceil(len(args.gpu_id)/2))
#     # model=torch.nn.parallel.DistributedDataParallel(model, device_ids=[0], output_device=0)
#
#     # Resume
#     title = 'ImageNet-' + args.arch
#
#
#     logger = Logger(os.path.join(sub_checkpoint, 'log.txt'), title=title)
#     logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])
#
#
#     if args.evaluate:
#         print('\nEvaluation only')
#         test_loss, test_acc = test(val_loader, model, criterion, start_epoch, use_cuda)
#         print(' Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))
#
#
#
#     test_loss, test_acc = test(val_loader, model, criterion, start_epoch, use_cuda)
#     print(' pretraining Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))
#     # exit()
#
#     # Train and val
#     for epoch in range(start_epoch, args.epochs):
#
#         # adjust_learning_rate(optimizer, epoch)
#         if epoch in args.schedule:
#             lr_current *= args.gamma
#         for param_group in optimizer.param_groups:
#             param_group['lr'] = lr_current
#
#
#         # exit()
#         # if epoch>12:
#         #     return best_acc
#         exp_id='pretrain'
#         print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, lr_current))
#
#         train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch, use_cuda)
#         writer.add_scalar('train_loss' + '_' + 'pretrain', train_loss, epoch)
#         writer.add_scalar('train_acc' + '_' + 'pretrain', train_acc, epoch)
#         test_loss, test_acc = test(val_loader, model, criterion, epoch, use_cuda)
#         writer.add_scalar('test_loss' + '_' + 'pretrain', test_loss, epoch)
#         writer.add_scalar('test_acc' + '_' + 'pretrain', test_acc, epoch)
#
#         # append logger file
#         logger.append([lr_current, train_loss, test_loss, train_acc, test_acc])
#
#         # save model
#         is_best = test_acc > best_acc
#         best_acc = max(test_acc, best_acc)
#         writer.add_scalar('best_acc' + '_' + 'pretrain', best_acc, epoch)
#         save_checkpoint({
#             'epoch': epoch + 1,
#             'state_dict': model.state_dict(),
#             'acc': test_acc,
#             'best_acc': best_acc,
#             'optimizer': optimizer.state_dict(),
#         }, is_best, checkpoint=sub_checkpoint, epoch=epoch)
#
#         # if best_acc>65 and epoch>=19:
#         #     return best_acc
#
#     logger.close()
#     writer.close()
#
#     print('Best acc:')
#     print(best_acc)
#     return best_acc
def prune_model(model,prune_policy):
    print('=====> Pruning')
    prunable_idx = []
    for i, m in enumerate(model.modules()):
        if type(m) in [QConv2d]:
            prunable_idx.append(i)
    # print(len(prunable_idx))
    # print(model)
    # prunable_idx=prunable_idx[3:-2]
    print((prunable_idx))
    print(prune_policy)
    prune_layer_dict = {key: value for key, value in zip(prunable_idx, prune_policy)}
    for i,layer in enumerate(model.modules()):
        if i in prunable_idx:
            prune.ln_structured(layer, name='weight', amount=prune_layer_dict[i], dim=0, n=2)
    return model





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

#     prune_policy = [0.13463187, 0.04389148, 0.22991219, 0.11824001, 0.3669421, 0.20548008,
#                     0.30818644, 0.2920741, 0.06783391, 0.21290512, 0.18711464, 0.42500067,
#                     0.18580721, 0.1929598, 0.40242872, 0.10622276, 0.11016646, 0.19689302,
#                     0.44865927]
#     quant_vector=[4, 2, 2, 4, 7, 6, 4, 2, 4, 6, 6, 6, 5, 4, 3, 5, 7, 6, 7, 6, 5, 8, 4, 4,
# 4, 5, 5, 2, 2, 8, 4, 3, 4, 6, 2, 3, 5, 3]
    prune_policy=[0.13979352, 0.27091193, 0.42088735, 0.61789185, 0.11904202, 0,
     0.05577452, 0.02558663, 0.09508231, 0.    ,     0.     ,    0.17391534,
     0.20223643, 0.41929013, 0.16388999, 0.11159942, 0.63201606, 0.,
     0.19483185]
    quant_vector=[4,
     4,
     2,
     2,
     7,
     5,
     3,
     4,
     6,
     2,
     4,
     8,
     3,
     5,
     7,
     7,
     5,
     2,
     5,
     4,
     5,
     2,
     2,
     2,
     6,
     2,
     7,
     4,
     6,
     4,
     2,
     5,
     8,
     2,
     4,
     3,
     3,
     5, ]

    quant_policy=[]
    for i in range(0,38,2):
        quant_policy.append([quant_vector[i],quant_vector[i+1]])
    print_logger.info(quant_policy,prune_policy)


    # quant2_policy = policy_list[exp_id][1]




    print_logger.info([quant_policy,prune_policy])

    best_acc = finetune(quant_policy, exp_id=0,
                        prune_policy=prune_policy)  # ,train_loader, val_loader, n_class)

    print_logger.info([quant_policy,prune_policy,best_acc])







        #

        #
        #

        #
        #     # print('loader len:',len(train_loader),len(val_loader))
        #     best_acc=finetune(strategy_policy,exp_id=exp_id,prune_policy=prune_policy)#,train_loader, val_loader, n_class)
        #     result_list.append([strategy_policy,prune_policy,best_acc])

        # # print(len(result_list[0][0]),'\n')


