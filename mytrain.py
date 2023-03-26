#!/usr/bin/env python
from __future__ import print_function

import argparse
import configparser
import os
import random
import shutil
from os.path import join, isfile
from os import makedirs
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import h5py
from tensorboardX import SummaryWriter
import numpy as np
from tqdm.auto import trange

from mycode.NetVLAD.netvlad import get_model_netvlad
import model3d.PointNetVlad as PNV
from sphereModel.sphereresnet import sphere_resnet18
from mycode.msls import MSLS

from mycode.train_epoch import train_epoch
from mycode.val import val
from crossmodal.training_tools.get_clusters import get_clusters
from crossmodal.training_tools.tools import save_checkpoint
from crossmodal.tools.datasets import input_transform
from crossmodal.models.models_generic import get_backend, get_model

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
def get_learning_rate(epoch):
    learning_rate = 0.0001 * ((0.8) ** (epoch // 2))  # 0.00005
    # learning_rate = max(learning_rate, 0.00001) * 50  # CLIP THE LEARNING RATE!
    return learning_rate

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CrossModal-train')

    parser.add_argument('--config_path', type=str, default='crossmodal/configs/train.ini',
                        help='File name (with extension) to an ini file that stores most of the configuration data')
    parser.add_argument('--cache_path', type=str, default='/data/zzp/cache',
                        help='Path to save cache, centroid data to.')
    parser.add_argument('--save_path', type=str, default='/data/zzp/result',
                        help='Path to save checkpoints to')
    parser.add_argument('--resume_path2d', type=str, default='',
                        help='Full path and name (with extension) to load checkpoint from, for resuming training.') # /home/zhipengz/result2/Aug26_15-48-13_vgg_clu64_4/checkpoints/model_best.pth.tar
    parser.add_argument('--pretrained_path3d', type=str, default='',
                        help='Full path and name (with extension) to load checkpoint from, for 3d pretrained.') # /home/zhipengz/result2/Aug26_15-48-13_vgg_clu64_4/checkpoints3d/model_best.ckpt
    parser.add_argument('--cluster_path', type=str, default='',
                        help='Full path and name (with extension) to load cluster data from, for resuming training.')# /data/zzp/cache/centroids/vgg_20m_KITTI360_64_desc_cen.hdf5
    parser.add_argument('--dataset_root_dir', type=str, default='/data/kitti360',
                        help='Root directory of dataset')
    parser.add_argument('--id', type=str, default='vgg',
                        help='Description of this model, e.g. vgg16_netvlad')
    parser.add_argument('--nEpochs', type=int, default=50, help='number of epochs to train for')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--save_every_epoch', action='store_true', help='Flag to set a separate checkpoint file for each new epoch')
    parser.add_argument('--threads', type=int, default=6, help='Number of threads for each data loader to use')
    parser.add_argument('--nocuda', action='store_true', help='If true, use CPU only. Else use GPU.')
    parser.add_argument('--network', type=str, default='vgg', help='2D CNN network, e.g. vgg')
    parser.add_argument('--pretrained_cnn_network', type=bool, default=True, help='whether use pretrained 2D CNN network ')

    opt = parser.parse_args()
    print(opt)
    print('os.environ[CUDA_VISIBLE_DEVICES]')
    print(os.environ['CUDA_VISIBLE_DEVICES'])
    size = 512
    attention = False
    print('size')
    print(size)
    print('attention')
    print(attention)
    print(opt.network)

    configfile = opt.config_path
    assert os.path.isfile(configfile)
    config = configparser.ConfigParser()
    config.read(configfile)

    # device_ids = [0, 1, 2, 3]
    cuda = not opt.nocuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run with --nocuda")
    device = torch.device("cuda" if cuda else "cpu")

    random.seed(int(config['train']['seed']))
    np.random.seed(int(config['train']['seed']))
    torch.manual_seed(int(config['train']['seed']))
    if cuda:
        # noinspection PyUnresolvedReferences
        torch.cuda.manual_seed(int(config['train']['seed']))

    optimizer = None
    scheduler = None

    print('===> Building 2d model')
    # feature extract network
    pre = opt.pretrained_cnn_network
    print('pretrained_cnn:')
    print(pre)
    if opt.network == 'spherical':
        encoder = sphere_resnet18(pretrained=pre)
        encoder_dim = 512
        # sphe = True
        # encoder_dim, encoder = get_spherical_cnn(network='original')  # TODO: freeze pretrained
    elif opt.network == 'resnet':
        encoder_dim, encoder = get_backend(net='resnet', pre=pre) #resnet
    elif opt.network == 'vgg':
        encoder_dim, encoder = get_backend(net='vgg', pre=pre) #resnet
    else:
        raise ValueError('Unknown cnn network')

    if opt.resume_path2d:  # if already started training earlier and continuing
        if isfile(opt.resume_path2d):
            print("=> loading checkpoint '{}'".format(opt.resume_path2d))
            checkpoint = torch.load(opt.resume_path2d, map_location=lambda storage, loc: storage)
            config['global_params']['num_clusters'] = str(checkpoint['state_dict']['pool.centroids'].shape[0])

            # model = get_model(encoder, encoder_dim, config['global_params'], append_pca_layer=False)
            model = get_model_netvlad(encoder, encoder_dim, config['global_params'],attention=attention)

            model.load_state_dict(checkpoint['state_dict'], strict=False)
            # opt.start_epoch = checkpoint['epoch']

            print("=> loaded checkpoint '{}'".format(opt.resume_path2d, ))
        else:
            raise FileNotFoundError("=> no checkpoint found at '{}'".format(opt.resume_path2d))
    else:  # if not, assume fresh training instance and will initially generate cluster centroids
        print('===> Loading model')
        config['global_params']['num_clusters'] = config['train']['num_clusters']

        # model = get_model(encoder, encoder_dim, config['global_params'], append_pca_layer=False)
        model = get_model_netvlad(encoder, encoder_dim, config['global_params'], attention=attention)

        initcache = join(opt.cache_path, 'centroids', opt.network + '_20m_KITTI360_' + config['train'][
            'num_clusters'] + '_desc_cen.hdf5')

        if opt.cluster_path:
            if isfile(opt.cluster_path):
                if opt.cluster_path != initcache:
                    shutil.copyfile(opt.cluster_path, initcache)
            else:
                raise FileNotFoundError("=> no cluster data found at '{}'".format(opt.cluster_path))
        else:
            print('===> Finding cluster centroids')

            print('===> Loading dataset(s) for clustering')
            train_dataset = MSLS(opt.dataset_root_dir, mode='test', cities='train', transform=input_transform(size, train=False),
                                 bs=int(config['train']['cachebatchsize']), threads=opt.threads,
                                 margin=float(config['train']['margin']))

            model = model.to(device)

            print('===> Calculating descriptors and clusters')
            get_clusters(train_dataset, model, encoder_dim, device, opt, config, initcache, size)

            # a little hacky, but needed to easily run init_params
            model = model.to(device="cpu")

        with h5py.File(initcache, mode='r') as h5:
            clsts = h5.get("centroids")[...]
            traindescs = h5.get("descriptors")[...]
            model.pool.init_params(clsts, traindescs)
            del clsts, traindescs
    print('model')
    print(model)
    isParallel = False
    '''if int(config['global_params']['nGPU']) > 1 and torch.cuda.device_count() > 1:
        model.encoder = nn.DataParallel(model.encoder)
        model.pool = nn.DataParallel(model.pool)
        # model3d = nn.DataParallel(model3d)
        isParallel = True'''

    if config['train']['optim'] == 'ADAM':
        optimizer = optim.Adam(filter(lambda par: par.requires_grad,
                                      model.parameters()), lr=float(config['train']['lr']))  # , betas=(0,0.9))
    elif config['train']['optim'] == 'SGD':
        optimizer = optim.SGD(filter(lambda par: par.requires_grad,
                                     model.parameters()), lr=float(config['train']['lr']),
                              momentum=float(config['train']['momentum']),
                              weight_decay=float(config['train']['weightDecay']))

        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=int(config['train']['lrstep']),
                                              gamma=float(config['train']['lrgamma']))
    else:
        raise ValueError('Unknown optimizer: ' + config['train']['optim'])

    criterion = nn.TripletMarginLoss(margin=float(config['train']['margin']) ** 0.5, p=2, reduction='sum').to(device)

    model = model.to(device)

    '''if opt.resume_path2d:
        optimizer.load_state_dict(checkpoint['optimizer'])'''

    # my code of 3dmodel
    print('===> Building 3d model')
    learning_rate = get_learning_rate(opt.start_epoch)
    print('3dLR:')
    print(learning_rate)
    if attention:
        model3d = PNV.PointNetVlad_attention(global_feat=True, feature_transform=True, max_pool=False, output_dim=256, num_points=4096)
        model3d.attention.init_weights()
    else:
        model3d = PNV.PointNetVlad(global_feat=True, feature_transform=True, max_pool=False, output_dim=256, num_points=4096)
    print('model3d')
    print(model3d)
    model3d = model3d.to(device)

    parameters3d = filter(lambda p: p.requires_grad, model3d.parameters())

    optimizer3d = optim.Adam(parameters3d, learning_rate)
    # scheduler3d = torch.optim.lr_scheduler.LambdaLR(optimizer3d, get_learning_rate, last_epoch=-1)
    if opt.pretrained_path3d:
        print("=> loading 3d model '{}'".format(opt.pretrained_path3d))
        checkpoint3d = torch.load(opt.pretrained_path3d)
        # saved_state_dict = checkpoint['state_dict']
        # starting_epoch = checkpoint3d['epoch']
        # TOTAL_ITERATIONS = starting_epoch * len(TRAINING_QUERIES)
        model3d.load_state_dict(checkpoint3d['state_dict'], strict=False)
        '''optimizer3d.load_state_dict(checkpoint3d['optimizer'])'''

    print('===> Loading dataset(s)')
    # exlude_panos_training = not config['train'].getboolean('includepanos')
    train_dataset = MSLS(opt.dataset_root_dir, mode='train', nNeg=int(config['train']['nNeg']),
                         transform=input_transform(size, train=True),
                         bs=int(config['train']['cachebatchsize']), threads=opt.threads,
                         margin=float(config['train']['margin']))

    validation_dataset = MSLS(opt.dataset_root_dir, mode='val', transform=input_transform(size, train=False),
                              bs=int(config['train']['cachebatchsize']), threads=opt.threads,
                              margin=float(config['train']['margin']), posDistThr=20)

    print('===> Training query set:', len(train_dataset.qIdx))
    print('===> Evaluating on val set, query count:', len(validation_dataset.qIdx))
    print('===> Training model')
    writer = SummaryWriter(log_dir=join(opt.save_path, datetime.now().strftime('%b%d_%H-%M-%S') + '_' + opt.id))

    # write checkpoints in logdir
    logdir = writer.file_writer.get_logdir()
    opt.save_file_path2d = join(logdir, 'checkpoints')
    makedirs(opt.save_file_path2d)
    opt.save_file_path3d = join(logdir, 'checkpoints3d')
    makedirs(opt.save_file_path3d)

    not_improved = 0
    best_score = 0
    if opt.resume_path2d:
        not_improved = checkpoint['not_improved']
        best_score = checkpoint['best_score']

    for epoch in trange(opt.start_epoch + 1, opt.nEpochs + 1, desc='Epoch number'.rjust(15), position=0):
        train_epoch(train_dataset, model, model3d, optimizer, optimizer3d, criterion, encoder_dim, device, epoch, opt, config, writer)
        if scheduler is not None:
            scheduler.step(epoch)
        # learning rate decay for 3d model
        lr_3d = get_learning_rate(epoch)
        parameters3d = filter(lambda p: p.requires_grad, model3d.parameters())
        optimizer3d = optim.Adam(parameters3d, lr_3d)
        if (epoch % int(config['train']['evalevery'])) == 0:
            recalls = val(validation_dataset, model, model3d, encoder_dim, device, opt.threads, config, writer, size, epoch,
                          write_tboard=True, pbar_position=1)
            is_best = recalls[5] > best_score
            if is_best:
                not_improved = 0
                best_score = recalls[5]
            else:
                not_improved += 1

            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'recalls': recalls,
                'best_score': best_score,
                'not_improved': not_improved,
                'optimizer': optimizer.state_dict(),
                'parallel': isParallel,
            }, opt, is_best)

            if isinstance(model3d, nn.DataParallel):
                model_to_save = model3d.module
            else:
                model_to_save = model3d
            # save 3d
            save_name = opt.save_file_path3d + "/" + "model.ckpt"
            torch.save({
                'epoch': epoch,
                'state_dict': model_to_save.state_dict(),
                'optimizer': optimizer3d.state_dict(),
            },
                save_name)
            if is_best:
                shutil.copyfile(save_name, join(opt.save_file_path3d, 'model_best.ckpt'))

            if int(config['train']['patience']) > 0 and not_improved > (
                    int(config['train']['patience']) / int(config['train']['evalevery'])):
                print('Performance did not improve for', config['train']['patience'], 'epochs. Stopping.')
                break

    print("=> Best Recall@5: {:.4f}".format(best_score), flush=True)
    writer.close()

    torch.cuda.empty_cache()  # garbage clean GPU memory, a bug can occur when Pytorch doesn't automatically clear the
    # memory after runs

    print('Done')
