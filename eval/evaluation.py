import sys
sys.path.append("..")
from mycode.detail_val import val
from crossmodal.models.models_generic import get_backend, get_model
import model3d.PointNetVlad as PNV
from crossmodal.tools.datasets import input_transform
# from tensorboardX import SummaryWriter
from mycode.msls_test import MSLS
from mycode.loading_pointclouds import load_pc_files
import torch
from torchvision.utils import make_grid, save_image
from mycode.NetVLAD.netvlad import get_model_netvlad
# from mycode.SCNN import get_spherical_cnn
import os
from PIL import Image
import numpy as np
import h5py
# from sphereModel.example import SphereNet, Net
from sphereModel.sphereresnet import sphere_resnet18
# from sphereModel.resnet_o import resnet18
print('import done')
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
'''tform = input_transform()
img_path = '/datassd4t/zhipengz/mydataset/train_val/s03/database/images/959ddddddd.png'
def save_img_my(tensor, name):
    tensor = tensor.permute((1, 0, 2, 3))
    im = make_grid(tensor, normalize=True, scale_each=True, nrow=8, padding=0).permute((1, 2, 0))
    im = (im.data.numpy() * 255.).astype(np.uint8)
    Image.fromarray(im).save(name + '.png')
def model_test(img_path, model):
    print('model')
    print(model)
    query = tform(Image.open(img_path))
    input_data = query.unsqueeze(0)
    model.eval()
    with torch.no_grad():
        input_data = input_data.to(device)
        image_encoding = model.encoder(input_data)
        print('image_encoding.shape')
        print(image_encoding.shape)
        print('image_encoding')
        print(image_encoding)

        vlad_encoding, v_t = model.pool(image_encoding)
        print('vlad_encoding.shape')
        print(vlad_encoding.shape)
        print('v_t.shape')
        print(v_t.shape)
        print('vlad_encoding')
        print(vlad_encoding)
        print('v_t')
        print(v_t)
    save_img_my(image_encoding.cpu(), 'encoding')
    # vlad_encoding = vlad_encoding.view(-1, 16, 16).unsqueeze(0)
    vlad_encoding = vlad_encoding[0].view(16, 16)
    v_t = v_t.unsqueeze(0)
    save_img_my(v_t.cpu(), 'v_t')
def model3d_test(model3d):
    pc_path = '/datassd4t/zhipengz/dataset_exam3d/mydataset/kitti360/s03/database/pc/0000000145.bin'
    query3d = torch.tensor(load_pc_files([pc_path]))
    input_data = query3d.unsqueeze(0)  # torch.Size([1, 1, 4096, 3])
    print('input.size')
    print(input_data.shape)
    print('model3d')
    print(model3d)
    model3d.eval()
    with torch.no_grad():
        input_data = input_data.to(device)
        pc_encoding = model3d.point_net(input_data)  # torch.Size([1, 1024, 4096, 1])
        print('pc_encoding.shape')
        print(pc_encoding.shape)
        print('pc_encoding')
        print(pc_encoding)

        vlad_encoding = model3d.net_vlad(pc_encoding)  # torch.Size([1, 256])
        print('vlad_encoding.shape')
        print(vlad_encoding.shape)
        # print('v_t.shape')
        # print(v_t.shape)
        print('vlad_encoding')
        print(vlad_encoding)
        # print('v_t')
        # print(v_t)
    save_img_my(pc_encoding.view(1, 1024, 64, 64).cpu(), 'enc_pc')
    vlad_encoding = vlad_encoding.view(1, 16, 16).unsqueeze(0)
    save_img_my(vlad_encoding.cpu(), 'vlad_pc')
def encoder_test():
    model = encoder
    print('model')
    print(model)
    query = tform(Image.open(img_path))
    input_data = query.unsqueeze(0)
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        input_data = input_data.to(device)
        image_encoding = model(input_data)
        print('image_encoding.shape')
        print(image_encoding.shape)
        # print('image_encoding')
        # print(image_encoding)
    save_img_my(image_encoding.cpu(), 'encoding')'''
trained = True
patchnv = False
clu = False
dataset_root_dir = '/datassd4t/zhipengz/mydataset'

device = torch.device("cuda")
config = {'network': 'vgg', 'num_clusters': 64, 'pooling': 'netvlad', 'vladv2': False}
config['train'] = {'cachebatchsize': 10}
writer = []

size = 512
if config['network'] == 'spherical':
    encoder = sphere_resnet18(pretrained=True)
    encoder_dim = 512
    size = 256
    # encoder_dim, encoder = get_spherical_cnn(network='original')  # TODO: freeze pretrained
elif config['network'] == 'resnet':
    encoder_dim, encoder = get_backend(net='resnet')  # resnet
elif config['network'] == 'vgg':
    encoder_dim, encoder = get_backend(net='vgg', pre=True)  # resnet
else:
    raise ValueError('Unknown cnn network')
# model_test(img_path, encoder)

# encoder_test()
# out = np.squeeze(image_encoding.cpu().numpy(), 0).transpose([1, 2, 0])
# plt.imsave('demo_pool_3x3.png', out)
if patchnv:
    model = get_model(encoder, encoder_dim, config, append_pca_layer=False)
else:
    model = get_model_netvlad(encoder, encoder_dim, config, attention=False)
# model = get_model_netvlad(encoder, encoder_dim, config)
model3d = PNV.PointNetVlad(global_feat=True, feature_transform=True, max_pool=False, output_dim=256, num_points=4096)

if trained:
    model_path2d = '/home/zhipengz/result3/Sep04_12-20-51_vgg_20m/checkpoints/model_best.pth.tar'  # Aug21_22-09-09_vgg_clu64_2
    model_path3d = '/home/zhipengz/result3/Sep04_12-20-51_vgg_20m/checkpoints3d/model_best.ckpt' # Aug27_16-38-16_vgg_f2_resume_h
    checkpoint = torch.load(model_path2d, map_location=lambda storage, loc: storage)

    checkpoint3d = torch.load(model_path3d, map_location=lambda storage, loc: storage)
    # print('chepoint3d')
    # print(checkpoint3d.keys())
    # print(checkpoint3d)
    # for name, param in model.named_parameters():
    #     print(name, '      ', param.size())
    model.load_state_dict(checkpoint['state_dict'])
    # print('pre_model')
    # print(model)
    model3d.load_state_dict(checkpoint3d['state_dict'])
    # print('pre_model3d')
    # print(model3d)
elif clu:
    cache_path = '/home/zhipengz/cache'
    initcache = os.path.join(cache_path, 'centroids',
                             'vgg16_' + 'KITTI360' + str(config['num_clusters']) + '_desc_cen.hdf5')
    with h5py.File(initcache, mode='r') as h5:
        clsts = h5.get("centroids")[...]
        traindescs = h5.get("descriptors")[...]
        model.pool.init_params(clsts, traindescs)
        del clsts, traindescs


model = model.to(device)
model3d = model3d.to(device)

validation_dataset = MSLS(dataset_root_dir, mode='val', transform=input_transform(size, train=False), bs=10, threads=6, margin=0.1, posDistThr=20)

recalls = val(validation_dataset, model, model3d, device, 6, config, writer, size, 0, write_tboard=False, pbar_position=1)
