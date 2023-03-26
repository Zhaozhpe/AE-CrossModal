'''
MIT License

Copyright (c) 2021 Stephen Hausler, Sourav Garg, Ming Xu, Michael Milford and Tobias Fischer

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Significant parts of our code are based on [Nanne's pytorch-netvlad repository]
(https://github.com/Nanne/pytorch-NetVlad/), as well as some parts from the [Mapillary SLS repository]
(https://github.com/mapillary/mapillary_sls)

Validation of NetVLAD, using the Mapillary Street-level Sequences Dataset.
'''
# 560
# len(eval.dbPcs)
# 1133


import numpy as np
import torch
import faiss
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from mycode.msls import ImagesFromList
from mycode.msls import PcFromFiles
from crossmodal.tools.datasets import input_transform
# import pickle
import os
import json
import shutil
from shutil import copyfile
from PIL import Image

def save_img_my(tensor, name):
    tensor = tensor.permute((1, 0, 2, 3))
    im = make_grid(tensor, normalize=True, scale_each=True, nrow=8, padding=0).permute((1, 2, 0))
    im = (im.data.numpy() * 255.).astype(np.uint8)
    Image.fromarray(im).save(name + '.jpg')


def val(eval_set, model, model3d, device, threads, config, writer, size, epoch_num=0, write_tboard=False, pbar_position=0):
    if device.type == 'cuda':
        cuda = True
    else:
        cuda = False
    eval_set_queries = ImagesFromList(eval_set.qImages, transform=input_transform(size, train=False))  # (512,512)
    eval_set_dbs = ImagesFromList(eval_set.dbImages, transform=input_transform(size, train=False)) # (512,512)
    eval_set_queries_pc = PcFromFiles(eval_set.qPcs)
    eval_set_dbs_pc = PcFromFiles(eval_set.dbPcs)
    print('eval.qImg')
    print(eval_set.qImages)
    print('eval.dbImages')
    print(eval_set.dbImages)
    print('eval.qPcs')
    print(eval_set.qPcs)
    print('eval.dbPcs')
    print(eval_set.dbPcs)
    print('queryNum')
    print(len(eval_set.qImages))
    print('databaseNum')
    print(len(eval_set.dbImages))
    test_data_loader_queries = DataLoader(dataset=eval_set_queries,
                                          num_workers=threads, batch_size=int(config['train']['cachebatchsize']),
                                          shuffle=False, pin_memory=cuda)
    test_data_loader_dbs = DataLoader(dataset=eval_set_dbs,
                                         num_workers=threads, batch_size=int(config['train']['cachebatchsize']),
                                         shuffle=False, pin_memory=cuda)
    test_data_loader_queries_pc = DataLoader(dataset=eval_set_queries_pc,
                                         num_workers=threads, batch_size=int(config['train']['cachebatchsize']),
                                         shuffle=False, pin_memory=cuda)
    test_data_loader_dbs_pc = DataLoader(dataset=eval_set_dbs_pc,
                                      num_workers=threads, batch_size=int(config['train']['cachebatchsize']),
                                      shuffle=False, pin_memory=cuda)
    # for each query get those within threshold distance
    save_pic = False
    save_path = './output/'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    gt = eval_set.all_pos_indices
    gt_index = []
    gt_lists = []
    for i in range(len(gt)):
        gt_index.append(eval_set.qImages[i][-14:])
        pos = gt[i]
        pics = [eval_set.dbImages[p][-14:] for p in pos]
        gt_lists.append(pics)
    gt_dic = dict(zip(gt_index, gt_lists))
    # print('gt')
    # print(gt_dic)
    with open('./output/ground_truth.txt', 'w', encoding= 'utf-8') as file:
        file.write(json.dumps(gt_dic))

    model.eval()
    model3d.eval()
    with torch.no_grad():
        tqdm.write('====> Extracting Features')
        # pool_size = encoder_dim
        # if config['global_params']['pooling'].lower() == 'netvlad':
        #     pool_size *= int(config['global_params']['num_clusters'])
        pool_size = 256
        qFeat = np.empty((len(eval_set_queries), pool_size), dtype=np.float32)
        qFeat_FeatureMap = np.empty((len(eval_set_queries), 512, 32, 64), dtype=np.float32)
        dbFeat = np.empty((len(eval_set_dbs), pool_size), dtype=np.float32)
        dbFeat_FeatureMap = np.empty((len(eval_set_dbs), 512, 32, 64), dtype=np.float32)
        qFeat_pc = np.empty((len(eval_set_queries_pc), pool_size), dtype=np.float32)
        # qFeat_pc_FeatureMap = np.empty((len(eval_set_queries_pc), 1024, 4096, 1), dtype=np.float32)
        qFeat_pc_FeatureMap = []
        dbFeat_pc = np.empty((len(eval_set_dbs_pc), pool_size), dtype=np.float32)
        # dbFeat_pc_FeatureMap = np.empty((len(eval_set_dbs_pc), 1024, 4096, 1), dtype=np.float32)
        dbFeat_pc_FeatureMap = []

        for feat, feat_fm, test_data_loader in zip([qFeat, dbFeat], [qFeat_FeatureMap, dbFeat_FeatureMap], [test_data_loader_queries, test_data_loader_dbs]):
            for iteration, (input_data, indices) in \
                    enumerate(tqdm(test_data_loader, position=pbar_position, leave=False, desc='Test1 Iter'.rjust(15)), 1):
                # print('input_data')
                # print(input_data)
                # print('input_data.shape')
                # print(input_data.shape)
                # print('indices')
                # print(indices)
                input_data = input_data.to(device)
                image_encoding = model.encoder(input_data)
                # img_enc = image_encoding
                vlad_encoding= model.pool(image_encoding)  # , v_t
                # feat_fm[indices.detach().numpy(), :] = image_encoding.detach().cpu().numpy()
                feat[indices.detach().numpy(), :] = vlad_encoding.detach().cpu().numpy()

                del input_data, image_encoding, vlad_encoding
        for feat, feat_fm, test_data_loader in zip([qFeat_pc, dbFeat_pc], [qFeat_pc_FeatureMap, dbFeat_pc_FeatureMap], [test_data_loader_queries_pc, test_data_loader_dbs_pc]):
            for iteration, (input_data, indices) in \
                    enumerate(tqdm(test_data_loader, position=pbar_position, leave=False, desc='Test2 Iter'.rjust(15)), 1):
                # print('input_data3d')
                # print(input_data)
                # print('input_data3d.shape')
                # print(input_data.shape)
                # print('indices3d')
                # print(indices)
                input_data = input_data.float()
                # feed_tensor = torch.cat((input_data), 1)
                input_data = input_data.view((-1, 1, 4096, 3))
                input_data = input_data.to(device)
                pc_enc = model3d.point_net(input_data)
                vlad_encoding = model3d.net_vlad(pc_enc)
                # vlad_encoding = model3d(input_data)
                # feat_fm[indices.detach().numpy(), :] = pc_enc.detach().cpu().numpy()
                feat[indices.detach().numpy(), :] = vlad_encoding.detach().cpu().numpy()

                del input_data, pc_enc, vlad_encoding


    del test_data_loader_queries, test_data_loader_dbs, test_data_loader_queries_pc, test_data_loader_dbs_pc

    tqdm.write('====> Building faiss index')
    '''print('q_2d')
    print(qFeat)
    print('db_3d')
    print(dbFeat_pc)'''
    # faiss_index = faiss.IndexFlatL2(pool_size)
    # # noinspection PyArgumentList
    # faiss_index.add(dbFeat_pc)

    tqdm.write('====> Calculating recall @ N')
    n_values = [1, 5, 10, 20, 50]



    # any combination of mapillary cities will work as a val set
    # 2d->3d
    predictions_t = {}
    predictions = {}
    des = ['2dto2d', '2dto3d', '3dto2d', '3dto3d']
    for i in range(4):
        if i == 0:   # 2d->2d
            qTest = qFeat
            dbTest = dbFeat
        if i == 1:   # 2d->3d
            qTest = qFeat
            dbTest = dbFeat_pc
        if i == 2:   # 3d->2d
            qTest = qFeat_pc
            dbTest = dbFeat
        if i == 3:   # 3d->3d
            qTest = qFeat_pc
            dbTest = dbFeat_pc
        qEndPosTot = 0
        dbEndPosTot = 0
        for cityNum, (qEndPos, dbEndPos) in enumerate(zip(eval_set.qEndPosList, eval_set.dbEndPosList)):
            faiss_index = faiss.IndexFlatL2(pool_size)
            faiss_index.add(dbTest[dbEndPosTot:dbEndPosTot+dbEndPos, :])
            dis, preds = faiss_index.search(qTest[qEndPosTot:qEndPosTot+qEndPos, :], max(n_values)+1) # add +1
            print(des[i])
            print(dis)
            # print('cityNum')
            # print(cityNum)
            # print('pred')
            # print(preds)
            if cityNum == 0:
                predictions_t[i] = preds
            else:
                predictions_t[i] = np.vstack((predictions_t[i], preds))
            qEndPosTot += qEndPos
            dbEndPosTot += dbEndPos
    # get rid of the same frame of query and database for same modality
    # predictions = predictions_t
    for i in range(4):
        print(des[i])
        print(predictions_t[i])
    predictions[0] = [list(pre[1:]) for pre in predictions_t[0]]
    predictions[3] = [list(pre[1:]) for pre in predictions_t[3]]
    predictions[1] = [list(pre[:50]) for pre in predictions_t[1]]
    predictions[2] = [list(pre[:50]) for pre in predictions_t[2]]

    q_ind = []
    q_ind_path = []
    # pics_path = {}

    pics_path = {}
    p_fnames = {}
    for i in range(4):
        pics_path[i] = []
        p_fnames[i] = []
    pre_lists_3d = []
    for i in range(len(gt)):
        q_ind.append(eval_set.qImages[i][-14:])
        q_ind_path.append(eval_set.qImages[i])
        for j in range(4):
            pre_pos = predictions[j][i]
            pic_path = eval_set.dbImages[pre_pos]
            pics_path[j].append(pic_path)
        # pre_pos_3d = predictions[1][i]
        # pics_path_3d = eval_set.dbImages[pre_pos_3d]
        # pre_lists_path_3d.append(pics_path_3d)
            if j == 0 or j == 2:
                p_fname = [eval_set.dbImages[p][-14:] for p in pre_pos]
            else:
                p_fname = [eval_set.dbPcs[p][-14:] for p in pre_pos]
            p_fnames[j].append(p_fname)
            # pre_pics_3d = [eval_set.dbImages[p][-14:] for p in pre_pos_3d]
            # pre_lists_3d.append(pre_pics_3d)
    for i,task in enumerate(des):
        path = './output/' + task + '.txt'
        pre_dic = dict(zip(q_ind, p_fnames[i]))
        with open(path, 'w', encoding='utf-8') as file:
            file.write(json.dumps(pre_dic))
    '''pre_dic_3d = dict(zip(q_ind, p_fnames))
    # print('pre_dic')
    # print(pre_dic)
    with open('./output/2dto2d.txt', 'w', encoding='utf-8') as file:
        file.write(json.dumps(pre_dic))
    with open('./output/2dto3d.txt', 'w', encoding='utf-8') as file:
        file.write(json.dumps(pre_dic_3d))'''
    # save 2d->2d 2d->3d
    if save_pic:
        save_num = 10
        save_ind = np.random.choice(len(qFeat), save_num, replace=False)
        print('save_ind:')
        print(save_ind)
        # indx_p = 0
        for indx_p, filename in enumerate(['2dto2d/', '2dto3d/','3dto2d/', '3dto3d/']):
            print('indx_p')
            print(indx_p)
            save2dto2d = save_path + filename
            if not os.path.exists(save2dto2d):
                os.mkdir(save2dto2d)
            for ind in save_ind:
                save_dir = save2dto2d + str(ind) + '/'
                if not os.path.exists(save_dir):
                    os.mkdir(save_dir)
                else:
                    save_dir = save_dir + 'second/'
                    os.mkdir(save_dir)

                save_q = save_dir + 'q/'
                os.mkdir(save_q)
                query = q_ind_path[ind]
                shutil.copy(query, save_q)
                if indx_p == 0 or indx_p == 1:
                    enc = torch.from_numpy(qFeat_FeatureMap[ind])
                    vlad = torch.from_numpy(qFeat[ind])
                    save_img_my(enc.unsqueeze(0), save_q + 'encoding')
                    save_img_my(vlad.view(16, 16).unsqueeze(0).unsqueeze(0), save_q + 'vlad')
                else:
                    enc = torch.from_numpy(qFeat_pc_FeatureMap[ind])
                    vlad = torch.from_numpy(qFeat_pc[ind])
                    save_img_my(enc.unsqueeze(0).view(1, 1024, 64, 64), save_q + 'encoding')
                    save_img_my(vlad.view(16, 16).unsqueeze(0).unsqueeze(0), save_q + 'vlad')
                # print('qFeat[ind]')
                # print(vlad)
                # print('qFeat_FeatureMap[ind]')
                # print(enc)
                # print('qFeat[ind].shape')
                # print(vlad.shape)
                # print('qFeat_FeatureMap[ind].shape')
                # print(enc.shape)
                # print('qFeat[ind].type')
                # print(type(vlad))
                # print('qFeat_FeatureMap[ind].type')
                # print(type(enc))
                # save_img_my(enc.unsqueeze(0), save_q + 'encoding')
                # save_img_my(vlad.view(16, 16).unsqueeze(0).unsqueeze(0), save_q + 'vlad')

                positives = pics_path[indx_p][ind][:5]
                for i, pos in enumerate(positives):
                    img_name = list(pos[-14:])
                    # print(img_name[9])
                    img_name[9] = str(i)
                    img_name = ''.join(img_name)
                    save_name = save_dir + img_name
                    shutil.copy(pos, save_name)
                pos_index_indbs = predictions[indx_p][ind][:5]

                save_dir_fm = save_dir + 'fm/'
                if not os.path.exists(save_dir_fm):
                    os.mkdir(save_dir_fm)
                if indx_p == 0 or indx_p == 2:
                    for i, ind_db in enumerate(pos_index_indbs):
                        enc_db = torch.from_numpy(dbFeat_FeatureMap[ind_db])
                        vlad_db = torch.from_numpy(dbFeat[ind_db])
                        save_enc_path = save_dir_fm  + 'enc' + str(i)
                        save_vlad_path = save_dir_fm + 'vlad' + str(i)
                        save_img_my(enc_db.unsqueeze(0), save_enc_path)
                        save_img_my(vlad_db.view(16, 16).unsqueeze(0).unsqueeze(0), save_vlad_path)
                else:
                    for i, ind_db in enumerate(pos_index_indbs):
                        enc_db = torch.from_numpy(dbFeat_pc_FeatureMap[ind_db])
                        vlad_db = torch.from_numpy(dbFeat_pc[ind_db])
                        save_enc_path = save_dir_fm  + 'enc' + str(i)
                        save_vlad_path = save_dir_fm + 'vlad' + str(i)
                        save_img_my(enc_db.unsqueeze(0).view(1, 1024, 64, 64), save_enc_path)
                        save_img_my(vlad_db.view(16, 16).unsqueeze(0).unsqueeze(0), save_vlad_path)

    recall_at_n = {}
    for test_index in range(4):
        correct_at_n = np.zeros(len(n_values))
        # TODO can we do this on the matrix in one go?
        for qIx, pred in enumerate(predictions[test_index]):
            for i, n in enumerate(n_values):
                # if in top N then also in top NN, where NN > N
                if np.any(np.in1d(pred[:n], gt[qIx])):
                    correct_at_n[i:] += 1
                    break
        recall_at_n[test_index] = correct_at_n / len(eval_set.qIdx)
    # 2d->2d
    for i, n in enumerate(n_values):
        tqdm.write("====> 2D->2D/Recall@{}: {:.4f}".format(n, recall_at_n[0][i]))
        if write_tboard:
            writer.add_scalar('2Dto2D/Recall@' + str(n), recall_at_n[0][i], epoch_num)
    # 2d->3d
    all_recalls = {}  # make dict for output
    for i, n in enumerate(n_values):
        all_recalls[n] = recall_at_n[1][i]
        tqdm.write("====> 2D->3D/Recall@{}: {:.4f}".format(n, recall_at_n[1][i]))
        if write_tboard:
            writer.add_scalar('2Dto3D/Recall@' + str(n), recall_at_n[1][i], epoch_num)
    # 3d->2d
    for i, n in enumerate(n_values):
        tqdm.write("====> 3D->2D/Recall@{}: {:.4f}".format(n, recall_at_n[2][i]))
        if write_tboard:
            writer.add_scalar('3Dto2D/Recall@' + str(n), recall_at_n[2][i], epoch_num)
    # 3d->3d
    for i, n in enumerate(n_values):
        tqdm.write("====> 3D->3D/Recall@{}: {:.4f}".format(n, recall_at_n[3][i]))
        if write_tboard:
            writer.add_scalar('3Dto3D/Recall@' + str(n), recall_at_n[3][i], epoch_num)

    return all_recalls
