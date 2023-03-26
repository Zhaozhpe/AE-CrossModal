import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch.utils.data as data
import pandas as pd
from os.path import join
from sklearn.neighbors import NearestNeighbors
import math
import torch
import random
import sys
import itertools
from tqdm import tqdm
import re
from mycode.loading_pointclouds import load_pc_file, load_pc_files
import random

path_to_3d = "/data/kitti360_pc"
default_cities = {
    'train': ["s00", "s02", "s04", "s05", "s06", "s07" , "s09", "s10"],
    'val': ["s00"],
    'test': []
} #
#
# record train1:'train': ["s00","s02"],
#   'val': ["s06"],
#    'test': ["s03"]

# , "s02", "s04", "s07", "s09", "s10"
class ImagesFromList(Dataset):
    def __init__(self, images, transform):
        self.images_list = np.asarray(images)
        self.transform = transform

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        #try:
        # img = [Image.open(im) for im in self.images[idx].split(",")]
        img_dir = self.images_list[idx]
        img = Image.open(img_dir)
        '''except:
            img = [Image.open(self.images[0])]'''
        # img = [self.transform(im) for im in img]
        img = self.transform(img)

        '''if len(img) == 1:
            img = img[0]'''

        return img, idx

class PcFromFiles(Dataset):
    def __init__(self, pcs):
        self.pcs_list = np.asarray(pcs)

    def __len__(self):
        return len(self.pcs_list)

    def __getitem__(self, idx):
        #try:
            # img = [Image.open(im) for im in self.images[idx].split(",")]
        # pc = load_pc_files([self.pcs[idx]])
        pc = load_pc_file(self.pcs_list[idx])
        '''except:
            # img = [Image.open(self.images[0])]
            pc = load_pc_files([self.pcs[0]])'''
        # img = [self.transform(im) for im in img]

        '''if len(pc) == 1:
            pc = pc[0]'''

        return pc, idx

class MSLS(Dataset):
    def __init__(self, root_dir, cities='', nNeg=5, transform=None, mode='train', task='im2im', subtask='all',
                 posDistThr=20, negDistThr=40, cached_queries=4000, cached_negatives=25000,
                 positive_sampling=True, bs=60, threads=8, margin=0.1):  #exclude_panos=True cached_queries=1000 cached_negatives=1000 posDistThr=10 negDistThr=25

        # initializing
        assert mode in ('train', 'val', 'test')

        if cities in default_cities:
            self.cities = default_cities[cities]
        elif cities == '':
            self.cities = default_cities[mode]
        else:
            self.cities = cities.split(',')

        self.qIdx = []
        # self.qIdx_pc = []

        self.qImages = []
        self.qPcs = []

        self.dbImages = []
        self.dbPcs = []

        self.pIdx = []
        self.nonNegIdx = []

        self.sideways = []
        self.night = []
        self.qEndPosList = []
        self.dbEndPosList = []

        self.all_pos_indices = []

        # hyper-parameters
        self.nNeg = nNeg
        self.margin = margin
        self.posDistThr = posDistThr
        self.negDistThr = negDistThr
        self.cached_queries = cached_queries
        self.cached_negatives = cached_negatives

        # flags
        self.cache = None
        self.mode = mode
        self.subtask = subtask

        # other
        self.transform = transform

        # load data
        for city in self.cities:
            print("=====> {}".format(city))

            subdir = 'train_val'
            # subdir_3d =

            # get len of images from cities so far for indexing
            _lenQ = len(self.qImages)
            _lenDb = len(self.dbImages)

            # when GPS / UTM is available
            if self.mode in ['train', 'val']:
                # city = city
                # if self.mode == 'val':
                #     city_2d = 's05'
                # load query data
                if self.mode == 'val':
                    qData = pd.read_csv(join(root_dir, subdir, city, 'database', 'test', 'query.csv'), index_col=0)
                    # qData_pc = pd.read_csv(join(path_to_3d, subdir_3d, city, 'database', 'postprocessed.csv'), index_col=0)
                    # qDataRaw = pd.read_csv(join(root_dir, subdir, city, 'query', 'raw.csv'), index_col=0)

                    # load database data
                    dbData = pd.read_csv(join(root_dir, subdir, city, 'database', 'test', 'database.csv'), index_col=0)
                else:
                    qData = pd.read_csv(join(root_dir, subdir, city, 'database', 'query.csv'), index_col=0)
                    # qData_pc = pd.read_csv(join(path_to_3d, subdir_3d, city, 'database', 'postprocessed.csv'), index_col=0)
                    # qDataRaw = pd.read_csv(join(root_dir, subdir, city, 'query', 'raw.csv'), index_col=0)

                    # load database data
                    dbData = pd.read_csv(join(root_dir, subdir, city, 'database', 'database.csv'), index_col=0)
                    # dbData_pc = pd.read_csv(join(path_to_3d, subdir_3d, city, 'database', 'postprocessed.csv'), index_col=0)
                    # dbDataRaw = pd.read_csv(join(root_dir, subdir, city, 'database', 'raw.csv'), index_col=0)
                    # print("dbData:")
                    # print(dbData)

                # arange based on task
                # qSeqKeys, qSeqIdxs = self.arange_as_seq(qData, join(root_dir, subdir, city, 'query'),  False)
                # qSeqKeys_pc, qSeqIdxs_pc = self.arange_as_seq(qData_pc, join(path_to_3d, subdir_3d, city, 'query'), False)
                qSeqKeys, qSeqIdxs, qSeqKeys_pc, qSeqIdxs_pc = \
                    self.arange_as_seq(qData, join(root_dir, subdir, city, 'database'),join(path_to_3d, city))

                dbSeqKeys, dbSeqIdxs, dbSeqKeys_pc, dbSeqIdxs_pc= \
                    self.arange_as_seq(dbData, join(root_dir, subdir, city, 'database'),join(path_to_3d, city))


                # if there are no query/dabase images,
                # then continue to next city
                if len(qSeqIdxs) == 0 or len(dbSeqIdxs) == 0:
                    continue

                self.qImages.extend(qSeqKeys)
                self.qPcs.extend(qSeqKeys_pc)
                
                self.dbImages.extend(dbSeqKeys)
                self.dbPcs.extend(dbSeqKeys_pc)

                self.qEndPosList.append(len(qSeqKeys))
                self.dbEndPosList.append(len(dbSeqKeys))
                print('self.qEndPosList:---------------')
                print(self.qEndPosList)
                print('self.dbEndPosList:---------------')
                print(self.dbEndPosList)

                # utm coordinates
                utmQ = qData[['easting', 'northing']].values.reshape(-1, 2)
                utmDb = dbData[['easting', 'northing']].values.reshape(-1, 2)


                # find positive images for training
                neigh = NearestNeighbors(algorithm='brute')
                neigh.fit(utmDb)
                pos_distances, pos_indices = neigh.radius_neighbors(utmQ, self.posDistThr)

                self.all_pos_indices.extend(pos_indices)

                if self.mode == 'train':
                    nD, nI = neigh.radius_neighbors(utmQ, self.negDistThr)

                for q_seq_idx in range(len(qSeqKeys)):

                    q_frame_idxs = q_seq_idx
                    q_uniq_frame_idx = q_frame_idxs

                    p_uniq_frame_idxs = pos_indices[q_uniq_frame_idx]

                    # the query image has at least one positive
                    if len(p_uniq_frame_idxs) > 0:
                        p_seq_idx = np.unique(dbSeqIdxs[p_uniq_frame_idxs])
                        self.qIdx.append(q_seq_idx + _lenQ)
                        self.pIdx.append(p_seq_idx + _lenDb)

                        # in training we have two thresholds, one for finding positives and one for finding images
                        # that we are certain are negatives.
                        if self.mode == 'train':

                            # n_uniq_frame_idxs = [n for nonNeg in nI[q_uniq_frame_idx] for n in nonNeg]
                            n_uniq_frame_idxs = nI[q_uniq_frame_idx]
                            n_seq_idx = np.unique(dbSeqIdxs[n_uniq_frame_idxs])


                            self.nonNegIdx.append(n_seq_idx + _lenDb)

            elif self.mode in ['test']:
                qData = pd.read_csv(join(root_dir, subdir, city, 'database', 'query.csv'), index_col=0)
                # load database data
                dbData = pd.read_csv(join(root_dir, subdir, city, 'database', 'database.csv'), index_col=0)

                qSeqKeys, qSeqIdxs, qSeqKeys_pc, qSeqIdxs_pc = \
                    self.arange_as_seq(qData, join(root_dir, subdir, city, 'database'),
                                       join(path_to_3d, city))

                dbSeqKeys, dbSeqIdxs, dbSeqKeys_pc, dbSeqIdxs_pc = \
                    self.arange_as_seq(dbData, join(root_dir, subdir, city, 'database'),
                                       join(path_to_3d, city))

                self.qImages.extend(qSeqKeys)
                self.dbImages.extend(dbSeqKeys)

                # add query index
                # self.qIdx.extend(list(range(_lenQ, len(qSeqKeys) + _lenQ)))
        if len(self.qImages) == 0 or len(self.dbImages) == 0:
            print("Exiting...")
            print("there are no query/database images.")
            print("Try more sequence")
            sys.exit()

        # cast to np.arrays for indexing during training
        self.qIdx = np.asarray(self.qIdx)
        self.pIdx = np.asarray(self.pIdx)
        self.nonNegIdx = np.asarray(self.nonNegIdx)

        self.qImages = np.asarray(self.qImages)
        self.qPcs = np.asarray(self.qPcs)
        self.dbImages = np.asarray(self.dbImages)
        self.dbPcs = np.asarray(self.dbPcs)

        self.device = torch.device("cuda")
        self.threads = threads
        self.bs = bs

        print('self.qImages')
        print(self.qImages)
        print('self.dbImages')
        print(self.dbImages)
        print('self.qPcs')
        print(self.qPcs)
        print('self.qIdx')
        print(self.qIdx)
        print('self.pIdx')
        print(self.pIdx)
        print('len(self.qIdx)')
        print(len(self.qIdx))
        print('len(self.pIdx)')
        print(len(self.pIdx))

    @staticmethod
    def arange_as_seq(data, path, path_pc):
        
        seq_keys, seq_idxs = [], []
        seq_keys_pc, seq_idxs_pc = [], []
        # pc_idx = 0
        # print('data_pc:')
        # print(list(data_pc['key']))
        for idx in data.index:
            # find surrounding frames in sequence
            seq_idx = idx
            seq = data.iloc[seq_idx]
            img_num = int(re.sub('[a-z]', '', seq['key']))
            # if img_num not in list(data_pc['key']):
            #     continue
            # row_index = data_pc[data_pc.key == img_num].index.tolist()[0]
            # print(row_index)
            # seq_key = ','.join([join(path, 'images', seq['key'] + '.png')])
            seq_key = join(path, 'images', seq['key'] + '.png')
            # seq_key_pc = ','.join([join(path_pc, 'pc', '%010d' % img_num + '.bin')])
            seq_key_pc = join(path_pc, 'pc', '%010d' % img_num + '.bin')

            seq_keys.append(seq_key)
            seq_keys_pc.append(seq_key_pc)
            seq_idxs.append([seq_idx])
            seq_idxs_pc.append([seq_idx])

        return seq_keys, np.asarray(seq_idxs), seq_keys_pc, np.asarray(seq_idxs_pc)


    def __len__(self):
        return len(self.triplets)

    def new_epoch(self):

        # find how many subset we need to do 1 epoch
        self.nCacheSubset = math.ceil(len(self.qIdx) / self.cached_queries)

        # get all indices
        # arr = np.arange(len(self.qIdx))
        arr = list(range(len(self.qIdx)))

        random.shuffle(arr)
        arr = np.array(arr)

        # apply positive sampling of indices
        # arr = random.choices(arr, self.weights, k=len(arr))

        # calculate the subcache indices
        self.subcache_indices = np.array_split(arr, self.nCacheSubset)

        # reset subset counter
        self.current_subset = 0

    def update_subcache(self, net=None, net3d=None, outputdim=None):

        # reset triplets
        self.triplets = []

        # if there is no network associate to the cache, then we don't do any hard negative mining.
        # Instead we just create some naive triplets based on distance.
        # take n query images
        if self.current_subset >= len(self.subcache_indices):
            tqdm.write('Reset epoch - FIX THIS LATER!')
            self.current_subset = 0
        qidxs = np.asarray(self.subcache_indices[self.current_subset])


        if net is None and net3d is None:
            # qidxs = np.random.choice(len(self.qIdx), self.cached_queries, replace=False)

            for q in qidxs:

                # get query idx
                qidx = self.qIdx[q]

                # get positives
                pidxs = self.pIdx[q]

                # choose a random positive (within positive range (default 10 m))
                pidx = np.random.choice(pidxs, size=1)[0]

                # get negatives
                while True:
                    nidxs = np.random.choice(len(self.dbImages), size=self.nNeg)
                    # nidxs = np.random.choice(self.nonNegIdx[q], size=self.nNeg)

                    # ensure that non of the choice negative images are within the negative range (default 25 m)
                    if sum(np.in1d(nidxs, self.nonNegIdx[q])) == 0:
                        break

                # package the triplet and target
                triplet = [qidx, pidx, *nidxs]
                target = [-1, 1] + [0] * len(nidxs)
                # if q < 20:
                #     print('triplet:----------------------------------')
                #     print(triplet)

                self.triplets.append((triplet, target))

            # increment subset counter
            self.current_subset += 1

            return
        # take their positive in the database
        pidxs = np.unique([i for idx in self.pIdx[qidxs] for i in np.random.choice(idx, size=5, replace=False)])
        print('pidxs------------------------')
        print(pidxs)
        print('len(pidxs)------------------------')
        print(len(pidxs))
        nidxs = []
        while len(nidxs) < self.cached_queries // 10:
            # take m = 5*cached_queries is number of negative images
            nidxs = np.random.choice(len(self.dbImages), self.cached_negatives, replace=False)

            # and make sure that there is no positives among them
            nidxs = nidxs[np.in1d(nidxs, np.unique([i for idx in self.nonNegIdx[qidxs] for i in idx]), invert=True)]
            # print('nidxs2------------------------')
            # print(nidxs)
            print('len(nidxs2)------------------------')
            print(len(nidxs))

        # make dataloaders for query, positive and negative images
        opt = {'batch_size': 1, 'shuffle': False, 'num_workers': self.threads, 'pin_memory': True} # self.bs
        qloader = torch.utils.data.DataLoader(ImagesFromList(self.qImages[qidxs], transform=self.transform), **opt)
        # ploader = torch.utils.data.DataLoader(ImagesFromList(self.dbImages[pidxs], transform=self.transform), **opt)
        # nloader = torch.utils.data.DataLoader(ImagesFromList(self.dbImages[nidxs], transform=self.transform), **opt)
        ploader_pc = torch.utils.data.DataLoader(PcFromFiles(self.dbPcs[pidxs]), **opt)
        nloader_pc = torch.utils.data.DataLoader(PcFromFiles(self.dbPcs[nidxs]), **opt)

        # calculate their descriptors
        net.eval()
        net3d.eval()
        with torch.no_grad():

            # initialize descriptors
            qvecs = torch.zeros(len(qidxs), outputdim).to(self.device)
            pvecs = torch.zeros(len(pidxs), outputdim).to(self.device)
            nvecs = torch.zeros(len(nidxs), outputdim).to(self.device)

            bs = opt['batch_size']

            # compute descriptors
            for i, batch in tqdm(enumerate(qloader), desc='compute query descriptors', total=len(qidxs) // bs,
                                 position=2, leave=False):
                X, _ = batch
                image_encoding = net.encoder(X.to(self.device))
                vlad_encoding = net.pool(image_encoding)
                qvecs[i * bs:(i + 1) * bs, :] = vlad_encoding
                del batch, X, image_encoding, vlad_encoding
            del qloader
            '''for i, batch in tqdm(enumerate(ploader), desc='compute positive descriptors', total=len(pidxs) // bs,
                                 position=2, leave=False):
                X, y = batch
                image_encoding = net.encoder(X.to(self.device))
                vlad_encoding = net.pool(image_encoding)
                pvecs[i * bs:(i + 1) * bs, :] = vlad_encoding'''

            for i, batch in tqdm(enumerate(ploader_pc), desc='compute positive descriptors', total=len(pidxs) // bs,
                                 position=2, leave=False):
                X, _ = batch
                # X = X.to(self.device)
                X = X.view((-1, 1, 4096, 3))
                vlad_encoding = net3d(X.to(self.device))
                pvecs[i * bs:(i + 1) * bs, :] = vlad_encoding
                del batch, X, vlad_encoding
            del ploader_pc
            for i, batch in tqdm(enumerate(nloader_pc), desc='compute negative descriptors', total=len(nidxs) // bs,
                                 position=2, leave=False):
                X, _ = batch
                X = X.view((-1, 1, 4096, 3))
                vlad_encoding = net3d(X.to(self.device))
                nvecs[i * bs:(i + 1) * bs, :] = vlad_encoding
                del batch, X, vlad_encoding
            del nloader_pc

        tqdm.write('>> Searching for hard negatives...')
        # compute dot product scores and ranks on GPU
        pScores = torch.mm(qvecs, pvecs.t())
        pScores, pRanks = torch.sort(pScores, dim=1, descending=True)

        # calculate distance between query and negatives
        nScores = torch.mm(qvecs, nvecs.t())
        nScores, nRanks = torch.sort(nScores, dim=1, descending=True)

        # convert to cpu and numpy
        pScores, pRanks = pScores.cpu().numpy(), pRanks.cpu().numpy()
        nScores, nRanks = nScores.cpu().numpy(), nRanks.cpu().numpy()

        # selection of hard triplets
        for q in range(len(qidxs)):

            qidx = qidxs[q]

            # find positive idx for this query (cache idx domain)
            cached_pidx = np.where(np.in1d(pidxs, self.pIdx[qidx]))

            # find idx of positive idx in rank matrix (descending cache idx domain)
            pidx = np.where(np.in1d(pRanks[q, :], cached_pidx))

            # take the closest positve
            dPos = pScores[q, pidx][0][0]

            # get distances to all negatives
            dNeg = nScores[q, :]

            # how much are they violating
            loss = dPos - dNeg + self.margin ** 0.5
            violatingNeg = 0 < loss

            # if less than nNeg are violating then skip this query
            if np.sum(violatingNeg) <= self.nNeg:
                continue

            # select hardest negatives
            hardest_negIdx = np.argsort(loss)[:self.nNeg]
            # print('hardest_negIdx:----------------------------------')
            # print(hardest_negIdx)

            # select the hardest negatives
            cached_hardestNeg = nRanks[q, hardest_negIdx]

            # select the closest positive (back to cache idx domain)
            cached_pidx = pRanks[q, pidx][0][0]

            # transform back to original index (back to original idx domain)
            qidx = self.qIdx[qidx]
            pidx = pidxs[cached_pidx]
            hardestNeg = nidxs[cached_hardestNeg]

            # package the triplet and target
            triplet = [qidx, pidx, *hardestNeg]
            if q < 20:
                print('triplet:----------------------------------')
                print(triplet)
            target = [-1, 1] + [0] * len(hardestNeg)

            self.triplets.append((triplet, target))
        del qvecs, nvecs, pScores, pRanks, nScores, nRanks
        # increment subset counter
        self.current_subset += 1

    @staticmethod
    def collate_fn(batch):
        """Creates mini-batch tensors from the list of tuples (query, positive, negatives).

        Args:
            batch: list of tuple (query, positive, negatives).
                - query: torch tensor of shape (3, h, w).
                - positive: torch tensor of shape (3, h, w).
                - negative: torch tensor of shape (n, 3, h, w).
        Returns:
            query: torch tensor of shape (batch_size, 3, h, w).
            positive: torch tensor of shape (batch_size, 3, h, w).
            negatives: torch tensor of shape (batch_size, n, 3, h, w).
        """

        batch = list(filter(lambda x: x is not None, batch))
        if len(batch) == 0:
            return None, None, None, None, None

        query, query_pc, positive, positive_pc, negatives, negatives_pcs, indices = zip(*batch)

        query = data.dataloader.default_collate(query)
        # o code
        positive = data.dataloader.default_collate(positive)

        query_pc = data.dataloader.default_collate(query_pc)
        positive_pc = data.dataloader.default_collate(positive_pc)
        negatives_pcs = data.dataloader.default_collate(negatives_pcs)
        negCounts = data.dataloader.default_collate([x.shape[0] for x in negatives])

        negatives = torch.cat(negatives, 0)
        # negatives = data.dataloader.default_collate(negatives)


        # negatives = torch.cat(negatives, 0)
        indices = list(itertools.chain(*indices))

        return query, query_pc, positive, positive_pc, negatives, negatives_pcs, negCounts, indices

    def __getitem__(self, idx):
        # get triplet
        triplet, target = self.triplets[idx]

        # get query, positive and negative idx
        qidx = triplet[0]
        pidx = triplet[1]
        nidx = triplet[2:]

        # load images into triplet list
        query = self.transform(Image.open(self.qImages[qidx]))
        # query_pc_ind = int(re.sub('[a-z]', '', os.path.basename(self.qImages[qidx])[:-4]))
        # city = self.qImages[qidx].split("/")[-3]
        # path = join(path_to_3d, 'kitti360', city)
        # query_pc_dir = ','.join([join(path, 'pc', '%010d'%query_pc_ind + '.bin')])
        query_pc = load_pc_files([self.qPcs[qidx]])
#original code
        '''positive = self.transform(Image.open(self.dbImages[pidx]))'''
#mycode
        positive = self.transform(Image.open(self.dbImages[pidx]))
        positive_pc = load_pc_files([self.dbPcs[pidx]])
#original code
        '''negatives = [self.transform(Image.open(self.dbImages[idx])) for idx in nidx]'''
# mycode
        negatives = [self.transform(Image.open(self.dbImages[idx])) for idx in nidx]
        negatives_pcs = load_pc_files([self.dbPcs[idx] for idx in nidx])
        # negatives = torch.from_numpy(negatives)
        negatives = torch.stack(negatives, 0)

        return query, query_pc, positive, positive_pc, negatives, negatives_pcs, [qidx, pidx] + nidx
