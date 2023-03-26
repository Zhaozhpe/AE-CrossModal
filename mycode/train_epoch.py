import torch
from tqdm.auto import trange, tqdm
from torch.utils.data import DataLoader
from crossmodal.training_tools.tools import humanbytes
from mycode.msls import MSLS
import numpy as np
import torch.nn as nn

pdist = nn.PairwiseDistance(p=2)
debug = False

def train_epoch(train_dataset, model, model3d, optimizer, optimizer3d, criterion, encoder_dim, device, epoch_num, opt, config, writer):
    if device.type == 'cuda':
        cuda = True
    else:
        cuda = False
    train_dataset.new_epoch()

    epoch_loss = 0
    # epoch_loss_je = 0
    startIter = 1  # keep track of batch iter across subsets for logging

    nBatches = (len(train_dataset.qIdx) + int(config['train']['batchsize']) - 1) // int(config['train']['batchsize'])

    for subIter in trange(train_dataset.nCacheSubset, desc='Cache refresh'.rjust(15), position=1):
        '''pool_size = encoder_dim
        if config['global_params']['pooling'].lower() == 'netvlad':
            pool_size *= int(config['global_params']['num_clusters'])'''
        pool_size = 256

        tqdm.write('====> Building Cache')
# original code
        '''train_dataset.update_subcache(model, pool_size)'''
# mycode
        train_dataset.update_subcache(net=None, net3d=None, outputdim=pool_size)
        training_data_loader = DataLoader(dataset=train_dataset, num_workers=opt.threads,
                                          batch_size=int(config['train']['batchsize']), shuffle=True,
                                          collate_fn=MSLS.collate_fn, pin_memory=cuda) #

        tqdm.write('Allocated: ' + humanbytes(torch.cuda.memory_allocated()))
        tqdm.write('Cached:    ' + humanbytes(torch.cuda.memory_reserved()))
        # bs = int(config['train']['batchsize'])

        model.train()
        model3d.train()
        accum_steps = 64
        for iteration, (query, query_pc, positives, positives_pc, negatives, negatives_pcs, negCounts, indices) in \
                enumerate(tqdm(training_data_loader, position=2, leave=False, desc='Train Iter'.rjust(15)), startIter):
            # some reshaping to put query, pos, negs in a single (N, 3, H, W) tensor
            # where N = batchSize * (nQuery + nPos + nNeg)
            if query is None:
                continue  # in case we get an empty batch
# original code
#             B, C, H, W = query.shape
            B = query.shape[0]
            if debug:
                batch1 = {}
                batch1['query'] = train_dataset.qImages[indices[0]][-14:]
                batch1['positive'] = train_dataset.dbImages[indices[1]][-14:]
                batch1['negatives'] = [train_dataset.dbImages[indices[i]][-14:] for i in range(2,7)]
                print('batch1')
                print(batch1)
                # print('batch2')
                # print(batch2)
            nNeg = torch.sum(negCounts)
            data2d_input = torch.cat([query, positives, negatives])

            data2d_input = data2d_input.to(device)
            image_encoding = model.encoder(data2d_input)

            vlad2d_encoding = model.pool(image_encoding)

            vladQ, vladP, vladN = torch.split(vlad2d_encoding, [B, B, B*5])

            query_pc = query_pc.float()
            positives_pc = positives_pc.float()
            negatives_pcs = negatives_pcs.float()
            # positives_tensor = torch.from_numpy(np.ndarray(positives)).float()
            # negatives_tensor = torch.from_numpy(np.ndarray(negatives)).float()
            feed_tensor = torch.cat(
                (query_pc, positives_pc, negatives_pcs), 1)
            feed_tensor = feed_tensor.view((-1, 1, 4096, 3))
            feed_tensor.requires_grad_(True)
#point process
            feed_tensor = feed_tensor.to(device)
            output = model3d(feed_tensor)
            # print('output.size')
            # print(output.shape)
            output = output.view(B , -1, 256)  # int(config['train']['batchsize'])
            output_query, output_positives, output_negatives = torch.split(
                output, [1, 1, 5], dim=1)
            output_query = output_query.view(-1, 256)
            output_positives = output_positives.view(-1, 256)
            output_negatives = output_negatives.contiguous().view(-1, 256)

            '''optimizer.zero_grad()
            optimizer3d.zero_grad()'''

            # calculate loss for each Query, Positive, Negative triplet
            # due to potential difference in number of negatives have to
            # do it per query, per negative
            loss_dic = {}
            loss_recode = {}
            loss_je = 0
            loss_2dto3d = 0
            loss_3dto2d = 0
            loss_2dto2d = 0
            loss_3dto3d = 0
            for i, negCount in enumerate(negCounts):
                loss_je += pdist(vladQ[i: i + 1] , output_query[i: i + 1])
                for n in range(negCount):
                    negIx = (torch.sum(negCounts[:i]) + n).item()
                    loss_2dto3d += criterion(vladQ[i: i + 1], output_positives[i: i + 1], output_negatives[negIx:negIx + 1])
                    loss_3dto2d += criterion(output_query[i: i + 1], vladP[i: i + 1], vladN[negIx:negIx + 1])
                    loss_2dto2d += criterion(vladQ[i: i + 1], vladP[i: i + 1], vladN[negIx:negIx + 1])
                    loss_3dto3d += criterion(output_query[i: i + 1], output_positives[i: i + 1], output_negatives[negIx:negIx + 1])
                    if debug:
                        loss_recode['2dto3d'] = loss_2dto3d.data
                        loss_recode['3dto2d'] = loss_3dto2d.data
                        loss_recode['2dto2d'] = loss_2dto2d.data
                        loss_recode['3dto3d'] = loss_3dto3d.data
                        print('loss_recode:')
                        print(loss_recode)
                        print('')
            loss_sm = loss_2dto2d + loss_3dto3d
            loss_cm = loss_2dto3d + loss_3dto2d
            loss = 0.1 * loss_sm + loss_cm + loss_je
            if debug:
                loss_dic['je'] = loss_je.data
                loss_dic['cm'] = loss_cm.data
                loss_dic['sm'] = loss_sm.data
                print('loss_dic')
                print(loss_dic)
            loss /= nNeg.float().to(device)  # normalise by actual number of negatives
            loss_je_t = loss_je / nNeg.float().to(device) # normalise by actual number of negatives negCounts
            loss_cm_t = loss_cm / nNeg.float().to(device)

            loss = loss / accum_steps
            loss.backward()
            if (iteration + 1) % accum_steps == 0 or (iteration + 1) == len(training_data_loader):
                optimizer.step()
                optimizer.zero_grad()
                optimizer3d.step()
                optimizer3d.zero_grad()
#original code
            # del data_input, image_encoding, vlad_encoding, vladQ, vladP, vladN
            # del query, positives, negatives
# my code
            del data2d_input, feed_tensor, output, image_encoding, vlad2d_encoding, vladQ, vladP, vladN, output_query, output_positives, output_negatives
            # del attention
            del query, query_pc, positives, positives_pc, negatives, negatives_pcs

            batch_loss = loss.item() * accum_steps
            epoch_loss += batch_loss
            batch_loss_je = loss_je_t.item()
            # epoch_loss_je += batch_loss_je
            batch_loss_cm = loss_cm_t.item()

            # check model
            # for name, param in model3d.named_parameters():
            #     print(name)
            # for name, param in model3d.named_parameters():
            #     if param.requires_grad == True:
            #         print(name, param)
            # for name, param in model3d.named_parameters():
            #     print(name, param)

            if iteration % 100 == 0 or nBatches <= 10:
                tqdm.write("==> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch_num, iteration,
                                                                       nBatches, batch_loss))
                writer.add_scalar('Train/Loss', batch_loss,
                                  ((epoch_num - 1) * nBatches) + iteration)
                writer.add_scalar('Train/Loss_je', batch_loss_je,
                                  ((epoch_num - 1) * nBatches) + iteration)
                writer.add_scalar('Train/Loss_cm', batch_loss_cm,
                                  ((epoch_num - 1) * nBatches) + iteration)
                writer.add_scalar('Train/nNeg', nNeg,
                                  ((epoch_num - 1) * nBatches) + iteration)
                tqdm.write('Allocated: ' + humanbytes(torch.cuda.memory_allocated()))
                tqdm.write('Cached:    ' + humanbytes(torch.cuda.memory_reserved()))

        startIter += len(training_data_loader)
        del training_data_loader, loss
        optimizer.zero_grad()
        optimizer3d.zero_grad()
        torch.cuda.empty_cache()

    avg_loss = epoch_loss / nBatches

    tqdm.write("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch_num, avg_loss))
    writer.add_scalar('Train/AvgLoss', avg_loss, epoch_num)
