import torch
import numpy as np
import numba as nb
from tqdm import tqdm
from torch.utils.data import DataLoader
from ir_model import BaseIRModel
from ir_dataset import NUSWideHashDataset, COCOHashDataset, Flickr25kHashDataset, test_transform, Cifar10

@nb.njit('int32[:,::1](float32[:,::1])', parallel=True)
def _argsort(a):
    b = np.empty(a.shape, dtype=np.int32)
    for i in nb.prange(a.shape[0]):
        b[i,:] = np.argsort(a[i,:])
    return b

def generate_code(
    model:BaseIRModel,
    db_dataloder: DataLoader, 
    query_dataloader: DataLoader,
    is_code=True
    ):
    db_binary_img = []
    db_label = []
    query_binary_img = []
    query_label = []

    with torch.no_grad():
        for _, img, label in tqdm(query_dataloader):
            img = img.to('cuda:0')
            _, _, _image_reps = model.get_code(img)
            if is_code:
                query_binary_img.extend(torch.sign(_image_reps).cpu().tolist())
            else:
                query_binary_img.extend(_image_reps.cpu().tolist())
            query_label.extend(label.tolist())

        for _, img, label in tqdm(db_dataloder):
            img = img.to('cuda:0')
            _, _, _image_reps = model.get_code(img)
            if is_code:
                db_binary_img.extend(torch.sign(_image_reps).cpu().tolist())
            else:
                db_binary_img.extend(_image_reps.cpu().tolist())
            db_label.extend(label.tolist())

    db_binary_img = np.array(db_binary_img, dtype=np.float32)
    db_label = np.array(db_label, dtype=np.float32)

    query_binary_img = np.array(query_binary_img, dtype=np.float32)
    query_label = np.array(query_label, dtype=np.float32)
    

    return db_binary_img, db_label, query_binary_img, query_label

def map_topk(inner_dot_neg, relevant_mask, topk=None):
    AP = []
    relevant_mask = (relevant_mask>0).astype(np.bool8)
    topkindex = _argsort(inner_dot_neg)[:, :topk].astype(np.int32)
    # topkindex = np.argsort(inner_dot_neg, axis=1)[:, :topk].astype(np.int32)
    relevant_topk_mask = np.take_along_axis(relevant_mask, topkindex, axis=1)
    # relevant_topk_mask = relevant_mask[np.expand_dims(np.arange(topkindex.shape[0]), axis=-1), topkindex]
    cumsum = np.cumsum(relevant_topk_mask, axis=1)
    precision = cumsum / np.arange(1, topkindex.shape[1]+1)
    for query in range(relevant_mask.shape[0]):
        if np.sum(relevant_topk_mask[query]) == 0:
            continue
        AP.append(np.sum(precision[query]*relevant_topk_mask[query]) / np.sum(relevant_topk_mask[query]))
    return float(np.mean(AP))

def DCG(rel, dist, topk=None):
    '''
    input: rel, N x M relevance matrix
           dist, N x M distance matrix
           topk, default all result
    return: Discounted Cumulative Gain@topk sorted by distance
    '''
    rank_index = _argsort(dist)[:, :topk]
    rel_rank = np.take_along_axis(rel, rank_index, axis=1)
    return np.mean(np.sum(np.divide(np.power(2, rel_rank) - 1, np.log2(np.arange(rel_rank.shape[1], dtype=np.float32) + 2)), axis=1))


def NDCG(rel, dist, topk=None):
    dcg = DCG(rel, dist, topk)
    idcg = DCG(rel, -rel, topk)

    if dcg == 0.0:
        return 0.0
    ndcg = dcg / idcg
    return float(ndcg)


def map_test(model, args):
    print('computing map for retrieval...')
    model.eval()
    if args.dataset == 'coco2017':
        query_dataset = COCOHashDataset(test_transform, 'query')
        db_dataset = COCOHashDataset(test_transform, 'db')
    elif args.dataset == 'flickr':
        query_dataset = Flickr25kHashDataset(test_transform, 'query')
        db_dataset = Flickr25kHashDataset(test_transform, 'db')
    elif args.dataset == 'nuswide':
        query_dataset = NUSWideHashDataset(test_transform, 'query')
        db_dataset = NUSWideHashDataset(test_transform, 'db')
    else:
        raise NotImplementedError
    query_dataloader = DataLoader(query_dataset, batch_size=32, shuffle=False, num_workers=8)
    db_dataloader = DataLoader(db_dataset, batch_size=32, shuffle=False, num_workers=8)
    db_binary_img, db_label, query_binary_img, query_label \
        = generate_code(model, db_dataloader, query_dataloader, args.iscode)
    inner_dot_neg_i2i = -np.dot(query_binary_img, db_binary_img.T)
    relevant_mask = np.dot(query_label, db_label.T)
    print(relevant_mask.shape)
    map = map_topk(inner_dot_neg_i2i, relevant_mask, 5000)
    ngcg = NDCG(relevant_mask, inner_dot_neg_i2i, 5000)
    model.train()
    return {'map': map, 'ndcg': ngcg}

if __name__ == "__main__":
    pass