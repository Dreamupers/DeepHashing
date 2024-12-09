import torch
import numpy as np
import numba as nb
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn.functional as F
from ir_model import BaseIRModel
from ir_dataset import NUSWideHashDataset, COCOHashDataset, Flickr25kHashDataset, IMAGENET1K_V1_test_transform

def argsort(x):
    return np.argsort(x, kind="stable").astype(np.int32)

@nb.njit('int32[:,::1](int16[:,::1])', parallel=True)
def _argsort16(a):
    b = np.empty(a.shape, dtype=np.int32)
    for i in nb.prange(a.shape[0]):
        b[i,:] = np.argsort(a[i,:]).astype(np.int32)
    return b

@nb.njit('int32[:,::1](int8[:,::1])', parallel=True)
def _argsort8(a):
    b = np.empty(a.shape, dtype=np.int32)
    for i in nb.prange(a.shape[0]):
        b[i,:] = np.argsort(a[i,:]).astype(np.int32)
    return b

# dot for int16, int8, float16
@nb.njit(parallel=True)
def matrix_multiply(A, B):
    n, m = A.shape
    m2, p = B.shape
    assert m == m2, "A's columns must match B's rows"
    
    C = np.zeros((n, p), dtype=A.dtype)
    for i in nb.prange(n):
        for j in range(p):
            for k in range(m):
                C[i, j] += A[i, k] * B[k, j]
    return C

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
        for batch_dict in tqdm(query_dataloader):
            img, label = batch_dict["image"], batch_dict["label"]
            img = img.to('cuda:0')
            _, h, _image_reps = model.get_code(img)
            if is_code:
                query_binary_img.append(torch.sign(_image_reps).cpu().numpy().astype(np.int16))
            else:
                query_binary_img.append(h.cpu().numpy().astype(np.int16))
            query_label.append(label.numpy().astype(np.int8))

        for batch_dict in tqdm(db_dataloder):
            img, label = batch_dict["image"], batch_dict["label"]
            img = img.to('cuda:0')
            _, h, _image_reps = model.get_code(img)
            if is_code:
                db_binary_img.append(torch.sign(_image_reps).cpu().numpy().astype(np.int16))
            else:
                db_binary_img.append(h.cpu().numpy().astype(np.int16))
            db_label.append(label.numpy().astype(np.int8))

    db_binary_img = np.concatenate(db_binary_img, axis=0, dtype=np.int16)
    db_label = np.concatenate(db_label, axis=0, dtype=np.int8)
    query_binary_img = np.concatenate(query_binary_img, axis=0, dtype=np.int16)
    query_label = np.concatenate(query_label, axis=0, dtype=np.int8)
    
    return db_binary_img, db_label, query_binary_img, query_label

def ACG(inner_dot_neg, relevant_mask, agrsort_index=None, topk=None):
    if agrsort_index is not None:
        topkindex = agrsort_index[:, :topk]
    else:
        topkindex = _argsort16(inner_dot_neg)[:, :topk]
    relevant_topk_mask = np.take_along_axis(relevant_mask, topkindex, axis=1)
    return float(np.mean(relevant_topk_mask))

def map_topk(inner_dot_neg, relevant_mask, agrsort_index=None, topk=None):
    AP = []
    relevant_mask = (relevant_mask>0).astype(np.bool_)
    if agrsort_index is not None:
        topkindex = agrsort_index[:, :topk]
    else:
        topkindex = _argsort16(inner_dot_neg)[:, :topk]
    # topkindex = np.argsort(inner_dot_neg, axis=1)[:, :topk].astype(np.int32)
    relevant_topk_mask = np.take_along_axis(relevant_mask, topkindex, axis=1)
    # relevant_topk_mask = relevant_mask[np.expand_dims(np.arange(topkindex.shape[0]), axis=-1), topkindex]
    cumsum = np.cumsum(relevant_topk_mask, axis=1)
    precision = cumsum / np.arange(1, topkindex.shape[1]+1)
    for query in range(relevant_mask.shape[0]):
        if np.sum(relevant_topk_mask[query]) == 0:
            AP.append(np.float32(0))
            # print("nothing")
        else:
            AP.append(np.sum(precision[query]*relevant_topk_mask[query]) / np.sum(relevant_topk_mask[query]))
    return float(np.mean(AP))

def DCG(rel, dist, agrsort_index=None, topk=None):
    '''
    input: rel, N x M relevance matrix
           dist, N x M distance matrix
           topk, default all result
    return: Discounted Cumulative Gain@topk sorted by distance
    '''
    if agrsort_index is not None:
        rank_index = agrsort_index[:, :topk]
    else:
        # rank_index = np.array(Parallel(n_jobs=15, prefer='threads')(delayed(argsort)(dist[i]) for i in range(dist.shape[0])), dtype=np.int32)[:, :topk]
        rank_index = _argsort8(dist)[:, :topk]
    rel_rank = np.take_along_axis(rel, rank_index, axis=1)
    return np.mean(np.sum(np.divide(np.power(2, rel_rank) - 1, np.log2(np.arange(rel_rank.shape[1], dtype=np.float32) + 2)), axis=1))


def NDCG(rel, dist, agrsort_index=None, idcg_index=None, topk=None):
    dcg = DCG(rel, dist, agrsort_index=agrsort_index, topk=topk)
    idcg = DCG(rel, -rel, agrsort_index=idcg_index, topk=topk)

    if dcg == 0.0:
        return 0.0
    ndcg = dcg / idcg
    return float(ndcg)


def map_test(model, args):
    # print('computing map for retrieval...')
    model.eval()

    if args.dataset == 'coco':
        query_dataset = COCOHashDataset(IMAGENET1K_V1_test_transform, 'query')
        db_dataset = COCOHashDataset(IMAGENET1K_V1_test_transform, 'db')
    elif args.dataset == 'flickr25k':
        query_dataset = Flickr25kHashDataset(IMAGENET1K_V1_test_transform, 'query')
        db_dataset = Flickr25kHashDataset(IMAGENET1K_V1_test_transform, 'db')
    elif args.dataset == 'nuswide':
        query_dataset = NUSWideHashDataset(IMAGENET1K_V1_test_transform, 'query')
        db_dataset = NUSWideHashDataset(IMAGENET1K_V1_test_transform, 'db')
    else:
        raise NotImplementedError
    
    query_dataloader = DataLoader(query_dataset, batch_size=128, shuffle=False, num_workers=16)
    db_dataloader = DataLoader(db_dataset, batch_size=128, shuffle=False, num_workers=16)
    db_binary_img, db_label, query_binary_img, query_label \
        = generate_code(model, db_dataloader, query_dataloader, args.iscode)
    inner_dot_neg_i2i = -matrix_multiply(query_binary_img, db_binary_img.T) 
    relevant_mask = matrix_multiply(query_label, db_label.T)
    agrsort_index = _argsort16(inner_dot_neg_i2i) 
    idcg_agrsort_index = _argsort8(-relevant_mask) 
    
    # print("parallel computing done")
    map = map_topk(inner_dot_neg_i2i, relevant_mask, agrsort_index, 1000)
    ngcg = NDCG(relevant_mask, inner_dot_neg_i2i, agrsort_index, idcg_agrsort_index, 1000)
    acg_1000 = ACG(inner_dot_neg_i2i, relevant_mask, agrsort_index, 1000)
    acg_100 = ACG(inner_dot_neg_i2i, relevant_mask, agrsort_index, 100)
    del inner_dot_neg_i2i
    del relevant_mask
    del agrsort_index
    del idcg_agrsort_index
    model.train()
    return {'map': round(map*100, 2), 'ndcg': round(ngcg*100, 2), 'acg_1000': round(acg_1000, 3), 'acg_100': round(acg_100, 3)}