from torchvision import transforms
from PIL import Image
import numpy as np
import random
from torch.utils.data import Dataset
import os
import json

DATASETDIR = '/home/zx/dataset/hash/'
WORKDIR = '/home/zx/deephash/'
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Resize(224),
    transforms.RandomCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
test_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


class BaseHashDataset(Dataset):
    def __init__(self, label_file, img_root, img_transform, split_list, split, sample_size) -> None:
        super().__init__()
        self.img_transform = img_transform
        self.img_root = img_root
        img_labels = open(label_file, 'r').readlines()
        self.img_label_list = [(
                os.path.join(img_root, val.split()[0]), 
                np.array([int(la) for la in val.split()[1:]])
                ) for val in img_labels]
        if sample_size is None:
            self.global_index = np.array(split_list[split])
        else:
            _size = sum([len(split_list[key]) for key in ['query', 'db']])
            sample_index = random.sample(list(range(_size)), sample_size)
            self.global_index = np.array(sample_index)

    def get_image(self, imgpath):
        with open(imgpath, 'rb') as imgf:
            img = Image.open(imgf).convert('RGB')
        return img

    def __getitem__(self, index):
        img_path, label = self.img_label_list[self.global_index[index]]
        image =self.get_image(img_path)
        if self.img_transform is not None:
            image = self.img_transform(image)
        return index, image, label
    
    def __len__(self):
        return len(self.global_index)
    
class COCOHashDataset(BaseHashDataset):
    def __init__(
        self,
        img_transform, split,
        sample_size = None,
        label_file= WORKDIR + 'data/coco/allannots20.txt', 
        img_root=DATASETDIR+'coco2017'
        ):
        assert split in ['train', 'db', 'query']
        if os.path.exists(WORKDIR + 'data/hash_split_for_coco.json'):
            split_list = json.load(open(WORKDIR + 'data/hash_split_for_coco.json', 'r'))
        else:
            all_index = list(range(len(open(label_file, 'r').readlines())))
            query_index = random.sample(all_index, 2000)
            train_index = random.sample(list(set(all_index)-set(query_index)), 5000)
            db_index = list(set(all_index)-set(query_index))
            split_list = {
                'train': train_index,
                'db': db_index,
                'query': query_index
            }
            json.dump(split_list, open(WORKDIR + 'data/hash_split_for_coco.json', 'w'))
        super().__init__(label_file, img_root, img_transform, split_list, split, sample_size)

class Flickr25kHashDataset(BaseHashDataset):
    def __init__(
        self,
        img_transform, split,
        sample_size = None,
        label_file=WORKDIR + 'data/flickr25k/allannots.txt', 
        img_root=DATASETDIR+'mirflickr'
        ):
        assert split in ['train', 'db', 'query']
        if os.path.exists(WORKDIR + 'data/hash_split_for_flickr25k.json'):
            split_list = json.load(open(WORKDIR + 'data/hash_split_for_flickr25k.json', 'r'))
        else:
            all_index = list(range(len(open(label_file, 'r').readlines())))
            query_index = random.sample(all_index, 2000)
            train_index = random.sample(list(set(all_index)-set(query_index)), 5000)
            db_index = list(set(all_index)-set(query_index))
            split_list = {
                'train': train_index,
                'db': db_index,
                'query': query_index
            }
            json.dump(split_list, open(WORKDIR + 'data/hash_split_for_flickr25k.json', 'w'))
        super().__init__(label_file, img_root, img_transform, split_list, split, sample_size)

class NUSWideHashDataset(BaseHashDataset):
    def __init__(
        self,
        img_transform, split,
        sample_size = None,
        label_file=WORKDIR + 'data/nuswide/allannots21.txt',
        img_root=DATASETDIR+'nuswide'
        ):
        assert split in ['train', 'db', 'query']
        if os.path.exists(WORKDIR + 'data/hash_split_for_nuswide.json'):
            split_list = json.load(open(WORKDIR + 'data/hash_split_for_nuswide.json', 'r'))
        else:
            all_index = list(range(len(open(label_file, 'r').readlines())))
            query_index = random.sample(all_index, 5000)
            train_index = random.sample(list(set(all_index)-set(query_index)), 10000)
            db_index = list(set(all_index)-set(query_index))
            split_list = {
                'train': train_index,
                'db': db_index,
                'query': query_index
            }
            json.dump(split_list, open(WORKDIR + 'data/hash_split_for_nuswide.json', 'w'))
        super().__init__(label_file, img_root, img_transform, split_list, split, sample_size)