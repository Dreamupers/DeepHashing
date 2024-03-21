# Dataset

## MIRFlickr25k

[MIRFLICKR Download (liacs.nl)](https://press.liacs.nl/mirflickr/mirdownload.html)

**24 class, 24581 images**

```
['animals', 'baby', 'bird', 'car', 'clouds',
'dog', 'female', 'flower', 'food', 'indoor',
'lake', 'male', 'night', 'people', 'plant_life',
'portrait', 'river', 'sea', 'sky', 'structures',
'sunset', 'transport', 'tree', 'water']
```

**5000 for train, 2000 for query, and the rest for database**

## MS COCO2017

[COCO - Common Objects in Context (cocodataset.org)](https://cocodataset.org/#download)

**the most frequent 20 class, 92306 images**

```
['person', 'chair', 'car', 'dining table', 'cup',
'bottle', 'bowl', 'handbag', 'truck', 'bench',
'backpack', 'book', 'cell phone', 'sink', 'clock',
'tv', 'potted plant', 'couch', 'dog', 'knife']
```

**80 class, 117266 images**

```
['person', 'bicycle', 'car', 'motorcycle', 'airplane',
'bus', 'train', 'truck', 'boat', 'traffic light',
'fire hydrant', 'stop sign', 'parking meter', 'bench',
'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant',
'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
'sports ball', 'kite', 'baseball bat', 'baseball glove',
'skateboard', 'surfboard', 'tennis racket', 'bottle',
'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
'banana', 'apple', 'sandwich', 'orange', 'broccoli',
'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
'couch', 'potted plant', 'bed', 'dining table', 'toilet',
'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
'toothbrush']
```

**5000 for train, 2000 for query, and the rest for database**

## NUSWIDE

[NUS-WIDE-128 – 南京理工大学智能媒体分析实验室 (imag-njust.net)](https://imag-njust.net/nus-wide-128/)

**the most frequent 21 class, 195834 images**

```
['sky', 'clouds', 'person', 'water', 'animal', 'grass',
'buildings', 'window', 'plants', 'lake', 'ocean', 'road',
'flowers', 'sunset', 'reflection', 'rocks', 'vehicle', 'snow',
'tree', 'beach', 'mountain']
```

**81 class, 209347 images**

```
['airport', 'animal', 'beach', 'bear', 'birds', 'boats', 'book',
 'bridge', 'buildings', 'cars', 'castle', 'cat', 'cityscape', 'clouds',
 'computer', 'coral', 'cow', 'dancing', 'dog', 'earthquake', 'elk',
 'fire', 'fish', 'flags', 'flowers', 'food', 'fox', 'frost',
 'garden', 'glacier', 'grass', 'harbor', 'horses', 'house', 'lake',
 'leaf', 'map', 'military', 'moon', 'mountain', 'nighttime', 'ocean',
 'person', 'plane', 'plants', 'police', 'protest', 'railroad',
 'rainbow', 'reflection', 'road', 'rocks', 'running', 'sand', 'sign',
 'sky', 'snow', 'soccer', 'sports', 'statue',
 'street', 'sun', 'sunset', 'surf', 'swimmers', 'tattoo',
 'temple', 'tiger', 'tower', 'town', 'toy', 'train', 'tree',
 'valley', 'vehicle', 'water', 'waterfall', 'wedding', 'whales',
 'window', 'zebra']
```

**10000 for train, 5000 for query, and the rest for database**

## Note

`data\hash_split_for_coco.json`

`data\hash_split_for_flickr25k.json`

`data\hash_split_for_nuswide.json`

the dataset split is for flickr 24 class, coco 20 class, and nuswide 21 class, used in our paper. look for `ir_dataset.py` for details.
