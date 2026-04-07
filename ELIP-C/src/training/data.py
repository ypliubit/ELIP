import ast
import json
import logging
import math
import os
import random
import sys
import braceexpand
from dataclasses import dataclass
from multiprocessing import Value

import numpy as np
import pandas as pd
import torch
import torchvision.datasets as datasets
import webdataset as wds
from PIL import Image
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, IterableDataset, get_worker_info
from torch.utils.data.distributed import DistributedSampler, DistributedHarderSampler
from webdataset.filters import _shuffle
from webdataset.tariterators import base_plus_ext, url_opener, tar_file_expander, valid_sample

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

import pickle
import pycocotools.mask as pymask
import torchvision.transforms as T
import cv2
import torch
import ipdb
import csv
from torchvision.transforms import Normalize,ToTensor,Compose,Resize,RandomResizedCrop,PILToTensor
from PIL import ImageDraw,ImageFont
from torchvision.transforms.functional import InterpolationMode

import shutil

class CsvDatasetWithMask(Dataset):
    def __init__(self, input_filename, transforms, img_key, caption_key, label_key, sep="\t", tokenizer=None, is_train=True):
        logging.debug(f'Loading csv data from {input_filename}.')
        df = pd.read_csv(input_filename, sep=sep)

        self.images = df[img_key].tolist()
        self.captions = df[caption_key].tolist()
        self.labels = df['class'].tolist()
        self.transforms = transforms

        # split the transforms
        if is_train:
            self.transform_shared = T.Compose([transforms.transforms[0]])
        else:
            self.transform_shared = T.Compose([transforms.transforms[0], transforms.transforms[1]])
        self.transform_image = T.Compose([transforms.transforms[-1]])

        # load mask
        logging.debug('Loading segmentation masks.')
        ids = map(int, df['id'])
        dataType = 'train'
        if 'val' in input_filename:
            dataType = 'val'
        adjusted_occludee_mask_dict = pickle.load(open(f'/home/ypliu/projects/OccludedCLIP/open_clip-main/tri_mask_amodal/{dataType}2017_adjusted_occludee_mask_dict.pkl', 'rb'))
        adjusted_occluder_mask_dict = pickle.load(open(f'/home/ypliu/projects/OccludedCLIP/open_clip-main/tri_mask_amodal/{dataType}2017_adjusted_occluder_mask_dict.pkl', 'rb'))
        adjusted_object_mask_dict = pickle.load(open(f'/home/ypliu/projects/OccludedCLIP/open_clip-main/tri_mask_amodal/{dataType}2017_adjusted_object_mask_dict.pkl', 'rb'))

        self.objects = []
        self.occluders = []
        self.occludees = []
        for i in ids:
            self.objects.append(adjusted_object_mask_dict[i])
            self.occluders.append(adjusted_occluder_mask_dict[i])
            self.occludees.append(adjusted_occludee_mask_dict[i])

        logging.debug('Done loading data.')

        self.tokenize = tokenizer

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        img = (cv2.cvtColor(cv2.imread(str(self.images[idx])), cv2.COLOR_BGR2RGB)/255.0).astype(np.float32)
        objects_mask = pymask.decode(self.objects[idx])
        occluders_mask = pymask.decode(self.occluders[idx])
        occludees_mask = pymask.decode(self.occludees[idx])

        img_merged = torch.from_numpy(np.concatenate([img, objects_mask, occluders_mask, occludees_mask], axis=2))
        img_merged = img_merged.permute(2,0,1)
        img_merged_transformed = self.transform_shared(img_merged)

        img2 = self.transform_image(img_merged_transformed[:3, :, :])
        # img2 = img_merged_transformed[:3, :, :]
        objects, occluders, occludees = img_merged_transformed[3, :, :], img_merged_transformed[4, :, :], img_merged_transformed[5, :, :]

        classes = self.labels[idx]
        texts = self.tokenize([str(self.captions[idx])])[0]
        return img2, texts, classes, objects, occluders, occludees


# original csvdataset until 2.20
class CsvDataset(Dataset):
    def __init__(self, input_filename, transforms, img_key, caption_key, sep="\t", tokenizer=None):
        logging.debug(f'Loading csv data from {input_filename}.')
        df = pd.read_csv(input_filename, sep=sep)

        self.images = df[img_key].tolist()
        self.captions = df[caption_key].tolist()
        self.transforms = transforms
        logging.debug('Done loading data.')

        self.tokenize = tokenizer


    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        # print(idx)
        # ipdb.set_trace()
        # shutil.copy(str(self.images[idx]), os.path.join(vis_dir, str(idx) + '.png'))
        images = self.transforms(Image.open(str(self.images[idx])))
        texts = self.tokenize([str(self.captions[idx])])[0]
        # idx_tensor = torch.Tensor([[idx]])

        return images, texts


# add real captions for training - 2.20
class CsvDatasetBothSynReal(Dataset):
    def __init__(self, input_filename, transforms, img_key, caption_key, sep="\t", tokenizer=None):
        logging.debug(f'Loading csv data from {input_filename}.')
        df = pd.read_csv(input_filename, sep=sep)

        self.images = df[img_key].tolist()
        self.captions = df[caption_key].tolist()
        # ipdb.set_trace()
        # ----- add real caption 2.20 -----
        self.real_captions = df['real_title'].tolist()
        # ----- add real caption 2.20 -----
        self.transforms = transforms
        logging.debug('Done loading data.')

        self.tokenize = tokenizer


    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        # print(idx)
        # ipdb.set_trace()
        # shutil.copy(str(self.images[idx]), os.path.join(vis_dir, str(idx) + '.png'))
        images = self.transforms(Image.open(str(self.images[idx])))
        texts = self.tokenize([str(self.captions[idx])])[0]
        # idx_tensor = torch.Tensor([[idx]])

        # ----- add real caption 2.20 -----
        real_texts = self.tokenize([str(self.real_captions[idx])])[0]
        # ----- add real caption 2.20 -----

        # return images, texts
        # ----- add real caption 2.20 -----
        return images, texts, real_texts


# add randomly choose from syn and real - 2.20
class CsvDatasetRandomSynReal(Dataset):
    def __init__(self, input_filename, transforms, img_key, caption_key, sep="\t", tokenizer=None):
        logging.debug(f'Loading csv data from {input_filename}.')
        df = pd.read_csv(input_filename, sep=sep)

        self.images = df[img_key].tolist()
        self.captions = df[caption_key].tolist()
        # ipdb.set_trace()
        # ----- add real caption 2.20 -----
        self.real_captions = df['real_title'].tolist()
        # ----- add real caption 2.20 -----
        self.transforms = transforms
        logging.debug('Done loading data.')

        self.tokenize = tokenizer

        print('CsvDatasetRandomSynReal\n\n\n\n\n\n\n')


    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        # print(idx)
        # ipdb.set_trace()
        # shutil.copy(str(self.images[idx]), os.path.join(vis_dir, str(idx) + '.png'))
        images = self.transforms(Image.open(str(self.images[idx])))
        texts = self.tokenize([str(self.captions[idx])])[0]
        # idx_tensor = torch.Tensor([[idx]])

        # ----- add real caption 2.20 -----
        real_texts = self.tokenize([str(self.real_captions[idx])])[0]
        # ----- add real caption 2.20 -----

        # return images, texts
        # ----- add real caption 2.20 -----
        if np.random.rand() < 0.5:
            return images, texts
        else:
            return images, real_texts


# add randomly choose from 5 syns and real - 2.20
class CsvDatasetRandomSynReal5(Dataset):
    def __init__(self, input_filename, transforms, img_key, caption_key, sep="\t", tokenizer=None):
        logging.debug(f'Loading csv data from {input_filename}.')
        df = pd.read_csv(input_filename, sep=sep)

        self.images = df[img_key].tolist()
        self.captions = df['title1'].tolist()
        self.captions2 = df['title2'].tolist()
        self.captions3 = df['title3'].tolist()
        self.captions4 = df['title4'].tolist()
        self.captions5 = df['title5'].tolist()
        # ipdb.set_trace()
        # ----- add real caption 2.20 -----
        self.real_captions = df['real_title'].tolist()
        # ----- add real caption 2.20 -----
        self.transforms = transforms
        logging.debug('Done loading data.')

        self.tokenize = tokenizer

        print('CsvDatasetRandomSynReal5\n\n\n\n\n\n\n')


    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        # print(idx)
        # ipdb.set_trace()
        # shutil.copy(str(self.images[idx]), os.path.join(vis_dir, str(idx) + '.png'))
        images = self.transforms(Image.open(str(self.images[idx])))
        texts = self.tokenize([str(self.captions[idx])])[0]
        texts_2 = self.tokenize([str(self.captions2[idx])])[0]
        texts_3 = self.tokenize([str(self.captions3[idx])])[0]
        texts_4 = self.tokenize([str(self.captions4[idx])])[0]
        texts_5 = self.tokenize([str(self.captions5[idx])])[0]
        # idx_tensor = torch.Tensor([[idx]])

        # ----- add real caption 2.20 -----
        real_texts = self.tokenize([str(self.real_captions[idx])])[0]
        # ----- add real caption 2.20 -----

        # return images, texts
        # ----- add real caption 2.20 -----
        if np.random.rand() < 0.5:
            another_random_value = np.random.rand()
            if another_random_value < 0.2:
                return images, texts
            elif 0.2 <= another_random_value < 0.4:
                return images, texts_2
            elif 0.4 <= another_random_value < 0.6:
                return images, texts_3
            elif 0.6 <= another_random_value < 0.8:
                return images, texts_4
            else:
                return images, texts_5
        else:
            return images, real_texts




class CsvDatasetLogoPrompt(Dataset):
    def __init__(self, input_filename, transforms, img_key, caption_key, sep="\t", tokenizer=None):
        logging.debug(f'Loading csv data from {input_filename}.')
        df = pd.read_csv(input_filename, sep=sep)

        self.images = df[img_key].tolist()
        self.captions = df[caption_key].tolist()
        self.transforms = transforms
        logging.debug('Done loading data.')

        self.tokenize = tokenizer

        # logo prompt
        self.logo_prompt = True
        self.logo_prompt_transform = Compose([
                                        Resize((16, 192), interpolation=InterpolationMode.BICUBIC),
                                        ToTensor(),
                                        Normalize(
                                            mean=[0.48145466, 0.4578275, 0.40821073],
                                            std=[0.26862954, 0.26130258, 0.27577711])
                                    ])
        self.font = ImageFont.truetype(r'PROJECT_PATH/Arial.ttf', size=16)


    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        # print(idx)
        # ipdb.set_trace()
        images = self.transforms(Image.open(str(self.images[idx])))
        texts = self.tokenize([str(self.captions[idx])])[0]

        if self.logo_prompt:
            # generate logo prompt
            size = self.font.getsize(text=(str(self.captions[idx])))
            r = random.randint(0, 255)
            g = random.randint(0, 255)
            b = random.randint(0, 255)
            im = Image.new("RGB", size, (0, 0, 0))
            draw = ImageDraw.Draw(im)
            draw.text((0, 0), str(self.captions[idx]), fill=(255 - r, 255 - g, 255 - b), font=self.font, align="right")
            logo_prompt_im = self.logo_prompt_transform(im)
            return images, texts, logo_prompt_im
        else:
            return images, texts



class CsvDatasetSD(Dataset):
    def __init__(self, input_filename, transforms, img_key, caption_key, sep="\t", tokenizer=None):
        logging.debug(f'Loading csv data from {input_filename}.')
        df = pd.read_csv(input_filename, sep=sep)

        self.images = df[img_key].tolist()
        self.captions = df[caption_key].tolist()
        self.transforms = transforms
        logging.debug('Done loading data.')

        self.tokenize = tokenizer

        with open(input_filename, mode='r', newline='', encoding='utf-8') as csvfile:
            self.csvreader = list(csv.reader(csvfile))
        # ipdb.set_trace()

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        # print(idx)
        # ipdb.set_trace()
        images_sd_folder = 'DATASET_PATH/cc3m_sd_generated_imgs_9.22'
        images = self.transforms(Image.open(str(self.images[idx])))
        images_sd_pth = os.path.join(images_sd_folder, str(idx+1) + '.png')
        # images_sd_pth = images_sd_pth[:-4] + '.png'
        try:
            images_sd = self.transforms(Image.open(images_sd_pth))
        except:
            print('img_not_find')
            images_sd = self.transforms(Image.open(str(self.images[idx])))
        texts = self.tokenize([str(self.captions[idx])])[0]
        return images, texts, images_sd


class CsvDatasetSDVal(Dataset):
    def __init__(self, input_filename, transforms, img_key, caption_key, sep="\t", tokenizer=None):
        logging.debug(f'Loading csv data from {input_filename}.')
        df = pd.read_csv(input_filename, sep=sep)

        self.images = df[img_key].tolist()
        self.captions = df[caption_key].tolist()
        self.transforms = transforms
        logging.debug('Done loading data.')

        self.tokenize = tokenizer

        with open(input_filename, mode='r', newline='', encoding='utf-8') as csvfile:
            self.csvreader = list(csv.reader(csvfile))
        # ipdb.set_trace()

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        # print(idx)
        # ipdb.set_trace()
        images_sd_folder = 'DATASET_PATH/coco_sd_generated_imgs_9.22_2'
        images_sd1_folder = 'DATASET_PATH/coco_sd_generated_imgs_9.22_2_1'
        images_sd2_folder = 'DATASET_PATH/coco_sd_generated_imgs_9.22_2_2'
        images_sd3_folder = 'DATASET_PATH/coco_sd_generated_imgs_9.22_2_3'
        images_sd4_folder = 'DATASET_PATH/coco_sd_generated_imgs_9.22_2_4'
        images = self.transforms(Image.open(str(self.images[idx])))
        images_sd_pth = os.path.join(images_sd_folder, str(idx+1) + '.png')
        images_sd1_pth = os.path.join(images_sd1_folder, str(idx+1) + '.png')
        images_sd2_pth = os.path.join(images_sd2_folder, str(idx+1) + '.png')
        images_sd3_pth = os.path.join(images_sd3_folder, str(idx+1) + '.png')
        images_sd4_pth = os.path.join(images_sd4_folder, str(idx+1) + '.png')
        # images_sd_pth = images_sd_pth[:-4] + '.png'
        try:
            images_sd = self.transforms(Image.open(images_sd_pth))
            images_sd1 = self.transforms(Image.open(images_sd1_pth))
            images_sd2 = self.transforms(Image.open(images_sd2_pth))
            images_sd3 = self.transforms(Image.open(images_sd3_pth))
            images_sd4 = self.transforms(Image.open(images_sd4_pth))
        except:
            print('img_not_find')
            images_sd = self.transforms(Image.open(images_sd_pth))
            images_sd1 = self.transforms(Image.open(images_sd_pth))
            images_sd2 = self.transforms(Image.open(images_sd_pth))
            images_sd3 = self.transforms(Image.open(images_sd_pth))
            images_sd4 = self.transforms(Image.open(images_sd_pth))
        texts = self.tokenize([str(self.captions[idx])])[0]
        return images, texts, images_sd, images_sd1, images_sd2, images_sd3, images_sd4


class CsvDatasetWithClass(Dataset):
    def __init__(self, input_filename, transforms, img_key, caption_key, label_key, sep="\t", tokenizer=None):
        logging.debug(f'Loading csv data from {input_filename}.')
        df = pd.read_csv(input_filename, sep=sep)

        self.images = df[img_key].tolist()
        self.captions = df[caption_key].tolist()
        self.labels = df[label_key].tolist()
        self.transforms = transforms
        logging.debug('Done loading data.')

        self.tokenize = tokenizer

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        images = self.transforms(Image.open(str(self.images[idx])))
        texts = self.tokenize([str(self.captions[idx])])[0]
        classes = self.labels[idx]
        return images, texts, classes


class CsvDatasetReID(Dataset):
    def __init__(self, input_filename, transforms, sep="\t"):
        logging.info(f'Loading ReID csv data from {input_filename}.')
        print(f'Loading ReID csv data from {input_filename}.')
        df = pd.read_csv(input_filename, sep=sep)

        self.images = df['filepath'].tolist()
        self.labels = df['class'].tolist()
        self.is_query = df['title'].tolist()
        self.transforms = transforms
        logging.debug('Done loading data.')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        images = self.transforms(Image.open(str(self.images[idx])))
        classes = self.labels[idx]
        is_query = self.is_query[idx]
        return images, classes, is_query



class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value('i', epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler = None
    shared_epoch: SharedEpoch = None

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)


def expand_urls(urls, weights=None):
    if weights is None:
        expanded_urls = wds.shardlists.expand_urls(urls)
        return expanded_urls, None
    if isinstance(urls, str):
        urllist = urls.split("::")
        weights = weights.split('::')
        assert len(weights) == len(urllist),\
            f"Expected the number of data components ({len(urllist)}) and weights({len(weights)}) to match."
        weights = [float(weight) for weight in weights]
        all_urls, all_weights = [], []
        for url, weight in zip(urllist, weights):
            expanded_url = list(braceexpand.braceexpand(url))
            expanded_weights = [weight for _ in expanded_url]
            all_urls.extend(expanded_url)
            all_weights.extend(expanded_weights)
        return all_urls, all_weights
    else:
        all_urls = list(urls)
        return all_urls, weights


def get_dataset_size(shards):
    shards_list, _ = expand_urls(shards)
    dir_path = os.path.dirname(shards_list[0])
    sizes_filename = os.path.join(dir_path, 'sizes.json')
    len_filename = os.path.join(dir_path, '__len__')
    if os.path.exists(sizes_filename):
        sizes = json.load(open(sizes_filename, 'r'))
        total_size = sum([int(sizes[os.path.basename(shard)]) for shard in shards_list])
    elif os.path.exists(len_filename):
        # FIXME this used to be eval(open(...)) but that seemed rather unsafe
        total_size = ast.literal_eval(open(len_filename, 'r').read())
    else:
        total_size = None  # num samples undefined
        # some common dataset sizes (at time of authors last download)
        # CC3M (train): 2905954
        # CC12M: 10968539
        # LAION-400M: 407332084
        # LAION-2B (english): 2170337258
    num_shards = len(shards_list)
    return total_size, num_shards


def get_imagenet(args, preprocess_fns, split):
    assert split in ["train", "val", "v2"]
    is_train = split == "train"
    preprocess_train, preprocess_val = preprocess_fns

    if split == "v2":
        from imagenetv2_pytorch import ImageNetV2Dataset
        dataset = ImageNetV2Dataset(location=args.imagenet_v2, transform=preprocess_val)
    else:
        if is_train:
            data_path = args.imagenet_train
            preprocess_fn = preprocess_train
        else:
            data_path = args.imagenet_val
            preprocess_fn = preprocess_val
        assert data_path

        dataset = datasets.ImageFolder(data_path, transform=preprocess_fn)

    if is_train:
        idxs = np.zeros(len(dataset.targets))
        target_array = np.array(dataset.targets)
        k = 50
        for c in range(1000):
            m = target_array == c
            n = len(idxs[m])
            arr = np.zeros(n)
            arr[:k] = 1
            np.random.shuffle(arr)
            idxs[m] = arr

        idxs = idxs.astype('int')
        sampler = SubsetRandomSampler(np.where(idxs)[0])
    else:
        sampler = None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        sampler=sampler,
    )

    return DataInfo(dataloader=dataloader, sampler=sampler)


def get_coco(args, preprocess_fns, split):
    is_train = split == "train"
    preprocess_train, preprocess_val = preprocess_fns

    if is_train:
        data_path = args.coco_train
        preprocess_fn = preprocess_train
    else:
        data_path = args.coco_val
        preprocess_fn = preprocess_val
    assert data_path

    dataset = datasets.ImageFolder(data_path, transform=preprocess_fn)

    if is_train:
        idxs = np.zeros(len(dataset.targets))
        target_array = np.array(dataset.targets)
        k = 50
        for c in range(1000):
            m = target_array == c
            n = len(idxs[m])
            arr = np.zeros(n)
            arr[:k] = 1
            np.random.shuffle(arr)
            idxs[m] = arr

        idxs = idxs.astype('int')
        sampler = SubsetRandomSampler(np.where(idxs)[0])
    else:
        sampler = None

    if args.csv_label_key == "class":
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=64,
            num_workers=args.workers,
            sampler=sampler,
        )
    else:
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.workers,
            sampler=sampler,
        )

    return DataInfo(dataloader=dataloader, sampler=sampler)


def get_ucf101(args, preprocess_fns, split):
    is_train = split == "train"
    preprocess_train, preprocess_val = preprocess_fns

    if is_train:
        data_path = args.ucf101_train
        preprocess_fn = preprocess_train
    else:
        data_path = args.ucf101_val
        preprocess_fn = preprocess_val
    assert data_path

    dataset = datasets.ImageFolder(data_path, transform=preprocess_fn)

    if is_train:
        idxs = np.zeros(len(dataset.targets))
        target_array = np.array(dataset.targets)
        k = 50
        for c in range(1000):
            m = target_array == c
            n = len(idxs[m])
            arr = np.zeros(n)
            arr[:k] = 1
            np.random.shuffle(arr)
            idxs[m] = arr

        idxs = idxs.astype('int')
        sampler = SubsetRandomSampler(np.where(idxs)[0])
    else:
        sampler = None

    if args.csv_label_key == "class":
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=64,
            num_workers=args.workers,
            sampler=sampler,
        )
    else:
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.workers,
            sampler=sampler,
        )

    return DataInfo(dataloader=dataloader, sampler=sampler)


def get_reid(args, preprocess_fns, split):
    is_train = split == "train"
    preprocess_train, preprocess_val = preprocess_fns

    if is_train:
        data_path = args.reid_train
        preprocess_fn = preprocess_train
    else:
        data_path = args.reid_val
        preprocess_fn = preprocess_val
    assert data_path

    dataset = datasets.ImageFolder(data_path, transform=preprocess_fn)

    if is_train:
        idxs = np.zeros(len(dataset.targets))
        target_array = np.array(dataset.targets)
        k = 50
        for c in range(1000):
            m = target_array == c
            n = len(idxs[m])
            arr = np.zeros(n)
            arr[:k] = 1
            np.random.shuffle(arr)
            idxs[m] = arr

        idxs = idxs.astype('int')
        sampler = SubsetRandomSampler(np.where(idxs)[0])
    else:
        sampler = None

    if args.csv_label_key == "class":
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=64,
            num_workers=args.workers,
            sampler=sampler,
        )
    else:
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.workers,
            sampler=sampler,
        )

    return DataInfo(dataloader=dataloader, sampler=sampler)


def get_cub(args, preprocess_fns, split):
    is_train = split == "train"
    preprocess_train, preprocess_val = preprocess_fns

    if is_train:
        data_path = args.cub_train
        preprocess_fn = preprocess_train
    else:
        data_path = args.cub_val
        preprocess_fn = preprocess_val
    assert data_path

    dataset = datasets.ImageFolder(data_path, transform=preprocess_fn)

    if is_train:
        idxs = np.zeros(len(dataset.targets))
        target_array = np.array(dataset.targets)
        k = 50
        for c in range(1000):
            m = target_array == c
            n = len(idxs[m])
            arr = np.zeros(n)
            arr[:k] = 1
            np.random.shuffle(arr)
            idxs[m] = arr

        idxs = idxs.astype('int')
        sampler = SubsetRandomSampler(np.where(idxs)[0])
    else:
        sampler = None

    if args.csv_label_key == "class":
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=64,
            num_workers=args.workers,
            sampler=sampler,
        )
    else:
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.workers,
            sampler=sampler,
        )

    return DataInfo(dataloader=dataloader, sampler=sampler)


def count_samples(dataloader):
    os.environ["WDS_EPOCH"] = "0"
    n_elements, n_batches = 0, 0
    for images, texts in dataloader:
        n_batches += 1
        n_elements += len(images)
        assert len(images) == len(texts)
    return n_elements, n_batches


def filter_no_caption_or_no_image(sample):
    has_caption = ('txt' in sample)
    has_image = ('png' in sample or 'jpg' in sample or 'jpeg' in sample or 'webp' in sample)
    return has_caption and has_image


def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, issue a warning, and continue."""
    logging.warning(f'Handling webdataset error ({repr(exn)}). Ignoring.')
    return True


def group_by_keys_nothrow(data, keys=base_plus_ext, lcase=True, suffixes=None, handler=None):
    """Return function over iterator that groups key, value pairs into samples.

    :param keys: function that splits the key into key and extension (base_plus_ext)
    :param lcase: convert suffixes to lower case (Default value = True)
    """
    current_sample = None
    for filesample in data:
        assert isinstance(filesample, dict)
        # try:
        fname, value = filesample["fname"], filesample["data"]
        # except:
        #     ipdb.set_trace()
        prefix, suffix = keys(fname)
        if prefix is None:
            continue
        if lcase:
            suffix = suffix.lower()
        # FIXME webdataset version throws if suffix in current_sample, but we have a potential for
        #  this happening in the current LAION400m dataset if a tar ends with same prefix as the next
        #  begins, rare, but can happen since prefix aren't unique across tar files in that dataset
        if current_sample is None or prefix != current_sample["__key__"] or suffix in current_sample:
            if valid_sample(current_sample):
                yield current_sample
            current_sample = dict(__key__=prefix, __url__=filesample["__url__"])
        if suffixes is None or suffix in suffixes:
            current_sample[suffix] = value
    if valid_sample(current_sample):
        yield current_sample


def tarfile_to_samples_nothrow(src, handler=log_and_continue):
    # NOTE this is a re-impl of the webdataset impl with group_by_keys that doesn't throw
    streams = url_opener(src, handler=handler)
    files = tar_file_expander(streams, handler=handler, eof_value=None)
    samples = group_by_keys_nothrow(files, handler=handler)
    return samples


def pytorch_worker_seed(increment=0):
    """get dataloader worker seed from pytorch"""
    worker_info = get_worker_info()
    if worker_info is not None:
        # favour using the seed already created for pytorch dataloader workers if it exists
        seed = worker_info.seed
        if increment:
            # space out seed increments so they can't overlap across workers in different iterations
            seed += increment * max(1, worker_info.num_workers)
        return seed
    # fallback to wds rank based seed
    return wds.utils.pytorch_worker_seed()


_SHARD_SHUFFLE_SIZE = 2000
_SHARD_SHUFFLE_INITIAL = 500
_SAMPLE_SHUFFLE_SIZE = 5000
_SAMPLE_SHUFFLE_INITIAL = 1000


class detshuffle2(wds.PipelineStage):
    def __init__(
            self,
            bufsize=1000,
            initial=100,
            seed=0,
            epoch=-1,
    ):
        self.bufsize = bufsize
        self.initial = initial
        self.seed = seed
        self.epoch = epoch

    def run(self, src):
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        rng = random.Random()
        if self.seed < 0:
            # If seed is negative, we use the worker's seed, this will be different across all nodes/workers
            seed = pytorch_worker_seed(epoch)
        else:
            # This seed to be deterministic AND the same across all nodes/workers in each epoch
            seed = self.seed + epoch
        rng.seed(seed)
        return _shuffle(src, self.bufsize, self.initial, rng)


class ResampledShards2(IterableDataset):
    """An iterable dataset yielding a list of urls."""

    def __init__(
        self,
        urls,
        weights=None,
        nshards=sys.maxsize,
        worker_seed=None,
        deterministic=False,
        epoch=-1,
    ):
        """Sample shards from the shard list with replacement.

        :param urls: a list of URLs as a Python list or brace notation string
        """
        super().__init__()
        urls, weights = expand_urls(urls, weights)
        self.urls = urls
        self.weights = weights
        if self.weights is not None:
            assert len(self.urls) == len(self.weights),\
                f"Number of urls {len(self.urls)} and weights {len(self.weights)} should match."
        assert isinstance(self.urls[0], str)
        self.nshards = nshards
        self.rng = random.Random()
        self.worker_seed = worker_seed
        self.deterministic = deterministic
        self.epoch = epoch

    def __iter__(self):
        """Return an iterator over the shards."""
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        if self.deterministic:
            # reset seed w/ epoch if deterministic
            if self.worker_seed is None:
                # pytorch worker seed should be deterministic due to being init by arg.seed + rank + worker id
                seed = pytorch_worker_seed(epoch)
            else:
                seed = self.worker_seed() + epoch
            self.rng.seed(seed)
        for _ in range(self.nshards):
            if self.weights is None:
                yield dict(url=self.rng.choice(self.urls))
            else:
                yield dict(url=self.rng.choices(self.urls, weights=self.weights, k=1)[0])


def get_wds_dataset(args, preprocess_img, is_train, epoch=0, floor=False, tokenizer=None):
    input_shards = args.train_data if is_train else args.val_data
    assert input_shards is not None
    resampled = getattr(args, 'dataset_resampled', False) and is_train

    num_shards = None
    if is_train:
        if args.train_num_samples is not None:
            num_samples = args.train_num_samples
        else:
            num_samples, num_shards = get_dataset_size(input_shards)
            if not num_samples:
                raise RuntimeError(
                    'Currently, the number of dataset samples must be specified for the training dataset. '
                    'Please specify it via `--train-num-samples` if no dataset length info is present.')
    else:
        # Eval will just exhaust the iterator if the size is not specified.
        num_samples = args.val_num_samples or 0 

    shared_epoch = SharedEpoch(epoch=epoch)  # create a shared epoch store to sync epoch to dataloader worker proc

    if is_train and args.train_data_upsampling_factors is not None:
        assert resampled, "--train_data_upsampling_factors is only supported when sampling with replacement (with --dataset-resampled)."
    
    if resampled:
        pipeline = [ResampledShards2(
            input_shards,
            weights=args.train_data_upsampling_factors,
            deterministic=True,
            epoch=shared_epoch,
        )]
    else:
        pipeline = [wds.SimpleShardList(input_shards)]

    # at this point we have an iterator over all the shards
    if is_train:
        if not resampled:
            pipeline.extend([
                detshuffle2(
                    bufsize=_SHARD_SHUFFLE_SIZE,
                    initial=_SHARD_SHUFFLE_INITIAL,
                    seed=args.seed,
                    epoch=shared_epoch,
                ),
                wds.split_by_node,
                wds.split_by_worker,
            ])
        pipeline.extend([
            # at this point, we have an iterator over the shards assigned to each worker at each node
            tarfile_to_samples_nothrow,  # wds.tarfile_to_samples(handler=log_and_continue),
            wds.shuffle(
                bufsize=_SAMPLE_SHUFFLE_SIZE,
                initial=_SAMPLE_SHUFFLE_INITIAL,
            ),
        ])
    else:
        pipeline.extend([
            wds.split_by_worker,
            # at this point, we have an iterator over the shards assigned to each worker
            wds.tarfile_to_samples(handler=log_and_continue),
        ])
    pipeline.extend([
        wds.select(filter_no_caption_or_no_image),
        wds.decode("pilrgb", handler=log_and_continue),
        wds.rename(image="jpg;png;jpeg;webp", text="txt"),
        wds.map_dict(image=preprocess_img, text=lambda text: tokenizer(text)[0]),
        wds.to_tuple("image", "text"),
        wds.batched(args.batch_size, partial=not is_train)
    ])

    dataset = wds.DataPipeline(*pipeline)

    if is_train:
        if not resampled:
            num_shards = num_shards or len(expand_urls(input_shards)[0])
            print('num_shards:', num_shards)
            assert num_shards >= args.workers * args.world_size, 'number of shards must be >= total workers'
        # roll over and repeat a few samples to get same number of full batches on each node
        round_fn = math.floor if floor else math.ceil
        global_batch_size = args.batch_size * args.world_size
        num_batches = round_fn(num_samples / global_batch_size)
        num_workers = max(1, args.workers)
        num_worker_batches = round_fn(num_batches / num_workers)  # per dataloader worker
        num_batches = num_worker_batches * num_workers
        num_samples = num_batches * global_batch_size
        dataset = dataset.with_epoch(num_worker_batches)  # each worker is iterating over this
    else:
        # last batches are partial, eval is done on single (master) node
        num_batches = math.ceil(num_samples / args.batch_size)

    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
    )

    # FIXME not clear which approach is better, with_epoch before vs after dataloader?
    # hoping to resolve via https://github.com/webdataset/webdataset/issues/169
    # if is_train:
    #     # roll over and repeat a few samples to get same number of full batches on each node
    #     global_batch_size = args.batch_size * args.world_size
    #     num_batches = math.ceil(num_samples / global_batch_size)
    #     num_workers = max(1, args.workers)
    #     num_batches = math.ceil(num_batches / num_workers) * num_workers
    #     num_samples = num_batches * global_batch_size
    #     dataloader = dataloader.with_epoch(num_batches)
    # else:
    #     # last batches are partial, eval is done on single (master) node
    #     num_batches = math.ceil(num_samples / args.batch_size)

    # add meta-data to dataloader instance for convenience
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch)


def get_csv_dataset(input_filename, args, preprocess_fn, is_train, epoch=0, tokenizer=None):
    # input_filename = args.train_data
    # input_filename = args.train_data if is_train else args.val_data
    # ipdb.set_trace()
    # assert input_filename
    if args.csv_label_key != "none":
        dataset = CsvDatasetWithClass(
            input_filename,
            preprocess_fn,
            img_key=args.csv_img_key,
            caption_key=args.csv_caption_key,
            label_key=args.csv_label_key,
            sep=args.csv_separator,
            tokenizer=tokenizer
        )
    elif args.csv_mask_key != "none":
        dataset = CsvDatasetWithMask(
            input_filename,
            preprocess_fn,
            img_key=args.csv_img_key,
            caption_key=args.csv_caption_key,
            label_key=args.csv_label_key,
            sep=args.csv_separator,
            tokenizer=tokenizer,
            is_train=is_train
        )
    # elif args.reid_val != "none":
    #     dataset = CsvDatasetReID(
    #         input_filename,
    #         preprocess_fn,
    #         sep=args.csv_separator
    #     )
    
    elif args.sd == True or args.sd1 == True or args.sd2 == True :
        # ipdb.set_trace()
        if is_train:
            dataset = CsvDatasetSD(
                input_filename,
                preprocess_fn,
                img_key=args.csv_img_key,
                caption_key=args.csv_caption_key,
                sep=args.csv_separator,
                tokenizer=tokenizer
            )
        else:
            dataset = CsvDatasetSDVal(
                input_filename,
                preprocess_fn,
                img_key=args.csv_img_key,
                caption_key=args.csv_caption_key,
                sep=args.csv_separator,
                tokenizer=tokenizer
            )
        # ipdb.set_trace()
    elif args.logo_prompt == True:
        # ipdb.set_trace()
        dataset = CsvDatasetLogoPrompt(
            input_filename,
            preprocess_fn,
            img_key=args.csv_img_key,
            caption_key=args.csv_caption_key,
            sep=args.csv_separator,
            tokenizer=tokenizer
        )
    else:
        dataset = CsvDataset(
            input_filename,
            preprocess_fn,
            img_key=args.csv_img_key,
            caption_key=args.csv_caption_key,
            sep=args.csv_separator,
            tokenizer=tokenizer
        )
    num_samples = len(dataset)
    # print(args.distributed)
    # ipdb.set_trace()
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    # args.cate is positive class
    # if isinstance(dataset, CsvDatasetWithClass):
    #     count = [0] * 2
    #     for item in dataset.labels:
    #         if item == args.cate:
    #             count[1] += 1
    #         else:
    #             count[0] += 1
    #     weight_per_class = [0.] * 2
    #     N = float(sum(count))
    #     for i in range(2):
    #         weight_per_class[i] = N / float(count[i])
    #     weight = [0] * len(dataset.labels)
    #     for idx, val in enumerate(dataset.labels):
    #         if val == args.cate:
    #             weight[idx] = weight_per_class[1]
    #         else:
    #             weight[idx] = weight_per_class[0]
    #
    #     weights = torch.DoubleTensor(weight)
    #     sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
        # shuffle = False
        # sampler = None
        # is_train = False
        # print(f'--------------- shuffle {shuffle}, sampler {sampler} drop_last {is_train}---------------')

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)



def get_csv_harder_dataset(input_filename, args, preprocess_fn, is_train, epoch=0, tokenizer=None):

    print('get_csv_harder_dataset')
    print('\n')
    print('\n')
    if args.csv_label_key != "none":
        dataset = CsvDatasetWithClass(
            input_filename,
            preprocess_fn,
            img_key=args.csv_img_key,
            caption_key=args.csv_caption_key,
            label_key=args.csv_label_key,
            sep=args.csv_separator,
            tokenizer=tokenizer
        )
    elif args.csv_mask_key != "none":
        dataset = CsvDatasetWithMask(
            input_filename,
            preprocess_fn,
            img_key=args.csv_img_key,
            caption_key=args.csv_caption_key,
            label_key=args.csv_label_key,
            sep=args.csv_separator,
            tokenizer=tokenizer,
            is_train=is_train
        )
    elif args.logo_prompt == True:
        # ipdb.set_trace()
        dataset = CsvDatasetLogoPrompt(
            input_filename,
            preprocess_fn,
            img_key=args.csv_img_key,
            caption_key=args.csv_caption_key,
            sep=args.csv_separator,
            tokenizer=tokenizer
        )
    elif args.simple_contrast_random == True:
        # ipdb.set_trace()
        dataset = CsvDatasetRandomSynReal(
            input_filename,
            preprocess_fn,
            img_key=args.csv_img_key,
            caption_key=args.csv_caption_key,
            sep=args.csv_separator,
            tokenizer=tokenizer
        )
    elif args.simple_contrast_random5 == True:
        # ipdb.set_trace()
        dataset = CsvDatasetRandomSynReal5(
            input_filename,
            preprocess_fn,
            img_key=args.csv_img_key,
            caption_key=args.csv_caption_key,
            sep=args.csv_separator,
            tokenizer=tokenizer
        )
    else:
        # ipdb.set_trace()
        dataset = CsvDataset(
            input_filename,
            preprocess_fn,
            img_key=args.csv_img_key,
            caption_key=args.csv_caption_key,
            sep=args.csv_separator,
            tokenizer=tokenizer
        )
    num_samples = len(dataset)
    print(args.distributed)
    # ipdb.set_trace()
    sampler = DistributedHarderSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


class SyntheticDataset(Dataset):

    def __init__(
            self,
            transform=None,
            image_size=(224, 224),
            caption="Dummy caption",
            dataset_size=100,
            tokenizer=None,
    ):
        self.transform = transform
        self.image_size = image_size
        self.caption = caption
        self.image = Image.new('RGB', image_size)
        self.dataset_size = dataset_size

        self.preprocess_txt = lambda text: tokenizer(text)[0]

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        if self.transform is not None:
            image = self.transform(self.image)
        return image, self.preprocess_txt(self.caption)


def get_synthetic_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None):
    image_size = preprocess_fn.transforms[0].size
    dataset = SyntheticDataset(
        transform=preprocess_fn, image_size=image_size, dataset_size=args.train_num_samples, tokenizer=tokenizer)
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


def get_dataset_fn(data_path, dataset_type):
    # ipdb.set_trace()
    if dataset_type == "webdataset":
        return get_wds_dataset
    elif dataset_type == "csv":
        return get_csv_dataset
    elif dataset_type == "csv_harder":
        # ipdb.set_trace()
        return get_csv_harder_dataset
    elif dataset_type == "synthetic":
        return get_synthetic_dataset
    elif dataset_type == "auto":
        ext = data_path.split('.')[-1]
        if ext in ['csv', 'tsv']:
            return get_csv_harder_dataset
        elif ext in ['tar']:
            return get_wds_dataset
        else:
            raise ValueError(
                f"Tried to figure out dataset type, but failed for extension {ext}.")
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")
    

def get_data(args, preprocess_fns, epoch=0, tokenizer=None):
    preprocess_train, preprocess_val = preprocess_fns
    data = {}

    if args.train_data or args.dataset_type == "synthetic":
        if args.dataset_type == "webdataset":
            data["train"] = get_dataset_fn(args.train_data, args.dataset_type)(
                args, preprocess_train, is_train=True, epoch=epoch, tokenizer=tokenizer)
            data["train_val"] = get_dataset_fn(args.train_data, args.dataset_type)(
                args, preprocess_val, is_train=False, epoch=epoch, tokenizer=tokenizer)
        else:
            data["train"] = get_dataset_fn(args.train_data, args.dataset_type)(
                args.train_data, args, preprocess_train, is_train=True, epoch=epoch, tokenizer=tokenizer)
            data["train_val"] = get_dataset_fn(args.train_data, args.dataset_type)(
                args.train_data, args, preprocess_val, is_train=False, epoch=epoch, tokenizer=tokenizer)

    if args.val_data:
        if args.dataset_type == "webdataset":
            data["val"] = get_dataset_fn(args.val_data, args.dataset_type)(
                args, preprocess_val, is_train=False, tokenizer=tokenizer)
        else:
            data["val"] = get_dataset_fn(args.val_data, args.dataset_type)(
                args.val_data, args, preprocess_val, is_train=False, tokenizer=tokenizer)
    if args.val_data_novel:
        data["val_novel"] = get_dataset_fn(args.val_data_novel, args.dataset_type)(
            args.val_data, args, preprocess_val, is_train=False, tokenizer=tokenizer)

    if args.imagenet_val is not None:
        data["imagenet-val"] = get_imagenet(args, preprocess_fns, "val")

    if args.imagenet_v2 is not None:
        data["imagenet-v2"] = get_imagenet(args, preprocess_fns, "v2")

    if args.coco_val is not None:
        data["coco-val"] = get_coco(args, preprocess_fns, "val")

    if args.ucf101_val is not None:
        data["ucf101-val"] = get_ucf101(args, preprocess_fns, "val")

    if args.reid_val is not None:
        data["reid-val"] = get_reid(args, preprocess_fns, "val")

    if args.cub_val is not None:
        data["cub-val"] = get_cub(args, preprocess_fns, "val")

    return data
