import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import lmdb
from PIL import Image
import numpy as np
import six


class CocotextDataset(Dataset):
    def __init__(self, root=None, transform=None, target_transform=None):
        self.lmdb = lmdb.open(
            root,
            max_readers=1,
            lock=False,
            meminit=False,
            readonly=True,
            readahead=False)

        with self.lmdb.begin(write=False) as txn:
            num_samples = int(txn.get(('num-samples').encode('utf-8')))
            self.num_samples = num_samples

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, item):
        item += 1

        with self.lmdb.begin(write=False) as txn:
            img_key = 'image-%09d' % item
            img_buffer = txn.get(img_key.encode('utf-8'))

            buffer = six.BytesIO()
            buffer.write(img_buffer)
            buffer.seek(0)

            try:
                img = Image.open(buffer).convert('L')  # convert to gray_scale
            except IOError:
                print('Corrupt image %d' % item)
                return self[item + 1]

            if self.transform is not None:
                img = self.transform(img)

            target_key = 'label-%09d' % item
            tgt = txn.get((target_key).encode('utf-8'))
            tgt = tgt.decode()

            if self.target_transform is not None:
                tgt = self.target_transform(tgt)

        return (img, tgt)

    def __len__(self):
        return self.num_samples


class rescale(object):

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, image):
        image = image.resize(self.size, self.interpolation)
        image = self.toTensor(image)
        image.sub_(0.5).div_(0.5)
        return image


class cocoCollateFn(object):

    def __init__(self, imgheight=32, imgwidth=100, hold_ratio=False, minimum_rto=1):
        self.img_height = imgheight
        self.img_width = imgwidth
        self.hold_ratio = hold_ratio
        self.min_ratio = minimum_rto

    def __call__(self, b):

        imgs, tgts = zip(*b)

        img_width = self.img_width
        img_height = self.img_height

        if self.hold_ratio:
            rtos = []

            for i in imgs:
                width, height = i.size
                rtos.append(width / float(height))

            rtos.sort()
            maximum_rto = rtos[-1]
            img_width = int(np.floor(maximum_rto * img_height))
            img_width = max(img_height * self.min_ratio, img_width)

        transform = rescale((img_width, img_height))
        imgs = [transform(image) for image in imgs]
        imgs = torch.cat([i.unsqueeze(0) for i in imgs], 0)
        return imgs, tgts
