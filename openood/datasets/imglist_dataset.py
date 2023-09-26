import ast
import io
import logging
import os
import time

import torch
from PIL import Image, ImageFile

from .base_dataset import BaseDataset

# to fix "OSError: image file is truncated"
ImageFile.LOAD_TRUNCATED_IMAGES = True


class Convert:
    def __init__(self, mode='RGB'):
        self.mode = mode

    def __call__(self, image):
        return image.convert(self.mode)

from PIL import Image
def pil_loader(img_str, str='RGB'):
    with Image.open(img_str) as img:
        img = img.convert(str)
    return img


class ImglistDataset(BaseDataset):
    def __init__(self,
                 name,
                 imglist_pth,
                 data_dir,
                 num_classes,
                 preprocessor,
                 data_aux_preprocessor,
                 maxlen=None,
                 dummy_read=False,
                 dummy_size=None,
                 **kwargs):
        super(ImglistDataset, self).__init__(**kwargs)

        self.name = name
        with open(imglist_pth) as imgfile:
            self.imglist = imgfile.readlines()
        self.data_dir = data_dir
        self.num_classes = num_classes
        self.preprocessor = preprocessor
        self.transform_image = preprocessor
        self.transform_aux_image = data_aux_preprocessor
        self.maxlen = maxlen
        self.dummy_read = dummy_read
        self.dummy_size = dummy_size

        self.samples = []
        for img_idx in range(len(self.imglist)):
            line = self.imglist[img_idx].strip('\n')
            tokens = line.split(' ', 1)
            if self.data_dir != '' and tokens[0].startswith('/'):
                raise RuntimeError('image_name starts with "/"')
            self.samples.append((tokens[0], tokens[1],
                                 os.path.join(self.data_dir, tokens[0]), tokens))

        if dummy_read and dummy_size is None:
            raise ValueError(
                'if dummy_read is True, should provide dummy_size')

    def __len__(self):
        if self.maxlen is None:
            return len(self.imglist)
        else:
            return min(len(self.imglist), self.maxlen)

    def getitem(self, index):
        image_name, extra_str, path, tokens = self.samples[index]
        sample = {'image_name': image_name}
        kwargs = {'name': self.name, 'path': path, 'tokens': tokens}
        try:
            # some preprocessor methods require setup
            self.preprocessor.setup(**kwargs)
        except:
            pass

        try:
            if self.dummy_size is not None:
                sample['data'] = torch.rand(self.dummy_size)
            else:
                # if not self.dummy_read:
                #     with open(path, 'rb') as f:
                #         content = f.read()
                #     filebytes = content
                #     buff = io.BytesIO(filebytes)
                #     image = Image.open(buff).convert('RGB')
                if not self.dummy_read:
                    image = pil_loader(path)
                sample['data'] = self.transform_image(image)
                # sample['data_aux'] = self.transform_aux_image(image)
            sample['label'] = int(extra_str)
            extras = ast.literal_eval(extra_str)
            try:
                for key, value in extras.items():
                    sample[key] = value
                # if you use dic the code below will need ['label']
                sample['label'] = 0
            except AttributeError:
                sample['label'] = int(extra_str)
            # Generate Soft Label
            soft_label = torch.Tensor(self.num_classes)
            if sample['label'] < 0:
                soft_label.fill_(1.0 / self.num_classes)
            else:
                soft_label.fill_(0)
                soft_label[sample['label']] = 1
            sample['soft_label'] = soft_label

        except Exception as e:
            logging.error('[{}] broken'.format(path))
            raise e
        return sample


    # def getitem(self, index):
    #     start_time = time.time()
    #     image_name, extra_str, path, tokens = self.samples[index]
    #     sample = {'image_name': image_name}
    #     try:
    #         if self.dummy_size is not None:
    #             image = torch.rand(self.dummy_size)
    #         else:
    #             if not self.dummy_read:
    #                 image = pil_loader(path)
    #                 img_time = time.time() - start_time
    #             sample['data'] = self.transform_image(image)
    #             # sample['data'] = torch.rand((3, 224, 224))
    #             # sample['data_aux'] = self.transform_aux_image(image)
    #         sample['label'] = int(extra_str)
    #     except Exception as e:
    #         logging.error('[{}] broken'.format(path))
    #         raise e
    #     sample_time = time.time() - start_time
    #     # print(f"Sample (t) {sample_time:.3f}, Image (t) {img_time:.3f}\t", flush=True)
    #     return sample