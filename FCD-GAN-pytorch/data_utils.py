import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as trans
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models

import os
import sys
import numpy as np
from tqdm import tqdm
import math
import random

from PIL import Image
import rasterio
from rasterio.windows import Window


# Dataset using rasterio to read remote sensing images (GeoTIFF format)
# The read patch is obtained from the large-scale image with overlaps
# When writing the patches, only the centering region without overlap padding is written
class RasterioDataset(Dataset):

    def __init__(self, imgPathX, imgPathY, refPath=None, outPath=None, transforms=None, enhance=None, patch_size=(200, 200), overlap_padding=(10, 10)):
        super(RasterioDataset, self).__init__()
        self.imgPathX = imgPathX
        self.imgDS_x = rasterio.open(imgPathX)
        if self.imgDS_x is None:
            print('No such a Image file:{}'.format(imgPathX))
            sys.exit(0)
        xsize = self.imgDS_x.width
        ysize = self.imgDS_x.height
        nband = self.imgDS_x.count

        self.imgPathY = imgPathY
        self.imgDS_y = rasterio.open(imgPathY)
        if self.imgDS_y is None:
            print('No such a Image file:{}'.format(imgPathY))
            sys.exit(0)
        xsize2 = self.imgDS_y.width
        ysize2 = self.imgDS_y.height
        nband2 = self.imgDS_y.count

        if xsize != xsize2 or ysize != ysize2 or nband != nband2:
            print('Image sizes don\'t match')
            sys.exit(0)

        self.transforms = transforms
        self.enhance = enhance

        xstart = list(range(0, xsize, patch_size[0] - 2 * overlap_padding[0]))
        xend = [(x + patch_size[0]  - 2 * overlap_padding[0]) for x in xstart if (x + patch_size[0] - 2 * overlap_padding[0] < xsize)]
        xend.append(xsize)

        ystart = list(range(0, ysize, patch_size[1] - 2 * overlap_padding[1]))
        yend = [(y + patch_size[1] - 2 * overlap_padding[1]) for y in ystart if (y + patch_size[1] - 2 * overlap_padding[1] < ysize)]
        yend.append(ysize)

        self.xstart = xstart
        self.xend = xend
        self.ystart = ystart
        self.yend = yend

        self.patch_size = patch_size
        self.overlap_padding = overlap_padding

        self.refPath = refPath
        if refPath is not None:
            self.imgDS_ref = rasterio.open(refPath)
            if self.imgDS_ref is None:
                print('No such a Image file:{}'.format(refPath))
                sys.exit(0)
            xsize3 = self.imgDS_ref.width
            ysize3 = self.imgDS_ref.height
            nband3 = self.imgDS_ref.count
            if xsize != xsize3 or ysize != ysize3 or nband3 != 1:
                print('Reference sizes don\'t match image')
                sys.exit(0)
        else:
            self.imgDS_ref = None

        self.outPath = outPath
        self.outDS = None
        self.outProfile = None

    def __getitem__(self, item):
        xitem_count, yitem_count = self.patch_count()

        item_x = math.floor(item / yitem_count)
        item_y = item % yitem_count

        slice, slice_read, slice_write = self.slice_assign(item_x, item_y)

        xsize, ysize, nband = self.size()

        # Read using rasterio window
        window_read = Window(slice_read[0], slice_read[1], slice_read[2], slice_read[3])
        
        tmp_x = self.imgDS_x.read(window=window_read)
        tmp_y = self.imgDS_y.read(window=window_read)

        tmp_x = np.array(tmp_x, dtype=float)
        tmp_y = np.array(tmp_y, dtype=float)

        if self.enhance is not None:
            tmp_x = self.enhance(tmp_x, switch=1)
            tmp_y = self.enhance(tmp_y, switch=2)

        msImage_x = np.zeros((nband, self.patch_size[1], self.patch_size[0]), dtype=float)
        msImage_y = np.zeros((nband, self.patch_size[1], self.patch_size[0]), dtype=float)

        msImage_x[:, slice_write[1]:slice_write[1] + slice_write[3],
        slice_write[0]:slice_write[0] + slice_write[2]] = tmp_x
        msImage_y[:, slice_write[1]:slice_write[1] + slice_write[3],
        slice_write[0]:slice_write[0] + slice_write[2]] = tmp_y

        msImage_x = torch.from_numpy(msImage_x).float()
        msImage_y = torch.from_numpy(msImage_y).float()
        item = torch.tensor(item)

        if self.transforms is not None:
            msImage_x, sync = self.transforms(msImage_x)
            msImage_y, sync = self.transforms(msImage_y, sync)

        refImage = np.zeros((1, self.patch_size[1], self.patch_size[0]), dtype=float)
        if self.imgDS_ref is not None:
            tmp_ref = self.imgDS_ref.read(window=window_read)
            refImage[:, slice_write[1]:slice_write[1] + slice_write[3],
            slice_write[0]:slice_write[0] + slice_write[2]] = tmp_ref
        refImage = torch.from_numpy(refImage).float()

        return msImage_x, msImage_y, item, refImage

    def __len__(self):
        return len(self.xstart) * len(self.ystart)

    def patch_count(self):
        return len(self.xstart), len(self.ystart)

    def size(self):
        xsize = self.imgDS_x.width
        ysize = self.imgDS_x.height
        nband = self.imgDS_x.count
        return xsize, ysize, nband

    def slice_assign(self, item_x, item_y):

        pad = self.overlap_padding
        xsize, ysize, nband = self.size()

        xstart = self.xstart[item_x]
        xend = self.xend[item_x]
        ystart = self.ystart[item_y]
        yend = self.yend[item_y]
        slice = (xstart, ystart, xend - xstart, yend - ystart)

        x_ori = 0 if xstart - pad[0] > 0 else pad[0]
        y_ori = 0 if ystart - pad[1] > 0 else pad[1]

        xstart = xstart - pad[0] if xstart - pad[0] > 0 else 0
        ystart = ystart - pad[1] if ystart - pad[1] > 0 else 0
        xend = xend + pad[0] if xend + pad[0] < xsize else xsize
        yend = yend + pad[1] if yend + pad[1] < ysize else ysize
        slice_read = (xstart, ystart, xend - xstart, yend - ystart)

        slice_write = (x_ori, y_ori, xend - xstart, yend - ystart)

        return slice, slice_read, slice_write

    def RasterioWriteDefault(self, outImage, item):
        # Only write one-band image

        if self.outPath == None:
            dir, fname = os.path.split(self.imgPathX)
            fname, ext = os.path.splitext(fname)
            fname = "{}_cmp{}".format(fname, ext)
            outPath = os.path.join(dir, fname)
            self.outPath = outPath

        xsize, ysize, nband = self.size()

        if self.outDS is None:
            # Create output profile based on input
            outProfile = self.imgDS_x.profile.copy()
            outProfile.update({
                'driver': 'GTiff',
                'height': ysize,
                'width': xsize,
                'count': 1,
                'dtype': 'float32'
            })
            self.outDS = rasterio.open(self.outPath, 'w', **outProfile)
            if self.outDS is None:
                print("Cannot create output raster")
                sys.exit(0)

        xitem_count, yitem_count = self.patch_count()

        item_x = math.floor(item / yitem_count)
        item_y = item % yitem_count

        slice, slice_read, slice_write = self.slice_assign(item_x, item_y)

        pad = self.overlap_padding
        self.outDS.write(outImage[0, pad[1]:pad[1]+slice[3], pad[0]:pad[0]+slice[2]], 1, window=Window(slice[0], slice[1], slice[2], slice[3]))

    def RasterioWrite(self, outImage, item, outDS=None):

        if outDS is None:
            self.RasterioWriteDefault(outImage.numpy(), item)
            return

        xitem_count, yitem_count = self.patch_count()

        item_x = math.floor(item / yitem_count)
        item_y = item % yitem_count

        slice, slice_read, slice_write = self.slice_assign(item_x, item_y)

        pad = self.overlap_padding

        for b in range(outDS.count):
            outDS.write(outImage[b, pad[1]:pad[1] + slice[3], pad[0]:pad[0] + slice[2]], b + 1, window=Window(slice[0], slice[1], slice[2], slice[3]))

    def close(self):
        """Close all opened rasterio datasets"""
        if self.imgDS_x:
            self.imgDS_x.close()
        if self.imgDS_y:
            self.imgDS_y.close()
        if self.imgDS_ref:
            self.imgDS_ref.close()
        if self.outDS:
            self.outDS.close()


# Dataset to read remote sensing images with rasterio, and also the regional reference
class RasterioDataset_RSS(Dataset):

    def __init__(self, imgPathX, imgPathY, regionPath=None, refPath=None, outPath=None, transforms=None, enhance=None, patch_size=(200, 200), overlap_padding=(10, 10)):

        super(RasterioDataset_RSS, self).__init__()
        self.DS = RasterioDataset(imgPathX, imgPathY, refPath=refPath, outPath=outPath, transforms=transforms, enhance=enhance, patch_size=patch_size, overlap_padding=overlap_padding)
        self.ds_len = self.DS.__len__()
        self.regionPath = regionPath
        self.patch_size = patch_size

        if regionPath is not None:
            self.imgDS_region = rasterio.open(regionPath)
            if self.imgDS_region is None:
                print('No such a Image file:{}'.format(regionPath))
                sys.exit(0)
            xsize = self.imgDS_region.width
            ysize = self.imgDS_region.height
            nband = self.imgDS_region.count
            if xsize != self.DS.size()[0] or ysize != self.DS.size()[1] or nband != 1:
                print('Reference sizes don\'t match image')
                sys.exit(0)
        else:
            self.imgDS_region = None

    def __getitem__(self, item):
        msImage_x, msImage_y, item, refImage = self.DS.__getitem__(item)

        xitem_count, yitem_count = self.DS.patch_count()

        item_x = math.floor(item / yitem_count)
        item_y = item % yitem_count
        slice, slice_read, slice_write = self.DS.slice_assign(item_x, item_y)

        regionImage = np.zeros((1, self.patch_size[1], self.patch_size[0]), dtype=float)
        if self.imgDS_region is not None:
            window_read = Window(slice_read[0], slice_read[1], slice_read[2], slice_read[3])
            tmp_ref = self.imgDS_region.read(window=window_read)
            regionImage[:, slice_write[1]:slice_write[1] + slice_write[3],
            slice_write[0]:slice_write[0] + slice_write[2]] = tmp_ref
        regionImage[regionImage > 125] = 1
        regionImage = torch.from_numpy(regionImage).float()

        return msImage_x, msImage_y, item, refImage, regionImage

    def __len__(self):
        return self.ds_len

    def RasterioWrite(self, outImage, item, outDS=None):
        self.DS.RasterioWrite(outImage, item, outDS)

    def close(self):
        """Close all opened rasterio datasets"""
        self.DS.close()
        if self.imgDS_region:
            self.imgDS_region.close()


# dataset to read images
class WHU_Dataset(Dataset):

    # 初始化
    def __init__(self, imgDirX, imgDirY, refDir, labelDir, label_selected='-1', scale=None, transforms=None):
        super(WHU_Dataset, self).__init__()

        # label_selected: '1' all the CHANGED images in the label list
        # label_selected: '0' all the UNCHANGED images in the label list
        # label_selected: '-1' all the images in the label list
        # label_selected: '-2' all the images no matter whether in the label list

        labelPath = os.path.join(labelDir, 'label.txt')
        with open(labelPath) as f:
            data = f.readlines()
            label_list = []
            for line in data:
                label_list.append(line.strip('\n').split(','))
        self.label_list = label_list

        imgFileNameX = [x for x in os.listdir(imgDirX) if self.is_image_file(x) and self.is_image_label(x, label_selected)]
        imgFileNameY = [y for y in os.listdir(imgDirY) if self.is_image_file(y) and self.is_image_label(y, label_selected)]
        # imgFileNameR = [r for r in os.listdir(refDir) if self.is_image_file(r) and self.is_image_label(r, label_selected)]

        self.label_list = self.label_list_arrange(imgFileNameX)

        if imgFileNameX != imgFileNameY:
            print('The multi-temporal images don\'t match')
            sys.exit(1)

        self.imgPathX = [os.path.join(imgDirX, x) for x in imgFileNameX]
        self.imgPathY = [os.path.join(imgDirY, y) for y in imgFileNameY]
        self.RefPath = [os.path.join(refDir, r) for r in imgFileNameX]

        self.transforms = transforms
        self.scale = scale

        self.meansX = []
        self.stdX = []
        self.meansY = []
        self.stdY = []

    def __getitem__(self, item):

        imgX = Image.open(self.imgPathX[item])
        imgY = Image.open(self.imgPathY[item])

        imgX = np.array(imgX, dtype='float32')
        imgY = np.array(imgY, dtype='float32')

        imgX = imgX.transpose((2, 0, 1))
        imgY = imgY.transpose((2, 0, 1))

        label_item = self.label_list[item]
        if int(label_item[3]) == 1:
            Ref = Image.open(self.RefPath[item])
            Ref = np.array(Ref)
            Ref[Ref > 0] = 1
            Ref = np.expand_dims(Ref, 0)
        else:
            Ref = np.zeros((1, imgX.shape[1], imgX.shape[2]))

        if self.scale is not None:
            imgX = self.scale(imgX, switch=1)
            imgY = self.scale(imgY, switch=2)

        imgX = torch.from_numpy(imgX).float()
        imgY = torch.from_numpy(imgY).float()
        Ref = torch.from_numpy(Ref).float()
        item = torch.tensor(item)
        label_list = [int(x) for x in self.label_list[item][1:]]
        label = torch.tensor(label_list)

        if self.transforms is not None:
            imgX, sync = self.transforms(imgX)
            imgY, sync = self.transforms(imgY, sync)

        return imgX, imgY, Ref, item, label

    def __len__(self):
        return len(self.imgPathX)

    def getFileName(self, item):
        path, imgFileName = os.path.split(self.imgPathX[item])
        return imgFileName

    # the ext name to indicate image
    def is_image_file(self, filename):
        return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.tif'])

    # function to filter images according to "label_selected"
    def is_image_label(self, filename, label_selected):
        if label_selected == '-2':
            return True

        for label_item in self.label_list:
            if filename in label_item:
                if label_selected == '-1':
                    return True
                if label_item[3] == label_selected:
                    return True
                else:
                    return False

        return False

    def label_list_arrange(self, filename_list):
        label_list = []
        for filename in filename_list:
            label_temp = [filename, '-1', '-1', '-2']
            for label_item in self.label_list:
                if filename in label_item:
                    label_temp = label_item
                    break
            label_list.append(label_temp)
        return label_list



# dataset to load changed pairs and unchanged pairs in weakly supervised change detection task
# in CHANGED and UNCHANGED samples, the one with larger count is selected as the base
# the other one with smaller count is selected by random ordering and repeating
class WHU_Dataset_WSS(Dataset):

    def __init__(self, imgDirX, imgDirY, refDir, labelDir, scale=None, transforms=None, random_assign=True):
        # random_assign = False, order_reset() should be call in every epoch to confirm random matching between CHANGED samples and UNCHANGED samples
        #   every samples will be used in this pattern
        # random_assign = True, the one with smaller count will be selected randomly in each __getitem__()
        #   maybe not all samples will be used in this pattern
        super(WHU_Dataset_WSS, self).__init__()
        self.cDS = WHU_Dataset(imgDirX, imgDirY, refDir, labelDir, scale=scale, label_selected='1')
        self.ncDS = WHU_Dataset(imgDirX, imgDirY, refDir, labelDir, scale=scale, label_selected='0', transforms=transforms)
        self.cds_len = self.cDS.__len__()
        self.ncds_len = self.ncDS.__len__()
        self.random_assign = random_assign
        if random_assign == False:
            self.order_reset()

    # repeat the sample list of the CHANGED/UNCHANGED class with smaller count to match the other one with larger count
    def order_reset(self):
        if self.cds_len > self.ncds_len:
            order_temp = [i for i in range(self.ncds_len)]
            iter = math.ceil(self.cds_len / self.ncds_len)
            ncds_order = []
            for i in range(iter):
                random.shuffle(order_temp)
                ncds_order = ncds_order + order_temp
            self.ncds_order = ncds_order[:self.cds_len]
            self.cds_order = [i for i in range(self.cds_len)]
        else:
            order_temp = [i for i in range(self.cds_len)]
            iter = math.ceil(self.ncds_len / self.cds_len)
            cds_order = []
            for i in range(iter):
                random.shuffle(order_temp)
                cds_order = cds_order + order_temp
            self.cds_order = cds_order[:self.ncds_len]
            self.ncds_order = [i for i in range(self.ncds_len)]

    def __getitem__(self, item):
        if self.random_assign == False:
            item_ncds = self.ncds_order[item]
            item_cds = self.cds_order[item]
        else:
            if self.cds_len > self.ncds_len:
                item_cds = item
                item_ncds = random.randint(0, self.ncds_len - 1)
            else:
                item_ncds = item
                item_cds = random.randint(0, self.cds_len - 1)

        cds_data = self.cDS.__getitem__(item_cds)
        ncds_data = self.ncDS.__getitem__(item_ncds)

        return cds_data, ncds_data

    def __len__(self):
        return max(self.cds_len, self.ncds_len)