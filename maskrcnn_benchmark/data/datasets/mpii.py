import os

import torch
import torch.utils.data
import cv2
import sys
import h5py
import numpy as np

from maskrcnn_benchmark.structures.bounding_box import BoxList


class EventMPIIObjectDataset(torch.utils.data.Dataset):

    CLASSES = (
        "__background__",
        "Car",
    )

    def __init__(self, data_dir, image_set, hdf5_file, npz_file, transforms=None):
        self.root = data_dir
        self.image_set = image_set
        self.transforms = transforms
        
        np_data = np.load(npz_file)
        views = np_data['views']
        imgnames = np_data['imgname']

        self._annopath = os.path.join(self.root, "label_2", "{}.txt")
        self._imgpath = os.path.join(self.root, "image_2", "{}")
        self._eventpath = os.path.join(self.root, "events_2", hdf5_file)
        self._imgsetpath = os.path.join(self.root, "{}.txt")

        self._image_ids = []

        cls = EventMPIIObjectDataset.CLASSES
        self.categories = dict(zip(range(len(cls)), cls))
        self.class_to_ind = dict(zip(cls, range(len(cls))))
        
        with open(self._imgsetpath.format(self.image_set)) as f:
            self.ids = f.readlines()
        self.ids = [x.strip("\n") for x in self.ids]
        self.ids = [x for x in self.ids if self.check_n_labels(x) > 0]
        
        for index in self.ids:
            ind = views[int(index), 2]
            imgname = imgnames[ind]
            self._image_ids.append(imgname)
        
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}

        self.loaded = False
        self.load_hdf5()
        self.close()
        
    def __getitem__(self, index):
        if not self.loaded:
            self.load_hdf5()
        img_id = self.ids[index]
        event_volume = self.event_volumes[int(img_id)]
        volume_shape = event_volume.shape
        
        target = self.get_groundtruth(index)
        target_resized = target.resize((volume_shape[1], volume_shape[0]))
        clipped_target = target_resized.clip_to_image(remove_empty=True)

        if self.transforms is not None:
            event_volume, clipped_target = self.transforms(event_volume, clipped_target)

        return event_volume, clipped_target, index

    def __len__(self):
        return len(self.ids)

    def load_hdf5(self):
        self.data = h5py.File(self._eventpath, 'r')
        self.event_volumes = self.data['davis']['events']
        self.volume_shape = {'height' : self.event_volumes.shape[1],
                             'width'  : self.event_volumes.shape[2]}

        self.loaded = True

    def close(self):
        self.event_volumes = None
        self.data.close()
        self.data = None
        
        self.loaded = False

    def check_n_labels(self, img_id):
        with open(self._annopath.format(img_id), 'r') as label_file:
            anno_list = label_file.readlines()
        anno_list = [x.strip("\n") for x in anno_list]

        n_labels = 0
        for anno in anno_list:
            data = anno.split(' ')
            if data[0] in self.class_to_ind:
                n_labels += 1

        return n_labels
        
    def get_groundtruth(self, index):
        img_id = self.ids[index]

        with open(self._annopath.format(img_id), 'r') as label_file:
            anno_list = label_file.readlines()
        anno_list = [x.strip("\n") for x in anno_list]

        label_list = []
        box_list = []
        for anno in anno_list:
            data = anno.split(' ')
            if not data[0] in self.class_to_ind:
                continue
            label_list.append(self.class_to_ind[data[0]])

            x1 = float(data[4])
            y1 = float(data[5])
            x2 = float(data[6])
            y2 = float(data[7])
            box_list.append((x1, y1, x2, y2))

        label_list = torch.tensor(label_list)
        img = cv2.imread(self._imgpath.format(self._image_ids[index]))
        height, width, _ = img.shape

        target = BoxList(box_list, (width, height), mode='xyxy')
        target.add_field("labels", label_list)
        
        return target

    def get_img_info(self, idx):
        return self.volume_shape
        
