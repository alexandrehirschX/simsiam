# By Alexandre Hirsch
from cmath import e
import os.path
from typing import Any, Callable, Optional, Tuple, List
import torchvision.datasets as datasets
import torchvision.io as io
from PIL import Image, ImageDraw 
import numpy as np
import torch


class CocoDetection_ex(datasets.CocoDetection):

    def _load_image(self, id: int) -> Tuple[np.ndarray, int, int]:
        img = self.coco.loadImgs(id)[0]
        return np.array(io.read_image(os.path.join(self.root, img["file_name"])), dtype=np.float32)/255, img["width"], img["height"]

    def _load_target(self, id: int) -> List[Any]:
        return self.coco.loadAnns(self.coco.getAnnIds(imgIds=id, iscrowd=False))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        id = self.ids[index]
        image, width, height = self._load_image(id)
        target = self._load_target(id)

        mask = Image.new('1', (width, height), 0)
        for s in target:
            seg = s['segmentation'][0]
            poly = list(zip(*[iter(seg)]*2))
            ImageDraw.Draw(mask).polygon(poly, outline=1, fill=1)

        M = np.expand_dims(np.array(mask), axis=0)
        image = torch.tensor(np.concatenate((image, M), axis=0))

        if self.transforms is not None:
            try:
                image, target = self.transforms(image, target)
            except Exception as e:
                print(e)
                print(f'\n\n\n{index = }, {id = }\n\n\n')

        return image, target