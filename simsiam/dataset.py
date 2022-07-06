# By Alexandre Hirsch
from cmath import e
import os.path
from typing import Any, Callable, Optional, Tuple, List
import torchvision.datasets as datasets
import torchvision.io as io
import torchvision.transforms as transforms
from PIL import Image, ImageDraw 
import numpy as np
import torch


class CocoDetection_ex(datasets.CocoDetection):
    T = transforms.PILToTensor()
    def _load_image(self, id: int) -> Tuple[np.ndarray, int, int]:
        img = self.coco.loadImgs(id)[0]
        #return torch.tensor(Image.open(os.path.join(self.root, img["file_name"])).convert("RGB"))/255, img["width"], img["height"]
        return io.read_image(os.path.join(self.root, img["file_name"]), mode=io.image.ImageReadMode.RGB)/255, img["width"], img["height"]


    def _load_target(self, id: int) -> List[Any]:
        return self.coco.loadAnns(self.coco.getAnnIds(imgIds=id, iscrowd=False))

    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        id = self.ids[index]
        image, width, height = self._load_image(id)
        target = self._load_target(id)
        
        #try with detectron2 rasterize instead
        #get mask (COCO dataset)
        with Image.new('1', (width, height), 0) as mask:
            for s in target:
                polygon_coords = list(zip(*[iter(s['segmentation'][0])]*2))
                ImageDraw.Draw(mask).polygon(polygon_coords, outline=1, fill=1)

        # mask = Image.new('1', (width, height), 0)
        # for s in target:
        #     seg = s['segmentation'][0]
        #     poly = list(zip(*[iter(seg)]*2))
        #     ImageDraw.Draw(mask).polygon(poly, outline=1, fill=1)
        M = self.T(mask)/255
        image = torch.cat((image, M), dim=0)
    
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        return image, target