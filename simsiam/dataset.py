# By Alexandre Hirsch
from cmath import e
import os.path
from typing import Any, Callable, Optional, Tuple, List
import torchvision.datasets as datasets
import torchvision.io as io
#from PIL import Image, ImageDraw 
#import numpy as np
import torch
#from detectron2.structures import masks


class CocoDetection_ex(datasets.CocoDetection):

    def __init__(
        self,
        root: str,
        annFile: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, annFile, transform, target_transform, transforms)
        #add iscrowd option
        self.iscrowd = False
        self.ids = [(i["image_id"], i["id"]) for i in self.coco.loadAnns(self.coco.getAnnIds(iscrowd=self.iscrowd))]

    def _load_image(self, id: int) -> torch.Tensor:
        path = os.path.join(self.root, self.coco.loadImgs(id)[0]["file_name"])
        return io.read_image(path, mode=io.image.ImageReadMode.RGB)/255

    def _load_target(self, id: int) -> Any:
        return self.coco.loadAnns(id)[0]
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        image_id, id = self.ids[index]
        image = self._load_image(image_id)
        target = self._load_target(id)
        #replaced detectron2 structures mask method
        mask = torch.from_numpy(self.coco.annToMask(target))[None, :, :]
        image = self.transform(torch.cat((image, mask), dim=0))
        return image, torch.tensor(target['category_id'])