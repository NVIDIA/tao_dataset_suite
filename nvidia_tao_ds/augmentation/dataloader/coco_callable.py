# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""kitti loader."""

import os
import numpy as np
from pycocotools import mask
from pycocotools.coco import COCO

from nvidia_tao_ds.augmentation.utils.file_handlers import load_file


class CocoInputCallable:
    """KITTI loader for DALI pipeline."""

    def __init__(self, image_dir, annotation_path, batch_size,
                 include_masks=False, shard_id=0, num_shards=1):
        """Init."""
        self.image_dir = image_dir
        self.coco = COCO(annotation_path)

        self.include_masks = include_masks
        self.batch_size = batch_size
        self.shard_id = shard_id
        self.num_shards = num_shards

        self.samples_per_iter = batch_size * num_shards
        self.size = (self.samples_per_iter - 1 + len(self.coco.dataset['images'])) // self.samples_per_iter * self.samples_per_iter

        self.shard_size = self.size // num_shards
        self.shard_offset = self.shard_size * shard_id
        self.full_iterations = self.shard_size // batch_size

    def __call__(self, sample_info):
        """Call."""
        if sample_info.iteration >= self.full_iterations:
            # Indicate end of the epoch
            raise StopIteration
        sample_idx = sample_info.idx_in_epoch + self.shard_offset
        sample_idx = min(sample_idx, len(self.coco.dataset['images']) - 1)
        image = self.coco.dataset['images'][sample_idx]
        image_path = os.path.join(self.image_dir, image['file_name'])
        image_id = image['id']

        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        boxes, masks = self._get_boxes(ann_ids, image['height'], image['width'])
        encoded_img = load_file(image_path)
        if self.include_masks:
            masks = np.transpose(masks, (1, 2, 0))
        return encoded_img, boxes, np.array([image_id], dtype=np.int32), np.uint8(masks)

    def _get_boxes(self, ann_ids, image_height, image_width):
        if len(ann_ids) == 0:
            return np.float32([[0, 0, 0, 0]]), np.uint8(np.zeros((1, 1, 1)))
        boxes = []
        masks = []
        for ann_id in ann_ids:
            ann = self.coco.loadAnns(ann_id)[0]
            boxes.append(ann['bbox'])
            if self.include_masks:
                if 'segmentation' not in ann:
                    raise ValueError(
                        f"segmentation groundtruth is missing in object: {ann}.")
                # pylygon (e.g. [[289.74,443.39,302.29,445.32, ...], [1,2,3,4]])
                if isinstance(ann['segmentation'], list):
                    rles = mask.frPyObjects(ann['segmentation'],
                                            image_height, image_width)
                    rle = mask.merge(rles)
                elif 'counts' in ann['segmentation']:
                    # e.g. {'counts': [6, 1, 40, 4, 5, 4, 5, 4, 21], 'size': [9, 10]}
                    if isinstance(ann['segmentation']['counts'], list):
                        rle = mask.frPyObjects(ann['segmentation'],
                                               image_height, image_width)
                    else:
                        rle = ann['segmentation']
                else:
                    raise ValueError('Please check the segmentation format.')
                binary_mask = mask.decode(rle)
                masks.append(binary_mask)
            else:
                masks.append(np.zeros((1, 1)))
        return np.float32(boxes), np.uint8(masks)
