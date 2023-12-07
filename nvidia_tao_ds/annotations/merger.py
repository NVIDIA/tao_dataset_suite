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

"""Annotation Merger."""
from abc import ABC, abstractmethod
import json
from pycocotools.coco import COCO
import logging
logger = logging.getLogger(__name__)


class Merger(ABC):
    """Base Merger class."""

    @abstractmethod
    def __init__(self):
        """Init."""
        pass

    @abstractmethod
    def merge(self):
        """Merge annotation files."""
        pass


class COCOMerger(Merger):
    """COCO Merger class."""

    def __init__(self, annotation_list, same_categories=True):
        """Init."""
        # TODO(@yuw): to enable more configs
        assert same_categories
        assert annotation_list or isinstance(annotation_list, list), "Annotation list is empty!"
        self.same_categories = same_categories
        self.annotation_list = annotation_list
        self.json_list = list(map(lambda x: COCO(x).dataset, annotation_list))

    def merge(self, output_path):
        """Merge COCO json files."""
        def helper(json_list):
            if len(json_list) == 1:
                return json_list[0]
            if len(json_list) == 2:
                assert json_list[0]['categories'] == json_list[1]['categories']
                output = {}
                output['images'] = []
                output['annotations'] = []
                output['categories'] = json_list[0]['categories']

                new_img_id, new_ann_id = 1, 1
                img_id_map = {}
                for i, data in enumerate(json_list):
                    logger.info(
                        f"Input #{i}: {len(data['images'])} images, {len(data['annotations'])} annotations"
                    )

                    for image in data["images"]:
                        img_id_map[image["id"]] = new_img_id
                        image["id"] = new_img_id
                        output["images"].append(image)
                        new_img_id += 1

                    for annotation in data["annotations"]:
                        annotation["id"] = new_ann_id
                        annotation["image_id"] = img_id_map[annotation["image_id"]]
                        output["annotations"].append(annotation)
                        new_ann_id += 1
                return output

            mid = len(json_list) // 2
            left_json = helper(json_list[:mid])
            right_json = helper(json_list[mid:])
            return helper([left_json, right_json])

        merged_json = helper(self.json_list)
        logger.info(f"Writing to {output_path}")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(merged_json, f, indent=4, ensure_ascii=False)

        # verfiy output json
        assert COCO(output_path)
