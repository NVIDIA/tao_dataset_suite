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

"""Convert KITTI labels to COCO format"""

from collections import OrderedDict
import os
from pathlib import Path
import pandas as pd
import numpy as np
import ujson
import yaml
from tqdm.auto import tqdm
import csv
import cv2


def read_kitti_labels(label_file):
    """
    Utility function to read a KITTI label file

    Args:
    label_file (string): Label file path

    Returns:
    label_list (list): List of labels
    """
    label_list = []
    if not os.path.exists(label_file):
        raise ValueError(f"Labelfile : {label_file} does not exist")
    with open(label_file, 'r', encoding='utf-8') as lf:
        for row in csv.reader(lf, delimiter=' '):
            label_list.append(row)
    lf.closed
    return label_list


def check_bbox_coordinates(coord, img_h, img_w):
    """
    Utility function to validate the bounding box coordinates

    Args:
    coord (tuple): Bounding box coordinates in KITTI format
    img_h (int): Image height
    img_w (int): Image widith
    label_file (string): Label file path

    Returns:
    Bounding box coordinates
    """
    "Checks coordinates."
    x1, y1, x2, y2 = coord
    x1 = min(max(x1, 0), img_w)
    x2 = min(max(x2, 0), img_w)
    y1 = min(max(y1, 0), img_h)
    y2 = min(max(y2, 0), img_h)
    if x2 > x1 and y2 > y1:
        return [x1, y1, x2, y2]
    return None


def convert_xyxy_to_xywh(coord):
    """
    Utility function to convert bounding box coordinates from KITTI format to COCO

    Args:
    coord (tuple): Bounding box coordinates in KITTI format
    img_h (int): Image height
    img_w (int): Image widith
    label_file (string): Label file path

    Returns:
    Bounding box coordinates in COCO format
    """
    "Checks coordinates."
    x1, y1, x2, y2 = coord
    w, h = x2 - x1, y2 - y1
    return [x1, y1, w, h]


def get_categories(cat_map):
    """
    Function to convert the category map to COCO annotation format

    Args:
    cat_map (dictionary): Category map

    Returns:
    categories_list (list): COCO annotation format of the category map
    """
    categories_list = []
    for i, class_name in enumerate(cat_map):
        category = {
            'id': i + 1,
            'name': class_name
        }
        categories_list.append(category)
    return categories_list


def construct_category_map(label_dir, mapping=None):
    """
    Function to create a category map for the given dataset

    Args:
    label_dir (str): Label directory
    mapping (str): Mapping file

    Returns:
    cat_map (dictionary): Category mapping
    """
    cat_map = OrderedDict()
    if mapping is not None:
        with open(mapping, "r", encoding='utf-8') as f:
            try:
                cat_map_list = yaml.safe_load(f)
            except yaml.YAMLError as e:
                print(e)
            for i in cat_map_list:
                k, v = list(i.items())[0]
                cat_map[k] = v
    else:
        for img in os.listdir(label_dir):
            labels = read_kitti_labels(os.path.join(label_dir, f"{img[:-4]}.txt"))
            df = pd.DataFrame(labels)
            for _, row_p in df.iterrows():
                if row_p[0] not in cat_map:
                    cat_map[row_p[0]] = [row_p[0]]
    return cat_map


def convert_kitti_to_coco(img_dir, label_dir, output_dir, mapping=None, name=None):
    """Function to convert KITTI annotations to COCO format.

    Args:
        img_dir (string): Directory containing the images.
        label_dir (string): Directory containing the KITTI labels
        output_dir (string): Directory to output the COCO annotation file
    """
    annot_list, img_list, skipped_list = [], [], []
    img_id, obj_id = 0, 0
    img_dir = str(Path(img_dir))
    label_dir = str(Path(label_dir))
    project = name or img_dir.split('/')[-2]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cat_map = construct_category_map(label_dir, mapping)
    categories = get_categories(cat_map)
    labels2cat = {label: k for k, v in cat_map.items() for label in v}
    cat2index = {c['name']: c['id'] for c in categories}
    print("category to id mapping:")
    print("*********************")
    print(cat2index)
    print("*********************")

    for img in tqdm(os.listdir(img_dir)):
        if str(Path(img).suffix).lower() not in [".jpg", ".png", ".jpeg"]:
            continue

        labels = os.path.join(label_dir, f"{img[:-4]}.txt")
        img_shape = cv2.imread(os.path.join(img_dir, img)).shape
        height, width = img_shape[0], img_shape[1]

        # update image list
        img_id += 1
        img_dict = {
            "file_name": img,
            "scene_id": project,
            "height": height,
            "width": width,
            "id": img_id
        }
        img_list.append(img_dict)

        # process labels
        bboxes = read_kitti_labels(labels)
        df = pd.DataFrame(bboxes)
        df = df.drop_duplicates()

        # update annotation list
        include_image = False
        for _, row_p in df.iterrows():

            mapped = labels2cat.get(row_p[0], None)
            if not mapped:
                continue

            bbox = np.array(row_p[4:8])
            bbox = bbox.astype(float)

            coord = check_bbox_coordinates(bbox.tolist(), height, width)
            if not coord:
                continue
            include_image = True
            coord = convert_xyxy_to_xywh(coord)
            area = coord[2] * coord[3]
            obj_id += 1
            annot_dict = {
                "bbox": coord,
                "image_id": img_id,
                "scene_id": project,
                "iscrowd": 0,
                "area": area,
                "category_id": cat2index[mapped],
                "id": obj_id
            }
            annot_list.append(annot_dict)

        if not include_image:
            img_skipped = img_list.pop()
            skipped_list.append(img_skipped['file_name'])

    final_dict = {
        "annotations": annot_list,
        "images": img_list,
        "categories": categories
    }
    save_path = os.path.join(output_dir, f"{project}.json")

    with open(save_path, "w", encoding='utf-8') as f:
        ujson.dump(final_dict, f)

    if skipped_list:
        with open(os.path.join(output_dir, 'skipped_files.txt'), 'w', encoding='utf-8') as g:
            g.write('\n'.join(str(fname) for fname in skipped_list))
