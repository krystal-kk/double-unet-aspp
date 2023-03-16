import numpy as np
from PIL import Image
from numpy import ndarray
import cv2
import xml.etree.ElementTree as ET
from typing import List
import os
from tqdm import tqdm


def parse_xml(xml_path: str) -> ET.Element:
    tree = ET.parse(xml_path)
    return tree.getroot()


def get_all_points(region: ET.Element) -> ndarray:
    points = []
    for i in range(len(region[1])):
        vertex = region[1][i]
        points.append([float(vertex.attrib['X']), float(vertex.attrib['Y'])])
    return np.array(points, dtype=np.int32)


def xml2mask(xml_path: str, size: List) -> ndarray:
    mask = np.zeros((size))
    root = parse_xml(xml_path)

    for i in range(1, len(root[0][1])):
        region = root[0][1][i]
        pts = get_all_points(region)
        cv2.fillPoly(mask, [pts], i+1)
    
    return mask


def main():
    anno_dir = '/home/xuexi/workspace/datasets/monuseg/test/Annotations'
    save_path = '/home/xuexi/workspace/datasets/monuseg/test/1000_masks'
    xmls = os.listdir(anno_dir)
    for xml in tqdm(xmls):
        file_name = xml.split('.')[0]
        mask = xml2mask(os.path.join(anno_dir, xml), [1000, 1000])
        np.save(f"{save_path}/{file_name}.npy", mask)


if __name__ == "__main__":
    main()
