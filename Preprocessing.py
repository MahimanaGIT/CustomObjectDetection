'''
Author: Mahimana Bhatt

This script converts xml to csv and also make a file named "label_map.pbtxt" which is mapping of class names to class ID numbers

Directory from which the files should be executed:
objectDetection/ i.e. the folder containing this file

The file will open ./data folder in which it will find test_labels in which all test image xml's will be saved in a csv file in ./data folder and similar for train_labels
label_map.pbtxt is also saved in ./data folder

ALL OUTPUT FILES ARE STORED IN 

./data/
'''

from __future__ import division, print_function, absolute_import

import pandas as pd
import numpy as np
import csv
import cv2 
import re
import os
import glob
import xml.etree.ElementTree as ET

import io
import tensorflow.compat.v1 as tf

print(tf.__version__)

def xml_to_csv(path):
  classes_names = []
  xml_list = []

  for xml_file in glob.glob(path + '/*.xml'):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    for member in root.findall('object'):
      classes_names.append(member[0].text)
      value = (root.find('filename').text,
               int(root.find('size')[0].text),
               int(root.find('size')[1].text),
               member[0].text,
               int(member[4][0].text),
               int(member[4][1].text),
               int(member[4][2].text),
               int(member[4][3].text))
      xml_list.append(value)
  column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
  xml_df = pd.DataFrame(xml_list, columns=column_name) 
  classes_names = list(set(classes_names))
  classes_names.sort()
  return xml_df, classes_names


if __name__ == '__main__':
  for label_path in ['./data/train_labels', './data/test_labels']:
    image_path = os.path.join(os.getcwd(), label_path)
    xml_df, classes = xml_to_csv(label_path)
    xml_df.to_csv("{}.csv".format(label_path), index=None)
    print("Successfully converted {} xml to csv.".format(label_path))

  label_map_path = os.path.join("./data/label_map.pbtxt")
  pbtxt_content = ""

  for i, class_name in enumerate(classes):
      pbtxt_content = (
          pbtxt_content
          + "item {{\n    id: {0}\n    name: '{1}'\n}}\n\n".format(i + 1, class_name)
      )

  pbtxt_content = pbtxt_content.strip()
  with open(label_map_path, "w") as f:
      f.write(pbtxt_content)
  

