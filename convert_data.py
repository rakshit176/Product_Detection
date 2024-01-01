from imutils import paths
from typing import List
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import xml.etree.ElementTree as ET
from xml.dom import minidom
import cv2

def convert_annotations_to_df(file_path,cols):
    master_df = pd.read_csv(file_path,names=cols)
    unique_class_id = master_df["class_id"].unique()
    train_image_names = [os.path.basename(image_path) for image_path in train_images]
    dataset_df = master_df[master_df["image_name"].isin(train_image_names)]
    dataset_df["image_name"] = dataset_df["image_name"].map(lambda x: os.path.join(str(os. getcwd()) + "/Train_data/train/" + x))
    return dataset_df , unique_class_id

def generate_xml(root, row):
    obj = ET.SubElement(root, 'object')
    ET.SubElement(obj, 'name').text = str(row['class_id'])
    ET.SubElement(obj, "pose").text = "Unspecified"
    ET.SubElement(obj, "truncated").text = "0"
    ET.SubElement(obj, "truncated").text = "difficult"
    bbox = ET.SubElement(obj, 'bndbox')
    ET.SubElement(bbox, 'xmin').text = str(row['xmin'])
    ET.SubElement(bbox, 'ymin').text = str(row['ymin'])
    ET.SubElement(bbox, 'xmax').text = str(row['xmax'])
    ET.SubElement(bbox, 'ymax').text = str(row['ymax'])
    
def get_image_size(row):
    print(row['image_name'])
    image = cv2.imread(row['image_name'])
    height, width, channels = image.shape
    return width , height ,channels

def process_and_save_xml(df):
    xml_dict = {}

    for index, row in df.iterrows():
        image_name = os.path.basename(row['image_name'])
        if image_name not in xml_dict:
            width , height ,channels = get_image_size(row)
            xml_dict[image_name] = ET.Element('annotation')
            ET.SubElement(xml_dict[image_name], 'folder').text = 'images'
            ET.SubElement(xml_dict[image_name], 'filename').text = image_name
            ET.SubElement(xml_dict[image_name], 'path').text = row['image_name']
            size = ET.SubElement(xml_dict[image_name], 'size')
            ET.SubElement(size, 'width').text = str(width)
            ET.SubElement(size, 'height').text = str(height)
            ET.SubElement(size, 'depth').text = str(channels)  # Assuming RGB images
            generate_xml(xml_dict[image_name], row)
        else:
            generate_xml(xml_dict[image_name], row)

    for image_name, xml_tree in xml_dict.items():
        xml_str = ET.tostring(xml_tree).decode()
        xml_str = minidom.parseString(xml_str).toprettyxml(indent="    ")
        
        with open(f'D:/BERT_LABS/Train_data/train/{image_name.split(".")[0]}.xml', 'w') as xml_file:
            xml_file.write(xml_str)


if __name__ == "__main__":  
    train_images = list(paths.list_images("./Train_data/train"))
    cols = ["image_name", "xmin", "ymin", "xmax", "ymax", "class_id"]
    dataset_df , unique_class_id = convert_annotations_to_df(r'D:\BERT_LABS\grocerydataset-master\grocerydataset-master\annotations.csv',cols)
    process_and_save_xml(dataset_df)