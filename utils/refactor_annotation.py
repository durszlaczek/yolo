import xml.etree.cElementTree as ET
import re
import pandas as pd


def preprocess_line(line, root_path, outfolder, folder='train_image_folder'):

    root = ET.Element('annotation')
    ET.SubElement(root, 'folder').text = folder
    ET.SubElement(root, 'filename').text = line['filename']
    ET.SubElement(root, 'path').text = root_path + folder + line['filename']
    size = ET.SubElement(root, 'size')

    ET.SubElement(size, 'width').text = str(line['width'])
    ET.SubElement(size, 'height').text = str(line['height'])
    ET.SubElement(size, 'depth').text = str(3)

    object = ET.SubElement(root, 'object')

    ET.SubElement(object, 'name').text = 'rect'
    bndbox = ET.SubElement(object, 'bndbox')

    ET.SubElement(bndbox, 'xmin').text = str(line['xmin'])
    ET.SubElement(bndbox, 'ymin').text = str(line['ymin'])
    ET.SubElement(bndbox, 'xmax').text = str(line['xmax'])
    ET.SubElement(bndbox, 'ymax').text = str(line['ymax'])

    tree = ET.ElementTree(root)

    tree.write(root_path + outfolder + re.sub('.jpg', '.xml', line['filename']))

if __name__ == '__main__':
    labels = pd.read_csv('/Users/aga/Documents/Projects/keras-yolo3/train_annot_folder/labels.csv')

    for i, elem in labels.iterrows():
        preprocess_line(elem, '/Users/aga/Documents/Projects/keras-yolo3/', 'labels/')