import xml.etree.ElementTree as ET

def read_content(xml_file: str):

    tree = ET.parse(xml_file)
    root = tree.getroot()

    list_with_all_boxes = []

    for boxes in root.iter('object'):

        filename = root.find('filename').text

        ymin = int(boxes.find("bndbox/ymin").text) - 1
        xmin = int(boxes.find("bndbox/xmin").text) - 1
        ymax = int(boxes.find("bndbox/ymax").text) - 1
        xmax = int(boxes.find("bndbox/xmax").text) - 1

        list_with_single_boxes = [xmin, ymin, xmax, ymax]
        list_with_all_boxes.append(list_with_single_boxes)

    return filename, list_with_all_boxes