"""
124-8_8_big_21_0.jpg: JPEG image data, JFIF standard 1.01, aspect ratio, density 1x1, segment length 16, baseline, precision 8, 782x1153, components 3

<annotation>
<folder>unknown</folder>
<filename>124-8_8_big_21_0.jpg</filename>
<path>/Users/dannybickson/Downloads/unknown/unknown/124-8_8_big_21_0.jpg</path>
<source>
<database>Unknown</database>
</source>
<size>
<width>782</width>
<height>1153</height>
<depth>3</depth>
</size>
<segmented>0</segmented>
<object>
<name>an</name>
<pose>Unspecified</pose>
<truncated>0</truncated>
<difficult>0</difficult>
<bndbox>
<xmin>110</xmin>
<ymin>415</ymin>
<xmax>538</xmax>
<ymax>944</ymax>
</bndbox>
</object>
</annotation>
"""
import os
import cv2
import numpy as np
from fastdup.image import get_shape

def image_to_label_img_xml(img_path, cur_label, save_dir=None):

    assert os.path.exists(img_path), '{} does not exist'.format(img_path)
    if save_dir:
        assert os.path.exists(save_dir), '{} does not exist'.format(save_dir)

    img = cv2.imread(img_path)
    assert img is not None, f"Failed to read image {img_path}"
    h, w, c = get_shape(img)
    xml =  f'<annotation>\n'
    xml += f'   <folder>{cur_label}</folder>\n'
    xml += f'   <filename>{os.path.basename(img_path)}</filename>\n'
    xml += f'   <path>{img_path}</path>\n'
    xml += f'   <source>\n'
    xml += f'       <database>Unknown</database>\n'
    xml += f'   </source>\n'
    xml += f'   <size>\n'
    xml += f'       <width>{w}</width>\n'
    xml += f'       <height>{h}</height>\n'
    xml += f'       <depth>{c}</depth>\n'
    xml += f'   </size>\n'
    xml += f'   <segmented>0</segmented>\n'
    xml += f'   <object>\n'
    xml += f'       <name>{cur_label}</name>\n'
    xml += f'       <pose>Unspecified</pose>\n'
    xml += f'       <truncated>0</truncated>\n'
    xml += f'       <difficult>0</difficult>\n'
    xml += f'       <bndbox>\n'
    xml += f'           <xmin>0</xmin>\n'
    xml += f'           <ymin>0</ymin>\n'
    xml += f'           <xmax>{w}</xmax>\n'
    xml += f'           <ymax>{h}</ymax>\n'
    xml += f'       </bndbox>\n'
    xml += f'   </object>\n'
    xml += f'</annotation>\n'

    ext = os.path.basename(img_path).split('.')[-1]
    assert ext.lower() in ['jpg', 'jpeg','png','gif','tif','bmp']
    local_xml = img_path[0:-len(ext)] + 'xml'
    if save_dir is not None:
        local_xml = os.path.join(save_dir, os.path.basename(local_xml))

    with open(local_xml, 'w') as f:
        f.write(xml)
    assert os.path.exists(local_xml)

    return xml


def export_label_classes(labels, save_path):
    with open(os.path.join(save_path, 'classes.txt'), 'w') as f:
        for label in labels:
            f.write(label + '\n')


def do_export_to_labelimg(files, labels, save_path):

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    assert os.path.exists(save_path)

    export_label_classes(list(np.unique(labels)), save_path)

    count = 0
    if labels is None:
        for f in files:
            try:
                image_to_label_img_xml(f, None, save_path)
                count+=1
            except Exception as ex:
                print('Failed to retag file', f, ' with exception', ex)
    else:
        assert len(labels) == len(files), "Number of labels and files should be the same"
        for f,l in zip(files, labels):
            try:
                image_to_label_img_xml(f, l, save_path)
                count+=1
            except Exception as ex:
                print('Failed to retag file', f, ' with exception', ex)

    print('Successfully exported to labeliImg', count, 'files')
    return 0
