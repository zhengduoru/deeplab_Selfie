# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 08:56:25 2018

@author: zdr2535
"""
# 讲Notebook脚本改为.py文件，并且能够批量预测。
import os
import tarfile
import cv2
import numpy as np
from PIL import Image
from skimage import measure, color
import face_recognition
import tensorflow as tf

if tf.__version__ < '1.5.0':
    raise ImportError('Please upgrade your tensorflow installation to v1.5.0 or newer!')

# Dataset names.
_CITYSCAPES = 'cityscapes'
_PASCAL = 'pascal'

# Max number of entries in the colormap for each dataset.
_DATASET_MAX_ENTRIES = {
    _CITYSCAPES: 19,
    _PASCAL: 256,
}


def create_cityscapes_label_colormap():
    """Creates a label colormap used in CITYSCAPES segmentation benchmark.

    Returns:
      A Colormap for visualizing segmentation results.
    """
    colormap = np.asarray([
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [70, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
    ])
    return colormap


def get_pascal_name():
    return _PASCAL


def get_cityscapes_name():
    return _CITYSCAPES


def bit_get(val, idx):
    """Gets the bit value.

    Args:
      val: Input value, int or numpy int array.
      idx: Which bit of the input val.

    Returns:
      The "idx"-th bit of input val.
    """
    return (val >> idx) & 1


def create_pascal_label_colormap():
    """Creates a label colormap used in PASCAL VOC segmentation benchmark.

    Returns:
      A Colormap for visualizing segmentation results.
    """
    colormap = np.zeros((_DATASET_MAX_ENTRIES[_PASCAL], 3), dtype=int)
    ind = np.arange(_DATASET_MAX_ENTRIES[_PASCAL], dtype=int)

    for shift in reversed(range(8)):
        for channel in range(3):
            colormap[:, channel] |= bit_get(ind, channel) << shift
        ind >>= 3

    return colormap


def create_label_colormap(dataset=_PASCAL):
    """Creates a label colormap for the specified dataset.

    Args:
      dataset: The colormap used in the dataset.

    Returns:
      A numpy array of the dataset colormap.

    Raises:
      ValueError: If the dataset is not supported.
    """
    if dataset == _PASCAL:
        return create_pascal_label_colormap()
    elif dataset == _CITYSCAPES:
        return create_cityscapes_label_colormap()
    else:
        raise ValueError('Unsupported dataset.')


def label_to_color_image(label, dataset=_PASCAL):
    """Adds color defined by the dataset colormap to the label.

    Args:
      label: A 2D array with integer type, storing the segmentation label.
      dataset: The colormap used in the dataset.

    Returns:
      result: A 2D array with floating type. The element of the array
        is the color indexed by the corresponding element in the input label
        to the PASCAL color map.

    Raises:
      ValueError: If label is not of rank 2 or its value is larger than color
        map maximum entry.
    """
    if label.ndim != 2:
        raise ValueError('Expect 2-D input label')

    if np.max(label) >= _DATASET_MAX_ENTRIES[dataset]:
        raise ValueError('label value too large.')

    colormap = create_label_colormap(dataset)
    return colormap[label]


# Needed to show segmentation colormap labels


model_path = './deeplabv3_pascal_trainval_2018_01_04.tar.gz.tar'
IMAGE_DIR = './test'

_FROZEN_GRAPH_NAME = 'frozen_inference_graph'


class DeepLabModel(object):
    """Class to load deeplab model and run inference."""

    INPUT_TENSOR_NAME = 'ImageTensor:0'
    OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
    INPUT_SIZE = 513

    def __init__(self, tarball_path):
        """Creates and loads pretrained deeplab model."""
        self.graph = tf.Graph()

        graph_def = None
        # Extract frozen graph from tar archive.
        tar_file = tarfile.open(tarball_path)
        for tar_info in tar_file.getmembers():
            if _FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
                file_handle = tar_file.extractfile(tar_info)
                graph_def = tf.GraphDef.FromString(file_handle.read())
                break

        tar_file.close()

        if graph_def is None:
            raise RuntimeError('Cannot find inference graph in tar archive.')

        with self.graph.as_default():
            tf.import_graph_def(graph_def, name='')

        self.sess = tf.Session(graph=self.graph)

    def run(self, image):
        """Runs inference on a single image.

        Args:
            image: A PIL.Image object, raw input image.

        Returns:
            resized_image: RGB image resized from original input image.
            seg_map: Segmentation map of `resized_image`.
        """
        width, height = image.size
        resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
        target_size = (int(resize_ratio * width), int(resize_ratio * height))
        resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
        batch_seg_map = self.sess.run(
            self.OUTPUT_TENSOR_NAME,
            feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
        seg_map = batch_seg_map[0]

        return resized_image, seg_map, width, height


with tf.device('/gpu:0'):
    model = DeepLabModel(model_path)

LABEL_NAMES = np.asarray([
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
    'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
    'train', 'tv'
])

FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
FULL_COLOR_MAP = label_to_color_image(FULL_LABEL_MAP)


def blur_edge(img, seg_map, dilate_ksize=(5, 5), blur_ksize=(5, 5), sigma=1):
    '''
    输入的img是原图，输出的img在seg_map边缘模糊过的img
    '''
    _, contours, _ = cv2.findContours(seg_map, cv2.RETR_TREE,
                                      cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0]
    thick_contours = np.zeros(img.shape[:2])
    thick_contours[contours[:, 0, 1], contours[:, 0, 0]] = 255
    img_tmp = cv2.dilate(thick_contours, np.ones(dilate_ksize, np.uint8), iterations=1)
    img[np.where(img_tmp == 255)] = cv2.GaussianBlur(img, blur_ksize, sigma)[np.where(img_tmp == 255)]
    return img


def how_to_cut(image, seg_map, orientation, cut_length):
    image1 = image.copy()
    image1[seg_map == 0] = 0
    try:
        face_locations = face_recognition.face_locations(image1)[0]
        if orientation == 'horizontal':
            a = face_locations[1]
            c = image.shape[1] - face_locations[3]
            if c - cut_length > a:
                return [0, -cut_length]
            elif a - cut_length > c:
                return [cut_length, -1]
            else:
                return [int(a / (a + c) * cut_length), -(cut_length - int(a / (a + c) * cut_length))]
        else:
            a = face_locations[0]
            if a - cut_length > image.shape[0] * 0.2:
                return [cut_length, -1]
            else:
                return [0, -cut_length]
    except:
        if orientation == 'horizontal':
            if np.nonzero(seg_map)[1].min() / seg_map.shape[1] < 0.2 and np.nonzero(seg_map)[1].max() / seg_map.shape[
                1] <= 0.8:
                return [0, -cut_length]
            elif np.nonzero(seg_map)[1].min() / seg_map.shape[1] >= 0.2 and np.nonzero(seg_map)[1].max() / \
                    seg_map.shape[1] > 0.8:
                return [cut_length, -1]
            else:
                return [cut_length // 2, -(cut_length - cut_length // 2)]
        else:
            if (np.nonzero(seg_map)[0].min() - cut_length) / seg_map.shape[0] > 0.2:
                return [cut_length, -1]
            else:
                return [0, -cut_length]


def cut_and_resize(image, seg_map, size):
    '''把图片resize到size大小，但是不改变shape，其余地方填空'''
    if image.shape[1] / image.shape[0] - size[0] / size[1] > 0:  # 横着的图
        cut_length = image.shape[1] - size[0] * image.shape[0] // size[1]
        cut = how_to_cut(image, seg_map, 'horizontal', cut_length)
        seg_map = cv2.resize(seg_map[:, cut[0]:cut[1]], size)
        image = cv2.resize(image[:, cut[0]:cut[1], :], size)

    elif image.shape[1] / image.shape[0] - size[0] / size[1] < 0:  # 竖着的图
        cut_length = image.shape[0] - size[1] * image.shape[1] // size[0]
        cut = how_to_cut(image, seg_map, 'vertical', cut_length)
        seg_map = cv2.resize(seg_map[cut[0]:cut[1], :], size)
        image = cv2.resize(image[cut[0]:cut[1], :, :], size)
    else:
        seg_map = cv2.resize(seg_map, size)
        image = cv2.resize(image, size)
    return image, seg_map


def resize_and_pad(image, size, pad_value=0):
    '''把图片resize到size大小，但是不改变shape，其余地方填空'''
    if image.shape[1] / image.shape[0] - size[0] / size[1] > 0 and len(image.shape) == 3:  # 横着的图
        size_real = (size[0], int(image.shape[0] / image.shape[1] * size[0]))
        image = np.pad(cv2.resize(image, size_real), [[size[1] - size_real[1], 0], [0, 0], [0, 0]], 'constant',
                       constant_values=pad_value)
    elif image.shape[1] / image.shape[0] - size[0] / size[1] > 0 and len(image.shape) == 2:  # 横着的图
        size_real = (size[0], int(image.shape[0] / image.shape[1] * size[0]))
        image = np.pad(cv2.resize(image, size_real), [[size[1] - size_real[1], 0], [0, 0]], 'constant',
                       constant_values=pad_value)
    elif image.shape[1] / image.shape[0] - size[0] / size[1] < 0 and len(image.shape) == 3:  # 竖着的图
        size_real = (int(image.shape[1] / image.shape[0] * size[1]), size[1])
        image = np.pad(cv2.resize(image, size_real), [[0, 0],
                                                      [int((size[0] - size_real[0]) // 2),
                                                       size[0] - size_real[0] - int((size[0] - size_real[0]) // 2)],
                                                      [0, 0]],
                       'constant', constant_values=pad_value)
    else:  # 竖着的图
        size_real = (int(image.shape[1] / image.shape[0] * size[1]), size[1])
        image = np.pad(cv2.resize(image, size_real), [[0, 0],
                                                      [int((size[0] - size_real[0]) // 2),
                                                       size[0] - size_real[0] - int((size[0] - size_real[0]) // 2)]],
                       'constant', constant_values=pad_value)
    return image


def image_to_background(image, seg_map, background, positions, fill, default_positions=None, fill_value=None):
    '''
    :param image:
    :param fill_value:
    :param seg_map:
    :param fill:
    :param background: './worldcup/background/id_card1.png'
    :param positions: [x_left, y_top, x_right, y_bottom] or np.array([])
    '''
    if isinstance(positions, list):
        if fill:
            image[np.where(seg_map == 0)] = fill_value
            image = blur_edge(image, seg_map)
        background = cv2.imread(background)
        size = (positions[2] - positions[0], positions[3] - positions[1])
        image, seg_map = cut_and_resize(image, seg_map, size)
        # image = resize_and_pad(image, size, pad_value=fill_value)
        # seg_map = resize_and_pad(seg_map, size)
        if not fill:
            back_size = background.shape[:2]
            image = np.pad(image, [[back_size[0] - size[1], 0],
                                   [(back_size[1] - size[0]) // 2,
                                    back_size[1] - size[0] - (back_size[1] - size[0]) // 2],
                                   [0, 0]], 'constant')
            seg_map = np.pad(seg_map, [[back_size[0] - size[1], 0],
                                       [(back_size[1] - size[0]) // 2,
                                        back_size[1] - size[0] - (back_size[1] - size[0]) // 2]], 'constant')
            background[np.where(seg_map != 0)] = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)[np.where(seg_map != 0)]
        else:
            background[positions[1]:positions[3], positions[0]:positions[2], :] = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return background, seg_map
    if isinstance(positions, np.ndarray):
        if fill:
            image[np.where(seg_map == 0)] = fill_value
            image = blur_edge(image, seg_map)
        default_size = (default_positions[2] - default_positions[0], default_positions[3] - default_positions[1])
        image, seg_map = cut_and_resize(image, seg_map, default_size)
        background = cv2.imread(background)
        height, width = image.shape[:2]
        positions_src = np.array([[0, 0], [0, height - 1], [width - 1, height - 1], [width - 1, 0]])
        h, status = cv2.findHomography(positions_src, positions, cv2.RANSAC, 5.0)
        im_out = cv2.warpPerspective(image, h, (background.shape[1], background.shape[0]), cv2.INTER_LINEAR)
        roi = cv2.imread('./worldcup/background/id_card2_roi.jpg', cv2.IMREAD_GRAYSCALE)
        background[np.where(roi > 100)] = cv2.cvtColor(im_out, cv2.COLOR_RGB2BGR)[np.where(roi > 100)]
        return background, seg_map


def vis_segmentation(image, seg_map, image_name, width, height, background_image):
    image = cv2.resize(image, (width, height))
    seg_map = cv2.resize(seg_map.astype('uint8'), (width, height))
    seg_map[np.where(seg_map != 15)] = 0
    seg_map[np.where(seg_map == 15)] = 255
    # GaussianBlur seg_map
    seg_map = cv2.GaussianBlur(seg_map, (5, 5), 1)
    # 最大连通域
    labels = measure.label(seg_map, connectivity=2)
    props = measure.regionprops(labels)
    seg_map[np.where(labels != props[np.argmax([prop.area for prop in props])].label)] = 0
    # closing, opening
    seg_map = cv2.morphologyEx(seg_map, cv2.MORPH_CLOSE, kernel=np.ones((5, 5), np.uint8))
    seg_map = cv2.morphologyEx(seg_map, cv2.MORPH_OPEN, kernel=np.ones((5, 5), np.uint8))
    # face_detect
    # face_positions = face_recognition.face_locations(image)

    if background_image == 'worldcup':
        # image_to_background
        background, seg_map = image_to_background(image, seg_map, './worldcup/background/worldcup.png',
                                                  [45, 349, 443, 844], fill=False)
        # blur_edge
        background = blur_edge(background, seg_map)
        cv2.imwrite('./worldcup/merge/worldcup/' + image_name.split('.')[0] + '.jpg', background)
    # if background_image == 'ground1':
    # if background_image == 'ground2':
    if background_image == 'id_card1':
        background, seg_map = image_to_background(image, seg_map, './worldcup/background/id_card1.png',
                                                  [181, 222, 345, 446], fill=True, fill_value=255)
        cv2.imwrite('./worldcup/merge/id_card1/' + image_name.split('.')[0] + '.jpg', background)
    if background_image == 'id_card2':
        default_positions = [181, 222, 345, 446]
        positions = np.array([[441, 216], [395, 378], [502, 414], [550, 228]])
        fill_value = [135 + np.random.randint(10), 125 + np.random.randint(10), 115 + np.random.randint(10)]
        background, seg_map = image_to_background(image, seg_map, './worldcup/background/id_card2.jpg',
                                                  positions, default_positions, fill=True, fill_value=fill_value)
        cv2.imwrite('./worldcup/merge/id_card2/' + image_name.split('.')[0] + '.jpg', background)
    if background_image == 'final1':
        background, seg_map = image_to_background(image, seg_map, './worldcup/background/final1.jpg',
                                                  [252, 404, 316, 483], fill=True, fill_value=130)
        cv2.imwrite('./worldcup/merge/final1/' + image_name.split('.')[0] + '.jpg', background)
    if background_image == 'final2':
        background, seg_map = image_to_background(image, seg_map, './worldcup/background/final2.jpg',
                                                  [382, 407, 461, 500], fill=True, fill_value=130)
        cv2.imwrite('./worldcup/merge/final2/' + image_name.split('.')[0] + '.jpg', background)


def run_demo_image(image_name):
    try:
        image_path = os.path.join(IMAGE_DIR, image_name)
        orignal_im = Image.open(image_path)
    except IOError:
        print('Failed to read image from %s.' % image_path)
        return
    print('running deeplab on image %s...' % image_name)
    resized_im, seg_map, width, height = model.run(orignal_im)

    vis_segmentation(np.array(resized_im), seg_map, image_name, width, height, 'worldcup')


def capture_positions(image_path):
    positions = []
    img = cv2.imread(image_path)
    drawing = False
    mode = True
    ix, iy = -1, -1

    def draw_circle(event, x, y, flags, param):
        global ix, iy, drawing, mode
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            positions.append([x, y])

    cv2.namedWindow('image')
    cv2.setMouseCallback('image', draw_circle)
    while 1:
        cv2.imshow('image', img)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('m'):
            mode = not mode
        elif k == ord('q'):
            cv2.destroyAllWindows()
            break
    return positions


images = os.listdir(IMAGE_DIR)

for k, image_name in enumerate(images):  # [:5]:
    run_demo_image(image_name)
