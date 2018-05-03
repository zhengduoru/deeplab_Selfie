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
import tensorflow as tf

if tf.__version__ < '1.5.0':
    raise ImportError('Please upgrade your tensorflow installation to v1.5.0 or newer!')

model_path = './deeplabv3_pascal_trainval_2018_01_04.tar.gz.tar'
IMAGE_DIR = './test'
facedetect_path = '/home/duoduo/opencv/data/haarcascades/haarcascade_frontalface_alt.xml'

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
        face_cascade = cv2.CascadeClassifier(facedetect_path)

        face_locations_tmp = face_cascade.detectMultiScale(image1)[0]  # x, y, w. h
        face_locations = face_locations_tmp[1], face_locations_tmp[0] + face_locations_tmp[2], face_locations_tmp[1] + \
                         face_locations_tmp[3], face_locations_tmp[0]
        if orientation == 'horizontal':
            a = face_locations[3]  # left
            c = image1.shape[1] - face_locations[1]
            if c - cut_length > a:
                return [0, -cut_length]
            elif a - cut_length > c:
                return [cut_length, -1]
            else:
                return [int(a / (a + c) * cut_length), -(cut_length - int(a / (a + c) * cut_length))]
        else:
            a = face_locations[0]
            if a - cut_length > image1.shape[0] * 0.2:
                return [cut_length, -1]
            else:
                return [0, -cut_length]
    except:
        raise IOError('no face')
        # if orientation == 'horizontal':
        #     if np.nonzero(seg_map)[1].min() / seg_map.shape[1] < 0.2 and np.nonzero(seg_map)[1].max() / seg_map.shape[
        #         1] <= 0.8:
        #         return [0, -cut_length]
        #     elif np.nonzero(seg_map)[1].min() / seg_map.shape[1] >= 0.2 and np.nonzero(seg_map)[1].max() / \
        #             seg_map.shape[1] > 0.8:
        #         return [cut_length, -1]
        #     else:
        #         return [cut_length // 2, -(cut_length - cut_length // 2)]
        # else:
        #     if (np.nonzero(seg_map)[0].min() - cut_length) / seg_map.shape[0] > 0.2:
        #         return [cut_length, -1]
        #     else:
        #         return [0, -cut_length]


def how_to_cut_with_face(image, seg_map, size, face_square=10000, w_mul=1, h_mul=1.5):
    ratio = size[1] / size[0]
    image1 = image.copy()
    image1[seg_map == 0] = 0
    if not np.sum(seg_map[0, :]):  # 不贴上
        try:
            face_cascade = cv2.CascadeClassifier(facedetect_path)

            face_locations_tmp = face_cascade.detectMultiScale(image1)[0]  # x, y, w. h
            # resize_ratio = np.sqrt(face_square / (face_locations_tmp[2] * face_locations_tmp[3]))
            # resize = (int(image.shape[1] * resize_ratio), int(image.shape[0] * resize_ratio))
            face_locations = (face_locations_tmp[1], face_locations_tmp[0] + face_locations_tmp[2], face_locations_tmp[1] + face_locations_tmp[3], face_locations_tmp[0])
            # image = cv2.resize(image,resize)
            # seg_map = cv2.resize(seg_map, resize)
            width = face_locations[1] - face_locations[3]
            height = face_locations[2] - face_locations[0]
            bottom_dst = min(int(face_locations[2] + height * h_mul), image.shape[0])
            left_dst = max(int(face_locations[3] - width * w_mul), 0)
            right_dst = min(int(face_locations[1] + width * w_mul), image.shape[1])
            width_dst = right_dst - left_dst
            height_dst = width_dst * ratio
            if height_dst > bottom_dst:
                top_dst = 0
                height_dst = bottom_dst
                width_dst = height_dst / ratio
                left_dst = max(int(face_locations[3] + width / 2 - width_dst / 2), 0)
                right_dst = min(int(face_locations[3] + width / 2 + width_dst / 2), image.shape[1])
                width_dst = right_dst - left_dst
                height_dst = width_dst * ratio
                top_dst = int(bottom_dst - height_dst)
            else:
                top_dst = int(bottom_dst - height_dst)
            return [top_dst, left_dst, bottom_dst, right_dst]
        except:
            # raise IOError('no face')
            print('no face')
            if image.shape[0]/image.shape[1] <= size[1]/size[0]:
                cut_length = image.shape[1] - size[0] * image.shape[0] // size[1]
                if np.nonzero(seg_map)[1].min() / seg_map.shape[1] < 0.2 and np.nonzero(seg_map)[1].max() / seg_map.shape[
                    1] <= 0.8:
                    return [0, 0, image.shape[0], image.shape[1]-cut_length] #[0 , -c]
                elif np.nonzero(seg_map)[1].min() / seg_map.shape[1] >= 0.2 and np.nonzero(seg_map)[1].max() / \
                        seg_map.shape[1] > 0.8:
                    return [0, cut_length, image.shape[0], image.shape[1]]
                else:
                    return [0, cut_length // 2, image.shape[0], image.shape[1] - (cut_length - cut_length // 2)]
            else:
                cut_length = image.shape[0] - size[1] * image.shape[1] // size[0]
                if (np.nonzero(seg_map)[0].min() - cut_length) / seg_map.shape[0] > 0.2:
                    return [cut_length, 0, image.shape[0], image.shape[1]]
                else:
                    return [0, 0, image.shape[0]-cut_length, image.shape[1]]
    else:  # 贴上
        try:
            face_cascade = cv2.CascadeClassifier(facedetect_path)

            face_locations_tmp = face_cascade.detectMultiScale(image1)[0]  # x, y, w. h
            # resize_ratio = np.sqrt(face_square / (face_locations_tmp[2] * face_locations_tmp[3]))
            # resize = (int(image.shape[1] * resize_ratio), int(image.shape[0] * resize_ratio))
            face_locations = (face_locations_tmp[1], face_locations_tmp[0] + face_locations_tmp[2], face_locations_tmp[1] + face_locations_tmp[3], face_locations_tmp[0])
            # image = cv2.resize(image,resize)
            # seg_map = cv2.resize(seg_map, resize)
            width = face_locations[1] - face_locations[3]
            height = face_locations[2] - face_locations[0]
            top_dst = 0  # might be negative
            left_dst = max(int(face_locations[3] - width * w_mul), 0)
            right_dst = min(int(face_locations[1] + width * w_mul), image.shape[1])
            width_dst = right_dst - left_dst
            height_dst = width_dst * ratio
            print('up')

            if height_dst > image.shape[0]:
                height_dst = image.shape[0]
                width_dst = height_dst / ratio
                left_dst = max(int(face_locations[3] + width / 2 - width_dst / 2), 0)
                right_dst = min(int(face_locations[3] + width / 2 + width_dst / 2), image.shape[1])
                width_dst = right_dst - left_dst
                height_dst = width_dst * ratio
                bottom_dst = int(height_dst)
            else:
                bottom_dst = int(height_dst)
            return [top_dst, left_dst, bottom_dst, right_dst]
        except:
            # raise IOError('no face')
            print('no face')
            if image.shape[0]/image.shape[1] <= size[1]/size[0]:
                cut_length = image.shape[1] - size[0] * image.shape[0] // size[1]
                if np.nonzero(seg_map)[1].min() / seg_map.shape[1] < 0.2 and np.nonzero(seg_map)[1].max() / seg_map.shape[
                    1] <= 0.8:
                    return [0, 0, image.shape[0], image.shape[1]-cut_length] #[0 , -c]
                elif np.nonzero(seg_map)[1].min() / seg_map.shape[1] >= 0.2 and np.nonzero(seg_map)[1].max() / \
                        seg_map.shape[1] > 0.8:
                    return [0, cut_length, image.shape[0], image.shape[1]]
                else:
                    return [0, cut_length // 2, image.shape[0], image.shape[1] - (cut_length - cut_length // 2)]
            else:
                cut_length = image.shape[0] - size[1] * image.shape[1] // size[0]
                if (np.nonzero(seg_map)[0].min() - cut_length) / seg_map.shape[0] > 0.2:
                    return [cut_length, 0, image.shape[0], image.shape[1]]
                else:
                    return [0, 0, image.shape[0]-cut_length, image.shape[1]]


def cut_and_resize(image, seg_map, size, fill, fill_value):
    '''把图片resize到size大小，但是不改变shape，其余地方填空'''
    # if image.shape[1] / image.shape[0] - size[0] / size[1] > 0:  # 横着的图
    #     cut_length = image.shape[1] - size[0] * image.shape[0] // size[1]
    #     cut = how_to_cut(image, seg_map, 'horizontal', cut_length)
    #     seg_map = cv2.resize(seg_map[:, cut[0]:cut[1]], size)
    #     image = cv2.resize(image[:, cut[0]:cut[1], :], size)
    #
    # elif image.shape[1] / image.shape[0] - size[0] / size[1] < 0:  # 竖着的图
    #     cut_length = image.shape[0] - size[1] * image.shape[1] // size[0]
    #     cut = how_to_cut(image, seg_map, 'vertical', cut_length)
    #     seg_map = cv2.resize(seg_map[cut[0]:cut[1], :], size)
    #     image = cv2.resize(image[cut[0]:cut[1], :, :], size)
    # else:
    #     seg_map = cv2.resize(seg_map, size)
    #     image = cv2.resize(image, size)
    # return image, seg_map
    cut = how_to_cut_with_face(image, seg_map, size)
    if cut[0] < 0:
        image = np.pad(image, [[abs(cut[0]), 0], [0, 0], [0, 0]], 'constant')
        seg_map = np.pad(seg_map, [[abs(cut[0]), 0], [0, 0]], 'constant')
        if fill:
            image[np.where(seg_map == 0)] = fill_value
        cut[0] = 0
    seg_map = cv2.resize(seg_map[cut[0]:cut[2], cut[1]:cut[3]], size)
    image = cv2.resize(image[cut[0]:cut[2], cut[1]:cut[3], :], size)
    return image, seg_map


# def resize_with_face_square(image, seg_map, face_square_dst):
#     try:
#         face_locations = face_recognition.face_locations(image)[0]
#         face_square_src = (face_locations[2] - face_locations[0]) * (face_locations[1] - face_locations[3])
#         rate = face_square_dst / face_square_src
#         size = (int(image.shape[1] * rate), int(image.shape[0] * rate))
#         image = cv2.resize(image, size)
#         seg_map = cv2.resize(seg_map, size)
#         print('have face')
#         return image, seg_map
#     except:
#         print('no face')
#         return image, seg_map


def resize_with_body_square(image, seg_map, body_square_dst):
    body_square_src = np.sum(seg_map) / 255
    rate = np.sqrt(body_square_dst / body_square_src)
    size = (int(image.shape[1] * rate), int(image.shape[0] * rate))
    image = cv2.resize(image, size)
    seg_map = cv2.resize(seg_map, size)
    return image, seg_map


def resize_and_pad(image, size, pad_value=0, height_how='both'):
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
        if height_how == 'both':
            image = np.pad(cv2.resize(image, size_real), [[0, 0],
                                                          [int((size[0] - size_real[0]) // 2),
                                                           size[0] - size_real[0] - int((size[0] - size_real[0]) // 2)],
                                                          [0, 0]],
                           'constant', constant_values=pad_value)
        elif height_how == 'left':
            image = np.pad(cv2.resize(image, size_real), [[0, 0],
                                                          [size[0] - size_real[0], 0],
                                                          [0, 0]],
                           'constant', constant_values=pad_value)
        else:
            image = np.pad(cv2.resize(image, size_real), [[0, 0],
                                                          [0, size[0] - size_real[0]],
                                                          [0, 0]],
                           'constant', constant_values=pad_value)

    else:  # 竖着的图
        if height_how == 'both':
            size_real = (int(image.shape[1] / image.shape[0] * size[1]), size[1])
            image = np.pad(cv2.resize(image, size_real), [[0, 0],
                                                          [int((size[0] - size_real[0]) // 2),
                                                           size[0] - size_real[0] - int(
                                                               (size[0] - size_real[0]) // 2)]],
                           'constant', constant_values=pad_value)
        elif height_how == 'left':
            size_real = (int(image.shape[1] / image.shape[0] * size[1]), size[1])
            image = np.pad(cv2.resize(image, size_real), [[0, 0],
                                                          [size[0] - size_real[0], 0]],
                           'constant', constant_values=pad_value)
        else:
            size_real = (int(image.shape[1] / image.shape[0] * size[1]), size[1])
            image = np.pad(cv2.resize(image, size_real), [[0, 0],
                                                          [0, size[0] - size_real[0]]],
                           'constant', constant_values=pad_value)
    return image


def how_to_cut_and_pad(image, seg_map, size):
    left_edge = np.nonzero(seg_map[:, 0])[0]
    right_edge = np.nonzero(seg_map[:, -1])[0]
    if left_edge.shape[0] == 0:
        left_length = 0
    else:
        left_length = left_edge.max() - left_edge.min()
    if right_edge.shape[0] == 0:
        right_length = 0
    else:
        right_length = right_edge.max() - right_edge.min()

    height_max = np.nonzero(seg_map)[0].max()
    height_min = np.nonzero(seg_map)[0].min()
    width_max = np.nonzero(seg_map)[1].max()
    width_min = np.nonzero(seg_map)[1].min()
    image = image[height_min:height_max, width_min:width_max, :]
    seg_map = seg_map[height_min:height_max, width_min:width_max]

    tmp_shape = image.shape
    if tmp_shape[0] <= size[1] and tmp_shape[1] <= size[0] and left_length <= right_length:
        if left_length == 0 and right_length == 0:
            image = np.pad(image, [[size[1] - tmp_shape[0], 0],
                                   [(size[0] - tmp_shape[1]) // 2,
                                    size[0] - tmp_shape[1] - (size[0] - tmp_shape[1]) // 2],
                                   [0, 0]], 'constant')
            seg_map = np.pad(seg_map, [[size[1] - tmp_shape[0], 0],
                                       [(size[0] - tmp_shape[1]) // 2,
                                        size[0] - tmp_shape[1] - (size[0] - tmp_shape[1]) // 2]], 'constant')
        else:
            image = np.pad(image, [[size[1] - tmp_shape[0], 0], [size[0] - tmp_shape[1], 0], [0, 0]], 'constant')
            seg_map = np.pad(seg_map, [[size[1] - tmp_shape[0], 0], [size[0] - tmp_shape[1], 0]], 'constant')
    elif tmp_shape[0] <= size[1] and tmp_shape[1] <= size[0] and left_length > right_length:
        image = np.pad(image, [[size[1] - tmp_shape[0], 0], [0, size[0] - tmp_shape[1]], [0, 0]], 'constant')
        seg_map = np.pad(seg_map, [[size[1] - tmp_shape[0], 0], [0, size[0] - tmp_shape[1]]], 'constant')
    elif tmp_shape[0] <= size[1] and tmp_shape[1] >= size[0]:
        image = resize_and_pad(image, size)
        seg_map = resize_and_pad(seg_map, size)
    elif tmp_shape[0] >= size[1] and tmp_shape[1] <= size[0] and left_length <= right_length:
        if left_length == 0 and right_length == 0:
            image = resize_and_pad(image, size, height_how='both')
            seg_map = resize_and_pad(seg_map, size, height_how='both')
        else:
            image = resize_and_pad(image, size, height_how='left')
            seg_map = resize_and_pad(seg_map, size, height_how='left')
    elif tmp_shape[0] >= size[1] and tmp_shape[1] <= size[0] and left_length >= right_length:
        if left_length == 0 and right_length == 0:
            image = resize_and_pad(image, size, height_how='both')
            seg_map = resize_and_pad(seg_map, size, height_how='both')
        else:
            image = resize_and_pad(image, size, height_how='right')
            seg_map = resize_and_pad(seg_map, size, height_how='right')
    elif tmp_shape[0] >= size[1] and tmp_shape[1] >= size[0] and left_length <= right_length:
        if left_length == 0 and right_length == 0:
            image = resize_and_pad(image, size, height_how='both')
            seg_map = resize_and_pad(seg_map, size, height_how='both')
        else:
            image = resize_and_pad(image, size, height_how='left')
            seg_map = resize_and_pad(seg_map, size, height_how='left')
    else:
        if left_length == 0 and right_length == 0:
            image = resize_and_pad(image, size, height_how='both')
            seg_map = resize_and_pad(seg_map, size, height_how='both')
        else:
            image = resize_and_pad(image, size, height_how='right')
            seg_map = resize_and_pad(seg_map, size, height_how='right')
    return image, seg_map


def image_to_background(image, seg_map, background, positions, fill, default_positions=None, fill_value=None,
                        body_square=None):
    '''
    :param image:
    :param fill_value:
    :param seg_map:
    :param fill:
    :param background: './worldcup/background/id_card1.png'
    :param positions: [x_left, y_top, x_right, y_bottom] or np.array([])
    '''
    if isinstance(positions, list):
        # 专门给证件照的 id_card1, final1, final2
        if fill:
            image[np.where(seg_map == 0)] = fill_value
            image = blur_edge(image, seg_map)
        background = cv2.imread(background)
        size = (positions[2] - positions[0], positions[3] - positions[1])
        image, seg_map = cut_and_resize(image, seg_map, size, fill, fill_value)
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
        # 专门给homography的 id_card2
        if fill:
            image[np.where(seg_map == 0)] = fill_value
            image = blur_edge(image, seg_map)
        default_size = (default_positions[2] - default_positions[0], default_positions[3] - default_positions[1])
        image, seg_map = cut_and_resize(image, seg_map, default_size, fill, fill_value)
        background = cv2.imread(background)
        height, width = image.shape[:2]
        positions_src = np.array([[0, 0], [0, height - 1], [width - 1, height - 1], [width - 1, 0]])
        h, status = cv2.findHomography(positions_src, positions, cv2.RANSAC, 5.0)
        im_out = cv2.warpPerspective(image, h, (background.shape[1], background.shape[0]), cv2.INTER_LINEAR)
        roi = cv2.imread('./worldcup/background/id_card2_roi.jpg', cv2.IMREAD_GRAYSCALE)
        background[np.where(roi > 100)] = cv2.cvtColor(im_out, cv2.COLOR_RGB2BGR)[np.where(roi > 100)]
        return background, seg_map
    if positions == None:
        # 首先计算整个半身照的面积，调整到合适大小，再在下方贴边，再在左右贴边。
        image, seg_map = resize_with_body_square(image, seg_map, body_square)
        background = cv2.imread(background)
        back_size = (background.shape[1], background.shape[0])
        image, seg_map = how_to_cut_and_pad(image, seg_map, back_size)
        background[np.where(seg_map != 0)] = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)[np.where(seg_map != 0)]
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
                                                  positions=None, fill=False, body_square=100000)
        # blur_edge
        background = blur_edge(background, seg_map)
        cv2.imwrite('./worldcup/merge/worldcup/' + image_name.split('.')[0] + '.jpg', background)
    if background_image == 'ground1':
        background, seg_map = image_to_background(image, seg_map, './worldcup/background/ground1.jpeg',
                                                  positions=None, fill=False, body_square=50000)

        # blur_edge
        background = blur_edge(background, seg_map)
        print(background.shape)
        cv2.imwrite('./worldcup/merge/ground_first/' + image_name.split('.')[0] + '.jpg', background)
    if background_image == 'ground2':
        background, seg_map = image_to_background(image, seg_map, './worldcup/background/ground2.jpg',
                                                  positions=None, fill=False, body_square=200000)

        # blur_edge
        background = blur_edge(background, seg_map)
        print(background.shape)
        cv2.imwrite('./worldcup/merge/ground_second/' + image_name.split('.')[0] + '.jpg', background)

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
    # seg_map_out = seg_map.copy()
    # seg_map_out[np.where(seg_map_out!=15)] = 0
    # seg_map_out[np.where(seg_map_out==15)] = 255
    # cv2.imwrite(IMAGE_DIR + '/' + image_name.split('.')[0] + '.png', seg_map_out)
    vis_segmentation(np.array(resized_im), seg_map, image_name, width, height, 'id_card1')


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
images = list(filter(lambda x: 'png' not in x, images))

for k, image_name in enumerate(images):  # [:5]:
    run_demo_image(image_name)

# cv2.imshow('seg_map',seg_map);cv2.waitKey(0)
