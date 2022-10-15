import cv2
import numpy as np
import tensorflow as tf
from urllib.request import urlopen

def get_url_image(picture_url):
    with urlopen(str(picture_url.numpy().decode("utf-8"))) as request:
        image_file_in_bytes = np.asarray(bytearray(request.read()), dtype=np.uint8)
    decoded_img = cv2.imdecode(image_file_in_bytes, cv2.IMREAD_COLOR)
    rgb_img = cv2.cvtColor(decoded_img, cv2.COLOR_BGR2RGB)
    return rgb_img

def resize_image(rgb_image, sqr_size=256):
    rezised_img = cv2.resize(rgb_image, (sqr_size, sqr_size))
    return rezised_img

def segmentate_thresh(picture):
    one_chanel_image = cv2.cvtColor(picture, cv2.COLOR_RGB2GRAY)
    _, image_threshold = cv2.threshold(one_chanel_image, 200, 255, cv2.THRESH_BINARY)
    return cv2.bitwise_and(picture, picture, mask=image_threshold)

def get_resized_image(picture_url):
    rgb_img = get_url_image(picture_url)
    return resize_image(rgb_img)

def get_threshold_image(picture_url):
    resized_img = get_resized_image(picture_url)
    return segmentate_thresh(resized_img)

def tf_get_url_image(picture_url, label):
    return tf.py_function(get_url_image, [picture_url], tf.float32)/255, tf.one_hot(label,2)

def tf_get_resized_image(picture_url, label):
    return tf.py_function(get_resized_image, [picture_url], tf.float32)/255, tf.one_hot(label,2)

def tf_get_threshold_image(picture_url, label):
    return tf.py_function(get_threshold_image, [picture_url], tf.float32)/255, tf.one_hot(label, 2)
