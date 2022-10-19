import cv2
import numpy as np
import tensorflow as tf
from urllib.request import urlopen

def tf_get_url_image(picture_url):
    """
    Get RGB image from a url stored in a tensorflow format

    Parameters
    ----------
    picture_url: tensor
        URL where an image is stored

    Returns
    -------
    rgb_img: numpy.ndarray
        rgb image
    """
    with urlopen(str(picture_url.numpy().decode("utf-8"))) as request:
        image_file_in_bytes = np.asarray(bytearray(request.read()), dtype=np.uint8)
    decoded_img = cv2.imdecode(image_file_in_bytes, cv2.IMREAD_COLOR)
    rgb_img = cv2.cvtColor(decoded_img, cv2.COLOR_BGR2RGB)
    return rgb_img

def get_url_image(picture_url):
    """
    Get RGB image from a url stored in a str variable

    Parameters
    ----------
    picture_url: str
        URL where an image is stored

    Returns
    -------
    rgb_img: numpy.ndarray
        rgb image
    """
    with urlopen(str(picture_url)) as request:
        image_file_in_bytes = np.asarray(bytearray(request.read()), dtype=np.uint8)
    decoded_img = cv2.imdecode(image_file_in_bytes, cv2.IMREAD_COLOR)
    rgb_img = cv2.cvtColor(decoded_img, cv2.COLOR_BGR2RGB)
    return rgb_img

def resize_image(rgb_image, sqr_size=256):
    """
    Get a resized input image using a square shape defined by sqr_size

    Parameters
    ----------
    rgb_image: numpy.ndarray
        rgb image

    Returns
    -------
    rgb_img: numpy.ndarray
        square resized rgb image
    """
    rezised_img = cv2.resize(rgb_image, (sqr_size, sqr_size))
    return rezised_img

def segmentate_thresh(picture, threshold = 200):
    """
    Get a segmented image from the input by getting the information of pixels lighter than threshold

    Parameters
    ----------
    picture: numpy.ndarray
        image
    threshold: int
        threshlod value to apply intensity segmentation

    Returns
    -------
    image where the pixels below threshold value turn into '0' (numpy.ndarray)
    """
    one_chanel_image = cv2.cvtColor(picture, cv2.COLOR_RGB2GRAY)
    _, image_threshold = cv2.threshold(one_chanel_image, threshold, 255, cv2.THRESH_BINARY)
    return cv2.bitwise_and(picture, picture, mask=image_threshold)

def get_resized_image(picture_url):
    """
    Get a squared image from url stored in tensor flow format

    Parameters
    ----------
    picture_url: tensor
        URL where an image is stored

    Returns
    -------
    squared rgb image (numpy.ndarray)
    """
    rgb_img = tf_get_url_image(picture_url)
    return resize_image(rgb_img)

def get_threshold_image(picture_url):
    """
    Get a thresholdlded and squared image from url stored in tensor flow format

    Parameters
    ----------
    picture_url: tensor
        URL where an image is stored

    Returns
    -------
    squared rgb image (numpy.ndarray)
    """
    resized_img = get_resized_image(picture_url)
    return segmentate_thresh(resized_img)

def tf_get_resized_image(picture_url, label):
    """
    Get a normalized  (0-1) squared image in tensor flow format from url stored in tensor flow format
    Also return a hot encoded label

    Parameters
    ----------
    picture_url: tensor
        URL where an image is stored
    label: tensor
        integer image label

    Returns
    -------
    squared rgb image (tensor)
    one hot encoded label (tensor)
    """
    return tf.py_function(get_resized_image, [picture_url], tf.float32)/255, tf.one_hot(label, 2)

def tf_get_threshold_image(picture_url, label):
    """
    Get a normalized, thresholded and squared image in tensor flow format from url stored in tensor flow format
    Also return a hot encoded label

    Parameters
    ----------
    picture_url: tensor
        URL where an image is stored
    label: tensor
        integer image label

    Returns
    -------
    normalized, thresholded and squared rgb image (tensor)
    one hot encoded label (tensor)
    """
    return tf.py_function(get_threshold_image, [picture_url], tf.float32)/255, tf.one_hot(label, 2)

def tf_get_thresh_valid_image(picture_url):
    """
    Get a normalized, thresholded and squared image in tensor flow format from url stored in tensor flow format
    without label processing

    Parameters
    ----------
    picture_url: tensor
        URL where an image is stored

    Returns
    -------
    normalized, thresholded and squared rgb image (tensor)
    """
    return tf.py_function(get_threshold_image, [picture_url], tf.float32)/255
