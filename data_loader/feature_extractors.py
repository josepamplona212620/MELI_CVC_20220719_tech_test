import mahotas as mh
import numpy as np
import cv2

def segment_with_threshold(picture, thresh):
    """
    Segment an imput image by turning it into gray scale image and applying a threshold
    in its pixel intensities.
    The function returns two segmented images (grayscale and rgb)

    Parameters
    ----------
    picture : uint8
        rgb image to segment

    Returns
    -------
    RGB segmented image ( np.ndarray)
    Gray scale segmented image ( np.ndarray)
    """
    one_chanel_image = cv2.cvtColor(picture, cv2.COLOR_RGB2GRAY)
    _, image_threshold = cv2.threshold(one_chanel_image, thresh, 255, cv2.THRESH_BINARY)
    return cv2.bitwise_and(picture, picture, mask=image_threshold),\
           cv2.bitwise_and(one_chanel_image, one_chanel_image, mask=image_threshold)

def get_image_sides(gray_picture):
    """
    Get 4 sub-images from the four sides of the input image

    Parameters
    ----------
    picture : uint8
        gray scale image to segment

    Returns
    -------
    top_side:  numpy.ndarray
        image from the middle to the top of the input image
    bottom_side:  numpy.ndarray
        image from the middle to the bottom of the input image
    left_side:  numpy.ndarray
        image from the middle to the left of the input image
    right_side:  numpy.ndarray
        image from the middle to the right of the input image
    """
    pic_h, pic_w = gray_picture.shape
    top_side = gray_picture[:int(pic_h/2)-1,:]
    bottom_side = gray_picture[int(pic_h/2):,:]
    right_side = gray_picture[:,:int(pic_w/2)-1]
    left_side = gray_picture[:,:int(pic_w/2)-1]
    return top_side, bottom_side, left_side, right_side

def get_moments_features(gray_picture):
    """
    Get three morphological features from the input image (Area, and Centroids)

    Parameters
    ----------
    picture : uint8
        gray scale image to apply segmentation

    Returns
    -------
    relative_aera: float
        Relative area of the input image (sum of intensities/sum of ones image )
    x_centroid/pic_w: float
        Relative x centroid of the input image respect to de image width
    y_centroid/pic_h: float
        Relative y centroid of the input image respect to de image height
    """
    pic_h, pic_w = gray_picture.shape
    total_area = pic_h*pic_w*255
    M = cv2.moments(gray_picture)
    if M["m00"] == 0:
        return [0, 0, 0]

    else:
        relative_aera = M["m00"]/total_area #image area calculated as moment(0,0)
        x_centroid = M["m10"]/M["m00"]
        y_centroid = M["m01"]/M["m00"]
        return [relative_aera, x_centroid/pic_w, y_centroid/pic_h]

def get_color_features(rgb_image):
    """
    Converts image into HSV color space and calculates 27 color features
    Get 15 values from the HUE histogram and 12 from the Value (intensity) histogram

    Parameters
    ----------
    picture : uint8
        RGB image

    Returns
    -------
    list of 12 values from HUE histogram
    list of 15 values from Value (intensity) histogram
    """
    pic_h, pic_w, _ = rgb_image.shape
    hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    hist_color, _ = np.histogram(hsv_image[:, :, 0], bins=16, range=(0, 180), density=True)
    hist_value, _ = np.histogram(hsv_image[:, :, 2], bins=32, range=(0, 255), density=True)
    return list(hist_color[1:])+list(hist_value[20:])

def get_texture_features(gray_picture):
    """
    Uses mahotas library to calculate 13 out of the 14 Haralick features from the image texture
    It is processed in 4 different position getting 52 features

    Parameters
    ----------
    picture : uint8
        Gray scale image

    Returns
    -------
    list of 52 Haralick features
    """
    labeled, n = mh.label(gray_picture)
    h_feature = mh.features.haralick(labeled)
    # lbp_feature = mh.features.lbp(labeled, 3, 8)
    return list(h_feature.flatten())  # +list(lbp_feature)

def get_features_record(image, threshold):
    """
    Segment clear image backgrounds and calculates color, morphological and texture features
    Return 91 image numeric features
        - 27 color features
        - 12 morphological features
        - 52 texture features

    Parameters
    ----------
    picture : uint8
        RGB image

    Returns
    -------
    features: list
    91 numerica features from the input image
    """
    h, w, ch = image.shape
    if sum(image.shape) > 1500:
        image = cv2.resize(image, (int(w/2), int(h/2)))
    segmented_image, segmented_img_gray = segment_with_threshold(image, threshold)
    features = []
    features = features+list(get_color_features(segmented_image))
    for side_image in get_image_sides(segmented_img_gray):
        features = features+get_moments_features(side_image)
    features = features+(get_texture_features(segmented_img_gray))
    return features

def get_features_names():
    """
    Defines the feature names for 91 image numeric features

    Returns
    -------
    col_names: list
        names of features columns for a pandas dataset
    """
    col_names_color = ['color' + str(i) for i in range(15)]
    col_names_intensity = ['value' + str(i) for i in range(12)]
    col_names_texture = ['texture' + str(i) for i in range(52)]
    col_names_moments = ['top_area', 'top_cx', 'top_cy',
                         'bottom_area', 'bottom_cx', 'bottom_cy',
                         'left_area', 'left_cx', 'left_cy',
                         'right_area', 'right_cx', 'right_cy', ]

    col_names = ['label'] + col_names_color + col_names_intensity + col_names_moments + col_names_texture
    return col_names
