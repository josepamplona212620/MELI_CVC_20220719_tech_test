import mahotas as mh
import numpy as np
import cv2

def segmentate_thresh_(picture, thresh):
    one_chanel_image = cv2.cvtColor(picture, cv2.COLOR_RGB2GRAY)
    _, image_threshold = cv2.threshold(one_chanel_image, thresh, 255, cv2.THRESH_BINARY)
    return cv2.bitwise_and(picture, picture, mask=image_threshold),\
           cv2.bitwise_and(one_chanel_image, one_chanel_image, mask=image_threshold)

def get_image_sides(gray_picture):
    pic_h, pic_w = gray_picture.shape
    top_side = gray_picture[:int(pic_h/2)-1,:]
    bottom_side = gray_picture[int(pic_h/2):,:]
    right_side = gray_picture[:,:int(pic_w/2)-1]
    left_side = gray_picture[:,:int(pic_w/2)-1]
    return top_side, bottom_side, left_side, right_side

def get_moments_features(gray_picture):
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
    pic_h, pic_w, _ = rgb_image.shape
    hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    hist_color, _ = np.histogram(hsv_image[:, :, 0], bins=16, range=(0, 180), density=True)
    hist_value, _ = np.histogram(hsv_image[:, :, 2], bins=32, range=(0, 255), density=True)
    return list(hist_color[1:])+list(hist_value[20:])

def get_texture_features(gray_picture):
    labeled, n = mh.label(gray_picture)
    h_feature = mh.features.haralick(labeled)
    # lbp_feature = mh.features.lbp(labeled, 3, 8)
    return list(h_feature.flatten())  # +list(lbp_feature)

def get_features_record(image, threshold):
    segmented_image, segmented_img_gray = segmentate_thresh_(image, threshold)
    features = []
    features = features+list(get_color_features(segmented_image))
    for side_image in get_image_sides(segmented_img_gray):
        features = features+get_moments_features(side_image)
    features = features+(get_texture_features(segmented_img_gray))
    return features

def get_features_names():
    col_names_color = ['color' + str(i) for i in range(15)]
    col_names_intensity = ['value' + str(i) for i in range(12)]
    col_names_texture = ['texture' + str(i) for i in range(52)]
    col_names_moments = ['top_area', 'top_cx', 'top_cy',
                         'bottom_area', 'bottom_cx', 'bottom_cy',
                         'left_area', 'left_cx', 'left_cy',
                         'right_area', 'right_cx', 'right_cy', ]

    col_names = ['label'] + col_names_color + col_names_intensity + col_names_moments + col_names_texture
    return col_names
