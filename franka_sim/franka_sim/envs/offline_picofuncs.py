# import the necessary packages
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
from imutils import resize
from imutils import paths
import imutils

import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def center_crop(img, fact):
    # return center cropped image
    # Args:
    # img: image to be center cropped
    # dim: [wid, hei] to be cropped from center

    dim = [img.shape[1] * fact, img.shape[0] * fact]

    width, height = img.shape[1], img.shape[0]

    # process crop width and height for max available dimension
    crop_width = dim[0] if dim[0] < img.shape[1] else img.shape[1]
    crop_height = dim[1] if dim[1] < img.shape[0] else img.shape[0]

    mid_x, mid_y = int(width / 2), int(height / 2)
    cw2, ch2 = int(crop_width / 2), int(crop_height / 2)
    crop_img = img[mid_y - ch2:mid_y + ch2, mid_x - cw2:mid_x + cw2]
    return crop_img


def count_sats(cnts):
    return len(cnts)


def meas_width(cnts, image):
    width, height = image.shape
    tl = [width, height]
    br = [0,0]
    mmppix = 2/width
    for c in cnts:
        # if the contour is not sufficiently large, ignore it
        #if cv2.contourArea(c) < 100:
        #    continue

        # compute the rotated bounding box of the contour
        x,y,w,h = cv2.boundingRect(c)
        tl[0] = x if x < tl[0] else tl[0]
        tl[1] = y if y < tl[1] else tl[1]

        br[0] = x + w if x + w > br[0] else br[0]
        br[1] = y + h if y + h > br[1] else br[1]

    dep_width = (br[0] - tl[0])*mmppix

    if dep_width == -2:
        # show the output image
        cv2.imshow("Image", image)
        cv2.waitKey(0)

    return (br[0] - tl[0])*mmppix


def area_diff(cnts):
    # calculate the real area of the contours
    real_area = 0
    for contour in cnts:
        real_area += cv2.contourArea(contour)

    # calculate the area of the ideal enclosing circle
    # This circle approximates the ideal deposition of a given width
    all_cnts = np.concatenate(cnts)
    (x, y), r = cv2.minEnclosingCircle(all_cnts)
    circ_area = np.pi * np.square(r)

    return circ_area - real_area


def load_contours(imagepath):
    image = cv2.imread(imagepath)
    #image = center_crop(image, 0.66)
    image = resize(image, width=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    # perform edge detection, then perform a dilation + erosion to
    # close gaps in between object edges
    edged = imutils.auto_canny(gray)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)

    # find contours in the edge map
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    return cnts, gray

def build_index(file):
    dataset = []
    for imagePath in paths.list_images(file):
        name = os.path.basename(imagePath)
        name = name.split()

        cnts, gray = load_contours(imagePath)
        satellites = count_sats(cnts)
        if satellites == 0:
            dep_w = 0
            difference = 0
        else:
            dep_w = meas_width(cnts, gray)
            difference = area_diff(cnts)

        dataset.append([int(name[1][1:]),
                        int(name[2][1:]),
                        imagePath,
                        satellites, dep_w, difference])
    df = pd.DataFrame(dataset, columns=['Stroke', 'Close_T', 'Img_File', 'Satellites', 'Dep Width', 'Area Diff'])
    return df

def get_state(stroke, close_t, index):

    strokeindex = index.loc[index['Stroke'].eq(stroke)]
    closeindex = strokeindex.iloc[(strokeindex['Close_T'] - close_t).abs().argsort()[:1]]

    satellites = closeindex['Satellites'].item()
    dep_w = closeindex['Dep Width'].item()
    difference = closeindex['Area Diff'].item()

    return satellites, dep_w, difference

def get_img(stroke, close_t, index):

    strokeindex = index.loc[index['Stroke'].eq(stroke)]
    closeindex = strokeindex.iloc[(strokeindex['Close_T'] - close_t).abs().argsort()[:1]]

    img_loc = closeindex['Img_File'].item()
    image = cv2.imread(img_loc)
    image = resize(image, width=500)

    return image


#index = build_index("C:/Users/samue/PycharmProjects/BlurDetection/Prt2_Combined")


# fig = plt.figure(1)
# fig.tight_layout()
# plt.scatter(index['Stroke'],index['Close_T'])
# plt.xlabel('Stroke')
# plt.ylabel('Close T')
# plt.title('Dataset Settings Distribution')
#
# threedee = plt.figure(2).gca(projection='3d')
# threedee.scatter(index['Stroke'],index['Close_T'], index['Satellites'])
# threedee.set_xlabel('Stroke')
# threedee.set_ylabel('Close T')
# threedee.set_zlabel('Satellites')
#
# threedee2 = plt.figure(3).gca(projection='3d')
# threedee2.scatter(index['Stroke'],index['Close_T'], index['Dep Width'])
# threedee2.set_xlabel('Stroke')
# threedee2.set_ylabel('Close T')
# threedee2.set_zlabel('Deposition Width')
#
# plt.show()

#for stroke in range(10):
#    cv2.imshow('Display Images', get_img(stroke+50, 200, index))
#    cv2.waitKey(500)