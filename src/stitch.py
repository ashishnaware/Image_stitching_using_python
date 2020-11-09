import sys

import cv2
import os
import numpy as np
import random

'''
# Name : Ashish Avinash Naware
# Citations:
#   Books : 
#       Computer Vision - Algorithms and Applications
#   Online Resources:
#       https://docs.opencv.org/master/da/df5/tutorial_py_sift_intro.html
#       https://docs.opencv.org/3.4/d2/d29/classcv_1_1KeyPoint.html
#       https://docs.opencv.org/3.4/d0/d13/classcv_1_1Feature2D.html
#       https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.svd.html
#       https://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html
#       https://github.com/hughesj919/HomographyEstimation/blob/master/Homography.py
#       https://towardsdatascience.com/image-stitching-using-opencv-817779c86a83
#
#       http://www.cse.psu.edu/~rtc12/CSE486/lecture16.pdf
#       https://cseweb.ucsd.edu/classes/wi07/cse252a/homography_estimation/homography_estimation.pdf
#
'''

'''
Function to generate set of four random keypoints among set of given keypoints
Input: set of keypoints of image 1 and image 2
Returns: random four keypoints
'''


def random_points_generator(keypoints1, keypoints2):
    rand1 = random.randrange(0, len(keypoints1))
    rand2 = random.randrange(0, len(keypoints1))
    rand3 = random.randrange(0, len(keypoints1))
    rand4 = random.randrange(0, len(keypoints1))

    p1 = keypoints1[rand1]
    p2 = keypoints1[rand2]
    p3 = keypoints1[rand3]
    p4 = keypoints1[rand4]

    p1_ = keypoints2[rand1]
    p2_ = keypoints2[rand2]
    p3_ = keypoints2[rand3]
    p4_ = keypoints2[rand4]

    return p1, p1_, p2, p2_, p3, p3_, p4, p4_


'''
Find homography matrix from given points
Input: set of four keypoints generated randomly
returns: homography matrix of the given points
'''


def perform_homography(p1, p1_, p2, p2_, p3, p3_, p4, p4_):
    homography_list = []

    # Construct 8*9 homography matrix
    ax1 = [-1 * p1.pt[0], -1 * p1.pt[1], -1, 0, 0, 0, p1.pt[0] * p1_.pt[0], p1.pt[1] * p1_.pt[0], p1_.pt[0]]
    ay1 = [0, 0, 0, -1 * p1.pt[0], -1 * p1.pt[1], -1, p1.pt[0] * p1_.pt[1], p1.pt[1] * p1_.pt[1], p1_.pt[1]]
    ax2 = [-1 * p2.pt[0], -1 * p2.pt[1], -1, 0, 0, 0, p2.pt[0] * p2_.pt[0], p2.pt[1] * p2_.pt[0], p2_.pt[0]]
    ay2 = [0, 0, 0, -1 * p2.pt[0], -1 * p2.pt[1], -1, p2.pt[0] * p2_.pt[1], p2.pt[1] * p2_.pt[1], p2_.pt[1]]
    ax3 = [-1 * p3.pt[0], -1 * p3.pt[1], -1, 0, 0, 0, p3.pt[0] * p3_.pt[0], p3.pt[1] * p3_.pt[0], p3_.pt[0]]
    ay3 = [0, 0, 0, -1 * p3.pt[0], -1 * p3.pt[1], -1, p3.pt[0] * p3_.pt[1], p3.pt[1] * p3_.pt[1], p3_.pt[1]]
    ax4 = [-1 * p4.pt[0], -1 * p4.pt[1], -1, 0, 0, 0, p4.pt[0] * p4_.pt[0], p4.pt[1] * p4_.pt[0], p4_.pt[0]]
    ay4 = [0, 0, 0, -1 * p4.pt[0], -1 * p4.pt[1], -1, p4.pt[0] * p4_.pt[1], p4.pt[1] * p4_.pt[1], p4_.pt[1]]

    homography_list.append(ax1)
    homography_list.append(ay1)
    homography_list.append(ax2)
    homography_list.append(ay2)
    homography_list.append(ax3)
    homography_list.append(ay3)
    homography_list.append(ax4)
    homography_list.append(ay4)

    homography_matrix = np.asarray(homography_list)
    # perform singular value decomposition
    u, s, vh = np.linalg.svd(homography_matrix)

    h_matrix = np.reshape(vh[8], (3, 3))

    return h_matrix;


'''
Calculate the homography for the given point and convert homogenous point into cartesian point
Input: homography matrix, input point
Returns: projected point
'''


def homography_estimation(homography_matrix, point):
    keypoint = np.asarray([point.pt[0], point.pt[1], 1])
    projected_point = np.dot(homography_matrix, keypoint)

    if (projected_point.item(2) != 0):
        projected_point = (1 / projected_point.item(2)) * projected_point

    return projected_point;


'''
Perform RANSAC to find best homography matrix
Input: set of keypoints of two images and the threshold
Returns: final homography matrix
'''


def perform_ransac(keypoints1, keypoints2, threshold):
    inliers = []
    list_of_homography = []
    for i in range(0, threshold):
        p1, p1_, p2, p2_, p3, p3_, p4, p4_ = random_points_generator(keypoints1, keypoints2)
        homography = perform_homography(p1, p1_, p2, p2_, p3, p3_, p4, p4_)

        count = 0
        for i in range(0, len(keypoints1)):
            projected_point1 = homography_estimation(homography, keypoints1[i])
            actual_point2 = np.asarray([keypoints2[i].pt[0], keypoints2[i].pt[1], 1])
            error = actual_point2 - projected_point1
            error_normalized = np.linalg.norm(error)

            if (error_normalized < 1):
                count = count + 1

        inliers.append(int(count))
        list_of_homography.append(homography)
    ind = inliers.index(max(inliers))
    final_homography = list_of_homography[ind]

    return final_homography;


'''
Controller function to perform panorama
Input: Two color images and their respective monochrome images
Returns: 1. If two images match then returns panorama
         2. If there is no match between the images then returns the second image
'''


def perform_panorama(image1_b, image1_c, image2_b, image2_c):
    distance = []
    list_of_des1 = []
    list_of_des2 = []
    list_of_kp1 = []
    list_of_kp2 = []
    list_of_all_elements = []
    list_of_all_sorted_elements = []
    descriptor_matches = []
    popElements = []
    count = 0

    sift = cv2.xfeatures2d.SIFT_create(contrastThreshold=0.04, edgeThreshold=10)
    kp1, des1 = sift.detectAndCompute(np.asarray(image1_b), None)
    kp2, des2 = sift.detectAndCompute(np.asarray(image2_b), None)

    for idx, i in enumerate(des1):
        for j in des2:
            distance.append(int(cv2.norm(i, j, normType=cv2.NORM_L2) / 128))
        descriptor_matches.append(min(distance))
        list_of_des2.append(des2[distance.index(min(distance))])
        list_of_kp2.append(kp2[distance.index(min(distance))])
        list_of_des1.append(i)
        list_of_kp1.append(kp1[idx])
        distance.clear()

    list_of_all_elements = [[list_of_kp1[i], list_of_kp2[i], list_of_des1[i], list_of_des2[i], descriptor_matches[i]]
                            for i in
                            range(0, len(descriptor_matches))]

    # sort list based on minimum distance of elements
    list_of_all_sorted_elements = sorted(list_of_all_elements, key=lambda dist: dist[4])
    list_of_all_sorted_elements_copy = list_of_all_sorted_elements.copy()

    for idx, element in enumerate(list_of_all_sorted_elements_copy):
        if (list_of_all_sorted_elements_copy[idx][4] > 0):
            popElements.append(idx)

    for ele in sorted(popElements, reverse=True):
        del list_of_all_sorted_elements_copy[ele]

    # Perform panorama if two images overlap by at least 20%
    if len(list_of_all_sorted_elements_copy) >= len(list_of_all_elements) * 0.20:

        percent_matches = int(len(list_of_all_sorted_elements_copy))
        list_of_all_sorted_elements1 = list_of_all_sorted_elements[:percent_matches]
        list_of_all_sorted_elements1 = np.array(list_of_all_sorted_elements1)
        kp1_for_homography = list_of_all_sorted_elements1[:, 0]
        kp2_for_homography = list_of_all_sorted_elements1[:, 1]
        H = perform_ransac(kp1_for_homography, kp2_for_homography, int(len(kp1_for_homography)) * 2)

        height1, width1 = image1_c.shape[:2]
        height2, width2 = image2_c.shape[:2]

        pts1 = np.float32([[0, 0], [0, height2], [width2, height2], [width2, 0]]).reshape(-1, 1, 2)
        pts2 = np.float32([[0, 0], [0, height1], [width1, height1], [width1, 0]]).reshape(-1, 1, 2)
        pts2_ = cv2.perspectiveTransform(pts2, H)

        pts = np.concatenate((pts1, pts2_), axis=0)
        [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
        [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)
        translation = [-xmin, -ymin]
        homography_translation = np.array([[1, 0, translation[0]], [0, 1, translation[1]], [0, 0, 1]])

        result = cv2.warpPerspective(image1_c, homography_translation.dot(H), (xmax - xmin, ymax - ymin))
        result[translation[1]:height2 + translation[1], translation[0]:width2 + translation[0]] = image2_c

    else:
        return image2_c
    return result;


'''
Main function
'''


def main():
    path_list = []
    data_d = sys.argv[1]
    data_directory = os.getcwd() + data_d
    iteration_count = 0

    for path in os.listdir(data_directory):
        path_list.append(os.path.join(data_directory, path))
    path_list.reverse()
    iteration_count = len(path_list) / 2
    if iteration_count % 2 != 0:
        iteration_count += 1
    previous_image = []
    image1_c = cv2.imread(path_list.pop(0))
    image1_b = cv2.cvtColor(image1_c, cv2.COLOR_BGR2GRAY)
    previous_image.append(image1_c)

    while len(path_list) > 0 and iteration_count > 0:

        if isinstance(path_list[0], str):
            image2_c = cv2.imread(path_list.pop(0))
        else:
            image2_c = path_list.pop(0)
        image2_b = cv2.cvtColor(image2_c, cv2.COLOR_BGR2GRAY)
        image1_c = perform_panorama(image1_b, image1_c, image2_b, image2_c)
        previous_image.append(image1_c)
        if np.array_equal(image1_c, image2_c):
            path_list.append(image1_c)
            previous_image.pop(-1)
            iteration_count -= 1
            continue
        else:
            image1_b = cv2.cvtColor(previous_image[-1], cv2.COLOR_BGR2GRAY)

    cv2.imwrite(os.getcwd() + sys.argv[1] + "\Panorama.jpg", previous_image[-1])


if __name__ == "__main__":
    main()
