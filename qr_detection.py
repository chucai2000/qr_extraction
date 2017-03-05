import os
import numpy as np
import cv2
import itertools

#------------------------------------------------------------------------------------------------------------
# Parameters
#------------------------------------------------------------------------------------------------------------
PROBE_LEN = 9
NUM_FG_BG_SWITCH = 3
""" It can be enlarged after the enhancement pre-processing """
EFFECTIVE_COLOR_DIFF = 180.0
INCR = 1
NUM_OF_INCRS = 100
BLOCK_RADIUS = 16
BLOCK_THRESHOLD = 1.0
SE_CLOSE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))
SE_OPEN = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11, 11))
BR_HEIGHT_MINIMUM = 50
BR_WIDTH_MINIMUM = 50
BR_ASPECT_RATIO_MAXIMUM = 2.0
""" First make it related to EFFECTIVE_COLOR_DIFF """
CANNY_HIGH_THRESHOLD = EFFECTIVE_COLOR_DIFF + 20.0
CANNY_LOW_THRESHOLD = CANNY_HIGH_THRESHOLD*0.4
FINDER_PATTERN_AREA_MINIMUM = 20 # bfi1, inner-square-area == 22.5, with whole qr code side length 65-by-65
FINDER_PATTERN_ASPECT_RATIO_MAXIMUM = 1.5
FINDER_PATTERN_SIDE_LENGTH_MAXIMUM = 35 # avp1-0008,9, 10, maximum length is 31.06
FINDER_PATTERN_NEST_VERIFY_SIZE_RATIO_MINIMUM = 0.7
FINDER_PATTERN_NEST_VERIFY_CENTROID_DIST_MAXIMUM = 4
FINDER_PATTERN_NEST_VERIFY_CENTROID_DIST_RATIO_MAXIMUM = 0.5
VERIFY_GROUPED_SQUARE_AREA_DIFF_TO_MINIMUM_RATIO = 2.0 # Almost all take
VERIFY_ESTIMATED_SQUARE_DISTANCE = 20
RECTIFY_MARGIN = 5

# ------------------------------------------------------------------------------------------------------------
# Utility methods
# ------------------------------------------------------------------------------------------------------------
def utility_get_norm_dist(vec1, vec2):
    """ Normal distance
    """

    return np.linalg.norm(vec1.astype(np.float32) - vec2.astype(np.float32))

def utility_contains(one_contour, another_contour):
    """ Verify whether one contour contains another contour or not,
        two kinds of 'contain' are defined, nest and reside, according to the size comparison between
        parent contour and children contour. The return value is
        0: NotContain
        1: Reside
        2: Nest
    """

    if not one_contour.shape[0] >= 3 and another_contour.shape[0] >= 3:
        print('Invalid rectangle contour')
        return 0

    one_contour_area = cv2.contourArea(one_contour)
    one_contour_centroid = np.int32(np.mean(one_contour, axis=0)+0.5)

    for pt_index in range(another_contour.shape[0]):
        pt = (another_contour[pt_index, 0], another_contour[pt_index, 1])
        if (cv2.pointPolygonTest(one_contour, pt, False) < 0):
            return 0

    another_contour_area = cv2.contourArea(another_contour)
    another_contour_centroid = np.int32(np.mean(another_contour, axis=0)+0.5)

    centroid_dist = utility_get_norm_dist(one_contour_centroid, another_contour_centroid)
    another_contour_side_length = np.linalg.norm(np.vstack((np.diff(another_contour, axis=0),
                                        another_contour[0, :]-another_contour[another_contour.shape[0]-1, :])), axis=1)
    another_contour_minimum_side_length = np.amin(another_contour_side_length)

    # This condition needs to be further refined,
    # especially when the nested squares are in very small size, the following conditions may result in false negative
    if 1.0*another_contour_area/one_contour_area > FINDER_PATTERN_NEST_VERIFY_SIZE_RATIO_MINIMUM or \
        centroid_dist < FINDER_PATTERN_NEST_VERIFY_CENTROID_DIST_MAXIMUM or \
        1.0*centroid_dist/another_contour_minimum_side_length < FINDER_PATTERN_NEST_VERIFY_CENTROID_DIST_RATIO_MAXIMUM:
        return 2
    else:
        return 1

# ------------------------------------------------------------------------------------------------------------
# Method of extracting Block maps
# ------------------------------------------------------------------------------------------------------------


def pre_process_img(img):
    """ Pre-process the captured image """

    """ Make Gaussian blur in raw image to remove some noise """
    img_blur = cv2.GaussianBlur(img, ksize=(5, 5), sigmaX=1,sigmaY=1)

    """ Perform image enhancement to make black-white qr-code prominent"""
    img_blur_assist = cv2.GaussianBlur(img_blur, ksize=(PROBE_LEN, PROBE_LEN), sigmaX=1, sigmaY=1)
    img_blur_enhance = cv2.addWeighted(img_blur, 1.5, img_blur_assist, -0.5, 0)

    #displays1((img_blur_enhance_ref, img_blur_enhance))
    img_gray = cv2.cvtColor(img_blur_enhance, cv2.COLOR_BGR2GRAY)
    img_binarize = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
										 cv2.THRESH_BINARY, 2*BLOCK_RADIUS+1, 0)
    #displays1((img_binarize, img_gray))
    return img_blur_enhance, img_gray, img_binarize


def integrate_img(img, img_binarize):
    """ Integral image of color variance """

    height, width, channels = img.shape
    img_integral = np.zeros((height, width), np.float32)

    """ Get integral at the first column """
    for i in range(1, height):
        k = 1
        img_integral[i][0] = img_integral[i-1][0]
        nIncr = NUM_OF_INCRS
        while i - k >= 0 and k <= PROBE_LEN and nIncr > 0:
            if utility_get_norm_dist(img[i, 0], img[i-k, 0]) > EFFECTIVE_COLOR_DIFF:
                img_integral[i, 0] += INCR
                nIncr -= 1
            k += 1

    """ Get integral at the first row """
    for j in range(1, width):
        k = 1
        img_integral[0][j] = img_integral[0, j-1]
        nIncr = NUM_OF_INCRS
        while j - k >= 0 and k <= PROBE_LEN and nIncr > 0:
            if utility_get_norm_dist(img[0, j], img[0, j-k]) > EFFECTIVE_COLOR_DIFF:
                img_integral[0, j] = img_integral[0, j] + INCR
                nIncr -= 1
            k += 1

    for i in range(1, height):
        for j in range(1, width):

            """ Initialize img_integral[i, j] """
            img_integral[i, j] = img_integral[i, j-1]+ img_integral[i-1, j] - img_integral[i-1, j-1]

            """ Check regional switch between BG and FG
                 set as "NUM_FG_BG_SWITCH=2" or more fg-bg switch
            """
            region_consistency = np.sum(img_binarize[i, j-PROBE_LEN:j]/255)
            if region_consistency <= NUM_FG_BG_SWITCH or region_consistency >= PROBE_LEN-NUM_FG_BG_SWITCH:
                continue
            region_consistency = np.sum(img_binarize[i-PROBE_LEN:i, j]/255)
            if region_consistency <= NUM_FG_BG_SWITCH or region_consistency >= PROBE_LEN-NUM_FG_BG_SWITCH:
                continue

            """ Get probe in horizontal direction to left"""
            incr_j = 0
            k = 1
            nIncr = NUM_OF_INCRS
            while j-k >= 0 and k <= PROBE_LEN and nIncr > 0:
                if utility_get_norm_dist(img[i, j], img[i, j-k]) > EFFECTIVE_COLOR_DIFF:
                    incr_j += INCR
                    nIncr -= 1
                k += 1

            """ Get probe in vertical direction to top """
            incr_i = 0
            k = 1
            nIncr = NUM_OF_INCRS
            while i-k >= 0 and k <= PROBE_LEN and nIncr > 0:
                if utility_get_norm_dist(img[i, j], img[i-k, j]) > EFFECTIVE_COLOR_DIFF:
                    incr_i += INCR
                    nIncr -= 1
                k += 1

            # Not Sure the Diagnal Probes are necessary

            """ Get probe in diagnal direction to top-left """
            incr_i_j_45 = 0
            k = 1
            nIncr = NUM_OF_INCRS
            while i-k>=0 and j-k>=0 and k<=PROBE_LEN and nIncr>0:
                if utility_get_norm_dist(img[i, j], img[i-k, j-k]) > EFFECTIVE_COLOR_DIFF:
                    incr_i_j_45 += INCR
                    nIncr -= 1
                k += 1

            """ Get probe in diagnal direction to top-left downwards """
            incr_i_j_23 = 0
            k = 1
            k_time = 0
            nIncr = NUM_OF_INCRS
            while i-k >= 0 and j-k_time >= 0 and k <= PROBE_LEN and nIncr > 0:
                if utility_get_norm_dist(img[i, j], img[i-k, j-k_time]) > EFFECTIVE_COLOR_DIFF:
                    incr_i_j_23 += INCR
                    nIncr -= 1
                k += 1
                k_time += (np.mod(k, 3) == 0)


            """ Get probe in diagnal direction to top-left upwards """
            incr_i_j_67 = 0
            k = 1
            k_time = 0
            nIncr = NUM_OF_INCRS
            while i-k_time>=0 and j-k>=0 and k<=PROBE_LEN and nIncr>0:
                if utility_get_norm_dist(img[i, j], img[i-k_time, j-k]) > EFFECTIVE_COLOR_DIFF:
                    incr_i_j_67 += INCR
                    nIncr -= 1
                k += 1
                k_time += (np.mod(k, 3) == 0)

            """ Update img_integral[i, j]"""
            if incr_i > 0 and incr_j > 0 and incr_i_j_45 > 0 and incr_i_j_23 > 0 and incr_i_j_67 > 0:
                """ The method of fusing integral in both directions can be improved,
                    also the probe in other directions should better be included
                """
                img_integral[i, j] += incr_i + incr_j + incr_i_j_45 + incr_i_j_23 + incr_i_j_67

    return img_integral


def get_block_mask(img_integral, radius=BLOCK_RADIUS):
    """ Extract the pixels surrounded by blocks with higher variance than their counterparts """

    [height, width] = img_integral.shape
    block_mask = np.zeros((height, width), np.float32)
    for i in range(radius, height-radius):
        for j in range(radius, width-radius):
            block_mask[i, j] = img_integral[i+radius, j+radius] + img_integral[i-radius, j-radius] - \
                img_integral[i-radius, j+radius] - img_integral[i+radius, j-radius]

    block_mask_raw = block_mask

    """ Sort the values in block_mask, and generate the thresold by the number of Maximum foreground pixels """
    sorted_flattened_block_mask = np.sort(block_mask, axis=None)
    thr = sorted_flattened_block_mask[np.floor(6.0*sorted_flattened_block_mask.size/7)]
    thresold_return_value, block_mask = cv2.threshold(block_mask, thr, 255, cv2.THRESH_BINARY)
    block_mask = block_mask.astype(np.uint8)
    print("threshold value is " + str(thresold_return_value))

    return block_mask, block_mask_raw


def get_connected_components_from_block_mask(block_mask):
    """ Pro-process and Find out foreground connected components from the map of block mask """

    """ Perform morphological processing over the block mask map, to 1)
    remedy the holes within the foreground connected components 2) remove the tiny noise """

    block_mask_morph = cv2.morphologyEx(block_mask, cv2.MORPH_CLOSE, SE_CLOSE)
    block_mask_morph = cv2.morphologyEx(block_mask, cv2.MORPH_OPEN, SE_OPEN)

    """ Find out all connected components through contours and their related bounding rectangles
        in the form of (x,y, width, height)
        Make a copy since the "findContours" method will use it as output """
    block_mask_morph_copy = block_mask_morph.copy()
    _, contours, hierarchy = cv2.findContours(block_mask_morph_copy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    bounding_rects_hypotheses = [cv2.boundingRect(c) for c in contours]

    """ Remove invalid bounding rects which are
        in too small size
    """
    bounding_rects_hypotheses = [br for br in bounding_rects_hypotheses if
                br[3] >= BR_HEIGHT_MINIMUM
            and     (br[2] >= BR_WIDTH_MINIMUM)
            and     (1.0/BR_ASPECT_RATIO_MAXIMUM <= 1.0*br[2]/br[3] <= BR_ASPECT_RATIO_MAXIMUM)
            ]

    return block_mask_morph, bounding_rects_hypotheses

# ------------------------------------------------------------------------------------------------------------
# Method of extracting finder Patterns
# ------------------------------------------------------------------------------------------------------------


def clear_nested_contours(img_patch, contours):
    """ Localize the three square finder patterns of each qr code hypotheses
        if finder patterns do not exist, discard the hypothesis as false alarm
        if finder patterns are localized, rectify the hypothesis rect and return it as detected qr-code region
    """

    """ Indicate whether a contour should be preserved or removed
        check_map element is a list in format:
        [contour_idx, contain_contours, reside_contained_by_contours, nest_contained_by_contours]
    """
    if not contours:
        return [], [], np.empty((0, 0), np.int32)

    num_contours = len(contours)
    check_map = np.zeros((num_contours, num_contours), np.int32)

    """ Compare each pair of contours to check their containing relations """
    for i in range(0, num_contours-1):
        for j in range(i+1, num_contours):
            if utility_contains(contours[i], contours[j]) == 1:
                check_map[i, j] = 1
                check_map[j, i] = -1
            if utility_contains(contours[i], contours[j]) == 2:
                check_map[i, j] = 2
                check_map[j, i] = -2

    #print(check_map)

    """ If a contour is Nest contained by two other contours, and
        the two contours do not have any contain relations (means check_map==0), then we will assign a nest contain relation to them
    """
    while True:
        action_marker = False
        for (i, j) in itertools.product(range(num_contours), range(num_contours)):
            if i < j and check_map[i, j] == 0 and check_map[j, i]==0:
                common_nest_map = np.logical_and(check_map[i, :]==2, check_map[i, :]==check_map[j, :])
                if np.any(common_nest_map):
                    check_map[i, j] = 2
                    check_map[j, i] = -2
                    action_marker = True
                    break
        if not action_marker:
            break

    noise_contour_set = set([])

    """ 1) For all Nest contain, only the external contour is preserved, and remove others """

    for (i, j) in itertools.product(range(num_contours), range(num_contours)):
        if i not in noise_contour_set and check_map[i, j] == -2:
            noise_contour_set |= set([i])
    for i in range(num_contours):
        if i in noise_contour_set:
            check_map[i, :] = 0

    """ 2) For all Reside contain, removing the external contours layer-by-layer """

    while True:
        action_marker = False
        for (i, j) in itertools.product(range(num_contours), range(num_contours)):
            if i not in noise_contour_set and check_map[i, j] == 1:
                noise_contour_set |= set([i])
                check_map[i, :] = 0
                check_map[j, i] = 0
                action_marker = True
                break
        if not action_marker:
            break

    contours_clear = [contours[i] for i in range(num_contours) if i not in noise_contour_set]
    contours_nest_external = [contours[i] for i in range(num_contours) if
                            i not in noise_contour_set and
                            2 in set(check_map[i, :])]

    # Some issues of the check_map
    return contours_clear, contours_nest_external, check_map


def estimate_a_third_contour_centroid(centroid1, centroid2):
    """ Given two squares that are from qr-code finder-pattern,
    estimate the third contour centroid by using cross product of the homography coordinate ,
    then verify its existence
    """
    centroid_distance = utility_get_norm_dist(centroid1, centroid2)
    centroid1_homo = np.append(centroid1.astype(np.float32), np.array([0], np.float32))
    centroid2_homo = np.append(centroid2.astype(np.float32), np.array([0], np.float32))
    centroid0_estimates = []

    """ 1) When the two given squares are at the Side """
    centroid_diff_vec = centroid2_homo - centroid1_homo
    ancillary = np.array([0., 0., centroid_distance], np.float32)
    centroid0_homo = np.cross(centroid_diff_vec, ancillary) / centroid_distance + centroid1_homo
    centroid0_estimates.append((centroid0_homo[0:2]+0.5).astype(np.int32))
    centroid0_homo = np.cross(ancillary, centroid_diff_vec) / centroid_distance + centroid1_homo
    centroid0_estimates.append((centroid0_homo[0:2]+0.5).astype(np.int32))

    centroid_diff_vec = centroid1_homo - centroid2_homo
    ancillary = np.array([0., 0., centroid_distance], np.float32)
    centroid0_homo = np.cross(centroid_diff_vec, ancillary) / centroid_distance + centroid2_homo
    centroid0_estimates.append((centroid0_homo[0:2]+0.5).astype(np.int32))
    centroid0_homo = np.cross(ancillary, centroid_diff_vec) / centroid_distance + centroid2_homo
    centroid0_estimates.append((centroid0_homo[0:2]+0.5).astype(np.int32))

    """ 2) When the two given squares are at the Diagonal """
    centroid_diff_vec = (centroid2_homo - centroid1_homo) * 0.5
    ancillary = np.array([0., 0., centroid_distance/2], np.float32)

    centroid0_homo = \
        np.cross(centroid_diff_vec, ancillary) / (centroid_distance/2) + (centroid1_homo+centroid2_homo)*0.5
    centroid0_estimates.append((centroid0_homo[0:2]+0.5).astype(np.int32))

    centroid0_homo = \
        np.cross(ancillary, centroid_diff_vec) / (centroid_distance/2) + (centroid1_homo+centroid2_homo)*0.5
    centroid0_estimates.append((centroid0_homo[0:2]+0.5).astype(np.int32))

    return centroid0_estimates


def extract_a_third_contour_centroid_by_estimates(centroid_estimates, contours, ct1, ct2):
    """
    centroid_estimates: The estimated positions of the third square centroid
    contours: The whole set of contours after the clear operation, not only the contour_nest_external
    area1 and area2: The areas of the two given square centroids
    return the index of the verified contour, otherwise -1
    """

    area1 = cv2.contourArea(ct1)
    area2 = cv2.contourArea(ct2)

    for ct0_est in centroid_estimates:
        if ct0_est[0] > 0 and ct0_est[1] > 0:
            for i in range(len(contours)):
                contour = contours[i]

                if np.array_equal(contour, ct1) or np.array_equal(contour, ct2):
                    continue

                area0 = cv2.contourArea(contour)
                ct0 = np.int32(np.mean(contour, axis=0)+0.5)

                """ Conditions of Area Difference """
                if np.fabs(area1-area0) <= np.amin([area1, area0])*VERIFY_GROUPED_SQUARE_AREA_DIFF_TO_MINIMUM_RATIO and \
                    np.fabs(area2-area0) <= np.amin([area2, area0])*VERIFY_GROUPED_SQUARE_AREA_DIFF_TO_MINIMUM_RATIO:
                    """ Conditions of Distances """
                    if (abs(ct0_est[0]-ct0[0]) <= VERIFY_ESTIMATED_SQUARE_DISTANCE and
                        abs(ct0_est[1]-ct0[1]) <= VERIFY_ESTIMATED_SQUARE_DISTANCE) or \
                        ((abs(ct0_est[0]-ct0[0]) <= VERIFY_ESTIMATED_SQUARE_DISTANCE or
                            abs(ct0_est[1]-ct0[1]) <= VERIFY_ESTIMATED_SQUARE_DISTANCE) and
                            utility_get_norm_dist(ct0_est, ct0) <= VERIFY_ESTIMATED_SQUARE_DISTANCE*np.sqrt(2)):
                        print("Extracted a third contour through the centroid estimates.")
                        return i

    return -1


def extract_grouped_squared_finder_patterns(img_patch, contours, contours_nest_external, check_map):
    """  Given the cleared contours from Nest/Reside containing relations,
        find out the finder patterns (composed of the three squares at corners) to extract true positive qr-code
        the input has: contours_nest_external = [contours[i] for i in range(num_contours) if 2 in set(check_map[i, :])]
    """
    num_contours = len(contours)
    num_contours_nest_external = len(contours_nest_external)

    """ Different processes according to the number of already detected squares  """
    if num_contours_nest_external == 0:
        return 'False',[]

    elif num_contours_nest_external == 1:

        # To be improved
        """ Get all the side lengths of the one given finder-pattern square
            , and take it if it is small enough

        ct = contours_nest_external[0]
        side0 = utility_get_norm_dist(ct[0, :], ct[1, :])
        side1 = utility_get_norm_dist(ct[1, :], ct[2, :])
        side2 = utility_get_norm_dist(ct[2, :], ct[3, :])
        side3 = utility_get_norm_dist(ct[3, :], ct[0, :])
        if np.amax([side0,side1,side2,side3]) <= FINDER_PATTERN_SIDE_LENGTH_MAXIMUM:
            return 'Probable', contours_nest_external
        else:
            return 'False',[]
        """
        return 'False', contours_nest_external

    elif num_contours_nest_external == 2:
        ct1 = contours_nest_external[0]
        ct2 = contours_nest_external[1]
        area1 = cv2.contourArea(ct1)
        area2 = cv2.contourArea(ct2)

        """ The two squares should have similar sizes """
        if np.fabs(area1-area2) > np.amin([area1, area2]) * VERIFY_GROUPED_SQUARE_AREA_DIFF_TO_MINIMUM_RATIO:
            return 'False', contours_nest_external

        centroid1 = np.int32(np.mean(ct1, axis=0)+0.5)
        centroid2 = np.int32(np.mean(ct2, axis=0)+0.5)
        centroid0_estimates = estimate_a_third_contour_centroid(centroid1, centroid2)
        centroid0_truth_indx = extract_a_third_contour_centroid_by_estimates(centroid0_estimates, contours, ct1, ct2)
        if centroid0_truth_indx != -1:
            contours_nest_external.append(contours[centroid0_truth_indx])
            return 'True', contours_nest_external
        else:
            """ Get all the side lengths of the two given finder-pattern squares
            , and take it if it is small enough
            """
            side10 = utility_get_norm_dist(ct1[0, :], ct1[1, :])
            side11 = utility_get_norm_dist(ct1[1, :], ct1[2, :])
            side12 = utility_get_norm_dist(ct1[2, :], ct1[3, :])
            side13 = utility_get_norm_dist(ct1[3, :], ct1[0, :])
            side20 = utility_get_norm_dist(ct2[0, :], ct2[1, :])
            side21 = utility_get_norm_dist(ct2[1, :], ct2[2, :])
            side22 = utility_get_norm_dist(ct2[2, :], ct2[3, :])
            side23 = utility_get_norm_dist(ct2[3, :], ct2[0, :])
            if np.amax([side10, side11, side12, side13]) <= FINDER_PATTERN_SIDE_LENGTH_MAXIMUM and \
                            np.amax([side20, side21, side22, side23]) <= FINDER_PATTERN_SIDE_LENGTH_MAXIMUM:
                return 'Probable', contours_nest_external
            else:
                return 'False', contours_nest_external

    elif num_contours_nest_external == 3:
        return 'True', contours_nest_external

    elif 4 <= num_contours_nest_external <= 7:
        #return False,[] # TODO
        return 'Probable', contours_nest_external

    else:
        return 'False', contours_nest_external


def get_finder_patterns_in_hypotheses(img, bounding_rects_hypotheses, is_pop_display):
    """ Get the hypotheses that truly contains finder patterns """
    qr_code_hypothesis_detections = []

    """ Process the hypotheses ONE-BY-ONE """
    for br in bounding_rects_hypotheses:

        img_patch = img[br[1]:br[1]+br[3], br[0]:br[0]+br[2]]
        img_patch_gray = cv2.cvtColor(img_patch, cv2.COLOR_BGR2GRAY)

        """ Get the edge map of each hypothesis patch"""
        img_edge = cv2.Canny(img_patch_gray, CANNY_LOW_THRESHOLD, CANNY_HIGH_THRESHOLD, L2gradient=True)
        img_edge_copy = img_edge.copy()
        _, contours, hierarchy = cv2.findContours(img_edge_copy, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        """ Figure out a Poly approximation for each contour in this hypothesis,
            and the finder pattern should an approximated 4 corner polygon (rectangle/square, named as square) ,
            and the size of the finder pattern square should not be too small, otherwise consider it as background noise
            and the aspect ratio of the finder pattern square should not be too large
        """
        contours = [cv2.approxPolyDP(ct, cv2.arcLength(ct, True)*0.05, True) for ct in contours]
        contours = [np.reshape(ct,(ct.shape[0], ct.shape[2])) for ct in contours]
        contours_quad = [ct for ct in contours if ct.shape[0] == 4]

        """ Perform quad clear """
        contours_quad_clear = []
        for ct in contours_quad:

            area = cv2.contourArea(ct)
            side0 = utility_get_norm_dist(ct[0, :], ct[1, :])
            side1 = utility_get_norm_dist(ct[1, :], ct[2, :])
            side2 = utility_get_norm_dist(ct[2, :], ct[3, :])
            side3 = utility_get_norm_dist(ct[3, :], ct[0, :])

            # angle between neighboring sides may be useful
            if (area > FINDER_PATTERN_AREA_MINIMUM
                    and (
                    1.0/FINDER_PATTERN_ASPECT_RATIO_MAXIMUM <= side0/side1 <= FINDER_PATTERN_ASPECT_RATIO_MAXIMUM and
                    1.0/FINDER_PATTERN_ASPECT_RATIO_MAXIMUM <= side1/side2 <= FINDER_PATTERN_ASPECT_RATIO_MAXIMUM and
                    1.0/FINDER_PATTERN_ASPECT_RATIO_MAXIMUM <= side2/side3 <= FINDER_PATTERN_ASPECT_RATIO_MAXIMUM and
                    1.0/FINDER_PATTERN_ASPECT_RATIO_MAXIMUM <= side3/side0 <= FINDER_PATTERN_ASPECT_RATIO_MAXIMUM
                        )):
                contours_quad_clear.append(ct)

        """ Perform nest/reside contain clear """
        contours_quad_nest_clear, contours_quad_nest_clear_external, check_map = \
                                                    clear_nested_contours(img_patch, contours_quad_clear)
        print("Length of cleared set of contours and external nest contours: " +
              str(len(contours_quad_nest_clear)) + " , " + str(len(contours_quad_nest_clear_external)))

        """ Fix and Verify the squares of finder patterns to extract true positive qr-code patches """
        is_containing_qr_code, contours_finalize = extract_grouped_squared_finder_patterns(
                                img_patch, contours_quad_nest_clear, contours_quad_nest_clear_external, check_map)
        print("is_containing_qr_code: " + is_containing_qr_code + " , " + "length: " + str(len(contours_finalize)))

        # visualize
        if is_pop_display:
            img_patch_dst_quad = visualize_contours(img_patch, contours_quad)
            img_patch_dst_quad_clear = visualize_contours(img_patch, contours_quad_clear)
            img_patch_dst_quad_nest_clear = visualize_contours(img_patch, contours_quad_nest_clear)
            img_patch_dst_quad_nest_clear_external = visualize_contours(img_patch, contours_quad_nest_clear_external)
            img_displays = (img_patch_dst_quad, img_patch_dst_quad_clear,
                            img_patch_dst_quad_nest_clear, img_patch_dst_quad_nest_clear_external)
            displays1(img_displays)

        """ Save the extracted true-positive qr-code patches """
        contours_finalize_offset = [np.add(ct, np.array([br[0], br[1]])) for ct in contours_finalize]
        result = {
                    'hypothesis_bounding_rect': [np.array([[br[0], br[1]], [br[0]+br[2]-1, br[1]],
                                                         [br[0]+br[2]-1, br[1]+br[3]-1], [br[0], br[1]+br[3]-1]])],
                    'img_patch': img_patch,
                    'finder_patterns': contours_finalize_offset,
                    'is_containing_qr_code': is_containing_qr_code}
        qr_code_hypothesis_detections.append(result)

    return qr_code_hypothesis_detections


# ------------------------------------------------------------------------------------------------------------
# Rectification
# ------------------------------------------------------------------------------------------------------------
def rectify_by_binarization(img, qr_code_hypothesis_detections):
    """
    Add 'img_patch_rectified' into the fields of 'qr_code_hypothesis_detections'
    """

    height, width, depth = img.shape

    for detection in qr_code_hypothesis_detections:

        is_qr_code = detection['is_containing_qr_code']

        """ 0, 1, 2, 3 corresponds to top-left, Top-right, bottom-right, bottom-left """
        rectified_rect = np.zeros((4, 2), np.int32)

        if is_qr_code is 'True' : #or is_qr_code is 'Probable'

            contours_fp = detection['finder_patterns']
            xmin = np.amin([np.amin(cr[:, 0]) for cr in contours_fp])
            xmax = np.amax([np.amax(cr[:, 0]) for cr in contours_fp])
            ymin = np.amin([np.amin(cr[:, 1]) for cr in contours_fp])
            ymax = np.amax([np.amax(cr[:, 1]) for cr in contours_fp])

            xmin = max([1, xmin-RECTIFY_MARGIN])
            xmax = min([width, xmax+RECTIFY_MARGIN])
            ymin = max([1, ymin-RECTIFY_MARGIN])
            ymax = min([height, ymax+RECTIFY_MARGIN])

            img_patch_rectified = img[ymin:ymax,xmin:xmax]
            detection['img_patch'] = img_patch_rectified

    return 0


# ------------------------------------------------------------------------------------------------------------
# Visualization
# ------------------------------------------------------------------------------------------------------------
def displays1(data_set):
    """ Display the specified image or data in windows """

    k = 0
    for data in data_set:
        if data.dtype is np.dtype('float32'):
            data_uchar = ((data-np.amin(data)) / (np.amax(data)-np.amin(data)) * 255).astype(np.uint8)
        else :
            data_uchar = data
        cv2.imshow("win_name" + str(k), data_uchar)
        k += 1
    cv2.waitKey(0)
    while k >= 0:
        cv2.destroyWindow("win_name"+str(k))
        k -= 1


def displays2(img, rects, color=(255, 255, 0)):
    """ Display the specified bounding boxes in an image """

    img_dst = img.copy()
    contours = [np.array([[r[0], r[1]], [r[0]+r[2]-1, r[1]],
                        [r[0]+r[2]-1, r[1]+r[3]-1], [r[0], r[1]+r[3]-1]]) for r in rects]
    cv2.drawContours(img_dst, contours, -1, color, thickness=3)
    return img_dst


def visualize_contours(img, contours, color=(255, 0, 0)):
    """ Display the specified contours in an image """

    img_dst = img.copy()
    cv2.drawContours(img_dst, contours, -1, color, thickness=3)
    return img_dst


def visualize_detections(img, detection, color_true=(255, 255, 0), color_probable=(0, 255, 0), color_false=(0, 0, 255)):
    """ Display the specified contours in an image,
        and the contours appear in different color according to is_qr_code
    """

    is_qr_code = detection['is_containing_qr_code']
    contours_finder_patterns = detection['finder_patterns']
    contours_bounding_rects = detection['hypothesis_bounding_rect']

    """ Visualize all rectangle boundaries of qr-code hypotheses """
    img_dst = img.copy()
    for i in range(len(contours_bounding_rects)):
        if is_qr_code is 'True':
            cv2.drawContours(img_dst, contours_bounding_rects, i, color_true, thickness=3)
        if is_qr_code is 'Probable': # consider the situations of more than 3 external squares
            cv2.drawContours(img_dst, contours_bounding_rects, i, color_probable, thickness=3)
        if is_qr_code is 'False':
            cv2.drawContours(img_dst, contours_bounding_rects, i, color_false, thickness=3)

    """ Visualize all finder patterns """
    img_dst = visualize_contours(img_dst, contours_finder_patterns, color=(255, 0, 0))

    return img_dst


# ------------------------------------------------------------------------------------------------------------
# Main entry
# ------------------------------------------------------------------------------------------------------------
def detect_qr_code(img, is_pop_display):

    """ QR code detection
    return a list of dictionaries, which contain
    'hypothesis_bounding_rect' as bounding box of qr-code hypothesis,
    'finder_patterns' as finder pattern contours,
    'is_containing_qr_code' as the indicator of true, false, or probable qr-codes
    """

    img_blur, img_blur_gray, img_blur_binarize = pre_process_img(img)
    img_integral = integrate_img(img_blur, img_blur_binarize)
    block_mask, block_mask_raw = get_block_mask(img_integral)
    block_mask, bounding_rects_hypotheses = get_connected_components_from_block_mask(block_mask)

    qr_code_hypothesis_detections = get_finder_patterns_in_hypotheses(img, bounding_rects_hypotheses, is_pop_display)

    return block_mask, block_mask_raw, qr_code_hypothesis_detections


