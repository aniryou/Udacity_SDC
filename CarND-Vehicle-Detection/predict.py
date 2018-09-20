import os
import glob
import numpy as np
import pandas as pd
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn.cross_validation import train_test_split
from sklearn.utils import shuffle

from sklearn.svm import LinearSVC, SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import KernelPCA
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
from scipy.ndimage.measurements import label


clf = joblib.load("model/clf_ycrcb.jbl")


def get_hog_features(img):
    assert img.shape[0] == 64
    return hog(img, block_norm='L2-Hys', feature_vector=False)

def get_spatial_features(img):
    assert img.shape == (64, 64, 3)
    return img.reshape(-1)

def get_hist_features(img, n_bins=12):
    assert img.shape == (64, 64, 3)
    hist_r = np.histogram(img[:,:,0], bins=n_bins)[0]
    hist_g = np.histogram(img[:,:,1], bins=n_bins)[0]
    hist_b = np.histogram(img[:,:,2], bins=n_bins)[0]
    return np.concatenate([hist_r, hist_g, hist_b])


def predict_heatmap(img_orig):
    heatmap = np.zeros(img_orig.shape[:2], np.uint8)

    START_X, END_X = 400, 680
    START_Y, END_Y = 0, 1280

    img = img_orig[START_X: END_X, :, :]
    img = np.float32(1. * img / img.reshape(-1, 3).max(axis=0))
    img_YCrCb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)

    WINDOW_SIZE_MIN = 64
    WINDOW_SIZE_MAX = END_X - START_X
    WINDOW_SIZE_STEP = 32

    PIXELS_PER_CELL = 8
    CELLS_PER_BLOCK = 3

    EPSILON = 1e-5

    IMG_CLF_SIZE = 64
    IMG_CLF_SHAPE = (IMG_CLF_SIZE, IMG_CLF_SIZE)

    # iterate over incrementally large slices of images from vehicles
    # further away, to those nearer to us
    lst_max = []
    lst_max2 = []
    for window_size in np.arange(WINDOW_SIZE_MIN, WINDOW_SIZE_MAX + 1, WINDOW_SIZE_STEP):

        img_slice = img[:window_size, :]
        img_YCrCb_slice = img_YCrCb[:window_size, :]

        # re-scale image
        scaling_factor = 1.0 * IMG_CLF_SIZE / window_size
        img_scaled = cv2.resize(img_slice, (0, 0),
                                fx=scaling_factor,
                                fy=scaling_factor)
        img_YCrCb_scaled = cv2.resize(img_YCrCb_slice, (0, 0),
                                      fx=scaling_factor,
                                      fy=scaling_factor)

        # create hog feature, one-time for rescaled image-slice
        feat_hog_all = get_hog_features(img_YCrCb_scaled[:, :, 0])

        # number of windows
        window_slide_step_y_scaled = int(1. / 3 * window_size * scaling_factor)
        n_window_y = (img_scaled.shape[1] - IMG_CLF_SIZE) // window_slide_step_y_scaled + 1

        for i in np.arange(n_window_y):
            window_start_x = 0
            window_end_x = IMG_CLF_SIZE
            window_start_y = int(i * window_slide_step_y_scaled)
            window_end_y = int(i * window_slide_step_y_scaled) + IMG_CLF_SIZE

            cells_per_window = IMG_CLF_SIZE // PIXELS_PER_CELL
            blocks_per_window = cells_per_window - CELLS_PER_BLOCK + 1
            cells_per_step = window_slide_step_y_scaled // PIXELS_PER_CELL
            blocks_per_step = cells_per_step - CELLS_PER_BLOCK + 1

            feat_hog = feat_hog_all[:blocks_per_window,
                       (i * cells_per_step):(i * cells_per_step + blocks_per_window)]

            img_window = img_scaled[window_start_x:window_end_x, window_start_y:window_end_y]

            feat_spatial = img_window.reshape(-1)
            feat_hist = get_hist_features(img_window)

            img_scaled_with_window = np.copy(img_scaled)
            img_scaled_with_window = cv2.rectangle(img_scaled_with_window,
                                                   (window_start_y, window_start_x),
                                                   (window_end_y, window_end_x), (0, 255, 0), 4)

            feat = np.concatenate((feat_spatial, feat_hist, feat_hog.reshape(-1)))

            if clf.predict(feat[np.newaxis, :]) == 1:
                window_unscaled_start_x = int(window_start_x / scaling_factor)
                window_unscaled_end_x = int(window_end_x / scaling_factor)

                window_unscaled_start_y = int(window_start_y / scaling_factor)
                window_unscaled_end_y = int(window_end_y / scaling_factor)

                heatmap[START_X + window_unscaled_start_x:START_X + window_unscaled_end_x,
                START_Y + window_unscaled_start_y:START_Y + window_unscaled_end_y] += 1

    img_pred = 1. * heatmap / heatmap.max()
    return heatmap, img_pred

def apply_threshold(heatmap, thresh):
    heatmap = heatmap.copy()
    heatmap[heatmap <= thresh] = 0
    return heatmap

def height(pt1, pt2):
    return  pt2[1] - pt1[1]

def width(pt1, pt2):
    return pt2[0] - pt1[0]

def distance(pt1, pt2):
    return np.sqrt(height(pt1, pt2) * width(pt1, pt2))

def pred(lst_pt, lim=5):
    lst_x, lst_y = zip(*lst_pt)
    if len(lst_pt) > 1:
        weights = 0.99**np.arange(len(lst_pt)-1)[::-1]
        pred_x = lst_x[-1] + np.average(np.array(lst_x[1:]) - np.array(lst_x[:-1]), weights=weights).astype(int)
        pred_y = lst_y[-1] + np.average(np.array(lst_y[1:]) - np.array(lst_y[:-1]), weights=weights).astype(int)
        # print("Weight", weights)
        # print("Diff X", np.array(lst_x[1:]) - np.array(lst_x[:-1]),
        #       np.average(np.array(lst_x[1:]) - np.array(lst_x[:-1]), weights=weights).astype(int))
        # print("Diff Y", np.array(lst_y[1:]) - np.array(lst_y[:-1]),
        #       np.average(np.array(lst_y[1:]) - np.array(lst_y[:-1]), weights=weights).astype(int))
    else:
        pred_x = lst_x[-1]
        pred_y = lst_y[-1]
    return (pred_x, pred_y)


class BBox(object):
    def __init__(self, pt1, pt2, thresh=2, margin_low=0.1, margin_high=0.5):
        self.pt1 = pt1
        self.pt2 = pt2
        self.lst_prev_pt1 = [pt1]
        self.lst_prev_pt2 = [pt2]
        self.thresh = thresh
        self.margin_low = margin_low
        self.margin_high = margin_high
        self.prev_widht = 64

    def fit(self, img, img_orig, margin=0.1):
        img_orig = img_orig.copy()
        thresh = self.thresh #max((self.prev_widht - 64) // 16, self.thresh)
        img_thresh = apply_threshold(img, thresh)
        img_label = label(img_thresh)[0]

        margin_x = int(margin * height(self.pt1, self.pt2))
        margin_y = int(margin * width(self.pt1, self.pt2))

        pred_next_pt1 = pred(self.lst_prev_pt1)
        pred_next_pt2 = pred(self.lst_prev_pt2)
        print("Pred", pred_next_pt1, pred_next_pt2)

        y1, x1 = self.pt1  # pred_next_pt1
        y2, x2 = self.pt2  # pred_next_pt2

        x1_low = max(x1 - margin_x, 0)
        x1_high = x1 + margin_x
        y1_low = max(y1 - margin_y, 0)
        y1_high = y1 + margin_y

        x2_low = max(x2 - margin_x, 0)
        x2_high = x2 + margin_x
        y2_low = max(y2 - margin_y, 0)
        y2_high = y2 + margin_y

        img_tl = img_label[x1_low:x1_high, y1_low:y1_high]
        img_tr = img_label[x1_low:x1_high, y2_low:y2_high]
        img_bl = img_label[x2_low:x2_high, y1_low:y1_high]
        img_br = img_label[x2_low:x2_high, y2_low:y2_high]

        car_number1 = img_tl[-1, -1]
        car_number2 = img_br[0, 0]

        car_number = car_number1
        if (car_number1 != car_number2) and len(self.lst_prev_pt1) > 1:
            if distance(self.lst_prev_pt1[-1], self.lst_prev_pt1[-2]) < \
                    distance(self.lst_prev_pt2[-1], self.lst_prev_pt2[-2]):
                car_number = car_number1
            else:
                car_number = car_number2

        nonzero_tl = (img_tl == car_number).nonzero()
        nonzero_tl_x = np.array(nonzero_tl[0])
        nonzero_tl_y = np.array(nonzero_tl[1])

        nonzero_tr = (img_tr == car_number).nonzero()
        nonzero_tr_x = np.array(nonzero_tr[0])
        nonzero_tr_y = np.array(nonzero_tr[1])

        nonzero_bl = (img_bl == car_number).nonzero()
        nonzero_bl_x = np.array(nonzero_bl[0])
        nonzero_bl_y = np.array(nonzero_bl[1])

        nonzero_br = (img_br == car_number).nonzero()
        nonzero_br_x = np.array(nonzero_br[0])
        nonzero_br_y = np.array(nonzero_br[1])

        img_orig = cv2.rectangle(img_orig, (y1_low, x1_low), (y1_high, x1_high), (255, 0, 0), 4)
        img_orig = cv2.rectangle(img_orig, (y2_low, x1_low), (y2_high, x1_high), (255, 0, 0), 4)
        img_orig = cv2.rectangle(img_orig, (y1_low, x2_low), (y1_high, x2_high), (255, 0, 0), 4)
        img_orig = cv2.rectangle(img_orig, (y2_low, x2_low), (y2_high, x2_high), (255, 0, 0), 4)
        plt.imshow(img_orig)
        plt.show()

        new_pt1, new_pt2 = None, None

        if (nonzero_tl_x.shape[0] * nonzero_tr_x.shape[0] * nonzero_bl_x.shape[0] * nonzero_br_x.shape[0]) > 0 \
                and car_number > 0:
            tmp_pt_tl_x = x1 - margin_x + np.min(nonzero_tl_x)
            tmp_pt_tl_y = y1 - margin_y + np.min(nonzero_tl_y)
            tmp_pt_tr_x = x1 - margin_x + np.min(nonzero_tr_x)
            tmp_pt_tr_y = y2 - margin_y + np.max(nonzero_tr_y)
            tmp_pt_bl_x = x2 - margin_x + np.max(nonzero_bl_x)
            tmp_pt_bl_y = y1 - margin_y + np.min(nonzero_bl_y)
            tmp_pt_br_x = x2 - margin_x + np.max(nonzero_br_x)
            tmp_pt_br_y = y2 - margin_y + np.max(nonzero_br_y)

            tmp_pt1 = max(tmp_pt_tl_y, tmp_pt_bl_y), max(tmp_pt_tl_x, tmp_pt_tr_x)
            tmp_pt2 = min(tmp_pt_tr_y, tmp_pt_br_y), min(tmp_pt_bl_x, tmp_pt_br_x)

            tmp_height = height(tmp_pt1, tmp_pt2)
            tmp_width = width(tmp_pt1, tmp_pt2)

            if (0.5 * tmp_height) <= tmp_width <= (2.0 * tmp_height):
                self.prev_widht = tmp_width
                new_pt1, new_pt2 = tmp_pt1, tmp_pt2

        if new_pt1 is None and margin < self.margin_high:
            new_pt1, new_pt2 = self.fit(img, img_orig, margin=margin + 0.1)
            self.lst_prev_pt1 = self.lst_prev_pt1[:-1]
            self.lst_prev_pt2 = self.lst_prev_pt2[:-1]

        self.pt1, self.pt2 = new_pt1, new_pt2

        self.lst_prev_pt1.append(self.pt1)
        self.lst_prev_pt2.append(self.pt2)

        return (self.pt1, self.pt2)

    def area(self):
        self_height = height(self.pt1, self.pt2)
        self_width = width(self.pt1, self.pt2)
        self_area = self_height * self_width
        return self_area

    def overlap(self, other):
        self_y1, self_x1 = self.pt1
        self_y2, self_x2 = self.pt2
        other_y1, other_x1 = other.pt1
        other_y2, other_x2 = other.pt2

        common_y1, common_x1 = max(self_y1, other_y1), max(self_x1, other_x1)
        common_y2, common_x2 = min(self_y2, other_y2), min(self_x2, other_x2)
        common_pt1, common_pt2 = (common_y1, common_x1), (common_y2, common_x2)
        common = BBox(common_pt1, common_pt2)
        common_area = common.area()

        overlap_fraction = 1. * common_area / self.area()

        return overlap_fraction

    def __repr__(self):
        y1, x1 = self.pt1
        y2, x2 = self.pt2
        return '[({0},{1}),({2},{3})]'.format(self.pt1[0], self.pt1[1], self.pt2[0], self.pt2[1])


class VehicleDetection():

    def __init__(self):
        self.bboxes = []
        self.thresh_high = 1
        self.thresh_low = 0

    def fit(self, img):
        new_bboxes = []

        heatmap, img_pred = predict_heatmap(img)
        # plt.imshow(img_pred)
        # plt.show()

        for bbox in self.bboxes:
            pt1, pt2 = bbox.fit(heatmap, img_pred)
            if pt1 is not None:
                new_bboxes.append(bbox)

        img_thresh = apply_threshold(heatmap, self.thresh_high)
        labels = label(img_thresh)

        print("Detected", labels[1], "Cars")
        for car_number in range(1, labels[1] + 1):
            nonzero = (labels[0] == car_number).nonzero()
            nonzerox = np.array(nonzero[0])
            nonzeroy = np.array(nonzero[1])

            if nonzerox.shape[0] > 0:
                pt1 = (np.min(nonzeroy), np.min(nonzerox))
                pt2 = (np.max(nonzeroy), np.max(nonzerox))

                bbox_height = height(pt1, pt2)
                bbox_width = width(pt1, pt2)

                if (0.5 * bbox_height) <= bbox_width <= (2.0 * bbox_width):
                    thresh = max((bbox_width - 128) / 32, self.thresh_low)
                    bbox = BBox(pt1, pt2, thresh=thresh)
                    new_bboxes.append(bbox)

        print("Found", "; ".join([str(bbox) for bbox in new_bboxes]))
        img_result = self.draw(img, new_bboxes)

        return img_result

    def draw(self, img, new_bboxes):
        img_result = img.copy()
        overlapping_bboxes = []
        for bbox1 in new_bboxes:
            flag = False
            for bbox2 in new_bboxes:
                if bbox2 in overlapping_bboxes:
                    continue
                elif bbox1 != bbox2 and bbox1.overlap(bbox2) >= 0.9:
                    flag = True
            if flag:
                overlapping_bboxes.append(bbox1)
        self.bboxes = [bbox for bbox in new_bboxes if bbox not in overlapping_bboxes]
        print("Overlapping", "; ".join([str(bbox) for bbox in overlapping_bboxes]))
        print("Drawing", "; ".join([str(bbox) for bbox in self.bboxes]))
        for bbox in self.bboxes:
            img_result = cv2.rectangle(img_result, bbox.pt1, bbox.pt2, (0, 0, 255), 6)
        return img_result


if __name__ == '__main__':
    import pickle
    vd = pickle.load(open("output_images/vd_724.pkl", 'rb'))
    img_path = "output_images/%d.png" % 724
    # for img_path in glob.glob("output_images/*.png")[:100]:
    print(img_path)
    img = mpimg.imread(img_path)[:, :, :3]

    img_result = vd.fit(img.copy())
    print("Now draw me")
    plt.imshow(img_result)
    plt.show()
