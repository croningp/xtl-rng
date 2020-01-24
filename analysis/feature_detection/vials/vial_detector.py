import os
import cv2
import numpy as np

def find_vial(image_path, compound):

    """ uses hsv thresholds to isolate vial rim, then contour detection to get vial rim.
        If successful it returns x, y, and radius of circle associated with rim.
        If unsuccessful it returns 0, 0, 0 """

    assert os.path.exists(image_path), "{} doesn't exist.format(image_path)"
    best_contour = None
    if compound == 'W19':
        final_mask = get_W19_mask(image_path)
        best_contour = get_best_contour(final_mask)
        (x,y), r = cv2.minEnclosingCircle(best_contour)
        vX,vY,vR = [int(i) for i in [x,y,r]]
    elif compound == 'Co4':

        final_mask = get_Co4_mask(image_path)
        best_contour = get_best_contour(final_mask)
        (x,y), r = cv2.minEnclosingCircle(best_contour)
        vX,vY,vR = [int(i) for i in [x,y,r]]
        vR = vR-55
    elif compound == 'CuSO4':
        final_mask = get_CuSO4_mask(image_path)
        best_contour = get_best_contour(final_mask)
        (x,y), r = cv2.minEnclosingCircle(best_contour)
        vX,vY,vR = [int(i) for i in [x,y,r]]
        vR = vR-30

    if best_contour is not None:
        return vX, vY, vR
    else:
        return 0, 0, 0


def get_CuSO4_mask(image_path):
    image = cv2.imread(image_path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0,0,100]), np.array([255,255,255]))
    return mask

def get_W19_mask(image_path):
    image = cv2.imread(image_path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0,50,0]), np.array([40,255,255]))
    return mask


def get_Co4_mask(image_path):
    image = cv2.imread(image_path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0,0,100]), np.array([255,255,255]))

    return mask

def show_rim(image_path, detections=None):
    if not detections:
        detections = find_rim(image_path)

    vX, vY, vR = detections
    image = cv2.imread(image_path)
    cv2.circle(image, (vX, vY), vR, (255,0,0), 8, cv2.LINE_AA)

def get_best_contour(mask):
    contours = get_contours(mask)
    best_idx = 0
    best_circularity = 0
    if len(contours) > 0:
        for idx, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area > 20000 :
                bg = np.zeros(mask.shape, np.uint8)
                cv2.drawContours(bg, [contour], 0, 255, -1)
                circularity = get_circularity(contour)
                if circularity > best_circularity:
                    best_circularity = circularity
                    best_idx = idx
        return contours[best_idx]
    else:
        return None


def apply_mask(image_path, mask):
    img = cv2.imread(image_path)
    image = cv2.bitwise_and(img, img, mask=mask)
    return image


def show_contours(image_path):
    mask = get_mask(image_path)

    masks, contours, heirarchy = cv2.findContours(mask, 1, 2)

    for contour in contours:
        if cv2.contourArea(contour) > 1000:
            print(cv2.contourArea(contour))
            bg = np.zeros(mask.shape, np.uint8)
            cv2.drawContours(bg, [approx], 0, 255, -1)

def get_contours(image):
    _, contours, hierarchy = cv2.findContours(image, 1, 2)
    return contours

def get_best_contour(mask):
    contours = get_contours(mask)
    best_idx = 0
    best_area = 999999
    for idx, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > 20000 :
            if area < best_area:
                best_area = area
                best_idx = idx

    return contours[best_idx]

def test_rxn(rxn_path):
    paths = [os.path.join(rxn_path, n) for n in os.listdir(rxn_path)]
    for idx, p in enumerate(paths):
        if idx % 5 == 0:
            find_rim(p, 'CuSO4')

if __name__ == '__main__':
    path = 'U:\\Chemobot\\crystalbot_imgs\\CuSO4\\180906a\\reaction_37\\Images'
    test_rxn(path)
