import cv2
import numpy as np


def HSV_Filtering(frame):
    # It converts the BGR color space of image to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Threshold of blue in HSV space
    lower_blue = np.array([100, 150, 0])
    upper_blue = np.array([140, 255, 255])
    # preparing the mask to overlay
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    # The black region in the mask has the value of 0,
    # so when multiplied with original image removes all non-blue regions
    result = cv2.bitwise_and(frame, frame, mask=mask)
    cv2.imshow('frame', frame)
    cv2.imshow('mask', mask)
    cv2.imshow('result', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return frame, mask, result


def findBiggestLabel(labels, num_labels):
    maxOcc, index = 0, 0
    for i in range(1, num_labels):
        tot = (labels == i).sum()
        if tot > maxOcc:
            maxOcc = tot
            index = i
    return index


def connected_component_label(img):
    # Converting those pixels with values 1-127 to 0 and others to 1
    img1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]
    # Applying cv2.connectedComponents()
    num_labels, labels = cv2.connectedComponents(img1)
    # Map component labels to hue val, 0-179 is the hue range in OpenCV
    label_hue = np.uint8(179 * labels / np.max(labels))
    blank_ch = 255 * np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
    # Converting cvt to BGR
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
    # set bg label to black
    labeled_img[label_hue == 0] = 0
    # Showing Original Image
    imgOrig = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    cv2.imshow('Original image: mask', imgOrig)
    # Showing Image after Component Labeling
    imgLabel = cv2.cvtColor(labeled_img, cv2.COLOR_BGR2RGB)
    cv2.imshow('Labeled image: mask', imgLabel)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # Find biggest region
    index = findBiggestLabel(labels, num_labels)
    # got the index, i have to crop inside that color --> done in a non smart way
    width, height = img.shape
    mask = np.zeros((width, height), np.uint8)
    for i in range(width):
        for j in range(height):
            if labels[i][j] == index:
                mask[i][j] = 255
            else:
                mask[i][j] = 0
    cv2.imshow("Biggest Label", mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # got the white mask, now crop inside and return image
    # get inside contour, crop, then harris detection on main function
    # get inside square
    # leftBound whiting
    for i in range(width):
        for j in range(height):
            if mask[i][j] == 255:
                break
            mask[i][j] = 255
    # rightBound whiting
    for i in range(width):
        for j in range(height - 1, 0, -1):
            if mask[i][j] == 255:
                break
            mask[i][j] = 255
    # reverse the matrix. you want to crop the internal area so you need to make it white. find biggest connected element again than crop
    mask = np.invert(mask)
    num_labels1, labels1 = cv2.connectedComponents(mask)
    index1 = findBiggestLabel(labels1, num_labels1)
    for i in range(width):
        for j in range(height):
            if labels1[i][j] != index1:
                mask[i][j] = 0
    cv2.imshow("Final mask: cropping area", mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return mask


def defRoi(img, finalMask):
    width, height, channels = img.shape
    X_values = []
    Y_values = []
    # checking upper left corner
    for i in range(width):
        for j in range(height):
            if not (img[i][j][0] == 0 and img[i][j][1] == 0 and img[i][j][2] == 0):
                X_values.append(i)
                Y_values.append(j)
                break
    # checking lower right corner
    for i in reversed(range(width)):
        for j in reversed(range(height)):
            if not (img[i][j][0] == 0 and img[i][j][1] == 0 and img[i][j][2] == 0):
                X_values.append(i)
                Y_values.append(j)
                break
    minX, maxX = min(X_values), max(X_values)
    minY, maxY = min(Y_values), max(Y_values)
    # decided not to crop, at first we did
    result = img  # [minX:maxX, minY:maxY]
    # finalMask = finalMask[minX:maxX, minY:maxY]
    """
    # whiting black pixels on the side
    width, height, channels = result.shape
    TRESH = 10
    cv2.imshow("result", result)
    cv2.imshow("finalMask", finalMask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    for i in range(width):
        for j in range(height):
            if (result[i][j][0] < TRESH and result[i][j][1] < TRESH and result[i][j][2] < TRESH) and (finalMask[i][j] == 0):
                result[i][j] = [255, 255, 255]
    for i in reversed(range(width)):
        for j in reversed(range(height)):
            if (result[i][j][0] < TRESH and result[i][j][1] < TRESH and result[i][j][2] < TRESH) and (finalMask[i][j] == 0):
                result[i][j] = [255, 255, 255]
    """
    # editable kernel size (1,1), (3,3), no more
    blurred_img = cv2.GaussianBlur(result, (1, 1), 0)
    # cv2.imshow("blur_img", blurred_img)
    mask = np.zeros(result.shape, np.uint8)
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("gray", gray)
    thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)[1]
    contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(mask, contours, -1, (255, 255, 255), 7)
    output = np.where(mask == np.array([255, 255, 255]), blurred_img, result)
    cv2.imshow("output", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return output, minX, minY, finalMask


def contourAnalysis(img):
    frame1, mask1, result1 = HSV_Filtering(img)
    mask = connected_component_label(mask1)
    result = cv2.bitwise_and(img, img, mask=mask)
    result, minX, minY, finalMask = defRoi(result, mask)
    cv2.imshow("Original Image", img)
    cv2.imshow("Result", result)
    cv2.waitKey()
    cv2.destroyAllWindows()
    return img, finalMask, result, minX, minY

# for i in range(-1, 13):
# contourAnalysis(cv2.imread("chessboard_frame_{}.png".format(-1)))
