import cairo
import calibImg as CI

import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import MeanShift
from io import BytesIO
from cairosvg import svg2png
import chess
import chess.svg
import HSV
import torch


points = []
cells = {}
mouseX, mouseY = 0, 0


def getFirstImage():
    cam = cv2.VideoCapture(0)
    if not (cam.isOpened()):
        print("Could not open video device")
    else :
        cv2.namedWindow("Empty Chessboard")
        while True:
            ret, frame = cam.read()
            if not ret:
                print("failed to grab frame")
                break
            cv2.imshow("Empty Chessboard", frame)
            k = cv2.waitKey(1)
            if k % 256 == 27:  # ESC pressed
                print("Escape hit, closing...")
                break
            elif k % 256 == 32:  # SPACE pressed
                imgName = "chessboard_frame_-1.png"
                cv2.imwrite(imgName, frame)
                break
    cam.release()
    cv2.destroyAllWindows()


def areaControl(x, y, mat):
    TRESH = 3
    for i in range(x - TRESH if x - TRESH > 0 else 0, x + TRESH if x + TRESH < len(mat) - 1 else len(mat) - 1, 1):
        for j in range(y - TRESH if y - TRESH > 0 else 0, y + TRESH if y + TRESH < len(mat[i]) - 1 else len(mat[i]) - 1,
                       1):
            if mat[i][j] == 0:
                return False
    return True


def harrisDetection(img, mask):
    width, height, channels = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    # result is dilated for marking the corners, not important
    dst = cv2.dilate(dst, None)
    # Threshold for an optimal value, it may vary depending on the image.
    for i in range(width):
        for j in range(height):
            if (areaControl(i, j, mask)) and (dst[i][j] > pow(10, -3) * dst.max()):
                img[i][j] = [0, 0, 255]
    cv2.imshow('Possible Corners Detected', img)
    # passing image by writing
    cv2.imwrite("corners_detected.png", img)
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()


def clusterPoints(points):
    ms = MeanShift(bandwidth=10, cluster_all=False)
    ms.fit(points)
    cluster_centers = ms.cluster_centers_
    return cluster_centers


def debugPoints(points, i):
    count = 0
    img = np.zeros([480, 640, 3], dtype=np.uint8)
    if not isinstance(points, dict):
        for x, y in points:
            cv2.putText(img, str(count), (int(x), int(y)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                        color=(255, 255, 255))
            count += 1
    else:
        for item in range(len(points)):
            cv2.putText(img, str(count), (int(points[item][0]), int(points[item][1])),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                        color=(255, 255, 255))
            count += 1
    cv2.imshow("test_position_of_points_{}".format(i), img)
    cv2.waitKey()
    if i == 2:
        cv2.destroyAllWindows()


# function to display the coordinates of the points clicked on the image
def click_event(event, x, y, flags, params):
    # double click saves value
    global mouseX, mouseY
    if event == cv2.EVENT_LBUTTONDBLCLK:
        mouseX, mouseY = x, y
        print("Got the point: {}, {}".format(x, y))


def validPoints(file):
    # Load image, ensure not palettised, and make into Numpy array
    pim = Image.open(str(file)).convert('RGB')
    img = np.array(pim)
    # Define the red colour we want to find - PIL uses RGB ordering
    red = [255, 0, 0]
    # Get X and Y coordinates of all red pixels
    X, Y = np.where(np.all(img == red, axis=2))
    zipped = np.column_stack((Y, X))
    # cluster all points to just 81 useful points
    points = clusterPoints(zipped)
    cv2.imshow("Points on masked image", printPoints(points, img))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """
    val = input("All points? y or n: ")
    if val == 'n' :
        cv2.namedWindow('image')
        cv2.setMouseCallback('image', click_event)
        while 1 :
            cv2.imshow('image', img)
            k = cv2.waitKey(20) & 0xFF
            if k == 27 :
                break
            elif k == ord('a') :
                # pressing a stores in points the values
                points = np.append(points, [[mouseX, mouseY]], axis=0)
                cv2.destroyAllWindows()
                break
    """
    debugPoints(points, 0)
    pointsSorted, pointsSorted2 = {}, {}
    pointsEdit = points
    # here inserted cycle; order by y the order by x
    flag = [False, 0]
    y_ord = np.sort(pointsEdit[:, 1])
    for item in range(len(y_ord)):
        pointsSorted[item] = pointsEdit[pointsEdit[:, 1] == y_ord[item]]
    for item in range(len(pointsSorted)):
        if flag[0] and flag[1]!=0:
            flag[1]-=1
            continue
        if pointsSorted[item].shape[0]!=1:
            flag[1] = pointsSorted[item].shape[0]-1
            for elem in range(pointsSorted[item].shape[0]):
                pointsSorted[item+elem] = pointsSorted[item+elem][elem]
            flag[0] = not flag[0]
        else:
            pointsSorted[item] = pointsSorted[item][0]
    debugPoints(pointsSorted, 1)
    minVal, index, avoid = 1000, 0, []
    for i in range(9):
        for j in range(9):
            if j==0:
                for k in range(9):
                   if pointsSorted[i*9+k][0] < minVal:
                        index = i*9+k
                        minVal = pointsSorted[index][0]
                pointsSorted2[9*i+j] = pointsSorted[index]
                avoid.append(index)
            else:
                dist = img.shape[0]**2 + img.shape[1]**2
                for k in range(9):
                    my_dist = (pointsSorted[9*i+k][0]-pointsSorted2[9*i+j-1][0])**2+(pointsSorted[9*i+k][1]-pointsSorted2[9*i+j-1][1])**2
                    if (my_dist < dist) and (avoid.count(9*i+k)==0):
                        dist = my_dist
                        index = 9*i+k
                pointsSorted2[9*i+j] = pointsSorted[index]
                avoid.append(index)
    debugPoints(pointsSorted2, 2)
    return pointsSorted2


def printPoints(points, img):
    if not isinstance(points, dict):
        for x, y in points:
            img = cv2.circle(img, (int(x), int(y)), radius=1, color=(0, 0, 255), thickness=-1)
    else:
        for item in range(len(points)):
            img = cv2.circle(img, (int(points[item][0]), int(points[item][1])), radius=1, color=(0, 0, 255),
                             thickness=-1)
    return img


def debugCell(cells, img):
    count = 0
    for item in cells:
        start_point = (int(cells[item][0][0]), int(cells[item][0][1]))
        end_point = (int(cells[item][1][0]), int(cells[item][1][1]))
        xmean, ymean = int((cells[item][0][0] + cells[item][1][0]) / 2), int(
            (cells[item][0][1] + cells[item][1][1]) / 2)
        cv2.rectangle(img, start_point, end_point, color=(0, 255, 0), thickness=5)
        cv2.putText(img, str(count), (xmean, ymean), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 0))
        count += 1
    cv2.imshow("Cells detected", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return img


# need to be corrected
def cell(points, img):
    cells, cells2 = {}, {}
    # define the 64 cells
    skip = 0
    for row in range(9) :
        for column in range(9):
            if column % 9 == 8:
                skip += 1
                continue
            if row == 8:
                break
            n_cell = column + row * 9 - skip
            point1 = column + row * 9
            point2 = (column + 1) + (row + 1) * 9
            xmin, xmax = min(points[point1][0], points[point2-1][0]), max(points[point1+1][0], points[point2][0])
            ymin, ymax = min(points[point1][1], points[point1+1][1]), max(points[point2][1], points[point2-1][1])
            cells[n_cell] = ((xmin, ymin), (xmax, ymax), [column, row])
    # correct, in the end there are 64 cells
    debugCell(cells, img)
    # reorder cells
    for i in range(8):
        for j in range(8):
            cells2[8*i+j] = cells[8*(7-i)+j]
    return cells


# YOLO net functions
def net_init():
    model = torch.hub.load('yolov5_master', 'custom', path='yolov5_master/best.pt', source='local')
    model.conf = 0.75
    model.classes = None
    model.names = ['bishop', 'white-pawn', 'white-rook', 'white-knight', 'white-bishop',
                   'white-queen', 'white-king', 'black-pawn', 'black-rook',
                   'black-knight', 'black-bishop', 'black-queen', 'black-king']
    model.to('cpu')
    return model


def plot_boxes(results, frame):
    """
    plots boxes and labels on frame.
    :param results: inferences made by model
    :param frame: frame on which to make the plots
    :return: new frame with boxes and labels plotted.
    """
    point_in_cell = {}
    label = {}
    labels, cord = results.xyxyn[0][:, -1].to('cpu').numpy(), results.xyxyn[0][:, :-1].to('cpu').numpy()
    n = len(labels)
    x_shape, y_shape = frame.shape[1], frame.shape[0]
    for i in range(n):
        row = cord[i]
        x1, y1, x2, y2 = int(row[0] * x_shape), int(row[1] * y_shape), int(row[2] * x_shape), int(row[3] * y_shape)
        point_in_cell[i] = [x2, y2]
        bgr = (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 1)
        label[i] = results.names[int(labels[i])]
        msg = label[i] + ' ' + f"{int(row[4] * 100)}"
        cv2.putText(frame, msg, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    cv2.putText(frame, f"Total Targets: {n}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
    return frame, point_in_cell, label


# Chess functions

def res_name(label):
    # function based on result.names, manually resolved
    if label == 'white-pawn': return 'P'
    if label == 'white-rook': return 'R'
    if label == 'white-knight': return 'N'
    if label == 'white-bishop': return 'B'
    if label == 'white-queen': return 'Q'
    if label == 'white-king': return 'K'
    if label == 'black-pawn': return 'p'
    if label == 'black-rook': return 'r'
    if label == 'black-knight': return 'n'
    if label == 'black-bishop': return 'b'
    if label == 'black-queen': return 'q'
    if label == 'black-king': return 'k'


def res_chess(cells, cellpoints, labels):
    board = chess.Board(None)
    for i in range(len(cellpoints)):
        label = res_name(labels[i])
        for j in range(len(cells)):
            # cellpoint needs fixing by adding quantization error bias
            x_bias = 0.5*abs(cells[j][0][0]-cells[j][1][0])
            y_bias = 0.5*abs(cells[j][0][1]-cells[j][1][1])
            cond = (cells[j][0][0] <= cellpoints[i][0]+x_bias <= cells[j][1][0]) and (
                    cells[j][0][1] <= cellpoints[i][1]+y_bias <= cells[j][1][1])
            if cond:
                my_square = chess.SQUARES[j]
                my_piece = chess.Piece.from_symbol(label)
                board.set_piece_at(my_square, my_piece)
                break
    return board


# main

if __name__ == "__main__":
    # setting up camera
    # mtx, dist = CI.calibImg('img_cali/*.png')

    # chessboard analysis
    getFirstImage()
    # calibration --> IT'S NOT WORKING FOR NOW, NOT USED
    # dest = CI.undistortImg(mtx, dist, cv2.imread("chessboard_frame_-1.png"))

    # finding chessboard, squares
    und_img, mask, result, refx, refy = HSV.contourAnalysis(cv2.imread("chessboard_frame_-1.png"))

    # find all intersection
    harrisDetection(result, mask)
    points = validPoints("corners_detected.png")
    img2 = printPoints(points, und_img)
    cv2.imshow("Points", img2)
    cv2.imwrite("square_corners.png", img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cells = cell(points, und_img)
    # at this point we have our cell definition, we need to use the YOLO to detect
    # dynamic analysis
    model = net_init()
    state = [False, False]
    cap = cv2.VideoCapture(0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    index = 0
    print("Press 'Space' for detection, 'r' for recording, 'esc' to quit.")
    if not (cap.isOpened()):
        print("Could not open video device")
    else:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            k = cv2.waitKey(1)
            if k % 256 == 32:  # 'Space' pressed
                state[0] = not state[0]
            if k % 256 == 114:  # 'r' pressed
                state[1] = not state[1]
                if state[1]:
                    writer = cv2.VideoWriter('runs/record_{}.mp4v'.format(index), cv2.VideoWriter_fourcc(*'DIVX'), 20, (width, height))
                else:
                    writer.release()
                    index += 1
            if state[0]:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                cv2.namedWindow("Detection")
                results = model(frame)
                frame, cellpoints, labels = plot_boxes(results, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                board = res_chess(cells, cellpoints, labels)
                boardsvg = chess.svg.board(board, size=400)
                png = svg2png(bytestring=boardsvg)
                pil_img = Image.open(BytesIO(png)).convert('RGBA')
                cv2_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGBA2BGRA)
                cv2.imshow("Resolution", cv2_img)
            if state[1]:
                writer.write(frame)
            cv2.imshow("Detection", frame)
            if k % 256 == 27:  # Esc pressed
                break
    cap.release()
    writer.release()
    cv2.destroyAllWindows()
