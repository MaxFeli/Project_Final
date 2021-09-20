import cv2

images = []


def getImages(vid):
    # series of function to detect: chessboard analysis
    i = 8
    cam = cv2.VideoCapture(vid)
    if not (cam.isOpened()):
        print("Could not open video device")
    else:
        cv2.namedWindow("Empty Chessboard")
        while True:
            ret, frame = cam.read()
            if not ret:
                print("failed to grab frame")
                break
            cv2.imshow("Empty Chessboard", frame)
            k = cv2.waitKey(1)
            if k % 256 == 27:
                # ESC pressed
                print("Escape hit, closing...")
                break
            elif k % 256 == 32:
                # SPACE pressed
                imgName = "img_cali/chessboard_frame_cali_{}.png".format(i)
                cv2.imwrite(imgName, frame)
                images.append(imgName)
                print("Printed nr {}.".format(i))
                i += 1
    cam.release()
    cv2.destroyAllWindows()
    return images
