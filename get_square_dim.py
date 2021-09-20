import cv2

points = {}


def click_event(event, x, y, flags, params):
    # double click saves value
    global mouseX, mouseY
    if event == cv2.EVENT_LBUTTONDBLCLK:
        mouseX, mouseY = x, y
        print("Got the point: {}, {}".format(x, y))


if __name__ == "__main__":
    cv2.namedWindow('image')
    img = cv2.imread("chessboard_frame_-1.png")
    index = 0
    while True:
        cv2.imshow('image', img)
        cv2.setMouseCallback('image', click_event)
        k = cv2.waitKey(20) & 0xFF
        if k == 27:
            break
        elif k == ord('a'):
            # pressing a stores in points the values
            points[index] = (mouseX, mouseY)
            cv2.destroyAllWindows()
            index += 1
        if index == 2:
            break
    square_side = ((points[0][0]-points[1][0])**2+(points[0][1]-points[1][1])**2)**0.5
    print("Square side is {} pixels".format(square_side))
