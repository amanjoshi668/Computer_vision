import cv2 as cv
import grabcut
import numpy as np
from copy import deepcopy

BLUE = [255, 0, 0]        # rectangle color
RED = [0, 0, 255]         # PR BG
GREEN = [0, 255, 0]       # PR FG
BLACK = [0, 0, 0]         # sure BG
WHITE = [255, 255, 255]   # sure FG

rect = (0, 0, 1, 1)
drawing = False         # flag for drawing curves
rectangle = False       # flag for drawing rect
rect_over = False       # flag to check if rect drawn
thickness = 3           # brush thickness
skip_learn_GMMs = False # whether to skip learning GMM parameters

def onmouse(event, x, y, flags, param):
    global img, drawing, value, mask, rectangle, rect, rect_or_mask, ix, iy, rect_over, skip_learn_GMMs

    # Draw Rectangle
    if event == cv.EVENT_RBUTTONDOWN:
        rectangle = True
        ix, iy = x, y

    elif event == cv.EVENT_MOUSEMOVE:
        if rectangle == True:
            cv.rectangle(img, (ix, iy), (x, y), BLUE, 2)
            rect = (min(ix, x), min(iy, y), abs(ix-x), abs(iy-y))

    elif event == cv.EVENT_RBUTTONUP:
        rectangle = False
        rect_over = True
        cv.rectangle(img, (ix, iy), (x, y), BLUE, 2)
        rect = (min(ix, x), min(iy, y), abs(ix-x), abs(iy-y))
        print(" Now press the key 'n' a few times until no further change \n")


if __name__ == "__main__":


    while(True):
        print("Enter image Path")
        cv.namedWindow('output')
        cv.namedWindow('input')
        image_path = input("Enter Image Path: ")
        img = cv.imread(image_path)
        cv.moveWindow('input', img.shape[1]+10, 90)
        image_path = None
        img2 = deepcopy(img)
        cv.imshow('input', img)
        cv.setMouseCallback('input', onmouse)
        k = cv.waitKey()
        if k == ord('n'):
            num_iterations = input("Enter number_of_iterations: ")
            gg = grabcut.GrabCut(img, np.zeros(img.shape[:2], dtype=np.uint8), rect)
            gg.run()
            output = gg.modified_image()
            cv.imshow('output', output)
            k = cv.waitKey()
        if k==27:
            break
        cv.destroyAllWindows()