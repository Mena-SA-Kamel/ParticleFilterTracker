import cv2
import matplotlib.image as mpimg
import numpy as np

drawing = False # true if mouse is pressed
mode = True # if True, draw rectangle. Press 'm' to toggle to curve
ix,iy = -1,-1
points = []
coords = [0,0,0,0]
img = mpimg.imread('Tracking Dataset/rgb/23.png')

# mouse callback function
def draw_circle(event,x,y,flags,param):
    global drawing,points,coords,img

    if event == cv2.EVENT_LBUTTONDBLCLK:
        drawing = True

    elif event == cv2.EVENT_LBUTTONDOWN:
        if drawing == True:
            cv2.circle(img,(x,y),2,(0,0,255),-1)
            points.append([x, y])

    elif event == cv2.EVENT_RBUTTONDBLCLK:
        points = np.array(points)
        cv2.fillPoly(img, [points], (0, 255, 0))
        x_list = []
        y_list = []
        for point in points:
            x_list.append(point[0])
            y_list.append(point[1])
        x0 = np.min(x_list)
        x1 = np.max(x_list)
        y0 = np.min(y_list)
        y1 = np.max(y_list)
        x = int((x0 + x1) /2)
        y = int((y0 + y1) /2)
        hx = x1 - x0
        hy = y1 - y0
        coords = [x,y,hx,hy]
        print (coords)
        # import code;
        # code.interact(local=dict(globals(), **locals()))
        # cv2.ellipse(img, (int((x0 + x1)/2), int((y0+y1)/2)), (int((x1-x0)/2), int((y1-y0)/2)), 0, 0, 360, (0,0,255), 1)
        cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 255), -1)

def draw_region(image_path):
    global img
    img = mpimg.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', draw_circle)
    while(1):
        cv2.imshow('image',img)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            return coords
            break