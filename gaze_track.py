import cv2
import numpy as np
import dlib
import math
import os

# extracts the coordinates of the 68 predicted points 
def shape2np(shape, dtype='int'):
    coords = np.zeros((68,2), dtype=dtype)
    for i in range(68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords
# pass function for Trackbar creation
def nothing(x):
    pass

def eye_on_mask(mask, side):
    #create list of points around eyes
    points = [shape[i] for i in side]
    #create np array with the points
    points = np.array(points, dtype=np.int32)
    #take black mask and white out the eyes
    mask = cv2.fillConvexPoly(mask, points, 255)
    #cv2.imshow('mask',mask)
    return mask

def contouring(thresh, mid, img, right=False):
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    try:
        cnt = max(contours, key = cv2.contourArea)
        M = cv2.moments(cnt)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        if right:
            cx += mid
        #* DRAW CG OF IRIS/PUPLE
        cv2.circle(img, (cx, cy), 2, (0, 0, 255), 2)
        #* DRAW EYE CONTOURS
        #if right:
        #    cv2.drawContours(img, contours, -1, (0,255,0), 2, offset = (mid, 0))
        #else:
        #    cv2.drawContours(img, contours, -1, (0,255,0), 2)
        
        return (cx, cy)
    except:
        #print('passed')
        pass

def draw_eye(img, shape, iris, right=False):

    if right:
        pt_left  = shape[42]
        pt_right = shape[45]
    else:
        pt_left  = shape[36]
        pt_right = shape[39]

    # delta of eye endpoints
    dx = pt_left[0] - pt_right[0]
    dy = pt_left[1] - pt_right[1]

    # calculate centerpoint of eye 
    #! OFFSET ADDED TO Y COORDINATE OF SPHERE
    #! TO ADD CALIBRATION TO SET (0,0) COORD PROPERLY
    xy_sphere = ((pt_left[0] + pt_right[0]) / 2 , (pt_left[1] + pt_right[1]) / 2 - 6)
    xy_sphere_int = (int((pt_left[0] + pt_right[0]) / 2), int((pt_left[1] + pt_right[1]) / 2))
    
    # calculate pixel diameter of eye
    R = int((math.sqrt(pow(dx, 2) + pow(dy, 2))) / 2)

    # draw full eye on img
    #cv2.circle(img, xy_sphere_int, R, (255,0,0), 2)

    # calculate sphere center coodinates
    c_sphere = (int(xy_sphere[0]), int(xy_sphere[1]))

    # offset to use center of sphere as local system
    c_sphere_offset = (0, 0, -R)

    # offset iris values to local coordinate system
    iris_x_offset = iris[0] - xy_sphere[0]
    iris_y_offset = xy_sphere[1] - iris[1]

    # project iris point onto sphere surface
    z_iris = (math.sqrt(pow(R,2) - pow(iris_x_offset,2) - pow(iris_y_offset,2)) - R)

    pt_iris  = (iris_x_offset, iris_y_offset, z_iris)

    # ---------------------------- Equation of 3D line --------------------------- #
    # calculate t vector using two points 
    t_vector=[(pt_iris[0] - c_sphere_offset[0]), (pt_iris[1] - c_sphere_offset[1]), (pt_iris[2] - c_sphere_offset[2])]

    # z(t) =  <0, 0, -R> + t<t_vector>

    # x(t) =  0 + t_vector[0] * t
    # y(t) =  0 + t_vector[1] * t
    # z(t) = -R + t_vector[2] * t

    # offset added to R to extend line in img
    if abs(iris_x_offset) < 4 and abs(iris_y_offset) > 1:
        line_gain = 1.5
    else: line_gain = 1.0
    #print(iris_x_offset, iris_y_offset)
    #print("-----------")
     
    t = ( 75*line_gain + R) / t_vector[2]
    # ---------------------------------------------------------------------------- #

    # point on x,y plane can now be computed:
    pt_cross_offset = (int(t_vector[0]*t), int(t_vector[1]*t))
    pt_cross = (int(t_vector[0]*t + xy_sphere[0]), int(xy_sphere[1] - t_vector[1]*t ))

    cv2.line(img,iris, pt_cross, (255,0,0), 2)
    #cv2.line(img,c_sphere ,iris , (0,255,0), 2)
    return img

# ---------------------------------------------------------------------------- #
#                                 INITIAL SETUP                                #
# ---------------------------------------------------------------------------- #

createVideo = False
index = 0

# keypoint indices for left eye
left = [36, 37, 38, 39, 40, 41] 
# keypoint indices for right eye
right = [42, 43, 44, 45, 46, 47] 

# create kernel
kernel = np.ones((9, 9), np.uint8)

# activate facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

cap = cv2.VideoCapture('images/eye_track3.mp4')
ret, img = cap.read()
thresh = img.copy()
cv2.namedWindow('image')
cv2.createTrackbar('threshold', 'image', 60, 255, nothing)

# ---------------------------------------------------------------------------- #
#                                PROCESS IMAGES                                #
# ---------------------------------------------------------------------------- #

while(True):
    # create gray image
    #img = cv2.imread('images/face.jpg')
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except:
        break
    # this holds all faces detected by dlib
    faces = detector(gray, 1)
    # for each face predict and extract the coordinates of the 68 points
    for i, face in enumerate(faces):
        shape = predictor(gray, face)
        shape = shape2np(shape)
        #* DRAW POINTS OF FACE MAP ON ORIGINAL IMAGE
        #for j, (x,y) in enumerate(shape):
        #    if j in left or j in right:
        #        cv2.circle(img, (x,y), 2, (0,255,0), -1)

        # create black mask
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        # whiteout each eye of the face 
        mask = eye_on_mask(mask, left)
        mask = eye_on_mask(mask, right)

        # dilate the eyes mask to pad the image
        mask = cv2.dilate(mask, kernel, 5)

        # bitwise& the image and mask to isolate the eyes
        eyes = cv2.bitwise_and(img, img, mask=mask)

        # calculate midpoint of eyes
        mid = (shape[42][0] + shape[39][0]) // 2
        #print(mid)

        # set all 0,0,0 pixels to max to isolate eyeball
        mask = (eyes == [0, 0, 0]).all(axis=2)
        eyes[mask] = [255, 255, 255]
        # convert to gray to threshold
        eyes_gray = cv2.cvtColor(eyes, cv2.COLOR_BGR2GRAY)
        #cv2.imshow('gray', eyes_gray)

        threshold = cv2.getTrackbarPos('threshold', 'image')
        #print(threshold)
        ret, thresh = cv2.threshold(eyes_gray, threshold, 255, cv2.THRESH_BINARY)
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=4)
        thresh = cv2.medianBlur(thresh, 3)

        # need eyes to be white when contouring, so inverse
        thresh = cv2.bitwise_not(thresh)

        # split image at midpoint and contour the eye on each side, locate iris
        iris_l = contouring(thresh[:, 0:mid], mid, img, False)
        iris_r = contouring(thresh[:, mid:], mid, img, True)

        # draw the eyes
        img = draw_eye(img, shape, iris_l, False)
        img = draw_eye(img, shape, iris_r, True)
    
    # Generate images
    img_crop = img[0:720, 360:880]
    if createVideo == True:
        fname = 'Eye-Track-{0:0=4d}.png'.format(index)
        fpath = 'E:\GitHub\OpenCV_Testing\images\data'
        cv2.imwrite(os.path.join(fpath, fname), img_crop)        
        index += 1

    #print(img.shape)
    cv2.imshow('eyes', img[0:720, 360:880])
    cv2.imshow("image", thresh)
    ret, img = cap.read()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows