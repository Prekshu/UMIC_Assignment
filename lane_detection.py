import matplotlib.pylab as plt
import cv2
import numpy as np

def ROI_mask(img,vertices):
    mask = np.zeros_like(img)
    mask_colour = 255
    cv2.fillPoly(mask,vertices,mask_colour)
    mask_img = cv2.bitwise_and(img,mask)
    return mask_img


def create_coordinates(img,line_parameters,y):
    m,c = line_parameters
    y1 = img.shape[0]
    y2 = int(y)
    x1 = int((y1-c)/m)
    x2 = int((y2-c)/m)
    return np.array([x1,y1,x2,y2])


def optimise_lines(img,lines,lane):
    left = []
    middle = []
    right = []
    y = 168
    for line in lines:
        x1,y1,x2,y2 = line.reshape(4)
        para = np.polyfit((x1,x2),(y1,y2),1)
        m = para[0]
        c = para[1]
        if lane=='right':          
            if abs(m)<0.1:
                y = c
                continue
            elif m>0:
                right.append((m,c))
            elif m<-1:
                middle.append((m,c))
            else:
                left.append((m,c))
        else:
            if abs(m)<0.4 and abs(m)>0.15:
                continue
            elif abs(m)<0.15:
                y=c
            elif m<0:
                left.append((m,c))
            elif m>1:
                middle.append((m,c))
            else:
                right.append((m,c))
    left_fit = np.average(left,axis=0)
    middle_fit = np.average(middle,axis=0)
    right_fit = np.average(right,axis=0)
    left_line = create_coordinates(img,left_fit,y)
    middle_line = create_coordinates(img,middle_fit,y)
    right_line = create_coordinates(img,right_fit,y)
    return np.array([left_line,middle_line,right_line])


def draw_lines(img,lines):
    copy_img = np.copy(img)
    blank_img = np.zeros((copy_img.shape[0],copy_img.shape[1],3),dtype=np.uint8)
    if lines is not None:
        i=1
        for line in lines:
            x1,y1,x2,y2 = line.reshape(4)
            if i==1:
                cv2.line(blank_img,(x1,y1),(x2,y2),(255,0,0),thickness=3)
            elif i==2:
                cv2.line(blank_img,(x1,y1),(x2,y2),(0,0,255),thickness=3)
            else:
                cv2.line(blank_img,(x1,y1),(x2,y2),(0,255,0),thickness=3)
            i = i+1

    img = cv2.addWeighted(img,0.8,blank_img,1,1)
    return img

path = input('Enter the path of the image ')
img = cv2.imread(path)
height = img.shape[0]
width = img.shape[1]

ROI_vertices_right = [
    (0,height-10),
    (width/2-30,height/2),
    (width,height)
]
ROI_vertices_left = [
    (50,height-10),
    (width/2+10 ,height/2+15),
    (width,height-10)
]
gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
canny_img = cv2.Canny(gray_img,50,100)
crop_img = ROI_mask(canny_img,np.array([ROI_vertices_right],np.int32),)
lines = cv2.HoughLinesP(crop_img,rho=6,theta=np.pi/180 ,threshold=100,lines=np.array([]),minLineLength=40,maxLineGap=100)

for line in lines:
    x1,y1,x2,y2 = line.reshape(4)
    if x1<20:
        lane = 'right'
        break
    else:
        lane = 'left'

if lane=='left':
    crop_img = ROI_mask(canny_img,np.array([ROI_vertices_left],np.int32),)
    lines = cv2.HoughLinesP(crop_img,rho=6,theta=np.pi/180 ,threshold=100,lines=np.array([]),minLineLength=40,maxLineGap=100)

optimised_line = optimise_lines(img,lines,lane)

print("The car is in " +lane+" lane")
img_line = draw_lines(img,optimised_line)
plt.imshow(img_line)
plt.show()