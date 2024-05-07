
import cv2
import numpy as np


def laplacian_edge_detection(image, threshold):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    laplacian_x = np.array([[0, -1, 0],
                         [-1, 4, -1],
                         [0, -1, 0]])

    laplacian_y = np.array([[0, -1, 0],
                         [-1, 4, -1],
                         [0, -1, 0]])

    grad_x = cv2.filter2D(gray_image, -1, laplacian_x)
    grad_y = cv2.filter2D(gray_image, -1, laplacian_y)

    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    thresholded_edges = np.where(grad_mag > threshold, grad_mag, 0)
    thresholded_edges *= 255.0 / thresholded_edges.max()
    return thresholded_edges.astype(np.uint8)
#code for implementing RANSAC
def slop(x1, y1, x2, y2):
    if (x2 != x1):
        m = (y2 - y1) / (x2 - x1)
        return m
    return None
def find_parameters(x1,y1,x2,y2):
    m=slop(x1,y1,x2,y2)
    if m is not None:
        a=1
    
        b=-1 / m
        c=y1/m-x1
        return a,b,c
    return None

def no_int(c1,c2,cordinates):
    x1,y1=c1
    x2,y2=c2
    t=find_parameters(x1,y1,x2,y2)
    if t is not None:
        a,b,c=t
        n=0
        for a1,a2 in cordinates:

            d=np.abs((a * a1 + b * a2 + c) / np.sqrt(a**2 + b**2))
            
            if(d<15):
                n=n+1
        return n
    return None
def ransac(points,image):
    X1=0
    Y1=0
    X2=0
    Y2=0

    max_ints=0
    for x1,y1 in points:
        for x2,y2 in points:
            if x1==x1 and y1==y2:
                continue
            ints=no_int((x1,y1),(x2,y2),points)
            if not ints:
                ints=0
            if ints>max_ints:
                max_ints=ints
                X1=x1
                Y1=y1
                X2=x2
                Y2=y2
    cv2.destroyAllWindows()
    return(X1,Y1,X2,Y2)




# Read image of table
image = cv2.imread('table.png')

threshold = 6 
blurred_image = cv2.GaussianBlur(image, (11, 11), 0) #the image was blurred to reduce noise
edges = laplacian_edge_detection(blurred_image, threshold)#laplacian edge detector was emplyed with appropriate threshhlod
_, binary_image = cv2.threshold(edges, 1, 255, cv2.THRESH_BINARY)
h,w=binary_image.shape

points=[]#to store the edge points
for i in range(w):
    for j in range(h):
        if binary_image[j][i]==255:
            points.append((i,j))
            image[j][i]=(255,0,0)
x1,y1,x2,y2=ransac(points,image)
m=slop(x1,y1,x2,y2)
b = y1 - m * x1
x1_extended = 0
y1_extended = int(m * x1_extended + b)
x2_extended = image.shape[1] - 1
y2_extended = int(m * x2_extended + b)
img=image.copy()
cv2.line(img, (x1_extended, y1_extended), (x2_extended, y2_extended), (0, 0, 255), 3)



lines = cv2.HoughLines(edges, 1, np.pi/180, 32)

# Draw detected lines on the original image houghs line 
if lines is not None:
    for rho, theta in lines[:, 0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

m=slop(x1,y1,x2,y2)
b = y1 - m * x1

x1_extended = 0
y1_extended = int(m * x1_extended + b)
x2_extended = image.shape[1] - 1
y2_extended = int(m * x2_extended + b)

# Draw the extended line representing the edge of table
cv2.line(image, (x1_extended, y1_extended), (x2_extended, y2_extended), (255, 0, 0), 3)
# Display the results using OpenCV
cv2.imshow('Edge with Hough line', image)
cv2.imshow('Edge with RANSAC', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
