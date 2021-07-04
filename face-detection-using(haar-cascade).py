import cv2
import matplotlib.pyplot as plt
import time 
%matplotlib inline


def convertToRGB(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#load test iamge
test1 = cv2.imread('test1.jpg')

#convert the test image to gray image as opencv face detector expects gray images
gray_img = cv2.cvtColor(test1, cv2.COLOR_BGR2GRAY)
plt.imshow(gray_img, cmap='gray')
#load cascade classifier 
haar_face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
faces = haar_face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5);
print('Faces found: ', len(faces))
for (x, y, w, h) in faces:
    cv2.rectangle(test1, (x, y), (x+w, y+h), (0, 255, 0), 2)
#conver image to RGB and show image
plt.imshow(convertToRGB(test1))
def detect_faces(f_cascade, colored_img, scaleFactor = 1.1):
    
    img_copy = colored_img.copy()
    
    #convert the test image to gray image as opencv face detector expects gray images
    gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    
    faces = f_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=5);
    
    #go over list of faces and draw them as rectangles on original colored img
    for (x, y, w, h) in faces:
        cv2.rectangle(img_copy, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
    return img_copy

test2 = cv2.imread('test3.jpg')
#call our function to detect faces
faces_detected_img = detect_faces(haar_face_cascade, test2)
plt.imshow(convertToRGB(faces_detected_img))

test2 = cv2.imread('test4.jpg')
faces_detected_img = detect_faces(haar_face_cascade, test2, scaleFactor=1.2)
#conver image to RGB and show image
plt.imshow(convertToRGB(faces_detected_img))


#load cascade classifier training file
haar_face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
test1 = cv2.imread('test51.jpg')

#------------HAAR-----------accuurately detect faces--
#note time before detection
t1 = time.time()
haar_detected_img = detect_faces(haar_face_cascade, test1)

#note time after detection
t2 = time.time()
#calculate time difference
dt1 = t2 - t1
#print the time differene

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

#show Haar image
ax1.set_title('Haar Detection time: ' + str(round(dt1, 3)) + ' secs')
ax1.imshow(convertToRGB(haar_detected_img))

#------------HAAR-----------for testing accurately to identify no of faces--
#note time before detection
t1 = time.time()
#call our function to detect faces
haar_detected_img = detect_faces(haar_face_cascade, test2)
#note time after detection
t2 = time.time()
#calculate time difference
dt1 = t2 - t1
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

#show Haar image
ax1.set_title('Haar Detection time: ' + str(round(dt1, 3)) + ' secs')
ax1.imshow(convertToRGB(haar_detected_img))
