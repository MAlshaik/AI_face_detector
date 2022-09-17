import cv2 

# Load some pre-trained data on face frontals from opencv
trained_face_data = cv2.CascadeClassifier('C:\Python310\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml')

# the photo where the programm will detect the face
img = cv2.imread('olimpia_stage.jpg')

#Converts image to grayscale (so that it ist a headache to make the computer detect the face)
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#detect face cordinates
face_cordinates = trained_face_data.detectMultiScale(grayscaled_img)

#draw rectangle around the face(s)
for (x,y,z,h) in face_cordinates:
    cv2.rectangle(img, (x,y), (x+z, y+h), (0, 255, 0), 2)

print(face_cordinates)

#opens the photo with the face in it
cv2.imshow('face detoctor', img)
#pauses the programm so that the image does not imediatley disappear
cv2.waitKey()

print("Code Completed")