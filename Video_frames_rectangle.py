import cv2
import os

def convertToGray(img): # function that convert image to grayscale.
	return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def reduce(image):
	return cv2.resize(image, (0,0), fx=0.1, fy=0.1)

def extractFrames(pathIn, pathOut):
	os.mkdir(pathOut)
 
	cap = cv2.VideoCapture(pathIn)
	count = 0
	i = 0
	bias = 50
	while (cap.isOpened() and i<10): # o contador i serve para tirar apenas os 10 primeiros frames do video

		# Capture frame-by-frame
		ret, frame = cap.read()

		if ret == True:
			print('Read %d frame: ' % count, ret)

			frame = convertToGray(frame) # convert image to gray
			#frame = reduce(frame) # reduce image
			haar_face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
			faces = haar_face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5)
			print(faces)
			for (x,y,w,h) in faces:
            			print(y)
            			# cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
			
			crop_img = frame[y-bias: y+h+bias, x-bias: x+w+bias]
			#cv2.imwrite("face.png",crop_img)
 
			cv2.imwrite(os.path.join(pathOut, "frame{:d}.jpg".format(count)), crop_img)  # save frame as JPEG file

			count += 1
			i += 1
		else:
			break
 
	# When everything done, release the capture
	cap.release()
	cv2.destroyAllWindows()

def main():
	extractFrames('Em1_Fala1_CarolinaHolly.mp4', 'teste2')
 
if __name__=="__main__":
	main()
