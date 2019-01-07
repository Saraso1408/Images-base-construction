import dlib
import cv2
import os

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def convertToGray(img): # function that convert image to grayscale.
	return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def get_landmarks(imag):
	landmarks = []
	detections = detector(imag, 1)
	for k,d in enumerate(detections): #For all detected face instances individually
		shape = predictor(imag, d) #Draw Facial Landmarks with the predictor class
		xlist = []
		ylist = []
		for i in range(1,68): #Store X and Y coordinates in two lists
			xlist.append(float(shape.part(i).x))
			ylist.append(float(shape.part(i).y))

		for x, y in zip(xlist, ylist): #Store all landmarks in one list in the format x1,y1,x2,y2,etc.
			landmarks.append(x)
			landmarks.append(y)
	if len(detections) > 0:
		return landmarks
	else: #If no faces are detected, return error message to other function to handle
		landmarks = "error"
		return landmarks

def extractFrames(pathIn, pathOut):
	os.mkdir(pathOut)

	cap = cv2.VideoCapture(pathIn)
	count = 0
	bias = 50
	while (cap.isOpened() and count<10): # o contador<10 serve para tirar apenas os 10 primeiros frames do video

		# Capture frame-by-frame
		ret, frame = cap.read()

		if ret == True:
			print('Read %d frame: ' % count, ret)

			frame = convertToGray(frame) # convert image to gray
			# uses the classifier HaarCascade to extract the faces.
			haar_face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
			faces = haar_face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5) # detecta quantas faces ha na imagem

			for (x,y,w,h) in faces: # Para cada
				crop_img = frame[y-bias: y+h+bias, x-bias: x+w+bias]

			landmark = get_landmarks(crop_img) # Aqui recebo lista com landmarks.
			# print(type(landmark)) # Podemos ver com isso que a variavel retornada pela funcao get_landmarks eh do tipo list.
			# print(len(landmark)) # Tamanho da lista

			clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
			clahe_image = clahe.apply(crop_img) # Aplicação do Clahe a imagem ja em cinza
			detections = detector(clahe_image, 1)#Detect the faces in the image
			for k,d in enumerate(detections): #For each detected face
            			shape = predictor(clahe_image, d) #Get coordinates
            			for i in range(1,68): #There are 68 landmark points on each face
                     			#For each point, draw a red circle with thickness2 on the original frame
                    			cv2.circle(crop_img, (shape.part(i).x, shape.part(i).y), 1, (0,0,255), thickness=2)

			cv2.imwrite(os.path.join(pathOut, "frame{:d}.jpg".format(count)), crop_img)  # save crop_img as JPEG file

			count += 1
		else:
			break

	# When everything done, release the capture
	cap.release()
	cv2.destroyAllWindows()

def main():
	extractFrames('Em1_Fala1_CarolinaHolly.mp4', 'teste1_landmarks')

if __name__=="__main__":
	main()
