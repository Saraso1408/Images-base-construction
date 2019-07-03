#!/usr/bin/env python
# coding: utf-8

# In[41]:


import dlib
import cv2
import os
import pandas as pd
import numpy as np
import csv


# In[42]:


# Aqui eu começo a colocar o programa de exemplo do OPF.


# In[43]:


import gc
import numpy as np
import pylab as pl
from time import time
import libopf_py
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.decomposition import PCA
from sklearn import tree
from sklearn import datasets


# In[ ]:


def read_dataset():
    X, y = datasets.load_breast_cancer(return_X_y=True)
    benchmark(X, y, len(y))


# In[ ]:


def benchmark(data, target, n_samples):
    list_n_samples = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    opf_results = np.zeros((len(list_n_samples), 4))
    svm_results = np.zeros((len(list_n_samples), 4))
    bayes_results = np.zeros((len(list_n_samples), 4))
    linear_results = np.zeros((len(list_n_samples), 4))
    sgd_results = np.zeros((len(list_n_samples), 4))
    tree_results = np.zeros((len(list_n_samples), 4))

    for i, size in enumerate(list_n_samples):
        n_split = int(size * n_samples)
        rand = np.random.permutation(n_samples)
        random_data = data[rand]
        random_label = target[rand]
        data_train, data_test = random_data[:n_split], random_data[n_split:]
        label_train, label_test = random_label[:n_split], random_label[n_split:]

        def _opf():
            label_train_32 = label_train.astype(np.int32)
            label_test_32 = label_test.astype(np.int32)
            O = libopf_py.OPF()
            t = time()
            O.fit(data_train, label_train_32)

            opf_results[i, 3] = time() - t
            t = time()
            print("----------OPF------------")
            print(t)
            predicted = O.predict(data_test)
            opf_results[i, 0] = precision_score(label_test_32, predicted, average='binary')
            opf_results[i, 1] = recall_score(label_test_32, predicted, average='binary')
            opf_results[i, 2] = f1_score(label_test_32, predicted, average='binary')
            gc.collect()

        def _svm():
            clf = svm.SVC(C=1000, gamma='auto')
            t = time()
            print("-----------SVM-----------")
            print(t)
            clf.fit(data_train, label_train)
            svm_results[i, 3] = time() - t
            predicted = clf.predict(data_test)
            svm_results[i, 0] = precision_score(label_test, predicted, average='binary')
            svm_results[i, 1] = recall_score(label_test, predicted, average='binary')
            svm_results[i, 2] = f1_score(label_test, predicted, average='binary')
            gc.collect()

        def _bayes():
            clf = GaussianNB()
            t = time()
            print("-----------BAYES-----------")
            print(t)
            clf.fit(data_train, label_train)
            bayes_results[i, 3] = time() - t
            predicted = clf.predict(data_test)
            bayes_results[i, 0] = precision_score(label_test, predicted, average='binary')
            bayes_results[i, 1] = recall_score(label_test, predicted, average='binary')
            bayes_results[i, 2] = f1_score(label_test, predicted, average='binary')
            gc.collect()

        def _linear():
            clf = LogisticRegression(C=1, penalty='l2', solver='liblinear')
            t = time()
            print("-----------LINEAR-----------")
            print(t)
            clf.fit(data_train, label_train)
            linear_results[i, 3] = time() - t
            predicted = clf.predict(data_test)
            linear_results[i, 0] = precision_score(label_test, predicted, average='binary')
            linear_results[i, 1] = recall_score(label_test, predicted, average='binary')
            linear_results[i, 2] = f1_score(label_test, predicted, average='binary')
            gc.collect()

        def _sgd():
            clf = SGDClassifier(loss="hinge", penalty="l2", max_iter=1000, tol=1e-3)
            t = time()
            print("-----------SGD-----------")
            print(t)
            clf.fit(data_train, label_train)
            linear_results[i, 3] = time() - t
            predicted = clf.predict(data_test)
            sgd_results[i, 0] = precision_score(label_test, predicted, average='binary')
            sgd_results[i, 1] = recall_score(label_test, predicted, average='binary')
            sgd_results[i, 2] = f1_score(label_test, predicted, average='binary')
            gc.collect()

        def _tree():
            clf = tree.DecisionTreeClassifier()
            t = time()
            print("-----------TREE-----------")
            print(t)
            clf.fit(data_train, label_train)
            tree_results[i, 3] = time() - t
            predicted = clf.predict(data_test)
            tree_results[i, 0] = precision_score(label_test, predicted, average='binary')
            tree_results[i, 1] = recall_score(label_test, predicted, average='binary')
            tree_results[i, 2] = f1_score(label_test, predicted, average='binary')
            gc.collect()

        _opf()
        _svm()
        _bayes()
        _linear()
        _sgd()
        _tree()

    pl.figure()
    pl.plot(list_n_samples, opf_results[:, 2], label="OPF")
    pl.plot(list_n_samples, svm_results[:, 2], label="SVM RBF")
    pl.plot(list_n_samples, bayes_results[:, 2], label="Naive Bayes")
    pl.plot(list_n_samples, linear_results[:, 2], label="Logistic Regression")
    pl.plot(list_n_samples, sgd_results[:, 2], label="SGD")
    pl.plot(list_n_samples, tree_results[:, 2], label="Decision Trees")
    pl.legend(loc='lower right', prop=dict(size=8))
    pl.xlabel("Training set size")
    pl.ylabel("F1 score")
    # pl.title("Precision")
    pl.show()


# In[ ]:


read_dataset()


# In[ ]:





# In[ ]:


# Até aqui eu coloquei o programa de exemplo do OPF.


# In[ ]:


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


# In[ ]:


def convertToGray(img): # function that convert image to grayscale.
	return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# In[ ]:


def get_landmarks(imag):
	landmarks = []
	detections = detector(imag, 1)
    
	for k,d in enumerate(detections): #For all detected face instances individually
		shape = predictor(imag, d) #Draw Facial Landmarks with the predictor class
		xlist = []
		ylist = []
		for i in range(0,68): #Store X and Y coordinates in two lists
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


# In[ ]:


def extractFrames(pathIn, pathOut):
	df = pd.DataFrame()
	os.mkdir(pathOut)

	emotion = []
	land_array = []
	col = []
        
	j = 1
	k1 = 1
	k2 = 1
	for j in range(1, 137):
		if (j%2) == 0:
			k1 = j/2
			col.append("land_%d_y" % k1)
		else:
			k2 = (j/2) + 0.5
			col.append("land_%d_x" % k2)
		j += 1
#	print(col)
    
	cap = cv2.VideoCapture(pathIn)
	count = 0
	bias = 50
	while (cap.isOpened()): # o contador vai tirar apenas os frames multiplos de 10 (1 a cada 10 frames)
		# Capture frame-by-frame       
		ret, frame = cap.read()

		if ret == True and (count % 10) == 0:
			print('Read %d frame: ' % count, ret)

			frame = convertToGray(frame) # convert image to gray
			# uses the classifier HaarCascade to extract the faces.
			haar_face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
			faces = haar_face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5) # detecta quantas faces ha na imagem

			for (x,y,w,h) in faces: # Para cada face faz um corte
				crop_img = frame[y-bias: y+h+bias, x-bias: x+w+bias]

			landmark = get_landmarks(crop_img) # Aqui recebo lista com landmarks.
			#print(type(landmark)) # Podemos ver com isso que a variavel retornada pela funcao get_landmarks eh do tipo list. 
#			print(len(landmark)) # Tamanho da lista
			land_array = pd.DataFrame([landmark], index = ["frame_%d" % count], columns = col)
			df = df.append(land_array)
            
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
		elif (count % 10) != 0:
			count += 1
		else:
			break

	k = 0
	num_em = extract_label(pathIn)
	while (k < (count/10)):            
		emotion.append(num_em)
		k += 1
	df.insert(136, "label", emotion, True) 

	print(df)
	return df
	# When everything done, release the capture
	cap.release()
	#cv2.destroyAllWindows()


# In[ ]:


def extract_label(video_name): # Essa função extrai do nome do video o numero da emocao.
	i = 0
	num = 0
	while(video_name[i] != '_'):
		i += 1
	if i == 3:
		num = ord(video_name[2]) - 48
	if i == 4:
		x = ord(video_name[3]) - 48 # pego o digito da unidade 
		y = (ord(video_name[2]) - 48) * 10 # pego o digito dos decimais
		num = x + y
	return num


# In[ ]:


def create_csv(j):
	export_csv = df.to_csv('dataframe_video{:d}.csv'.format(j)) 
#	f = open("guru99.txt","w+")


# In[ ]:


def main():
	dataset = pd.DataFrame()
	i = 1
	while(i < 3):
#	dataFrame = extractFrames('Em1_Fala1_CarolinaHolly.mp4', 'A16')
		dataFrame = extractFrames('Em{:d}_.mp4'.format(i), 'frames_em{:d}'.format(i))
		dataset = dataset.append(dataFrame) # aqui eu junto todos os dataframes em um so
#		create_csv(i)
		i += 1
    
	print(dataset)
    
    # aqui eu pretendo transformar o dataframe em um dataset para ser usado no opf.


# In[ ]:


if __name__=="__main__":
	main()

