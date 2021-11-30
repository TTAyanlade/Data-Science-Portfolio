import tkinter as tk
from tkinter.ttk import *
from tkinter import HORIZONTAL
# import tkinter.filedialog as fd
import os
import cv2
import glob
import numpy as np
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
# import cv2
import sys
# import os
import time
import shutil



root = tk.Tk()

l1=tk.Label(root, text=".")
l1.grid(row=3,column=0)

root.option_add("*Button.Background", "gray")
root.option_add("*Button.Foreground", "black")

root.title("Image Recognition")
# root.geometry("280x180")
# root.resizable(0,0)


def prepreprocess(imageP):
    image1 = cv2.imread(imageP)
    gray =cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)


    faceCascade = cv2.CascadeClassifier("./src_c/haarcascade_frontalface_default.xml")
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3, minSize=(30,30))

    # print("[INFO] Found {0} Faces.".format(len(faces)))
    if len(faces) == 0:
        # "./trainingimages/"+
        os.remove(str(imageP))



    for (x, y, w, h) in faces:
        cv2.rectangle(image1,(x,y), (x + w, y+h), (0,255,0),2)
        roi_color = image1[y:y+h, x:x+w]
        # print("[INFO] Object found. Saving locally.")
        cv2.imwrite(str(imageP), roi_color)

    return(len(faces))
    # status = cv2.imwrite(str(imageP)[:-4]+'faces_detected.jpg', image1)
    # print("[INFO] Image faces_detected.jpg written to filesystem: ", status)

def upload():
    global files
    files = os.listdir("./trainingimages")
    # files = fd.askopenfilenames(parent=root, title='Choose the file(s)')
    # print(files)

button1 =tk.Button(root, text='Choose the Training images', command=upload)
# button1.pack()
button1.grid(row=1, column=1, padx=20,pady=5)


def preprocess():
    for t in files:
        # t=t.split('/')[-1]
        # os.system("python preprocess.py "+str(t))
        # os.system("python preprocess.py ./trainingimages/"+str(t))
        prepreprocess("./trainingimages/"+str(t))


button2 =tk.Button(root, text='Preprocess The Training Images', command=preprocess)
button2.grid(row=3, column=1,padx=10,pady=5)
# button2.pack()



def model():


    #Train data
    train = []
    train_labels = []

    for l in os.listdir("./trainingimages"):
        image = cv2.imread("./trainingimages/"+l)
        image=cv2.resize(image, (100, 100))
        train.append (image)
        train_labels.append([1.])

    for e in os.listdir("./category2"):
        image2 = cv2.imread("./category2/"+e)
        image2=cv2.resize(image2, (100, 100))
        train.append (image2)
        train_labels.append([2.])

    train = np.array(train) #as mnist
    train_labels = np.array(train_labels,dtype='float64') #as mnist
    # convert (number of images x height x width x number of channels) to (number of images x (height * width *3))
    # for example (120 * 40 * 40 * 3)-> (120 * 4800)
    train = np.reshape(train,[train.shape[0],train.shape[1]*train.shape[2]*train.shape[3]])

    # save numpy array as .npy formats
    np.save('train.npy',train)
    np.save('train_labels.npy',train_labels)


    try:
        shutil.rmtree("Similar")
    except:
        pass

    shutil.copytree("Comparison", "Similar", symlinks=False, ignore=None)


    # scaler = StandardScaler()
    # # fitting on training set.
    # scaler.fit(train)
    #
    # # applying transformation on both the train and test image set
    # train_img = scaler.transform(train)
    #
    # # test_img = scaler.transform(test_img)
    #
    #
    # # Make an instance of the PCA model
    # pca = PCA(.95)
    # # fitting on the train dataset
    # pca.fit(train)
    #
    # # using the PCA model to transform both the training and test data
    # pca_train_img = pca.transform(train)
    # # pca_test_img = pca.transform(test_img)

button3 =tk.Button(root, text='Model', command=model)
# button3.pack()
button3.grid(row=5, column=1,padx=10,pady=5)




# def uploadComp():
#     global test
#     test = []
#     # duplicate comparison and use it for move
#     for k in os.listdir("./comparison"):
#         os.system("python preprocess.py "+str(k))
#         image3 = cv2.imread(k, 0)
#         image3=cv2.resize(image3, (100, 100))
#         test.append (image3)
#         test = np.array(test)
#         test = np.reshape(test,[test.shape[0],test.shape[1]*test.shape[2]*test.shape[3]])
#         # save numpy array as .npy formats
#         np.save('test',test)
#
#
# button4 =tk.Button(root, text='Upload Test images', command=uploadComp)
# button4.pack()




progress = Progressbar(root, orient = HORIZONTAL,
			length = len(os.listdir("./comparison"))/5, maximum=len(os.listdir("./comparison"))/5, mode = 'determinate')

def execute():
    train = np.load("train.npy")
    train_lbl = np.load("train_labels.npy")
    # start_time = os.times()[0]
    finale=[]
    knn_model = KNeighborsClassifier(n_neighbors=5, algorithm="brute")

    knn_model.fit(train, train_lbl.reshape(-1))
    # print(knn_model)
    count=0

    t=0
    r= 1/5
    for k in os.listdir("./comparison"):
        t=t+r
        progress['value'] = t
        root.update_idletasks()
        # time.sleep(1)


        test=[]
        # Path("./comparison/similar").mkdir(parents=True, exist_ok=True)
        # try:
        # os.system("python preprocess.py ./comparison/"+str(k))
        try:
            prepreprocess("./comparison/"+str(k))
        # if prepreprocess("./comparison/"+str(k)) == 0:
        #     continue

            image3 = cv2.imread("./comparison/"+str(k))
            image3=cv2.resize(image3, (100, 100))
            test.append(image3)
            test= np.array(test)
            test = np.reshape(test,[test.shape[0],test.shape[1]*test.shape[2]*test.shape[3]])
            np.save('test.npy',test)

            test = np.load("test.npy")
            # print(image3)
            predictions = knn_model.predict(test)
            # print(predictions)
            if predictions == 1.:
                finale.append(str(k))
            else:
                pass
        except:
            continue

    for el in os.listdir("./Similar"):
        if el not in finale:
            os.remove("./Similar/"+el)


            # test.append (image3)
            # test = np.array(test)
            # test = np.reshape(test,[test.shape[0],test.shape[1]*test.shape[2]*test.shape[3]])
            # save numpy array as .npy formats
            # np.save('test',test)



        # for item in test:
        #     count += 1
            # predictions = knn_model.predict(item)
            #
            # if predictions == 1.:
            #     test[count]
            #
        # except:
        #     pass
progress.grid(row=7, column=1,padx=10, pady=5)

button5 =tk.Button(root, text='Execute Comparison Process', command=execute)
# button5.pack()
button5.grid(row=8, column=1,padx=10, pady=5)

root.mainloop()
