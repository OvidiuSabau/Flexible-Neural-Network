from PIL import Image
import os
import numpy as np
import pandas as pd

#dir refers to the directory that contains the folders with the files
#make sure that csv_name ends with .csv

def imagesToInputMatrixWithLabels(dir, csv_name):

    #The foldername that a file is in is used as label in this current implementation. Feel free to change that.
    first_elem = True
    labels_list = []
    for foldername in os.listdir(dir):
        for filename in os.listdir(dir + "/" + foldername + "/"):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                print("eeu")
                image = Image.open(dir + foldername + "/" + filename, mode='r')
                greyscale = image.convert(mode='L')
                resized = greyscale.resize((28, 28))
                resized.save("./generated images/" + foldername + "/" + filename)
                vector = np.array(resized)
                vector = np.reshape(vector.flatten(), newshape=(784))
                labels_list.append(int(foldername))
                if first_elem:
                    vector_list = [vector]
                    first_elem = False
                else:
                    vector_list.append(vector)

    matrix = np.array(vector_list)
    matrix = 255 - matrix
    labels_list = np.array(labels_list)
    labels_list = np.reshape(labels_list,newshape=(labels_list.shape[0], 1))
    matrix = np.append(matrix, labels_list, axis=1)
    np.random.shuffle(matrix)

    #Create a list for the CSV header
    names = []
    for i in range(784):
        temp = "label_" + str(i)
        names.append(temp)
    names.append("labels")

    #Save the matrix as .csv
    df_matrix = pd.DataFrame(matrix,columns=names)
    df_matrix.to_csv(csv_name)
    return matrix

def imagesToInputMatrix(dir, csv_name):
    first_elem = True
    for filename in os.listdir(dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            print("eeu")
            image = Image.open(dir + filename, mode='r')
            greyscale = image.convert(mode='L')
            resized = greyscale.resize((28, 28))
            resized.save("./generated images/" + filename)
            vector = np.array(resized)
            vector = np.reshape(vector.flatten(), newshape=(784))
            if first_elem:
                vector_list = [vector]
                first_elem = False
            else:
                vector_list.append(vector)

    matrix = np.array(vector_list)
    matrix = 255 - matrix
    np.random.shuffle(matrix)
    names = []
    for i in range(784):
        temp = "label_" + str(i)
        names.append(temp)
    names.append("labels")

    df_matrix = pd.DataFrame(matrix,columns=names)
    df_matrix.to_csv(csv_name)
    return matrix

#Used to transform a single image
def singleImageToVector(filename):
    image = Image.open(filename, mode='r')
    greyscale = image.convert(mode='L')
    resized = greyscale.resize((28, 28))
    resized.save("./generated images/" + filename)
    vector = np.array(resized)
    vector = np.reshape(vector.flatten(), newshape=(784, 1))
    vector = np.transpose(vector)

    #We do this because in our dataset 255 is used for the darkest values and in pillow it's used for the lightest
    return 255 - vector