import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from mtcnn.mtcnn import MTCNN
import imutils
import tensorflow as tf

class Person:
    def __init__(self, name, image, path, face):
        self.name = name
        self.image = image
        self.path = path
        self.face = face

    def print_all(self):
        print(f'Name : {self.name}, path : {self.path}, face : {self.face}')

def get_path_list(root_path):
    return os.listdir(root_path)

def get_image_list(root_path, train_names):
    
    persons = []
    for idx,train_name in enumerate(train_names):
        name_img = root_path + '/' + train_name
        
        for name in os.listdir(name_img):
            image_full_path = name_img + '/' + name
            img = cv2.imread(image_full_path)
            img = cv2.resize(img, (224,224))

            persons.append(Person(train_name, img, name, None))

        if idx == 10:
            break;       
    return persons

if __name__ == "__main__":
    test_root_path = 'RMFD_result'
    test_name = get_path_list(test_root_path)
    test_person_list = get_image_list(test_root_path, test_name) 
    for person in test_person_list:
        cv2.imshow("person",person.image)
        cv2.waitKey(0)