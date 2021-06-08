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

        if(idx % 2 == 0):
            continue

        name_img = root_path + '/' + train_name
        
        for name in os.listdir(name_img):
            image_full_path =name_img + '/' + name
            img = cv2.imread(image_full_path)

            persons.append(Person(train_name, img, name, None))
                   
    return persons

def get_face_by_area(faces):
    max_area = -1
    max_idx = 0
    
    if(len(faces) == 1):
        return faces[0]
    
    for idx, face in enumerate(faces):
        width = face['box'][2]
        height = face['box'][3]
        area = width * height
        if area > max_area:
            max_area = area
            best_face = face
    
    return best_face

def get_face(faces):
    temp_confidence = 0
    
    if(len(faces) == 1):
        return faces[0]
    
    for face in faces:
        if face['confidence'] > temp_confidence:
            temp_confidence = face['confidence']
            best_face = face
    
    return best_face

def detect_faces(persons):
    persons_detect = []

    for person in persons:
        detect = detector.detect_faces(person.image)
        
        if(detect and len(detect) != 0):
            face = get_face_by_area(detect)
            detected_image = person.image
            detected_image_name = person.name
            detected_image_path = person.path
            persons_detect.append(Person(detected_image_name, detected_image, detected_image_path, face))

    return persons_detect

def image_rotated(persons):
    persons_rotated = []

    for person in persons:
        bounding_box = person.face['box']
        keypoints = person.face['keypoints']
        leftEyePts  = keypoints['left_eye']
        rightEyePts = keypoints['right_eye']
        dY = rightEyePts[1] - leftEyePts[1]
        dX = rightEyePts[0] - leftEyePts[0]
        angle = np.degrees(np.arctan2(dY, dX))
        person.image = imutils.rotate_bound(person.image, 360 - angle)
        persons_rotated.append(person)
    return persons_rotated

def image_cropped(persons):
    persons_cropped = []

    for person in persons:
        bounding_box =person.face['box']
        keypoints =person.face['keypoints']
        h,w,c = person.image.shape
        y1 = np.maximum(bounding_box[1], 0)
        x1 = np.maximum(bounding_box[0], 0)
        x2 = np.minimum(bounding_box[0] + bounding_box[2], w)
        # img_cropped = person.image[bounding_box[1] : keypoints['nose'][1], bounding_box[0] : bounding_box[0] + bounding_box[2]]
        img_cropped = person.image[y1 : keypoints['nose'][1], x1 : x2]
        img_grayscale = cv2.cvtColor(img_cropped, cv2.COLOR_RGB2GRAY)
        person.image = img_grayscale
        persons_cropped.append(person)

    return persons_cropped

def combine_and_show_result(image_list):
    w=10
    h=10
    fig=plt.figure(figsize=(15, 15))
    columns = 5
    rows = 5
    for i in range(1, columns*rows +1):
        img = image_list[i]
        fig.add_subplot(rows, columns, i)
        # RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
    plt.show()

def create_directory(persons):
    if not os.path.exists("RMFD_result"):
        os.mkdir("RMFD_result")
    
    for person in persons:
        dirname = os.path.join("RMFD_result",person.name)
        print(dirname)
        if not os.path.exists(dirname):
            os.mkdir(dirname)

def save_file(persons):
    for person in persons:
        result = cv2.imwrite("RMFD_result/"+ os.path.join(person.name, person.path), person.image)

    if result == False:
        print('Error in saving file')

if __name__ == '__main__':
    detector = MTCNN()
    print("step 1")
    train_root_path = 'RMFD/self-built-masked-face-recognition-dataset/AFDB_face_dataset/'
    # train_root_path = 'dataset/train/'
    print("step 2")
    train_names = get_path_list(train_root_path)
    print("step 3")
    person_list = get_image_list(train_root_path, train_names) 
    # print(len(image_list))
    # print(len(image_name))
    print("step 4")
    train_person_list = detect_faces(person_list)
    # for person in train_person_list:
    #     person.print_all()
    print("step 5")
    image_rotated_list = image_rotated(train_person_list)
    print("step 6")
    train_person_list_after_rotated = detect_faces(image_rotated_list)
    print("step 7")
    image_cropped_list = image_cropped(train_person_list_after_rotated)

    # # combine_and_show_result(image_cropped_list)

    print("step 8")
    create_directory(image_cropped_list)
    print("step 9")
    save_file(image_cropped_list)

