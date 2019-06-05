import numpy as np
import scipy
from scipy.io import loadmat
import tqdm
from datetime import datetime
import os
import cv2

db = 'imdb'

def calculate_age(taken, dob):
    # The actual DOB of characters stored in legacy MATLAB format and can be decoded using datetime object
    birth = datetime.fromordinal(max(int(dob) - 366, 1))
    if birth.month < 7:
        return taken - birth.year
    else:
        return taken - birth.year - 1

def get_meta(matlab_file_path, db_name):
    dataset = loadmat(matlab_file_path)
    full_path = dataset[db_name][0,0]['full_path'][0]
    dob = dataset[db][0, 0]["dob"][0]
    gender = dataset[db][0, 0]["gender"][0]
    photo_taken = dataset[db][0, 0]["photo_taken"][0]
    face_score = dataset[db][0, 0]["face_score"][0]
    second_face_score = dataset[db][0, 0]["second_face_score"][0]
    age = [calculate_age(photo_taken[i], dob[i]) for i in range(len(dob))]
    return full_path, dob, gender, photo_taken, face_score, second_face_score, age

output_path = "final_matlab.mat"
img_size = 32
min_score = 1.0
root_path = "data/{}_crop/".format(db)
mat_path = root_path + "{}.mat".format(db)
full_path, dob, gender, photo_taken, face_score, second_face_score, age = get_meta(mat_path, db)

out_genders = []
out_ages = []
out_imgs = []

# Cleaning the data using facescore
for i in tqdm.tqdm(range(len(face_score))):
    if face_score[i] < min_score:
        continue
    if (~np.isnan(second_face_score[i])) and second_face_score[i] > 0.0:
        continue
    if ~(0 <= age[i] <= 100):
        continue
    if np.isnan(gender[i]):
        continue
    out_genders.append(int(gender[i]))
    out_ages.append(age[i])
    img = cv2.imread(root_path + str(full_path[i][0]))
    out_imgs.append(cv2.resize(img, (img_size, img_size)))
output = {"image": np.array(out_imgs), "gender": np.array(out_genders), "age": np.array(out_ages),
              "db": db, "img_size": img_size, "min_score": min_score}
              
# Creating a new matlab file with all out data stored alongside with images in an array form
scipy.io.savemat(output_path, output)
print("The New Matlab file is succesfully created")