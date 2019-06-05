# GenderAge_IMDB_Wiki
Detecting Gender &amp; Age from an input image or video (live or pre-recorded) using Keras CNN architecture

### Estimated Results
![](https://i.imgur.com/xCeH2jM.jpg)
- Image Source : Google Images

### Dependencies
- Python 3.7
- NumPy, Pandas, tqdm, scipy
- Keras
- dlib
- OpenCV

## Usage

### Creating our training data from IMDB dataset
- Download the [imdb dataset](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/imdb_crop.tar) using the hyperlink and extract them or follow the below commands to run the activity in command prompt or remote server
- Creating the project directory
```sh
mkdir genderage
```
```sh
cd gender age
```
- Downloading the data from IMDB website into our directory
```sh
mkdir data
```

```sh
cd data
```
```sh
!wget https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/
```
- Extracting the downloaded zip file
```sh
!tar xf imdb-wiki.tar
cd ..
```

- The information of face images stored in a Matlab file and it can be extracted using scipy *loadmat* library module
- Run the following command to create a new Matlab file storing 32x32 Size images pixel values, gender, age, face_score e.t.c
- 
```sh
python create_db.py
```
- The downloaded imdb_crop folder can be deleted to save storage.

### Creating Neural Net Architecture(WideResNet)
- The model can be checked here, [code](neural_net.py)


## Train Network
- Run the following commands to train our model with Epochs = 30, batch_size = 128
```sh
python training.py
```
- Training [file](training.py) takes the images from the Matlab file, but not from the data/imdb_crop folder.
- Optimizer Adam(lr=0.001)
- By running the training script, model weights in .hdf5 format will be created for every epoch if the training_loss score improves when compared with the previous epoch and the weights can be stored in checkpoint folder of the project directory.
- Training our model also gives us a CSV file stored with loss and accuracy scores

### Plot training curves for epoch vs loss, epochs vs validation_accuracy

- Check this [file](plotting_history.py) to create a figure with loss and accuracy scores
- The graph does look like
![](https://i.imgur.com/8qTvRXw.png)

![](https://i.imgur.com/7U5dXSM.png)


## Testing Model

### Testing the finalized model using Images or Video Capture
- Check this [video](genderage_api.mp4) to test the model against random images or video
- The model could be evaluated against live camera open the code file by changing the *image_dir value to None*
```sh
python final.py
```

### Face Detector Module
- I have used dlib.get_frontal_face_detector to crop the images if there is a face in an image and load that cropped face to evaluate our final model. The code for the [face_detector](face_detector.py) can be checked from the hyperlink

### Reference
- [yu4u](https://github.com/yu4u/age-gender-estimation)

