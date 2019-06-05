import pandas as pd
import logging
from pathlib import Path
import numpy as np
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.optimizers import Adam
from keras.utils import np_utils
from neural_net import WideResNet
from scipy.io import loadmat

print("All the Libraries Imported Successfully")

class Schedule:
    def __init__(self, nb_epochs, initial_lr):
        self.epochs = nb_epochs
        self.initial_lr = initial_lr

    def __call__(self, epoch_idx):
        if epoch_idx < self.epochs * 0.25:
            return self.initial_lr
        elif epoch_idx < self.epochs * 0.50:
            return self.initial_lr * 0.2
        elif epoch_idx < self.epochs * 0.75:
            return self.initial_lr * 0.04
        return self.initial_lr * 0.008

def load_mat(input_path):
    d = loadmat(input_path)
    return d["image"], d["gender"][0], d["age"][0], d["db"][0], d["img_size"][0, 0], d["min_score"][0, 0]

print("Step-1")       
input_path = 'final_matlab.mat'
batch_size = 128
nb_epochs = 30
depth = 16
k = 8
lr = 0.001
validation_split = 0.1
output_path = Path(__file__).resolve().parent.joinpath('checkpoint')
output_path.mkdir(parents=True, exist_ok=True)
logging.debug("Loading data...")
print("Step-2")
image, gender, age, _, image_size, _ = load_mat(input_path)
X_data = image
y_data_g = np_utils.to_categorical(gender, 2)
y_data_a = np_utils.to_categorical(age, 101)
model = WideResNet(image_size, depth=depth, k=k)()
print("WideResNet Loaded")
opt = Adam(lr=00.001)
print("Model is getting ready for compiling")
model.compile(optimizer=opt, loss=["categorical_crossentropy", "categorical_crossentropy"],metrics=['accuracy'])
logging.debug("Model summary...")
model.count_params()
model.summary()

callbacks = [LearningRateScheduler(schedule=Schedule(nb_epochs, lr)),
                 ModelCheckpoint(str(output_path) + "/weights.{epoch:02d}-{val_loss:.2f}.hdf5",
                                 monitor="val_loss",
                                 verbose=1,
                                 save_best_only=True,
                                 mode="auto")
                 ]

logging.debug("Running training...")
data_num = len(X_data)
indexes = np.arange(data_num)
np.random.shuffle(indexes)
X_data = X_data[indexes]
y_data_g = y_data_g[indexes]
y_data_a = y_data_a[indexes]
train_num = int(data_num * (1 - validation_split))
X_train = X_data[:train_num]
X_test = X_data[train_num:]
y_train_g = y_data_g[:train_num]
y_test_g = y_data_g[train_num:]
y_train_a = y_data_a[:train_num]
y_test_a = y_data_a[train_num:]

hist = model.fit(X_train, [y_train_g, y_train_a], batch_size=batch_size, epochs=nb_epochs, callbacks=callbacks,
                         validation_data=(X_test, [y_test_g, y_test_a]))

logging.debug("Saving history...")

try:
    training_loss = hist.history['loss']
    pred_gender_loss = hist.history['pred_gender_loss']
    pred_age_loss = hist.history['pred_age_loss']
    pred_gender_acc = hist.history['pred_gender_acc']
    pred_age_acc = hist.history['pred_age_acc']
    graph_data = {'Training_Loss': training_loss, 'Predicted_Gender_Loss' : pred_gender_loss, 'Predicted_Age_Loss' : pred_age_loss,
    'Predicted_Gender_Accuracy' : pred_gender_acc, 'Predicted_Age_Accuracy': pred_age_acc}
    df = pd.DataFrame(graph_data)
except:
    pass
print("First Trail Successful")

try:
    df.to_csv('graph_data.csv',sep='\t', encoding='utf-8', index = False)
except:
    df.to_csv('graph_data1.csv', encoding = 'utf-8')
finally:
    df.to_csv('graph_data2.csv')

try:
    pd.DataFrame(hist.history).to_hdf(output_path.joinpath("history_{}_{}.h5".format(depth, k)), "history")
except:
    pass

