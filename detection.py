import os                               
import cv2
import time                              
import numpy as np  
from PIL import Image
import tensorflow as tf 
from google.colab import drive
from keras import backend as K           
import matplotlib.pyplot as plt
from sklearn.metrics import auc
import sklearn.metrics as metrics
from keras.models import load_model
from skimage.transform import resize
from keras.preprocessing import image  
from sklearn.metrics import roc_curve
from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_curve
from keras.preprocessing.image import img_to_array
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator





"""##### Set Path for Training, Testing and Validation Directories"""

base_dir = '/content/drive/MyDrive/IC DATASETS/Skin cancer dataset'
train_dir = '/content/drive/MyDrive/IC DATASETS/Skin cancer dataset/train'
validations_dir = '/content/drive/MyDrive/IC DATASETS/Skin cancer dataset/validation'
test_dir = '/content/drive/MyDrive/IC DATASETS/Skin cancer dataset/test'

train_benign_dir = '/content/drive/MyDrive/IC DATASETS/Skin cancer dataset/train/benign'
train_malignant_dir = '/content/drive/MyDrive/IC DATASETS/Skin cancer dataset/train/malignant'

validation_benign_dir = '/content/drive/MyDrive/IC DATASETS/Skin cancer dataset/validation/benign'
validation_malignant_dir = '/content/drive/MyDrive/IC DATASETS/Skin cancer dataset/validation/malignant'

test_benign_dir = '/content/drive/MyDrive/IC DATASETS/Skin cancer dataset/test/benign'
test_malignant_dir = '/content/drive/MyDrive/IC DATASETS/Skin cancer dataset/test/malignant'

num_benign_train = len(os.listdir(train_benign_dir))
num_malignant_train = len(os.listdir(train_malignant_dir))

num_benign_validaition = len(os.listdir(validation_benign_dir))
num_malignant_validation= len(os.listdir(validation_malignant_dir))

num_benign_test = len(os.listdir(test_benign_dir))
num_malignant_test= len(os.listdir(test_malignant_dir))

print("Total Training Benign Images",num_benign_train)
print("Total Training Malignant Images",num_malignant_train)
print("--")
print("Total validation Benign Images",num_benign_validaition)
print("Total validation Malignant Images",num_malignant_validation)
print("--")
print("Total Test Benign Images", num_benign_test)
print("Total Test Malignant Images",num_malignant_test)

total_train = num_benign_train+num_malignant_train
total_validation = num_benign_validaition+num_malignant_validation
total_test = num_benign_test+num_malignant_test
print("Total Training Images",total_train)
print("--")
print("Total Validation Images",total_validation)
print("--")
print("Total Testing Images",total_test)

"""######  defining batch and image Size for model training</b>"""

IMG_SHAPE  = 224
batch_size = 32

"""######  Make function for displaying Random Images</b>"""

def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
    plt.tight_layout()
    plt.axis("off")
    plt.show()

"""######  Applying Data Augmentation and Pre-Processing on training Data</b>"""

image_gen_train = ImageDataGenerator(rescale = 1./255,rotation_range = 90,
                                     width_shift_range=0.3,height_shift_range=0.3,
                                     shear_range = 0.3,zoom_range = 0.2,
                                     horizontal_flip = True,
                                     fill_mode = 'nearest')

train_data_gen = image_gen_train.flow_from_directory(batch_size = batch_size,
                                                     directory = train_dir,
                                                     shuffle= True,
                                                     target_size = (IMG_SHAPE,IMG_SHAPE),
                                                     class_mode = 'binary')

augmented_images = [train_data_gen[0][0][0] for i in range(5)]
plotImages(augmented_images)

"""###### Applying pre-processing on validation & Testing Data</b>"""

image_generator_validation = ImageDataGenerator(rescale=1./255)
val_data_gen = image_generator_validation.flow_from_directory(batch_size=batch_size,
                                                 directory=validations_dir,
                                                 target_size=(IMG_SHAPE, IMG_SHAPE),
                                                  class_mode='binary')

image_gen_test = ImageDataGenerator(rescale=1./255)
test_data_gen = image_gen_test.flow_from_directory(batch_size=batch_size,
                                                 directory=test_dir,
                                                 target_size=(IMG_SHAPE, IMG_SHAPE),
                                                 class_mode='binary')

"""######  Let check names of classes"""

train_data_gen.classes

"""###### Setting Models Parameters"""

skin_classifier = tf.keras.Sequential([
        
        tf.keras.layers.Conv2D(16,(3,3),activation = tf.nn.relu,input_shape=(IMG_SHAPE,IMG_SHAPE, 3)),
        tf.keras.layers.MaxPooling2D(2,2),
        
        tf.keras.layers.Conv2D(32,(3,3),activation = tf.nn.relu),
        tf.keras.layers.MaxPooling2D(2,2),
        
        tf.keras.layers.Conv2D(64,(3,3),activation = tf.nn.relu),
        tf.keras.layers.MaxPooling2D(2,2),
        
        tf.keras.layers.Conv2D(128,(3,3),activation = tf.nn.relu),
        tf.keras.layers.MaxPooling2D(2,2),
        
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.5),
        
        tf.keras.layers.Dense(512,kernel_regularizer = tf.keras.regularizers.l2(0.001), activation = tf.nn.relu),
        tf.keras.layers.Dense(2,activation = tf.nn.sigmoid)  
])

"""######  Compiling model parameters"""

skin_classifier.compile(optimizer='adam', loss=tf.keras.losses.sparse_categorical_crossentropy, metrics=['acc'])

"""###### View Summary of model before training</b>"""

skin_classifier.summary()

"""######  Let's Start Training"""

history_skin_classifier = skin_classifier.fit(train_data_gen,
                                              steps_per_epoch=65,
                                              epochs = 50,
                                              validation_data=val_data_gen,
                                              validation_steps=18,
                                              batch_size = batch_size,
                                              verbose = 1)

"""######  Checking history of models parameters"""

history_dict = history_skin_classifier.history
print(history_dict.keys())

"""#####  Visualizing Accuracy and Loss results</b>"""

acc = history_skin_classifier.history['acc']
val_acc = history_skin_classifier.history['val_acc']

loss = history_skin_classifier.history['loss']
val_loss = history_skin_classifier.history['val_loss']

epochs_range = range(50)
plt.figure(figsize=(8, 8))
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='best')
plt.title('Training and Loss')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()

"""#####  Saving model"""

model_json = skin_classifier.to_json()
with open("/content/drive/MyDrive/IC DATASETS MODEL FILES/Skin_cancer_classification.json", "w") as json_file:
    json_file.write(model_json)
skin_classifier.save("/content/drive/MyDrive/IC DATASETS MODEL FILES/Skin_cancer_classification.h5")
print("Saved model to disk")
skin_classifier.save_weights("/content/drive/MyDrive/IC DATASETS MODEL FILES/SCC-Weights.h5")

"""###### Testing Model"""

results = skin_classifier.evaluate(test_data_gen,batch_size=batch_size)
print("test_loss, test accuracy",results)

"""###### Checking Classification Matrix of Model"""

prediction = skin_classifier.predict(test_data_gen)
pred_class = np.argmax(prediction, axis=1)

true_classes = test_data_gen.classes
class_labels = list(test_data_gen.class_indices.keys())  
report = metrics.classification_report(pred_class,true_classes,target_names=class_labels)
print(report)

"""###### Showing multiple images"""

lr_probs = skin_classifier.predict(test_data_gen)
lr_probs = lr_probs[:, 1]

lr_precision, lr_recall, _ = precision_recall_curve(true_classes, lr_probs)

"""##### Visulazing Precision and Recall</b>"""

no_skill = len(true_classes[true_classes==1]) / len(true_classes)
plt.plot(lr_recall, lr_precision, linestyle='--',label='CNN')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
plt.show()

"""##### Visualizing AUC & ROC Curves"""

fpr_keras, tpr_keras, thresholds_keras = roc_curve(true_classes, lr_probs)
auc_keras = auc(fpr_keras, tpr_keras)
plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve (zoomed in at top left)')
plt.legend(loc='best')
plt.show()
