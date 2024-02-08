import numpy as np
import tensorflow as tf
import time

from tensorflow import keras
from tensorflow.keras import callbacks
from tensorflow.keras.models import load_model


x_train = np.load("data/x_train.npy")
y_train = np.load("data/y_train.npy")

x_val = np.load("data/x_val.npy")
y_val = np.load("data/y_val.npy")

x_test = np.load("data/x_test.npy")
y_test = np.load("data/y_test.npy")

model = load_model('modelv4')

# The batch size controls the number of images that are processed simultaneously
batch_size = 128
# The number of epochs that we want to train the network for
epochs = 20
# The learning rate (step size in gradient descent)
learning_rate = 0.001

# Define the loss function - for a multi-class classification task we need to
# use categorical crossentropy loss
loss_function = keras.losses.categorical_crossentropy
# The optimiser performs the gradient descent for us. There are a few different
# algorithms, but Adam is one of the more popular ones
optimiser = keras.optimizers.legacy.Adam(learning_rate=learning_rate)
# Now we compile the model with the loss function and optimiser
model.compile(loss=loss_function, optimizer=optimiser, metrics=['accuracy'])

start_time = time.time()

# Train the model using the training data with the true target outputs.
# Fill in the required arguments using the clues given above
history = model.fit(x = x_train, y = y_train, batch_size = batch_size, epochs = epochs,
              validation_data = (x_val, y_val), verbose = 1)

end_time = time.time()
training_duration = end_time - start_time

model.save('modelv4')

# Make a list of incorrect classifications
incorrect_indices = []
# Let's look at the whole test dataset, but you can reduce this to 1000 or so
# if you want run more quickly
n_images_to_check = x_test.shape[0]
# Use the CNN to predict the classification of the images. It returns an array
# containing the 10 class scores for each image. It is best to write this code
# using the array notation x[:i] that means use all values of x up until
# the index i, such that if you changed the number of images above then it all
# still works efficiently
raw_predictions = model.predict(x = x_test[:n_images_to_check], batch_size = batch_size)
for i in range(0,n_images_to_check):
  # Remember the raw output from the CNN gives us an array of scores. We want
  # to select the highest one as our prediction. We need to do the same thing
  # for the truth too since we converted our numbers to a categorical
  # representation earlier. We use the np.argmax() function for this
  prediction = np.argmax(raw_predictions[i])
  truth = np.argmax(y_test[i])
  if prediction != truth:
    incorrect_indices.append([i,prediction,truth])

loss, accuracy = model.evaluate(x_test, y_test)

with open('eval.txt', 'w') as file:
    file.write(f'Loss: {loss}\n')
    file.write(f'Accuracy: {accuracy}\n')
    file.write(f'Number of images that were incorrectly classified: {len(incorrect_indices)}\n')
    file.write(f'Training Duration (seconds): {training_duration}\n')
    
with open('hist.json', 'w') as file:
    import json
    history_dict = history.history
    json.dump(history_dict, file)