import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard

def train_model(model, trainloader, epochs=1, log_dir='log'):
    # Callback for TensorBoard
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1, profile_batch='1,10')

    # Training the model
    model.fit(trainloader, epochs=epochs, callbacks=[tensorboard_callback])
