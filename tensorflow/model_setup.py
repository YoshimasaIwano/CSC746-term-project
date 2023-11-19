import tensorflow as tf

def setup_model(use_gpus=1):
    if use_gpus > 1:
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            model = tf.keras.applications.ResNet50(input_shape=(32, 32, 3), weights=None, classes=100)
            # Using SGD optimizer
            sgd_optimizer = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9)
            model.compile(optimizer=sgd_optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    else:
        model = tf.keras.applications.ResNet50(input_shape=(32, 32, 3), weights=None, classes=100)
        # Using SGD optimizer
        sgd_optimizer = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9)
        model.compile(optimizer=sgd_optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model
