import tensorflow as tf

def setup_model(use_gpus=1):
    strategy = tf.distribute.MirroredStrategy() if use_gpus > 1 else None
    with strategy.scope() if strategy else dummy_context_mgr():
        model = tf.keras.applications.ResNet50(input_shape=(32, 32, 3), weights=None, classes=100)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
