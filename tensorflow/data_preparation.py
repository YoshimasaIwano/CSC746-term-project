import tensorflow as tf

def get_train_loader(batch_size=64, data_size_factor=1):
    (train_images, train_labels), _ = tf.keras.datasets.cifar100.load_data()

    # Reduce dataset size based on data_size_factor
    reduced_size = len(train_images) // data_size_factor
    train_images, train_labels = train_images[:reduced_size], train_labels[:reduced_size]

    # Data normalization and augmentation
    train_images = train_images.astype("float32") / 255
    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    train_dataset = train_dataset.shuffle(10000).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    return train_dataset
