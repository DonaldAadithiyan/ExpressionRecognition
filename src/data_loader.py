import tensorflow as tf
import os

IMG_SIZE = (48, 48)
BATCH_SIZE = 64
AUTOTUNE = tf.data.AUTOTUNE
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, '../data/raw/images')

def load_dataset_from_directory(data_dir=DATA_DIR, img_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=True):
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        labels='inferred',
        label_mode='categorical', 
        image_size=img_size,
        batch_size=batch_size,
        shuffle=shuffle
    )

    # dataset = dataset.cache().prefetch(buffer_size=AUTOTUNE)
    return dataset

def load_datasets(img_size=IMG_SIZE, batch_size=BATCH_SIZE):
    print(os.path.join(DATA_DIR, 'train'))
    train_ds = load_dataset_from_directory(
        os.path.join(DATA_DIR, 'train'),
        img_size=img_size,
        batch_size=batch_size,
        shuffle=True
    )
    
    val_ds = load_dataset_from_directory(
        os.path.join(DATA_DIR, 'validation'),
        img_size=img_size,
        batch_size=batch_size,
        shuffle=False
    )

    test_ds = load_dataset_from_directory(
        os.path.join(DATA_DIR, 'test'),
        img_size=img_size,
        batch_size=batch_size,
        shuffle=False
    )

    return train_ds, val_ds, test_ds
