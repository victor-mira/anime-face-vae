import pathlib
import tensorflow as tf

IMG_HEIGHT = 64
IMG_WIDTH = 64

class ImageLoader():
    def __init__(self, data_dir):
        self.data_dir = pathlib.Path(data_dir)
        self.list_ds = tf.data.Dataset.list_files(str(self.data_dir/'*'))

    def load(self):
        unlabeled_ds = self.list_ds.map(self.process_filepath)
        return unlabeled_ds

    def process_filepath(self, filepath):
        img = tf.io.read_file(filepath)
        img = self.decode_img(img)
        return img

    def decode_img(self, img):
        img = tf.image.decode_jpeg(img, channels=3)  # color images
        img = tf.image.convert_image_dtype(img, tf.float32)
        return tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])