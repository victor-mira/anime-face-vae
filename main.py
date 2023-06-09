import numpy as np
import keras
from keras import backend as K
from keras import layers

import imageLoader
from imageLoader import ImageLoader

il = ImageLoader('size64/')
ds = il.load()

encoding_dim = 128
latent_dim = 64
img_shape = (imageLoader.IMG_HEIGHT, imageLoader.IMG_WIDTH, 3)

# Encoder
inputs = keras.Input(shape=img_shape)
normalization_layer = layers.Rescaling(1. / 255, input_shape=img_shape)(inputs)
x = layers.Conv2D(64, (5, 5), activation="relu", padding="same")(normalization_layer)
# x = layers.MaxPool2D((2, 2))(x)
x = layers.BatchNormalization()(x)
x = layers.Conv2D(128, (5, 5), activation="relu", padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.Conv2D(256, (5, 5), activation="relu", padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.Flatten()(x)
# "encoded" is the encoded representation of the input
h = layers.Dense(encoding_dim, activation='relu')(x)
z_mean = layers.Dense(latent_dim, activation='sigmoid')(h)
z_log_sigma = layers.Dense(latent_dim, activation='sigmoid')(h)


def sampling(args):
    z_mean, z_log_sigma = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim),
                              mean=0., stddev=0.1)
    return z_mean + K.exp(z_log_sigma) * epsilon


z = layers.Lambda(sampling)([z_mean, z_log_sigma])

encoder = keras.Model(inputs, [z_mean, z_log_sigma, z], name='encoder')
encoder.summary()

image_size = 64
s2, s4, s8, s16 = int(image_size / 2), int(image_size / 4), int(image_size / 8), int(image_size / 16)  # 32,16,8,4
gf_dim = 64
c_dim = 3

# Decoder
latent_inputs = keras.Input(shape=(latent_dim,), name='z_sampling')
y = layers.Dense(encoding_dim, activation='relu')(latent_inputs)
y = layers.Dense(gf_dim * 4 * s8 * s8, activation='relu')(y)
y = layers.Reshape((s8, s8, gf_dim * 4))(y)
y = layers.BatchNormalization()(y)
y = layers.Conv2DTranspose(gf_dim * 4, (5, 5)  , strides=(2, 2),
                           padding='SAME')(y)
y = layers.BatchNormalization()(y)
y = layers.Conv2DTranspose(gf_dim * 2, (5, 5), strides=(2, 2),
                           padding='SAME')(y)
y = layers.BatchNormalization()(y)
y = layers.Conv2DTranspose(gf_dim, (5, 5), strides=(2, 2),
                           padding='SAME')(y)
y = layers.BatchNormalization()(y)
# y = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(y)
outputs = layers.Conv2DTranspose(3, (5, 5), activation="sigmoid", padding="same")(y)

decoder = keras.Model(latent_inputs, outputs, name='decoder')
decoder.summary()

# VAE model
outputs = decoder(encoder(inputs)[2])
vae = keras.Model(inputs, outputs, name='vae_mlp')

vae.compile(optimizer='adam', loss='binary_crossentropy')
vae.summary()

# Train

train_size = int(len(ds) * 0.75)
test_size = len(ds) - int(len(ds) * 0.75)
newShape = [imageLoader.IMG_WIDTH * imageLoader.IMG_HEIGHT, 3]

x_train = np.array([x.numpy() for x in ds.take(train_size)])
x_test = np.array([x.numpy() for x in ds.skip(train_size).take(test_size)])

vae.fit(x=x_train, y=x_train,
        batch_size=32,
        epochs=100,
        shuffle=True,
        validation_data=(x_test, x_test))

vae.save('Models4/VAE')
encoder.save('Models4/Encoder')
decoder.save('Models4/Decoder')

# Test

decoded_imgs = vae.predict(x_test)

# Use Matplotlib (don't ask)
import matplotlib.pyplot as plt

n = 10  # How many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # Display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
