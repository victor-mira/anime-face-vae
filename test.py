import numpy as np
from keras import models
from matplotlib import pyplot as plt
import tensorflow as tf

decoder = models.load_model('Models4/Decoder')

# Test with new generation
n=8
input_shape=64

img_size = 64
z_sample = []
for i in range(n):
    z_sample.append(np.random.rand(input_shape))

sample = tf.data.Dataset.from_tensor_slices(z_sample)
sample = sample.batch(1)


x_decoded = decoder.predict(sample)
plt.figure(figsize=(20, 4))

for i in range(n):
    # Display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_decoded[i])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()
