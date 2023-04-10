import numpy as np
from keras import models
from matplotlib import pyplot as plt

decoder = models.load_model('Models/Decoder')

# Test with new generation

img_size = 28
figure = np.zeros((img_size * 8, img_size * 8))

z_sample1 = np.linspace(-15, 15, 8)

x_decoded = decoder.predict(z_sample1)

plt.imshow(figure)
plt.show()
