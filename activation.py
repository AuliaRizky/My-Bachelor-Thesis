
from keras import backend as K
import matplotlib.pyplot as plt

def activation_layer(images, model, layer):
    activation = K.function([model.layers[0].input, K.learning_phase()], model.layers[layer].output)
    input_image = images[:, :, 1]
    activations = activation([input_image, 0])
    print(activations.shape)

    fig = plt.figure(figsize=(192, 192))
    row = 32
    column = 8
    for i in range(1, column * row + 1):
        fig.add_subplot(row, column, i)
        plt.imshow(activations)
    plt.show()
    return activations
