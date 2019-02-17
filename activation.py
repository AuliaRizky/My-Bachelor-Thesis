from keras.models import Model
    
def display_activation(activations, col_size, row_size, act_index): 
    activation = activations[act_index]
    activation_index=0
    fig, ax = plt.subplots(row_size, col_size, figsize=(row_size*2.5,col_size*1.5))
    for row in range(0,row_size):
        for col in range(0,col_size):
            ax[row][col].imshow(activation[0, :, :, activation_index], cmap='gray')
            activation_index += 1
            
def activation_layer(images, activation_model):
    # May need to load weight
    activations = activation_model.predict(images)            
    display_activation(activations, 8, 8, 1)
