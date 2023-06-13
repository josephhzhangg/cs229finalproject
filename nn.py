from numpy import loadtxt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation
from tensorflow.keras.utils import plot_model
from keras_visualizer import visualizer 
import tensorflow
import visualkeras


import util


def main():

    ### LOAD DATASET ###
    train_path = "train_split.csv"
    test_path = "test_split.csv"
    train_x,train_y=util.load_dataset(train_path,add_intercept=False)
    test_x,test_y=util.load_dataset(test_path,add_intercept=False)
    n_features = train_x.shape[1]
    #print(train_x.shape)

    ### CREATE MODEL ### 
    model = Sequential()
    model.add(Dense(64, input_shape = (n_features,), name = "Hidden_layer_1"))
    model.add(Activation('relu', name = "Relu_activation"))
    #model.add(Dense(8))
    # # #model.add(Activation('relu'))
    model.add(Dense(40, name = "Hidden_layer_2"))
    model.add(Activation('tanh', name="Tanh_activation"))
    model.add(Dense(1, activation = 'sigmoid', name = "Sigmoid_output")) #Binary classification single neuron output layer
    model.compile(loss = 'binary_crossentropy', optimizer = 'Adam', metrics = ['accuracy'])
    model.fit(train_x, train_y, epochs = 5000, batch_size = 40)

    #Load from file
    #model = tensorflow.keras.models.load_model("modelFile.h5")
    # with open("model_architecture.json", "r", encoding="utf-8") as json_file:
    #   model_json = json_file.read()
    # model = tensorflow.keras.models.model_from_json(model_json)
    # model.compile(loss = 'binary_crossentropy', optimizer = 'Adam', metrics = ['accuracy'])


    #Save the JSON to a file
    # model_json = model.to_json()
    # with open("model_architecture.json", "w", encoding="utf-8") as json_file:
    #    json_file.write(model_json)    
    # model.save("model.h5")

    plot_model(model, to_file = 'model_visualization.png', show_shapes = True, show_layer_names = True)
    _, accuracy = model.evaluate(train_x, train_y)
    print('Train accuracy: %.2f' % (accuracy*100))
    _, accuracy = model.evaluate(test_x, test_y)
    print('Test accuracy: %.2f' % (accuracy*100))

    return

    #Make copy model to test outputs
    modelCopy = Sequential()
    modelCopy.add(Dense(64, input_shape = (n_features,)))
    modelCopy.add(Activation('relu'))
    modelCopy.layers[0].set_weights(model.layers[0].get_weights())
    modelCopy.layers[1].set_weights(model.layers[1].get_weights())
    modelCopy.compile(loss = 'binary_crossentropy', optimizer = 'Adam', metrics = ['accuracy'])
    # _, accuracy = modelCopy.evaluate(train_x, train_y)
    # print('Train accuracy: %.2f' % (accuracy*100))
    # _, accuracy = model.evaluate(test_x, test_y)
    # print('Test accuracy: %.2f' % (accuracy*100))

main()
