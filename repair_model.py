import keras
import sys
import h5py
import numpy as np
from architecture import Net

model_path = str(sys.argv[1])
clean_path = str(sys.argv[2])
save_model = str(sys.argv[3])
save_weights = str(sys.argv[4])
# model_path = 'models\sunglasses_bd_weights.h5'
# clean_path = 'data\clean_validation_data.h5'
# save_model = 'models\G1_net.h5'
# save_weights = 'models\G1_weights.h5'

def prunedNet():
    x = keras.Input(shape=(55, 47, 3), name='input')
    conv_1 = keras.layers.Conv2D(20, (4, 4), activation='relu', name='conv_1')(x)
    pool_1 = keras.layers.MaxPooling2D((2, 2), name='pool_1')(conv_1)
    conv_2 = keras.layers.Conv2D(40, (3, 3), activation='relu', name='conv_2')(pool_1)
    pool_2 = keras.layers.MaxPooling2D((2, 2), name='pool_2')(conv_2)
    conv_3 = keras.layers.Conv2D(60, (3, 3), activation='relu', name='conv_3')(pool_2)
    model = keras.Model(inputs=x, outputs=conv_3)

    return model

def data_loader(filepath):
    data = h5py.File(filepath, 'r')
    x_data = np.array(data['data'])
    y_data = np.array(data['label'])
    x_data = x_data.transpose((0, 2, 3, 1))

    return x_data / 255, y_data

def pruning(model, x_valid, y_valid):
    prunedModel = prunedNet()
    weightsDict = {}
    biasDict = {}
    contribution = {}
    accurancy = {}

    model.load_weights(model_path)

    for i in range(1, 4):
        layer = "conv_" + str(i)
        weights, bias = model.get_layer(layer).get_weights()
        prunedModel.get_layer(layer).set_weights((weights, bias))
        weightsDict[layer] = weights
        biasDict[layer] = bias

    prunedOutput = prunedModel.predict(x_valid)

    for output in prunedOutput:
        for node in range(len(prunedOutput[0][0, 0])):
            contribution[node] = 0
            for i in range(len(prunedOutput[0])):
                for j in range(len(prunedOutput[0][0])):
                    contribution[node] += output[i][j][node]

    contribution_sorted = [(val, key) for key, val in contribution.items()]
    contribution_sorted.sort()

    clean_label = np.argmax(model.predict(x_valid), axis=1)
    class_accu = np.mean(np.equal(clean_label, y_valid)) * 100

    print('Classification accuracy:', class_accu)

    current_accu = class_accu
    weights_prune, bias_prune = weightsDict['conv_3'], biasDict['conv_3']
    i = 0

    while current_accu > class_accu - 4 and i < len(contribution_sorted):
        weights_prune[:, :, :, contribution_sorted[i][1]] = np.zeros(np.shape(weights_prune[:, :, :, contribution_sorted[i][1]]))
        bias_prune[contribution_sorted[i][1]] = 0
        model.get_layer('conv_3').set_weights((weights_prune, bias_prune))
        prune_label = np.argmax(model.predict(x_valid), axis=1)
        current_accu = np.mean(np.equal(prune_label, y_valid)) * 100
        accurancy[i] = current_accu
        print("After pruning {} neuro the accuracy is {}".format(contribution_sorted[i][1], current_accu))
        i += 1
    
    return model

def fine_tuning(model, x, y):
    model.compile(
        optimizer='adam',
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])
    model.fit(x, y, epochs=5)
    prune_label = np.argmax(model.predict(x), axis=1)
    current_accu = np.mean(np.equal(prune_label, y)) * 100
    print("After pruning neuroes the accuracy is {}".format(current_accu))

    return model


if __name__ == "__main__":
    x_valid, y_valid = data_loader(clean_path)
    model = Net()
    prunedModel = pruning(model, x_valid, y_valid)
    tunedModel = fine_tuning(prunedModel, x_valid, y_valid)
    tunedModel.save(save_model)
    tunedModel.save_weights(save_weights)