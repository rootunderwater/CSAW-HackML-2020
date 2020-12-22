import keras
import sys
import h5py
import numpy as np

clean_data_filename = str(sys.argv[1])
model_name = str(sys.argv[2])

N_plus = 1283

model_dict = {
    'b1':('models/sunglasses_bd_net.h5', 'models/repaired_sunglasses_bd_net.h5'), 
    'b2':('models/multi_trigger_multi_target_bd_net.h5', 'models/repaired_multi_trigger_multi_target_bd_net.h5'),
    'b3':('models/anonymous_bd_net.h5', 'models/repaired_anonymous_bd_net.h5')
}

def data_loader(filepath):
    data = h5py.File(filepath, 'r')
    x_data = np.array(data['data'])
    y_data = np.array(data['label'])
    x_data = x_data.transpose((0,2,3,1))

    return x_data, y_data

def data_preprocess(x_data):
    return x_data/255

def main():
    x_test, y_test = data_loader(clean_data_filename)
    x_test = data_preprocess(x_test)

    bd_model_filename, gd_model_filename = model_dict[model_name]

    bd_model = keras.models.load_model(bd_model_filename)
    gd_model = keras.models.load_model(gd_model_filename)

    bd_label_p = np.argmax(bd_model.predict(x_test), axis=1)

    before_accu = np.mean(np.equal(bd_label_p, y_test))*100
    print('Before defense the model accuracy is:', before_accu)

    gd_label_p = np.argmax(gd_model.predict(x_test), axis=1)
    final_label = np.copy(gd_label_p)

    for i in range(len(bd_label_p)):
        if bd_label_p[i] != gd_label_p[i]:
            final_label[i] = N_plus
    
    after_accu = np.mean(np.equal(final_label, y_test))*100
    print('After defense the model accuracy is:', after_accu)

if __name__ == '__main__':
    main()
