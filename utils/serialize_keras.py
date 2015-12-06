import h5py
import json
import gzip

layer_name_dict = {
    'RepeatVector' : 'repeatVector',
    'TimeDistributedDense' : 'timeDistributedDense',
    'Dense': 'denseLayer',
    'Dropout': 'dropoutLayer',
    'Flatten': 'flattenLayer',
    'Embedding': 'embeddingLayer',
    'BatchNormalization': 'batchNormalizationLayer',
    'LeakyReLU': 'leakyReLULayer',
    'PReLU': 'parametricReLULayer',
    'ParametricSoftplus': 'parametricSoftplusLayer',
    'ThresholdedLinear': 'thresholdedLinearLayer',
    'ThresholdedReLu': 'thresholdedReLuLayer',
    'LSTM': 'rLSTMLayer',
    'GRU': 'rGRULayer',
    'JZS1': 'rJZS1Layer',
    'JZS2': 'rJZS2Layer',
    'JZS3': 'rJZS3Layer',
    'Convolution2D': 'convolution2DLayer',
    'MaxPooling2D': 'maxPooling2DLayer',
    'Convolution1D': 'convolution1DLayer',
    'MaxPooling1D': 'maxPooling1DLayer'
}

layer_params_dict = {
    'RepeatVector' : ['n'],
    'TimeDistributedDense' : ['weights', 'activation'],
    'Dense': ['weights', 'activation'],
    'Dropout': ['p'],
    'Flatten': [],
    'Embedding': ['weights', 'mask_zero'],
    'BatchNormalization': ['weights', 'epsilon'],
    'LeakyReLU': ['alpha'],
    'PReLU': ['weights'],
    'ParametricSoftplus': ['weights'],
    'ThresholdedLinear': ['theta'],
    'ThresholdedReLu': ['theta'],
    'LSTM': ['weights', 'activation', 'inner_activation', 'return_sequences'],
    'GRU': ['weights', 'activation', 'inner_activation', 'return_sequences'],
    'JZS1': ['weights', 'activation', 'inner_activation', 'return_sequences'],
    'JZS2': ['weights', 'activation', 'inner_activation', 'return_sequences'],
    'JZS3': ['weights', 'activation', 'inner_activation', 'return_sequences'],
    'Convolution2D': ['weights', 'nb_filter', 'nb_row', 'nb_col', 'border_mode', 'subsample', 'activation'],
    'MaxPooling2D': ['pool_size', 'stride', 'ignore_border'],
    'Convolution1D': ['weights', 'nb_filter', 'filter_length', 'border_mode', 'subsample_length', 'activation'],
    'MaxPooling1D': ['pool_length', 'stride', 'ignore_border']
}

layer_weights_dict = {
    'RepeatVector' : [],
    'Dense': ['W', 'b'],
    'TimeDistributedDense' : ['W', 'b'],
    'Embedding': ['E'],
    'BatchNormalization': ['gamma', 'beta', 'mean', 'std'],
    'PReLU': ['alphas'],
    'ParametricSoftplus': ['alphas', 'betas'],
    'LSTM': ['W_xi', 'W_hi', 'b_i', 'W_xc', 'W_hc', 'b_c', 'W_xf', 'W_hf', 'b_f', 'W_xo', 'W_ho', 'b_o'],
    'GRU': ['W_xz', 'W_hz', 'b_z', 'W_xr', 'W_hr', 'b_r', 'W_xh', 'W_hh', 'b_h'],
    'JZS1': ['W_xz', 'b_z', 'W_xr', 'W_hr', 'b_r', 'W_hh', 'b_h', 'Pmat'],
    'JZS2': ['W_xz', 'W_hz', 'b_z', 'W_hr', 'b_r', 'W_xh', 'W_hh', 'b_h', 'Pmat'],
    'JZS3': ['W_xz', 'W_hz', 'b_z', 'W_xr', 'W_hr', 'b_r', 'W_xh', 'W_hh', 'b_h'],
    'Convolution2D': ['W', 'b'],
    'Convolution1D': ['W', 'b']
}

def appr_f32_prec(arr):
    arr_formatted = []
    for item in arr:
        if type(item) is list:
            arr_formatted.append(appr_f32_prec(item))
        elif type(item) is float:
            arr_formatted.append(float('{:.7f}'.format(item)))
        else:
            arr_formatted.append(item)
    return arr_formatted

def serialize(model_json_file, weights_hdf5_file, save_filepath, compress):
    with open(model_json_file, 'r') as f:
        model_metadata = json.load(f)
    weights_file = h5py.File(weights_hdf5_file, 'r')

    layers = []

    num_activation_layers = 0
    for k, layer in enumerate(model_metadata['layers']):
        if layer['name'] == 'Activation':
            num_activation_layers += 1
            prev_layer_name = model_metadata['layers'][k-1]['name']
            idx_activation = layer_params_dict[prev_layer_name].index('activation')
            layers[k-num_activation_layers]['parameters'][idx_activation] = layer['activation']
            continue

        layer_params = []

        for param in layer_params_dict[layer['name']]:
            if param == 'weights':
                weights = {}
                weight_names = layer_weights_dict[layer['name']]
                for p, name in enumerate(weight_names):
                    arr = weights_file.get('layer_{}/param_{}'.format(k, p)).value
                    if arr.dtype == 'float32':
                        weights[name] = appr_f32_prec(arr.tolist())
                    else:
                        weights[name] = arr.tolist()
                layer_params.append(weights)
            else:
                layer_params.append(layer[param])

        layers.append({
            'layerName': layer_name_dict[layer['name']],
            'parameters': layer_params
        })

    if compress:
        with gzip.open(save_filepath, 'wb') as f:
            f.write(json.dumps(layers).encode('utf8'))
    else:
        with open(save_filepath, 'w') as f:
            json.dump(layers, f)
