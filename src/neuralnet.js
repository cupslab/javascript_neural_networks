import pack from './lib/ndarray-pack';
import unpack from 'ndarray-unpack';
import * as layerFuncs from './layers';

export default class NeuralNet {
  constructor(config, layers) {
    config = config || {};

    if (config.arrayType === 'float32') {
      this._arrayType = Float32Array;
    } else if (config.arrayType === 'float64') {
      this._arrayType = Float64Array;
    } else {
      this._arrayType = Array;
    }

    if (typeof window === 'object') {
      this._environment = 'browser';
    } else if (typeof importScripts === 'function') {
      this._environment = 'webworker';
    } else if (typeof process === 'object' && typeof require === 'function') {
      this._environment = 'node';
    } else {
      this._environment = 'shell';
    }

    this._layers = layers;
  }

  predict(input) {
    let _predict = (X) => {
      for (let layer of this._layers) {
        let { layerName, parameters } = layer;
        X = layerFuncs[layerName](this._arrayType, X, ...parameters);
      }
      return X;
    };

    let X = pack(this._arrayType, input);
    let output_ndarray = _predict(X);

    return unpack(output_ndarray);
  }

}
