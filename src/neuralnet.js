import pack from './lib/ndarray-pack';
import unpack from 'ndarray-unpack';
import msgpack from 'msgpack-lite';
import * as layerFuncs from './layers';

var weight_first_list = [
  'timeDistributedDense', 'denseLayer', 'embeddingLayer',
  'batchNormalizationLayer', 'parametricReLULayer', 'parametricSoftplusLayer',
  'rLSTMLayer', 'rGRULayer', 'rJZS1Layer', 'rJZS2Layer', 'rJZS3Layer',
  'convolution2DLayer', 'convolution1DLayer'];

function recursive_translate(object, factor, zigzag) {
  for (var key in object) {
    var value = object[key];
    if (typeof value === 'number') {
      var temp;
      if (zigzag) {
        temp = value >> 1;
        if (value % 2 == 1) {
          temp *= -1;
        }
        object[key] = temp / factor;
      } else {
        temp = value;
      }
      object[key] = temp / factor;
    } else if (typeof value === 'object') {
      recursive_translate(value, factor, zigzag);
    }
  }
  return object;
}

function unpack_from_msg(data, scale_factor, zigzag_encoding) {
  var pre_layers = msgpack.decode(new Uint8Array(data));
  var answer = [];
  for (var ln = 0; ln < pre_layers.length; ln++) {
    var layer = pre_layers[ln];
    if (weight_first_list.indexOf(layer.layerName) != -1) {
      layer.parameters = recursive_translate(
        layer.parameters, scale_factor, zigzag_encoding);
    }
    answer.push(layer);
  }
  return answer;
}

export default class NeuralNet {
  constructor(config) {
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

    this._modelFilePath = config.modelFilePath || null;
    this._layers = [];
    this._msg_pck_fmt = config.msgPackFmt || false;
    this._zig_zag_encoding = config.zigzagEncoding || false;
  }

  init(callback) {
    if (!this._modelFilePath) {
      throw new Error('no modelFilePath specified in config object.');
    }

    var xhr = new XMLHttpRequest();
    xhr.open('GET', this._modelFilePath, true);
    xhr.responseType = 'text';
    xhr.onload = (function() {
      if (xhr.status !== 200) {
        console.error(xhr.status);
        return;
      }
      var resp = xhr.response;
      if (this._msg_pck_fmt) {
        this._layers = unpack_from_msg(
          resp, this._msg_pck_fmt, this._zig_zag_encoding);
      } else {
        this._layers = JSON.parse(resp);
      }
      callback();
    }).bind(this);
    xhr.responseType = 'arraybuffer';
    xhr.send();
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
