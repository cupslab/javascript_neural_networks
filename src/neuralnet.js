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

function translate(pre_layers, scale_factor, zigzag_encoding) {
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

function unpack_from_msg(data, scale_factor, zigzag_encoding) {
  return translate(msgpack.decode(new Uint8Array(data)),
                   scale_factor, zigzag_encoding);
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
    this._scale_factor = config.scaleFactor || false;
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
      if (this._scale_factor) {
        if (this._msg_pck_fmt) {
          this._layers = unpack_from_msg(
            xhr.response, this._scale_factor, this._zig_zag_encoding);
        } else {
          this._layers = translate(JSON.parse(xhr.responseText),
                                   this._scale_factor, this._zig_zag_encoding);
        }
      } else {
        this._layers = JSON.parse(xhr.responseText);
      }
      callback();
    }).bind(this);
    if (this._msg_pck_fmt) {
      xhr.responseType = 'arraybuffer';
    }
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
