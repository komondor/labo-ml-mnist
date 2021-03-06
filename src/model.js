import * as tf from "@tensorflow/tfjs-node";
//import * as tf from "@tensorflow/tfjs-node-gpu";

export default class Model {
  constructor() {}
  async load() {
    const model = tf.sequential();

    model.add(
      tf.layers.conv2d({
        inputShape: [28, 28, 1],
        kernelSize: 3,
        filters: 16,
        activation: "relu"
      })
    );

    model.add(tf.layers.maxPooling2d({ poolSize: 2, strides: 2 }));
    model.add(
      tf.layers.conv2d({ kernelSize: 3, filters: 32, activation: "relu" })
    );
    model.add(tf.layers.maxPooling2d({ poolSize: 2, strides: 2 }));
    model.add(
      tf.layers.conv2d({ kernelSize: 3, filters: 32, activation: "relu" })
    );
    model.add(tf.layers.flatten({}));
    model.add(tf.layers.dense({ units: 64, activation: "relu" }));
    model.add(tf.layers.dense({ units: 10, activation: "softmax" }));

    return model;
  }
}
