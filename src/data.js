import * as assert from "assert";
import * as fs from "fs";
import * as tf from "@tensorflow/tfjs-node";

const DATA_OFFSET = 16;
const IMAGES_HEIGHT = 28;
const IMAGES_WIDTH = 28;
const TRAIN_SAMPLES = 60000;
const TEST_SAMPLES = 10000;
const TOTAL_BYTES = TRAIN_SAMPLES * IMAGES_HEIGHT * IMAGES_WIDTH;

export default class Data {
  constructor() {
    this.imagesToTrain = null;
    this.labelsToTrain = null;
  }

  async load() {
    let images2Train = await this.prepareTrainingImages();
    let labels2Train = await this.prepareTrainingLabels();
    this.imagesToTrain = await this.trainImagesToTensor(images2Train);
    this.labelsToTrain = await this.trainLabelsToTensor(labels2Train);
    return [this.imagesToTrain, this.labelsToTrain];
  }

  async trainImagesToTensor(images) {
    let tensorImages = new Float32Array(TOTAL_BYTES);

    for (let i = 0; i < images.length; i++) {
      for (let j = 0; j < images[i].length; j++) {
        tensorImages[i * 784 + j] = images[i][j];
      }
    }

    return tf.tensor(tensorImages, [60000, 28, 28, 1]);
  }

  async trainLabelsToTensor(labels) {
    let tensorLabels = new Uint8Array(600000);

    let index = 0;

    while (index < 6000000) {
      for (let i = 0; i < labels.length; i++) {
        let array = new Array(10);
        array.fill(0);

        array[labels[i]] = 1;

        for (let j = 0; j < 10; j++) {
          tensorLabels[index++] = array[j];
        }
      }
    }

    return tf.tensor(tensorLabels, [60000, 10]);
  }

  async prepareTrainingImages() {
    const trainImagesRaw = fs.readFileSync("data/train-images");

    const headerEnding = 16;
    const imageSize = IMAGES_HEIGHT * IMAGES_WIDTH;
    const images = [];
    let index = headerEnding;

    while (index < trainImagesRaw.length) {
      let array = new Float32Array(imageSize);

      for (let i = 0; i < imageSize; i++) {
        array[i] = trainImagesRaw.readUInt8(index++) / 255;
      }
      images.push(array);
    }
    return images;
  }

  prepareTrainingLabels() {
    const labelImagesRaw = fs.readFileSync("data/train-label");
    const headerEnding = 8;
    let index = headerEnding;

    const labels = new Uint8Array(TRAIN_SAMPLES);

    while (index < labelImagesRaw.length) {
      for (let i = 0; i < TRAIN_SAMPLES; i++) {
        labels[i] = labelImagesRaw.readUInt8(index++);
      }
    }

    return labels;
  }
}
