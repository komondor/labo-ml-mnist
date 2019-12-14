//import * as tf from "@tensorflow/tfjs-node";
import * as tf from "@tensorflow/tfjs-node-gpu";
import Data from "./data.js";
import Model from "./model.js";
import * as fs from "fs";
import sharp from "sharp";

async function main() {
  const loadData = new Data();
  const loadModel = new Model();

  const [trainData, labelData] = await loadData.load();

  let image = fs.readFileSync("test_image.png", () => {});

  let resizeImage = await sharp(image)
    .resize(28, 28)
    .toBuffer();

  let testImage = tf.node.decodePng(resizeImage, 1);

  let reshapeImage = testImage.reshape([1, 28, 28, 1]);

  const model = await loadModel.load();

  model.compile({
    optimizer: "rmsprop",
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"]
  });

  await model.fit(trainData, labelData, {
    batchSize: 32,
    epochs: 1,
    shuffle: true,
    onEpochEnd: (epoch, logs) => {
      console.log(5);
      return null;
    }
    // callbacks: tf.node.tensorBoard("/tmp/f}it_logs_1")
  });
  model.predict(reshapeImage).print();
  return null;
}

main();
