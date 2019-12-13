import * as tf from "@tensorflow/tfjs-node";
//import * as tf from "@tensorflow/tfjs-node-gpu";
import Data from "./data.js";
import Model from "./model.js";

async function main() {
  const loadData = new Data();
  const loadModel = new Model();

  const [trainData, labelData] = await loadData.load();

  const model = await loadModel.load();

  model.compile({
    optimizer: "rmsprop",
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"]
  });

  model.fit(trainData, labelData, {
    batchSize: 32,
    epochs: 5,
    shuffle: true
    // callbacks: tf.node.tensorBoard("/tmp/fit_logs_1")
  });

  return null;
}

main();
