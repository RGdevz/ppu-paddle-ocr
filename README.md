# paddle-ocr.js

A lightweight, PaddleOCR implementation in Bun/Node.js for text detection and recognition in JavaScript environments.

## Usage

#### Basic usage

```ts
const ocrService = await PaddleOcrService.getInstance({
  verbose: true,
});
```

#### Using custom models

```ts
const customService = await PaddleOcrService.createInstance({
  detectionModelPath: "./models/custom-det.onnx",
  recognitionModelPath: "./models/custom-rec.onnx",
  verbose: true,
});
```

#### Changing models on an existing instance

```ts
await PaddleOcrService.changeModel({
  detectionModelPath: "./models/new-model.onnx",
  verbose: true,
});
```
