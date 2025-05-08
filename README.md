# paddle-ocr.js

A lightweight, PaddleOCR implementation in Bun/Node.js for text detection and recognition in JavaScript environments.

## Description

PaddleOCR.js brings the powerful PaddleOCR optical character recognition capabilities to JavaScript environments. This library simplifies the integration of ONNX models with Node.js applications, offering a lightweight solution for text detection and recognition without complex dependencies.

Built on top of `onnxruntime-node`, PaddleOCR.js handles all the complexity of model loading, preprocessing, and inference, providing a clean and simple API for developers to extract text from images with minimal setup.

## Key Features

1. **Lightweight**: Optimized for performance with minimal dependencies
2. **Easy Integration**: Simple API to detect and recognize text in images
3. **Cross-Platform**: Works in Node.js and Bun environments
4. **Customizable**: Support for custom models and dictionaries
5. **Pre-packed Models**: Includes optimized PaddleOCR models ready for immediate use
6. **TypeScript Support**: Full TypeScript definitions for enhanced developer experience

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
