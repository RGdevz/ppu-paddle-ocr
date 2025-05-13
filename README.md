# paddle-ocr.js

A lightweight, type-safe, PaddleOCR implementation in Bun/Node.js for text detection and recognition in JavaScript environments.

OCR should be as easy as:

```ts
import { PaddleOcrService } from "paddle-ocr.js";

const service = await PaddleOcrService.getInstance();
const result = await service.recognize(fileBuffer);
service.destroy();
```

You can combine it further by using open-cv https://github.com/PT-Perkasa-Pilar-Utama/ppu-ocv for more improved accuracy.

## Description

PaddleOCR.js brings the powerful PaddleOCR optical character recognition capabilities to JavaScript environments. This library simplifies the integration of ONNX models with Node.js applications, offering a lightweight solution for text detection and recognition without complex dependencies.

Built on top of `onnxruntime-node`, PaddleOCR.js handles all the complexity of model loading, preprocessing, and inference, providing a clean and simple API for developers to extract text from images with minimal setup.

### Why use this library?

1. **Lightweight**: Optimized for performance with minimal dependencies
2. **Easy Integration**: Simple API to detect and recognize text in images
3. **Cross-Platform**: Works in Node.js and Bun environments
4. **Customizable**: Support for custom models and dictionaries
5. **Pre-packed Models**: Includes optimized PaddleOCR models ready for immediate use
6. **TypeScript Support**: Full TypeScript definitions for enhanced developer experience

## Installation

Install using your preferred package manager:

```bash
npm install paddle-ocr.js
yarn add paddle-ocr.js
bun add paddle-ocr.js
```

> [!NOTE]
> This project is developed and tested primarily with Bun.  
> Support for Node.js, Deno, or browser environments is **not guaranteed**.
>
> If you choose to use it outside of Bun and encounter any issues, feel free to report them.  
> I'm open to fixing bugs for other runtimes with community help.

## Usage

#### Basic usage as singleton

```ts
const service = await PaddleOcrService.getInstance({
  debugging: {
    debug: false,
    verbose: true,
  },
});
```

#### Basic usage using constructor

```ts
const service = new PaddleOcrService();
await service.initialize();
```

#### Using custom models with createInstance

```ts
const customService = await PaddleOcrService.createInstance({
  model: {
    detection: "./models/custom-det.onnx",
    recoginition: "./models/custom-rec.onnx",
  },
});
```

#### Changing models on an existing instance

```ts
await PaddleOcrService.changeModel({
  model: {
    detection: "./models/custom-det.onnx",
    recoginition: "./models/custom-rec.onnx",
  },
});
```
