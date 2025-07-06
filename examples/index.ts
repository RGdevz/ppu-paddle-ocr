import { ImageProcessor } from "ppu-ocv";
import { PaddleOcrService } from "../src/";
// import { PaddleOcrService } from "ppu-paddle-ocr";

const service = await PaddleOcrService.getInstance({
  debugging: {
    debug: false,
    verbose: false,
  },
});

const imagePath = "./assets/receipt.jpg";
const imgFile = Bun.file(imagePath);
const fileBuffer = await imgFile.arrayBuffer();

let canvas = await ImageProcessor.prepareCanvas(fileBuffer);
const processor = new ImageProcessor(canvas);

canvas = processor.grayscale().blur().toCanvas();
processor.destroy();

const startTime = Date.now();
const result = await service.recognize(canvas);
const speed = Date.now() - startTime;

service.destroy();

console.log(JSON.stringify(result, null, 2));
console.log(`Operation completed in ${speed} ms`);
