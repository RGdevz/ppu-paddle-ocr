import { PaddleOcrService } from "../src/";
// import { PaddleOcrService } from "paddle-ocr.js";

const service = await PaddleOcrService.getInstance({
  debugging: {
    debug: false,
    verbose: false,
  },
});

const imagePath = "./assets/receipt.jpg";
const imgFile = Bun.file(imagePath);
const fileBuffer = await imgFile.arrayBuffer();

const startTime = Date.now();
const result = await service.recognize(fileBuffer);
const speed = Date.now() - startTime;

service.destroy();

console.log(result);
console.log(`Operation completed in ${speed} ms`);
