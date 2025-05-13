import { PaddleOcrService } from "../src/";
// import { PaddleOcrService } from "paddle-ocr.js";

const ocrService = await PaddleOcrService.getInstance({
  debugging: {
    debug: true,
    verbose: true,
  },
});

const imagePath = "./assets/receipt.jpg";
const imgFile = Bun.file(imagePath);
const fileBuffer = await imgFile.arrayBuffer();

const startTime = Date.now();
const result = await ocrService.recognize(fileBuffer);
const speed = Date.now() - startTime;

console.log(result.map((el) => el.text).join(" "));
console.log(`Operation completed in ${speed} ms`);
