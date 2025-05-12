import PaddleOcrService from "../src/paddle-ocr.service";
// import PaddleOcrService from "paddle-ocr.js";

const ocrService = await PaddleOcrService.getInstance({
  verbose: true,
});

const imagePath = "./assets/receipt.jpg";
const imgFile = Bun.file(imagePath);
const fileBuffer = await imgFile.arrayBuffer();

const result = await ocrService.recognize(fileBuffer);
console.log(result);
