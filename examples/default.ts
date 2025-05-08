import PaddleOcrService from "../src/paddle-ocr.service";
// import PaddleOcrService from "paddle-ocr.js";

const ocrService = await PaddleOcrService.getInstance({
  verbose: true,
});

console.log(ocrService.isInitialized());
