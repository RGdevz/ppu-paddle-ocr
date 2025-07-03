import { PaddleOcrService } from "../src";
// import { PaddleOcrService } from "ppu-paddle-ocr";

const service = await PaddleOcrService.getInstance({
  debugging: {
    debug: true,
    verbose: true,
  },
});

const imagePath = "./assets/tilted.png";
const imgFile = Bun.file(imagePath);
const fileBuffer = await imgFile.arrayBuffer();

const startTime = Date.now();
const result = await service.deskewImage(fileBuffer);
const speed = Date.now() - startTime;

service.destroy();

console.log(`Operation completed in ${speed} ms, check out folder`);
