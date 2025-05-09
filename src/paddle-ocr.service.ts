import { existsSync, readFileSync } from "fs";
import * as ort from "onnxruntime-node";
import * as path from "path";
import {
  Canvas,
  CanvasToolkit,
  Contours,
  createCanvas,
  cv,
  ImageProcessor,
} from "ppu-ocv";

import {
  DEFAULT_DETECTION_MODEL_PATH,
  DEFAULT_EN_DICT_PATH,
  DEFAULT_RECOGNITION_MODEL_PATH,
  DET_MEAN,
  DET_STD,
  MAX_SIDE_LEN,
} from "./constants";

import type {
  Box,
  PaddleServiceOptions,
  PreprocessDetectionResult,
} from "./interface";
import { log } from "./util";

/**
 * PaddleOcrService - Provides OCR functionality using PaddleOCR models
 *
 * This service can be used either as a singleton or as separate instances
 * depending on your application needs.
 */
class PaddleOcrService {
  private static instance: PaddleOcrService | null = null;
  private options: PaddleServiceOptions;

  private detectionSession: ort.InferenceSession | null = null;
  private recognitionSession: ort.InferenceSession | null = null;
  private charactersDictionary: string[] = [];

  /**
   * Create a new PaddleOcrService instance
   * @param options Optional configuration options
   */
  constructor(options?: PaddleServiceOptions) {
    this.options = {
      detectionModelPath: DEFAULT_DETECTION_MODEL_PATH,
      recognitionModelPath: DEFAULT_RECOGNITION_MODEL_PATH,
      dictionaryPath: DEFAULT_EN_DICT_PATH,
      verbose: false,
      ...options,
    };
  }

  /**
   * Initialize the OCR service by loading models
   * @param overrideOptions Optional parameters to override the constructor options
   */
  public async initialize(
    overrideOptions?: Partial<PaddleServiceOptions>
  ): Promise<void> {
    try {
      const effectiveOptions = {
        ...this.options,
        ...overrideOptions,
      };

      const resolvedDetectionPath = path.resolve(
        process.cwd(),
        effectiveOptions.detectionModelPath!
      );
      const resolvedRecognitionPath = path.resolve(
        process.cwd(),
        effectiveOptions.recognitionModelPath!
      );
      const resolvedDictionaryPath = path.resolve(
        process.cwd(),
        effectiveOptions.dictionaryPath!
      );

      log(
        effectiveOptions.verbose,
        `Loading Detection ONNX model from: ${resolvedDetectionPath}`
      );

      const detModelBuffer = readFileSync(resolvedDetectionPath).buffer;
      this.detectionSession = await ort.InferenceSession.create(detModelBuffer);

      log(
        effectiveOptions.verbose,
        `Detection ONNX model loaded successfully.`
      );
      log(
        effectiveOptions.verbose,
        `Loading Recognition ONNX model from: ${resolvedRecognitionPath}`
      );

      const recModelBuffer = readFileSync(resolvedRecognitionPath).buffer;
      this.recognitionSession = await ort.InferenceSession.create(
        recModelBuffer
      );

      log(
        effectiveOptions.verbose,
        `Recognition ONNX model loaded successfully.`
      );

      this.charactersDictionary = this.loadCharDictionary(
        resolvedDictionaryPath
      );

      log(
        effectiveOptions.verbose,
        `Character dictionary loaded with ${this.charactersDictionary.length} entries.`
      );
    } catch (error) {
      console.error("Failed to initialize PaddleOcrService:", error);
      throw error;
    }
  }

  /**
   * Get or create the singleton instance of PaddleOcrService
   * @param options Configuration options for the service
   * @returns A promise resolving to the singleton instance
   * @example
   * const service = await PaddleOcrService.getInstance({
   *   verbose: true,
   *   detectionModelPath: './models/myDetection.onnx'
   * });
   */
  public static async getInstance(
    options?: PaddleServiceOptions
  ): Promise<PaddleOcrService> {
    if (!PaddleOcrService.instance) {
      PaddleOcrService.instance = new PaddleOcrService(options);
      await PaddleOcrService.instance.initialize();
    } else if (options) {
      await PaddleOcrService.instance.initialize(options);
    }
    return PaddleOcrService.instance;
  }

  /**
   * Load character dictionary from file
   * @param filePath Path to the dictionary file
   * @returns Array of characters
   */
  private loadCharDictionary(filePath: string): string[] {
    if (!existsSync(filePath)) {
      throw new Error(`Character dictionary not found at: ${filePath}`);
    }

    const dictContent = readFileSync(filePath, "utf-8");
    const lines = dictContent.split("\n").map((line) => line.trimEnd());

    log(
      this.options.verbose,
      `Loaded ${lines.length} characters from dictionary.`
    );

    return lines;
  }

  /**
   * Check if the service is initialized with models loaded
   */
  public isInitialized(): boolean {
    return this.detectionSession !== null && this.recognitionSession !== null;
  }

  /**
   * Change models in the singleton instance
   * @param options New configuration options
   */
  public static async changeModel(
    options: Partial<PaddleServiceOptions>
  ): Promise<PaddleOcrService> {
    if (!PaddleOcrService.instance) {
      PaddleOcrService.instance = new PaddleOcrService(options);
      await PaddleOcrService.instance.initialize();
    } else {
      await PaddleOcrService.instance.initialize(options);
    }

    return PaddleOcrService.instance;
  }

  /**
   * Create a new instance instead of using the singleton
   * This is useful when you need multiple instances with different models
   * @param options Configuration options for this specific instance
   */
  public static async createInstance(
    options?: PaddleServiceOptions
  ): Promise<PaddleOcrService> {
    const instance = new PaddleOcrService(options);
    await instance.initialize();

    return instance;
  }

  private async prepocessDetection(
    image: ArrayBuffer
  ): Promise<PreprocessDetectionResult> {
    const initialCanvas = await ImageProcessor.prepareCanvas(image);

    let resizeW = initialCanvas.width;
    let resizeH = initialCanvas.height;
    let resizeRatio = 1.0;

    if (Math.max(resizeH, resizeW) > MAX_SIDE_LEN) {
      if (resizeH > resizeW) {
        resizeRatio = MAX_SIDE_LEN / resizeH;
      } else {
        resizeRatio = MAX_SIDE_LEN / resizeW;
      }
    }

    resizeW = Math.round(resizeW * resizeRatio);
    resizeH = Math.round(resizeH * resizeRatio);

    const processor = new ImageProcessor(initialCanvas);
    const resizedCanvas: Canvas = processor
      .resize({
        width: resizeW,
        height: resizeH,
      })
      .toCanvas();

    processor.destroy();

    const width = Math.ceil(resizeW / 32) * 32;
    const height = Math.ceil(resizeH / 32) * 32;

    const paddedCanvas = createCanvas(width, height);
    const paddedCtx = paddedCanvas.getContext("2d");
    paddedCtx.drawImage(resizedCanvas, 0, 0, resizeW, resizeH);

    const imageDataFromPaddedCanvas = paddedCtx.getImageData(
      0,
      0,
      width,
      height
    );

    const rgbaPaddedData = imageDataFromPaddedCanvas.data;

    const numChannels = 3;
    const tensor = new Float32Array(numChannels * height * width);

    for (let h = 0; h < height; h++) {
      for (let w = 0; w < width; w++) {
        const R_idx_in_rgba = (h * width + w) * 4 + 0;
        const G_idx_in_rgba = (h * width + w) * 4 + 1;
        const B_idx_in_rgba = (h * width + w) * 4 + 2;

        let r = rgbaPaddedData[R_idx_in_rgba]! / 255.0;
        let g = rgbaPaddedData[G_idx_in_rgba]! / 255.0;
        let b = rgbaPaddedData[B_idx_in_rgba]! / 255.0;

        r = (r - DET_MEAN[0]) / DET_STD[0];
        g = (g - DET_MEAN[1]) / DET_STD[1];
        b = (b - DET_MEAN[2]) / DET_STD[2];

        tensor[0 * height * width + h * width + w] = r;
        tensor[1 * height * width + h * width + w] = g;
        tensor[2 * height * width + h * width + w] = b;
      }
    }

    log(
      this.options.verbose,
      `Detection preprocessed: original(${initialCanvas.width}x${
        initialCanvas.height
      }), model_input(${width}x${height}), resize_ratio_to_padded_input: ${resizeRatio.toFixed(
        4
      )}`
    );

    return {
      tensor,
      width,
      height,
      resizeRatio,
      originalWidth: initialCanvas.width,
      originalHeight: initialCanvas.height,
    };
  }

  private async runDetection(
    tensor: Float32Array,
    inputWidth: number,
    inputHeight: number
  ): Promise<Float32Array | null> {
    try {
      const inputOrtTensor = new ort.Tensor("float32", tensor, [
        1,
        3,
        inputHeight,
        inputWidth,
      ]);

      log(this.options.verbose, "Running detection inference...");

      const feeds = { x: inputOrtTensor };
      const results = await this.detectionSession!.run(feeds);
      const outputTensor = results["sigmoid_0.tmp_0"];

      log(this.options.verbose, "Detection inference complete!");

      if (!outputTensor) {
        console.error(
          "Output tensor 'sigmoid_0.tmp_0' not found in det results."
        );
        return null;
      }

      return outputTensor.data as Float32Array;
    } catch (error) {
      console.error(
        "Error in detectText:",
        error instanceof Error ? error.message : String(error)
      );
      throw error;
    }
  }

  private postprocessDetection(
    detection: Float32Array,
    input: PreprocessDetectionResult,
    minBoxAreaOnPadded: number = 20,
    paddingRatio: number = 0.6
  ): Box[] {
    log(this.options.verbose, "Post-processing detection results...");
    const { width, height, resizeRatio, originalWidth, originalHeight } = input;

    const canvas = createCanvas(width, height);
    const ctx = canvas.getContext("2d");

    const imageData = ctx.createImageData(width, height);
    const data = imageData.data;

    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const mapIndex = y * width + x;
        const probability = detection[mapIndex] || 0;

        const grayValue = Math.round(probability * 255);
        const pixelStartIndex = (y * width + x) * 4;

        data[pixelStartIndex + 0] = grayValue;
        data[pixelStartIndex + 1] = grayValue;
        data[pixelStartIndex + 2] = grayValue;
        data[pixelStartIndex + 3] = 255;
      }
    }
    ctx.putImageData(imageData, 0, 0);

    const processor = new ImageProcessor(canvas);
    processor.grayscale().convert({ rtype: cv.CV_8UC1 });

    const contours = new Contours(processor.toMat(), {
      mode: cv.RETR_LIST,
      method: cv.CHAIN_APPROX_SIMPLE,
    });
    const detectedBoxes: Box[] = [];

    contours.iterate((contour) => {
      let rect = contours.getRect(contour);
      if (rect.width * rect.height > minBoxAreaOnPadded) {
        const verticalPadding = Math.round(rect.height * paddingRatio);
        const horizontalPadding = Math.round(rect.height * paddingRatio * 2);

        let paddedRectX = rect.x - horizontalPadding;
        let paddedRectY = rect.y - verticalPadding;
        let paddedRectWidth = rect.width + 2 * horizontalPadding;
        let paddedRectHeight = rect.height + 2 * verticalPadding;

        paddedRectX = Math.max(0, paddedRectX);
        paddedRectY = Math.max(0, paddedRectY);
        paddedRectWidth =
          rect.x - paddedRectX + paddedRectWidth - (rect.x - paddedRectX);
        paddedRectHeight =
          rect.y - paddedRectY + paddedRectHeight - (rect.y - paddedRectY);

        paddedRectWidth = Math.min(width - paddedRectX, paddedRectWidth);
        paddedRectHeight = Math.min(height - paddedRectY, paddedRectHeight);

        const originalX = paddedRectX / resizeRatio;
        const originalY = paddedRectY / resizeRatio;
        const originalWidthBox = paddedRectWidth / resizeRatio;
        const originalHeightBox = paddedRectHeight / resizeRatio;

        const finalX = Math.max(0, Math.round(originalX));
        const finalY = Math.max(0, Math.round(originalY));
        const finalWidth = Math.min(
          originalWidth - finalX,
          Math.round(originalWidthBox)
        );
        const finalHeight = Math.min(
          originalHeight - finalY,
          Math.round(originalHeightBox)
        );

        if (finalWidth > 5 && finalHeight > 5) {
          detectedBoxes.push({
            x: finalX,
            y: finalY,
            width: finalWidth,
            height: finalHeight,
          });
        }
      }
    });

    processor.destroy();
    contours.destroy();

    detectedBoxes.sort((a, b) => {
      if (Math.abs(a.y - b.y) < (a.height + b.height) / 4) {
        return a.x - b.x;
      }
      return a.y - b.y;
    });

    log(
      this.options.verbose,
      `Found ${detectedBoxes.length} potential text boxes.`
    );

    return detectedBoxes;
  }

  private async debugDetectionCanvas(
    detection: Float32Array,
    width: number,
    height: number
  ): Promise<void> {
    const canvas = createCanvas(width, height);
    const ctx = canvas.getContext("2d");

    const imageData = ctx.createImageData(width, height);
    const data = imageData.data;

    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const mapIndex = y * width + x;
        const probability = detection[mapIndex] || 0;

        const grayValue = Math.round(probability * 255);
        const pixelStartIndex = (y * width + x) * 4;

        data[pixelStartIndex + 0] = grayValue;
        data[pixelStartIndex + 1] = grayValue;
        data[pixelStartIndex + 2] = grayValue;
        data[pixelStartIndex + 3] = 255;
      }
    }
    ctx.putImageData(imageData, 0, 0);

    const dir = "out";
    await CanvasToolkit.getInstance().saveImage({
      canvas,
      filename: "detection-debug",
      path: dir,
    });

    log(
      this.options.verbose,
      `Probability map visualized and saved to: ${dir}`
    );
  }

  private async debugDetectedBoxes(image: ArrayBuffer, boxes: Box[]) {
    const canvas = await ImageProcessor.prepareCanvas(image);
    const ctx = canvas.getContext("2d");

    const toolkit = CanvasToolkit.getInstance();

    for (const box of boxes) {
      const { x, y, width, height } = box;
      toolkit.drawLine({
        ctx,
        x,
        y,
        width,
        height,
      });
    }

    const dir = "out";
    await CanvasToolkit.getInstance().saveImage({
      canvas,
      filename: "boxes-debug",
      path: dir,
    });

    log(this.options.verbose, `Boxes visualized and saved to: ${dir}`);
  }

  private static async preprocessRecognnition() {}

  private static async runRecognition() {}

  public async recognize(image: ArrayBuffer): Promise<boolean> {
    const startTime = Date.now();
    await ImageProcessor.initRuntime();

    const input = await this.prepocessDetection(image);
    const detection = await this.runDetection(
      input.tensor,
      input.width,
      input.height
    );

    if (!detection) {
      console.error("Text detection failed (output map is null).");
      return false;
    }

    const detectedBoxes = this.postprocessDetection(detection, input);

    // debug
    await this.debugDetectionCanvas(detection, input.width, input.height);
    await this.debugDetectedBoxes(image, detectedBoxes);

    const speed = Date.now() - startTime;
    log(this.options.verbose, `Operation completed in ${speed} ms`);
    return true;
  }
}

export default PaddleOcrService;
