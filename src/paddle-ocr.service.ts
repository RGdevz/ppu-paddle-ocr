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
  REC_IMG_HEIGHT,
} from "./constants";

import type {
  Box,
  PaddleServiceOptions,
  PreprocessDetectionResult,
} from "./interface";
import { PaddleOcrUtils } from "./paddle-utils.service";

/**
 * PaddleOcrService - Provides OCR functionality using PaddleOCR models
 *
 * This service can be used either as a singleton or as separate instances
 * depending on your application needs.
 */
export class PaddleOcrService extends PaddleOcrUtils {
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
    super({ verbose: options?.verbose || false });

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

      this.log(`Loading Detection ONNX model from: ${resolvedDetectionPath}`);

      const detModelBuffer = readFileSync(resolvedDetectionPath).buffer;
      this.detectionSession = await ort.InferenceSession.create(detModelBuffer);

      this.log(
        `Detection ONNX model loaded successfully.\nLoading Recognition ONNX model from: ${resolvedRecognitionPath}`
      );

      const recModelBuffer = readFileSync(resolvedRecognitionPath).buffer;
      this.recognitionSession = await ort.InferenceSession.create(
        recModelBuffer
      );

      this.log(
        `Recognition ONNX model loaded successfully.\nLoading characters dictionary from: ${resolvedDictionaryPath}`
      );

      this.charactersDictionary = this.loadCharDictionary(
        resolvedDictionaryPath
      );

      this.log(
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
    let lines = dictContent.split("\n").map((line) => line.trimEnd());

    if (lines.length > 0 && lines[0] !== "") {
    }

    const spacePlaceholder = "<SPACE>";
    const spacePlaceholderIndex = lines.indexOf(spacePlaceholder);

    if (spacePlaceholderIndex !== -1) {
      lines[spacePlaceholderIndex] = " ";
    } else {
      let foundActualSpace = false;

      for (let i = 1; i < lines.length; i++) {
        if (lines[i] === " ") {
          foundActualSpace = true;
          break;
        }
      }

      if (!foundActualSpace) {
        console.warn(
          `WARNING: Space placeholder '${spacePlaceholder}' not found AND no actual space ' ' found (after index 0) in dictionary. Spaces may not be recognized.`
        );
      }
    }

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

    this.log(
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

      this.log("Running detection inference...");

      const feeds = { x: inputOrtTensor };
      const results = await this.detectionSession!.run(feeds);
      const outputTensor = results["sigmoid_0.tmp_0"];

      this.log("Detection inference complete!");

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
    this.log("Post-processing detection results...");

    const { width, height, resizeRatio, originalWidth, originalHeight } = input;
    const canvas = this.tensorToCanvas(detection, width, height);

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

    this.log(`Found ${detectedBoxes.length} potential text boxes.`);

    return detectedBoxes;
  }

  private tensorToCanvas(
    tensor: Float32Array,
    width: number,
    height: number
  ): Canvas {
    const canvas = createCanvas(width, height);
    const ctx = canvas.getContext("2d");

    const imageData = ctx.createImageData(width, height);
    const data = imageData.data;

    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const mapIndex = y * width + x;
        const probability = tensor[mapIndex] || 0;

        const grayValue = Math.round(probability * 255);
        const pixelStartIndex = (y * width + x) * 4;

        data[pixelStartIndex + 0] = grayValue;
        data[pixelStartIndex + 1] = grayValue;
        data[pixelStartIndex + 2] = grayValue;
        data[pixelStartIndex + 3] = 255;
      }
    }
    ctx.putImageData(imageData, 0, 0);

    return canvas;
  }

  private async debugDetectionCanvas(
    detection: Float32Array,
    width: number,
    height: number
  ): Promise<void> {
    const canvas = this.tensorToCanvas(detection, width, height);

    const dir = "out";
    await CanvasToolkit.getInstance().saveImage({
      canvas,
      filename: "detection-debug",
      path: dir,
    });

    this.log(`Probability map visualized and saved to: ${dir}`);
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

    this.log(`Boxes visualized and saved to: ${dir}`);
  }

  private async preprocessRecognnition(
    cropCanvas: Canvas,
    targetHeight: number = REC_IMG_HEIGHT
  ): Promise<{
    imageTensor: Float32Array;
    tensorWidth: number;
    tensorHeight: number;
  }> {
    const processor = new ImageProcessor(cropCanvas);

    const originalCropWidth = processor.width;
    const originalCropHeight = processor.height;

    if (originalCropHeight === 0 || originalCropWidth === 0) {
      throw new Error(
        `Crop dimensions are zero: ${originalCropWidth}x${originalCropHeight}`
      );
    }

    const aspectRatio = originalCropWidth / originalCropHeight;
    let resizedWidth = Math.round(targetHeight * aspectRatio);
    resizedWidth = Math.max(8, resizedWidth); // Ensure a minimum width (e.g., 8 pixels)

    processor.resize({
      width: resizedWidth,
      height: targetHeight,
    });

    const finalCanvas = processor.toCanvas();
    const finalCtx = finalCanvas.getContext("2d");
    const imageData = finalCtx.getImageData(0, 0, resizedWidth, targetHeight);
    const pixelData = imageData.data; // RGBA format

    // --- MODIFICATION FOR 3 CHANNELS ---
    const numChannels = 3; // Changed from 1 to 3
    const imageTensor = new Float32Array(
      numChannels * targetHeight * resizedWidth
    ); // NCHW with N=1, C=3

    for (let h = 0; h < targetHeight; h++) {
      for (let w = 0; w < resizedWidth; w++) {
        const r_idx = (h * resizedWidth + w) * 4;
        const grayValue = pixelData[r_idx]!; // Assuming R, G, B are same for grayscale from canvas
        const normalizedValue = (grayValue / 255.0 - 0.5) / 0.5;

        // Triplicate the normalized grayscale value for R, G, B channels
        imageTensor[0 * targetHeight * resizedWidth + h * resizedWidth + w] =
          normalizedValue; // Channel 0 (R)
        imageTensor[1 * targetHeight * resizedWidth + h * resizedWidth + w] =
          normalizedValue; // Channel 1 (G)
        imageTensor[2 * targetHeight * resizedWidth + h * resizedWidth + w] =
          normalizedValue; // Channel 2 (B)
      }
    }
    processor.destroy();
    return {
      imageTensor,
      tensorWidth: resizedWidth,
      tensorHeight: targetHeight,
    };
  }

  private async runRecognition() {}

  /**
   * Performs greedy decoding on CTC model output logits.
   *
   * @param recOutputLogits - Raw logits from the recognition model
   * @param sequenceLength - Length of the input sequence
   * @param numClasses - Number of output classes
   * @param charDict - Character dictionary for mapping indices to characters
   * @returns Decoded text string
   */
  private ctcGreedyDecode(
    recOutputLogits: Float32Array,
    sequenceLength: number,
    numClasses: number,
    charDict: string[]
  ): string {
    const BLANK_INDEX = 0;
    const UNK_TOKEN = "<unk>";

    let decodedText = "";
    let lastCharIndex = -1;

    for (let t = 0; t < sequenceLength; t++) {
      const maxProbResult = this.findMaxProbabilityClass(
        recOutputLogits,
        t,
        numClasses
      );
      const predictedClassIndex = maxProbResult.index;

      if (
        predictedClassIndex === BLANK_INDEX ||
        predictedClassIndex === lastCharIndex
      ) {
        lastCharIndex = predictedClassIndex;
        continue;
      }

      if (this.isValidIndex(predictedClassIndex, charDict)) {
        if (
          predictedClassIndex === charDict.length - 1 &&
          charDict[predictedClassIndex] === UNK_TOKEN
        ) {
          // Do nothing for unknown token
        } else {
          const isLastChar =
            predictedClassIndex === this.charactersDictionary.length - 1;
          decodedText += isLastChar ? " " : charDict[predictedClassIndex];
        }
      } else {
        console.warn(
          `Decoded predictedClassIndex ${predictedClassIndex} out of bounds for charDict (length ${charDict.length}) at t=${t}`
        );
      }

      lastCharIndex = predictedClassIndex;
    }

    return decodedText;
  }

  /**
   * Finds the class with maximum probability for a given timestep.
   *
   * @param logits - Raw logits from the recognition model
   * @param timestep - Current timestep
   * @param numClasses - Number of output classes
   * @returns Object containing the max probability value and class index
   */
  private findMaxProbabilityClass(
    logits: Float32Array,
    timestep: number,
    numClasses: number
  ): { value: number; index: number } {
    let maxProb = -Infinity;
    let maxIndex = 0;

    for (let c = 0; c < numClasses; c++) {
      const prob = logits[timestep * numClasses + c];
      if (prob > maxProb) {
        maxProb = prob;
        maxIndex = c;
      }
    }

    return { value: maxProb, index: maxIndex };
  }

  /**
   * Checks if the predicted class index is valid for the character dictionary.
   *
   * @param index - The predicted class index
   * @param charDict - Character dictionary
   * @returns Boolean indicating if the index is valid
   */
  private isValidIndex(index: number, charDict: string[]): boolean {
    return index < charDict.length;
  }

  public async recognize(image: ArrayBuffer): Promise<
    Array<{
      text: string;
      box: Box;
    }>
  > {
    const startTime = Date.now();
    await ImageProcessor.initRuntime();
    const toolkit = CanvasToolkit.getInstance();

    const input = await this.prepocessDetection(image);
    const detection = await this.runDetection(
      input.tensor,
      input.width,
      input.height
    );

    if (!detection) {
      console.error("Text detection failed (output map is null).");
      return [];
    }

    const detectedBoxes = this.postprocessDetection(detection, input);

    // debug
    // await this.debugDetectionCanvas(detection, input.width, input.height);
    // await this.debugDetectedBoxes(image, detectedBoxes);

    const recognizedTexts: Array<{
      text: string;
      box: Box;
    }> = [];

    const cropsDebugPath = "./out/crops";
    for (let i = 0; i < detectedBoxes.length; i++) {
      const box = detectedBoxes[i];
      if (box.width <= 0 || box.height <= 0) {
        console.warn(
          `Skipping invalid box ${i + 1}: w=${box.width}, h=${box.height}`
        );
        continue;
      }

      const sourceCanvasForCrop = await ImageProcessor.prepareCanvas(image);
      const cropCanvas = toolkit.crop({
        bbox: {
          x0: box.x,
          y0: box.y,
          x1: box.x + box.width,
          y1: box.y + box.height,
        },
        canvas: sourceCanvasForCrop,
      });

      // await toolkit.saveImage({
      //   canvas: cropCanvas,
      //   filename: `crop_${String(i).padStart(3, "0")}.png`,
      //   path: cropsDebugPath,
      // });

      try {
        const {
          imageTensor: recInputTensor,
          tensorWidth: recTensorWidth,
          tensorHeight: recTensorHeight,
        } = await this.preprocessRecognnition(cropCanvas);

        const recInputOrtTensor = new ort.Tensor("float32", recInputTensor, [
          1,
          3,
          recTensorHeight,
          recTensorWidth,
        ]);
        const recFeeds = { x: recInputOrtTensor };

        // this.log(
        //   `Running recognition for box ${
        //     i + 1
        //   }, tensor shape: [1,1,${recTensorHeight},${recTensorWidth}]`
        // );
        const recResults = await this.recognitionSession!.run(recFeeds);

        const recOutputNodeName = Object.keys(recResults)[0];
        const recOutputTensor = recResults[recOutputNodeName];

        if (!recOutputTensor) {
          console.error(
            `Recognition output tensor '${recOutputNodeName}' not found for box ${
              i + 1
            }. Available keys: ${Object.keys(recResults)}`
          );
          continue;
        }

        const recOutputData = recOutputTensor.data as Float32Array;
        const recOutputShape = recOutputTensor.dims;

        const sequenceLength = recOutputShape[1];
        const numClasses = recOutputShape[2];

        if (numClasses !== this.charactersDictionary.length) {
          console.warn(
            `Warning: Rec model output numClasses (${numClasses}) does not match charactersDictionary length (${this.charactersDictionary.length}). Decoding might be incorrect.`
          );
        }

        const text = this.ctcGreedyDecode(
          recOutputData,
          sequenceLength,
          numClasses,
          this.charactersDictionary
        );
        // console.log(`  Recognized Box ${i + 1}: "${text}"`);
        recognizedTexts.push({ text, box });
      } catch (e: any) {
        console.error(`Error processing box ${i + 1}: ${e.message}`, e.stack);
      }
    }

    const speed = Date.now() - startTime;
    this.log(`Operation completed in ${speed} ms`);

    return recognizedTexts;
  }
}

export default PaddleOcrService;
