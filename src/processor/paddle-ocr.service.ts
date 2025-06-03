import { readFileSync } from "fs";
import * as ort from "onnxruntime-node";
import * as path from "path";
import { Canvas, ImageProcessor } from "ppu-ocv";

import { DEFAULT_PADDLE_OPTIONS } from "../constants";

import type { PaddleOptions } from "../interface";
import { DetectionService } from "./detection.service";
import {
  RecognitionService,
  type RecognitionResult,
} from "./recognition.service";

export interface PaddleOcrResult {
  text: string;
  lines: RecognitionResult[][];
}

/**
 * PaddleOcrService - Provides OCR functionality using PaddleOCR models
 *
 * This service can be used either as a singleton or as separate instances
 * depending on your application needs.
 */
export class PaddleOcrService {
  private static instance: PaddleOcrService | null = null;
  private options: PaddleOptions;

  private detectionSession: ort.InferenceSession | null = null;
  private recognitionSession: ort.InferenceSession | null = null;

  /**
   * Create a new PaddleOcrService instance
   * @param options Optional configuration options
   */
  constructor(options?: PaddleOptions) {
    this.options = {
      ...DEFAULT_PADDLE_OPTIONS,
      ...options,
    };
  }

  /**
   * Logs a message if verbose debugging is enabled
   */
  private log(message: string): void {
    if (this.options.debugging?.verbose) {
      console.log(`[DetectionService] ${message}`);
    }
  }

  /**
   * Initialize the OCR service by loading models
   * @param overrideOptions Optional parameters to override the constructor options
   */
  public async initialize(
    overrideOptions?: Partial<PaddleOptions>
  ): Promise<void> {
    try {
      const effectiveOptions = {
        ...this.options,
        ...overrideOptions,
      };

      const resolvedDetectionPath = path.resolve(
        process.cwd(),
        effectiveOptions.model!.detection
      );
      const resolvedRecognitionPath = path.resolve(
        process.cwd(),
        effectiveOptions.model!.recognition
      );
      const resolvedCharactersPath = path.resolve(
        process.cwd(),
        effectiveOptions.model!.charactersDictionary
      );

      this.log(`Loading Detection ONNX model from: ${resolvedDetectionPath}`);

      const detModelBuffer = readFileSync(resolvedDetectionPath).buffer;
      this.detectionSession = await ort.InferenceSession.create(detModelBuffer);
      await new Promise((resolve) => setImmediate(resolve));

      this.log(
        `Detection ONNX model loaded successfully\n\tLoading Recognition ONNX model from: ${resolvedRecognitionPath}`
      );

      this.log(
        `input: ${this.detectionSession.inputNames}\n\toutput: ${
          this.detectionSession.outputNames
        }\n\tinputMetadata: ${JSON.stringify(
          this.detectionSession.inputMetadata
        )}\n\toutputMetadata: ${JSON.stringify(
          this.detectionSession.outputMetadata
        )}`
      );

      const recModelBuffer = readFileSync(resolvedRecognitionPath).buffer;
      this.recognitionSession = await ort.InferenceSession.create(
        recModelBuffer
      );
      await new Promise((resolve) => setImmediate(resolve));

      this.log(
        `Recognition ONNX model loaded successfully\n\tinput: ${
          this.recognitionSession.inputNames
        }\n\toutput: ${
          this.recognitionSession.outputNames
        }\n\tinputMetadata: ${JSON.stringify(
          this.recognitionSession.inputMetadata
        )}\n\toutputMetadata: ${JSON.stringify(
          this.recognitionSession.outputMetadata
        )}`
      );

      this.log(`Loading character dictionary from: ${resolvedCharactersPath}`);
      const charactersDictionary = readFileSync(
        resolvedCharactersPath,
        "utf-8"
      ).split("\n");

      if (!charactersDictionary.length) {
        throw new Error(
          `Character dictionary at ${resolvedCharactersPath} is empty or not found.`
        );
      }

      this.options.recognition!.charactersDictionary = charactersDictionary;

      this.log(
        `Character dictionary loaded with ${
          this.options.recognition?.charactersDictionary.length || 0
        } entries.`
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
    options?: PaddleOptions
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
    options: Partial<PaddleOptions>
  ): Promise<PaddleOcrService> {
    if (!PaddleOcrService.instance) {
      PaddleOcrService.instance = new PaddleOcrService(options);
      await PaddleOcrService.instance.initialize();
    } else {
      await PaddleOcrService.instance.destroy();
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
    options?: PaddleOptions
  ): Promise<PaddleOcrService> {
    const instance = new PaddleOcrService(options);
    await instance.initialize();

    return instance;
  }

  /**
   * Runs object detection on the provided image buffer, then performs
   * recognition on the detected regions.
   *
   * @param image - The raw image data as an ArrayBuffer or Canvas.
   * @return A promise that resolves to an array of RecognitionResult objects,
   *          one for each detected and recognized region.
   */
  public async recognize(
    image: ArrayBuffer | Canvas
  ): Promise<PaddleOcrResult> {
    await ImageProcessor.initRuntime();

    const detector = new DetectionService(
      this.detectionSession!,
      this.options.detection,
      this.options.debugging
    );
    const recognitor = new RecognitionService(
      this.recognitionSession!,
      this.options.recognition,
      this.options.debugging
    );

    const detection = await detector.run(image);
    const recognition = await recognitor.run(image, detection);

    return this.groupResult(recognition);
  }

  /**
   * Groups recognition results into lines based on their vertical positions
   * and returns the full text and organized lines efficiently.
   */
  private groupResult(recognition: RecognitionResult[]): PaddleOcrResult {
    const result: PaddleOcrResult = {
      text: "",
      lines: [],
    };

    if (!recognition.length) {
      return result;
    }

    let currentLine: RecognitionResult[] = [recognition[0]];
    let fullText = recognition[0].text;

    let avgHeight = recognition[0].box.height;

    for (let i = 1; i < recognition.length; i++) {
      const current = recognition[i];
      const previous = recognition[i - 1];

      const verticalGap = Math.abs(current.box.y - previous.box.y);
      const threshold = avgHeight * 0.5;

      if (verticalGap <= threshold) {
        currentLine.push(current);
        fullText += ` ${current.text}`;

        avgHeight =
          currentLine.reduce((sum, r) => sum + r.box.height, 0) /
          currentLine.length;
      } else {
        result.lines.push([...currentLine]);

        fullText += `\n${current.text}`;

        currentLine = [current];
        avgHeight = current.box.height;
      }
    }

    if (currentLine.length > 0) {
      result.lines.push([...currentLine]);
    }

    result.text = fullText;
    return result;
  }

  /**
   * Releases the onnx runtime session for both
   * detection and recognition model.
   */
  public async destroy(): Promise<void> {
    await this.detectionSession?.release();
    await this.recognitionSession?.release();
  }
}

export default PaddleOcrService;
