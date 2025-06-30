import { readFileSync } from "fs";
import * as ort from "onnxruntime-node";
import * as path from "path";
import { Canvas, ImageProcessor } from "ppu-ocv";

import merge from "lodash.merge";
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
  confidence: number;
}

export interface FlattenedPaddleOcrResult {
  text: string;
  results: RecognitionResult[];
  confidence: number;
}

/**
 * PaddleOcrService - Provides OCR functionality using PaddleOCR models
 *
 * This service can be used either as a singleton or as separate instances
 * depending on your application needs.
 */
export class PaddleOcrService {
  private static instance: PaddleOcrService | null = null;
  private options: PaddleOptions = DEFAULT_PADDLE_OPTIONS;

  private detectionSession: ort.InferenceSession | null = null;
  private recognitionSession: ort.InferenceSession | null = null;

  /**
   * Create a new PaddleOcrService instance
   * @param options Optional configuration options
   */
  constructor(options?: PaddleOptions) {
    this.options = merge({}, DEFAULT_PADDLE_OPTIONS, options);
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
      const effectiveOptions = merge({}, this.options, overrideOptions);

      const detectionModel = effectiveOptions.model?.detection;
      if (!detectionModel) {
        throw new Error(
          "A detection model path (string) or ArrayBuffer must be provided in the options."
        );
      }

      if (typeof detectionModel === "string") {
        const resolvedDetectionPath = path.resolve(
          process.cwd(),
          detectionModel
        );
        this.log(`Loading Detection ONNX model from: ${resolvedDetectionPath}`);
        const modelBuffer = readFileSync(resolvedDetectionPath).buffer;
        this.detectionSession = await ort.InferenceSession.create(modelBuffer);
      } else {
        this.log(`Loading Detection ONNX model from ArrayBuffer`);
        this.detectionSession =
          await ort.InferenceSession.create(detectionModel);
      }

      this.log(
        `Detection ONNX model loaded successfully\n\tinput: ${this.detectionSession.inputNames}\n\toutput: ${this.detectionSession.outputNames}`
      );

      const recognitionModel = effectiveOptions.model?.recognition;
      if (!recognitionModel) {
        throw new Error(
          "A recognition model path (string) or ArrayBuffer must be provided in the options."
        );
      }

      if (typeof recognitionModel === "string") {
        const resolvedRecognitionPath = path.resolve(
          process.cwd(),
          recognitionModel
        );
        this.log(
          `Loading Recognition ONNX model from: ${resolvedRecognitionPath}`
        );
        const modelBuffer = readFileSync(resolvedRecognitionPath).buffer;
        this.recognitionSession =
          await ort.InferenceSession.create(modelBuffer);
      } else {
        this.log(`Loading Recognition ONNX model from ArrayBuffer`);
        this.recognitionSession =
          await ort.InferenceSession.create(recognitionModel);
      }

      this.log(
        `Recognition ONNX model loaded successfully\n\tinput: ${this.recognitionSession.inputNames}\n\toutput: ${this.recognitionSession.outputNames}`
      );

      const dictionarySource = effectiveOptions.model?.charactersDictionary;
      if (!dictionarySource) {
        throw new Error(
          "A character dictionary path (string) or ArrayBuffer must be provided in the options."
        );
      }

      let dictionaryContent: string;
      if (typeof dictionarySource === "string") {
        const isLikelyContent = dictionarySource.includes("\n");

        if (isLikelyContent) {
          dictionaryContent = dictionarySource;
        } else {
          const resolvedCharactersPath = path.resolve(
            process.cwd(),
            dictionarySource
          );
          this.log(
            `Loading character dictionary from: ${resolvedCharactersPath}`
          );
          dictionaryContent = readFileSync(resolvedCharactersPath, "utf-8");
        }
      } else {
        this.log(`Loading character dictionary from ArrayBuffer`);
        dictionaryContent = Buffer.from(dictionarySource).toString("utf-8");
      }

      const charactersDictionary = dictionaryContent.split("\n");

      if (charactersDictionary.length === 0) {
        throw new Error(
          "Character dictionary is empty or could not be loaded."
        );
      }

      this.options.recognition!.charactersDictionary = charactersDictionary;
      this.log(
        `Character dictionary loaded with ${charactersDictionary.length} entries.`
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
   * Runs OCR and returns a flattened list of recognized text boxes.
   *
   * @param image - The raw image data as an ArrayBuffer or Canvas.
   * @param options - Options object with `flatten` set to `true`.
   * @return A promise that resolves to a flattened result object.
   */
  public recognize(
    image: ArrayBuffer | Canvas,
    options: { flatten: true }
  ): Promise<FlattenedPaddleOcrResult>;

  /**
   * Runs OCR and returns recognized text grouped into lines.
   *
   * @param image - The raw image data as an ArrayBuffer or Canvas.
   * @param options - Optional options object. If `flatten` is `false` or omitted, this structure is returned.
   * @return A promise that resolves to a result object with text lines.
   */
  public recognize(
    image: ArrayBuffer | Canvas,
    options?: { flatten?: false }
  ): Promise<PaddleOcrResult>;

  /**
   * Runs object detection on the provided image buffer, then performs
   * recognition on the detected regions.
   *
   * @param image - The raw image data as an ArrayBuffer or Canvas.
   * @param options - Optional configuration for the recognition output, e.g., `{ flatten: true }`.
   * @return A promise that resolves to the OCR result, either grouped by lines or as a flat list.
   */
  public async recognize(
    image: ArrayBuffer | Canvas,
    options?: { flatten?: boolean }
  ): Promise<PaddleOcrResult | FlattenedPaddleOcrResult> {
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

    const processed = this.processRecognition(recognition);

    if (options?.flatten) {
      return {
        text: processed.text,
        results: recognition,
        confidence: processed.confidence,
      };
    }

    return processed;
  }

  /**
   * Processes raw recognition results to generate the final text,
   * grouped lines, and overall confidence.
   */
  private processRecognition(
    recognition: RecognitionResult[]
  ): PaddleOcrResult {
    const result: PaddleOcrResult = {
      text: "",
      lines: [],
      confidence: 0,
    };

    if (!recognition.length) {
      return result;
    }

    // Calculate overall confidence as the average of all individual confidences
    const totalConfidence = recognition.reduce(
      (sum, r) => sum + r.confidence,
      0
    );
    result.confidence = totalConfidence / recognition.length;

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
