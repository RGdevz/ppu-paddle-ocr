import { existsSync, readFileSync } from "fs";
import * as ort from "onnxruntime-node";
import * as path from "path";
import {
  DEFAULT_DETECTION_MODEL_PATH,
  DEFAULT_EN_DICT_PATH,
  DEFAULT_RECOGNITION_MODEL_PATH,
} from "./constants";
import type { PaddleServiceOptions } from "./interface";

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

      if (effectiveOptions.verbose) {
        console.log(
          "Loading Detection ONNX model from:",
          resolvedDetectionPath
        );
      }

      const detModelBuffer = readFileSync(resolvedDetectionPath).buffer;
      this.detectionSession = await ort.InferenceSession.create(detModelBuffer);

      if (effectiveOptions.verbose) {
        console.log("Detection ONNX model loaded successfully.");
      }

      if (effectiveOptions.verbose) {
        console.log(
          "Loading Recognition ONNX model from:",
          resolvedRecognitionPath
        );
      }

      const recModelBuffer = readFileSync(resolvedRecognitionPath).buffer;
      this.recognitionSession = await ort.InferenceSession.create(
        recModelBuffer
      );

      if (effectiveOptions.verbose) {
        console.log("Recognition ONNX model loaded successfully.");
      }

      this.charactersDictionary = this.loadCharDictionary(
        resolvedDictionaryPath
      );

      if (effectiveOptions.verbose) {
        console.log(
          `Character dictionary loaded with ${this.charactersDictionary.length} entries.`
        );
      }
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
      if (this.options.verbose) {
        console.error(`Character dictionary not found at: ${filePath}`);
      }
      return [];
    }

    const dictContent = readFileSync(filePath, "utf-8");
    const lines = dictContent.split("\n").map((line) => line.trimEnd());

    if (this.options.verbose) {
      console.log(`Loaded ${lines.length} characters from dictionary.`);
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

  private static async prepocessDetection() {}

  private static async runDetection() {}

  private static async preprocessRecognnition() {}

  private static async runRecognition() {}

  public static async recognize(image: ArrayBuffer): Promise<boolean> {
    return true;
  }
}

export default PaddleOcrService;
