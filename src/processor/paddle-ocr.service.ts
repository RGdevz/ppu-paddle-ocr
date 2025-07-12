import { existsSync, mkdirSync, readFileSync, writeFileSync } from "fs";
import * as ort from "onnxruntime-node";
import * as os from "os";
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

const GITHUB_BASE_URL =
  "https://raw.githubusercontent.com/PT-Perkasa-Pilar-Utama/ppu-paddle-ocr/main/models/";
const CACHE_DIR = path.join(os.homedir(), ".cache", "ppu-paddle-ocr");

/**
 * PaddleOcrService - Provides OCR functionality using PaddleOCR models.
 * To use this service, create an instance and call the `initialize()` method.
 */
export class PaddleOcrService {
  private options: PaddleOptions = DEFAULT_PADDLE_OPTIONS;

  private detectionSession: ort.InferenceSession | null = null;
  private recognitionSession: ort.InferenceSession | null = null;

  /**
   * Creates an instance of PaddleOcrService.
   * @param options - Configuration options for the service.
   */
  public constructor(options?: PaddleOptions) {
    this.options = merge({}, DEFAULT_PADDLE_OPTIONS, options);
  }

  /**
   * Logs a message if verbose debugging is enabled.
   */
  private log(message: string): void {
    if (this.options.debugging?.verbose) {
      console.log(`[PaddleOcrService] ${message}`);
    }
  }

  /**
   * Fetches a resource from a URL and caches it locally.
   * If the resource is already in the cache, it loads it from there.
   */
  private async _fetchAndCache(url: string): Promise<ArrayBuffer> {
    const fileName = path.basename(new URL(url).pathname);
    const cachePath = path.join(CACHE_DIR, fileName);

    if (existsSync(cachePath)) {
      this.log(`Loading cached resource from: ${cachePath}`);
      const buf = readFileSync(cachePath);
      return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
    }

    console.log(
      `[PaddleOcrService] Downloading resource: ${fileName}\n` +
        `                 Cached at: ${CACHE_DIR}`
    );
    this.log(`Fetching resource from URL: ${url}`);

    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Failed to fetch resource from ${url}`);
    }
    if (!response.body) {
      throw new Error("Response body is null or undefined");
    }

    const contentLength = response.headers.get("Content-Length");
    const totalLength = contentLength ? parseInt(contentLength, 10) : 0;
    let receivedLength = 0;
    const chunks: Uint8Array[] = [];

    const reader = response.body.getReader();
    // eslint-disable-next-line no-constant-condition
    while (true) {
      const { done, value } = await reader.read();
      if (done) {
        break;
      }
      chunks.push(value);
      receivedLength += value.length;

      if (totalLength > 0) {
        const percentage = ((receivedLength / totalLength) * 100).toFixed(2);
        process.stdout.write(`\rDownloading... ${percentage}%`);
      }
    }
    process.stdout.write("\n"); // Move to the next line

    const buffer = new Uint8Array(receivedLength);
    let position = 0;
    for (const chunk of chunks) {
      buffer.set(chunk, position);
      position += chunk.length;
    }

    this.log(`Caching resource to: ${cachePath}`);
    if (!existsSync(CACHE_DIR)) {
      mkdirSync(CACHE_DIR, { recursive: true });
    }
    writeFileSync(cachePath, Buffer.from(buffer));

    return buffer.buffer;
  }

  /**
   * Loads a resource from a buffer, a file path, a URL, or a default URL.
   */
  private async _loadResource(
    source: string | ArrayBuffer | undefined,
    defaultUrl: string
  ): Promise<ArrayBuffer> {
    if (source instanceof ArrayBuffer) {
      this.log("Loading resource from ArrayBuffer");
      return source;
    }

    if (typeof source === "string") {
      if (source.startsWith("http")) {
        return this._fetchAndCache(source);
      } else {
        const resolvedPath = path.resolve(process.cwd(), source);
        this.log(`Loading resource from path: ${resolvedPath}`);
        const buf = readFileSync(resolvedPath);
        return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
      }
    }

    return this._fetchAndCache(defaultUrl);
  }

  /**
   * Initializes the OCR service by loading models and dictionary.
   * This method must be called before any OCR operations.
   */
  public async initialize(): Promise<void> {
    try {
      this.log("Initializing PaddleOcrService...");

      // Load detection model
      const detModelBuffer = await this._loadResource(
        this.options.model?.detection,
        `${GITHUB_BASE_URL}PP-OCRv5_mobile_det_infer.onnx`
      );
      this.detectionSession = await ort.InferenceSession.create(detModelBuffer);
      this.options.model!.detection = detModelBuffer;
      this.log(
        `Detection ONNX model loaded successfully\n\tinput: ${this.detectionSession.inputNames}\n\toutput: ${this.detectionSession.outputNames}`
      );

      // Load recognition model
      const recModelBuffer = await this._loadResource(
        this.options.model?.recognition,
        `${GITHUB_BASE_URL}en_PP-OCRv4_mobile_rec_infer.onnx`
      );
      this.recognitionSession =
        await ort.InferenceSession.create(recModelBuffer);
      this.options.model!.recognition = recModelBuffer;
      this.log(
        `Recognition ONNX model loaded successfully\n\tinput: ${this.recognitionSession.inputNames}\n\toutput: ${this.recognitionSession.outputNames}`
      );

      // Load character dictionary
      const dictBuffer = await this._loadResource(
        this.options.model?.charactersDictionary,
        `${GITHUB_BASE_URL}en_dict.txt`
      );
      const dictionaryContent = Buffer.from(dictBuffer).toString("utf-8");
      const charactersDictionary = dictionaryContent.split("\n");

      if (charactersDictionary.length === 0) {
        throw new Error(
          "Character dictionary is empty or could not be loaded."
        );
      }

      this.options.model!.charactersDictionary = dictBuffer;
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
   * Checks if the service has been initialized with models loaded.
   */
  public isInitialized(): boolean {
    return this.detectionSession !== null && this.recognitionSession !== null;
  }

  /**
   * Changes the detection model for the current instance.
   * @param model - The new detection model as a path, URL, or ArrayBuffer.
   */
  public async changeDetectionModel(
    model: ArrayBuffer | string
  ): Promise<void> {
    this.log("Changing detection model...");
    const modelBuffer = await this._loadResource(
      model,
      `${GITHUB_BASE_URL}PP-OCRv5_mobile_det_infer.onnx`
    );

    await this.detectionSession?.release();
    this.detectionSession = await ort.InferenceSession.create(modelBuffer);
    this.options.model!.detection = modelBuffer;
    this.log("Detection model changed successfully.");
  }

  /**
   * Changes the recognition model for the current instance.
   * @param model - The new recognition model as a path, URL, or ArrayBuffer.
   */
  public async changeRecognitionModel(
    model: ArrayBuffer | string
  ): Promise<void> {
    this.log("Changing recognition model...");
    const modelBuffer = await this._loadResource(
      model,
      `${GITHUB_BASE_URL}en_PP-OCRv4_mobile_rec_infer.onnx`
    );

    await this.recognitionSession?.release();
    this.recognitionSession = await ort.InferenceSession.create(modelBuffer);
    this.options.model!.recognition = modelBuffer;
    this.log("Recognition model changed successfully.");
  }

  /**
   * Changes the text dictionary for the current instance.
   * @param dictionary - The new dictionary as a path, URL, ArrayBuffer, or string content.
   */
  public async changeTextDictionary(
    dictionary: ArrayBuffer | string
  ): Promise<void> {
    this.log("Changing text dictionary...");
    const dictBuffer = await this._loadResource(
      dictionary,
      `${GITHUB_BASE_URL}en_dict.txt`
    );

    const dictionaryContent = Buffer.from(dictBuffer).toString("utf-8");
    const charactersDictionary = dictionaryContent.split("\n");

    if (charactersDictionary.length === 0) {
      throw new Error("Character dictionary is empty or could not be loaded.");
    }

    this.options.model!.charactersDictionary = dictBuffer;
    this.options.recognition!.charactersDictionary = charactersDictionary;
    this.log(
      `Character dictionary changed successfully with ${charactersDictionary.length} entries.`
    );
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
    options: { flatten: true; dictionary?: string | ArrayBuffer }
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
    options?: { flatten?: false; dictionary?: string | ArrayBuffer }
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
    options?: { flatten?: boolean; dictionary?: string | ArrayBuffer }
  ): Promise<PaddleOcrResult | FlattenedPaddleOcrResult> {
    if (!this.isInitialized()) {
      throw new Error(
        "PaddleOcrService is not initialized. Call initialize() first."
      );
    }
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

    let charactersDictionary: string[] | undefined;
    if (options?.dictionary) {
      const dictBuffer = await this._loadResource(options.dictionary, "");
      const dictionaryContent = Buffer.from(dictBuffer).toString("utf-8");
      charactersDictionary = dictionaryContent.split("\n");

      if (charactersDictionary.length === 0) {
        throw new Error(
          "Custom character dictionary is empty or could not be loaded."
        );
      }
    }

    const detection = await detector.run(image);
    const recognition = await recognitor.run(
      image,
      detection,
      charactersDictionary
    );

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
   * Runs deskew algorithm on the provided image buffer | canvas
   *
   * @param image - The raw image data as an ArrayBuffer or Canvas.
   * @return A promise that resolves deskewed image as Canvas
   */
  public async deskewImage(image: ArrayBuffer | Canvas): Promise<Canvas> {
    if (!this.isInitialized()) {
      throw new Error(
        "PaddleOcrService is not initialized. Call initialize() first."
      );
    }
    await ImageProcessor.initRuntime();

    const detector = new DetectionService(
      this.detectionSession!,
      this.options.detection,
      this.options.debugging
    );

    const detection = await detector.deskew(image);
    return detection;
  }

  /**
   * Releases the onnx runtime session for both
   * detection and recognition model.
   */
  public async destroy(): Promise<void> {
    await this.detectionSession?.release();
    await this.recognitionSession?.release();
    this.detectionSession = null;
    this.recognitionSession = null;
  }
}

export default PaddleOcrService;
