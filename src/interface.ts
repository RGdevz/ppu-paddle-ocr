/**
 * Paths to the OCR model and dictionary files.
 */
export interface ModelPathOptions {
  /**
   * Filesystem path to the text detection model file.
   * Required if not using the library's built‑in default model.
   */
  detection: string;

  /**
   * Filesystem path to the text recognition model file.
   * Required if not using the library's built‑in default model.
   */
  recognition: string;
}

/**
 * Controls verbose output and image dumps for debugging OCR.
 */
export interface DebuggingOptions {
  /**
   * Enable detailed logging of each processing step.
   * @default false
   */
  verbose?: boolean;

  /**
   * Save intermediate image data to disk for inspection.
   * @default false
   */
  debug?: boolean;

  /**
   * Directory where debug images will be written.
   * Relative to the current working directory.
   * @default "out"
   */
  debugFolder?: string;
}

/**
 * Parameters for the text detection preprocessing and filtering stage.
 */
export interface DetectionOptions {
  /**
   * Per-channel mean values used to normalize input pixels [R, G, B].
   * @default [0.485, 0.456, 0.406]
   */
  mean?: [number, number, number];

  /**
   * Per-channel standard deviation values used to normalize input pixels [R, G, B].
   * @default [0.229, 0.224, 0.225]
   */
  stdDeviation?: [number, number, number];

  /**
   * Maximum dimension (longest side) for input images, in pixels.
   * Images above this size will be scaled down, maintaining aspect ratio.
   * @default 960
   */
  maxSideLength?: number;

  /**
   * Padding applied to each detected box vertical as a fraction of its height
   * @default 0.4
   */
  paddingVertical?: number;

  /**
   * Padding applied to each detected box vertical as a fraction of its height
   * @default 0.6
   */
  paddingHorizontal?: number;

  /**
   * Remove detected boxes with area below this threshold, in pixels.
   * @default 20
   */
  minimumAreaThreshold?: number;
}

/**
 * Parameters for the text recognition preprocessing stage.
 */
export interface RecognitionOptions {
  /**
   * Fixed height for input images, in pixels.
   * Models will resize width proportionally.
   * @default 48
   */
  imageHeight?: number;

  /**
   * A list of loaded character dictionary (string) for
   * recognition result decoding.
   */
  charactersDictionary: string[];
}

/**
 * Full configuration for the PaddleOCR service.
 * Combines model file paths with detection, recognition, and debugging parameters.
 */
export interface PaddleOptions {
  /**
   * File paths to the required OCR model components.
   */
  model?: ModelPathOptions;

  /**
   * Controls parameters for text detection.
   */
  detection?: DetectionOptions;

  /**
   * Controls parameters for text recognition.
   */
  recognition?: RecognitionOptions;

  /**
   * Controls logging and image dump behavior for debugging.
   */
  debugging?: DebuggingOptions;
}

/**
 * Simple rectangle representation.
 */
export interface Box {
  /** X-coordinate of the top-left corner. */
  x: number;
  /** Y-coordinate of the top-left corner. */
  y: number;
  /** Width of the box in pixels. */
  width: number;
  /** Height of the box in pixels. */
  height: number;
}
