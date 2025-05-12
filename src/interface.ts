/**
 * Configuration options for PaddleOcrService
 */
export interface PaddleServiceOptions {
  detectionModelPath?: string;
  recognitionModelPath?: string;
  dictionaryPath?: string;
  verbose?: boolean;
}

export interface PaddleUtilsOptions {
  verbose: boolean;
}

export interface PreprocessDetectionResult {
  tensor: Float32Array<ArrayBufferLike>;
  width: number;
  height: number;
  resizeRatio: number;
  originalWidth: number;
  originalHeight: number;
}

export interface Box {
  x: number;
  y: number;
  width: number;
  height: number;
}
