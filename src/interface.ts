/**
 * Configuration options for PaddleOcrService
 */
export interface PaddleServiceOptions {
  detectionModelPath?: string;
  recognitionModelPath?: string;
  dictionaryPath?: string;
  verbose?: boolean;
}
