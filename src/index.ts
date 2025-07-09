export {
  PaddleOcrService,
  type FlattenedPaddleOcrResult,
  type PaddleOcrResult,
} from "./processor/paddle-ocr.service";

export type {
  Box,
  DebuggingOptions,
  DetectionOptions,
  ModelPathOptions,
  PaddleOptions,
  RecognitionOptions,
} from "./interface";

export {
  DetectionService,
  type PreprocessDetectionResult,
} from "./processor/detection.service";

export {
  RecognitionService,
  type RecognitionResult,
} from "./processor/recognition.service";

export {
  DEFAULT_DEBUGGING_OPTIONS,
  DEFAULT_DETECTION_OPTIONS,
  DEFAULT_PADDLE_OPTIONS,
  DEFAULT_RECOGNITION_OPTIONS,
} from "./constants";
