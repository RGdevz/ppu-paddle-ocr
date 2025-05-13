import path from "path";
import type {
  DebuggingOptions,
  DetectionOptions,
  ModelPathOptions,
  PaddleOptions,
  RecognitionOptions,
} from "./interface";

export const DEFAULT_DETECTION_MODEL_PATH: string = path.join(
  __dirname,
  "models",
  "en_PP-OCRv3_det_infer.onnx"
);
export const DEFAULT_RECOGNITION_MODEL_PATH: string = path.join(
  __dirname,
  "models",
  "en_PP-OCRv3_rec_infer.onnx"
);
export const DEFAULT_EN_CHARS_DICT_PATH: string = path.join(
  __dirname,
  "models",
  "en_dict.txt"
);

export const DEFAULT_MODEL_OPTIONS: ModelPathOptions = {
  detection: DEFAULT_DETECTION_MODEL_PATH,
  recognition: DEFAULT_RECOGNITION_MODEL_PATH,
  dictionary: DEFAULT_EN_CHARS_DICT_PATH,
};

export const DEFAULT_DEBUGGING_OPTIONS: DebuggingOptions = {
  verbose: false,
  debug: false,
  debugFolder: "out",
};

export const DEFAULT_DETECTION_OPTIONS: DetectionOptions = {
  mean: [0.485, 0.456, 0.406],
  stdDeviation: [0.229, 0.224, 0.225],
  maxSideLength: 960,
  minimumAreaThreshold: 20,
  paddingRatio: 0.6,
};

export const DEFAULT_RECOGNITION_OPTIONS: RecognitionOptions = {
  imageHeight: 48,
};

export const DEFAULT_PADDLE_OPTIONS: PaddleOptions = {
  model: DEFAULT_MODEL_OPTIONS,
  detection: DEFAULT_DETECTION_OPTIONS,
  recognition: DEFAULT_RECOGNITION_OPTIONS,
  debugging: DEFAULT_DEBUGGING_OPTIONS,
};
