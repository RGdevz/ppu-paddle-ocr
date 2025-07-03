import type {
  DebuggingOptions,
  DetectionOptions,
  ModelPathOptions,
  PaddleOptions,
  RecognitionOptions,
} from "./interface";

import detModel from "./models/PP-OCRv5_mobile_det_infer.onnx" with { type: "file", embed: "false" };
import recModel from "./models/en_PP-OCRv4_mobile_rec_infer.onnx" with { type: "file", embed: "false" };
import dict from "./models/en_dict.txt" with { type: "file", embed: "false" };

export const DEFAULT_DETECTION_MODEL_PATH: string = detModel;
export const DEFAULT_RECOGNITION_MODEL_PATH: string = recModel;
export const DEFAULT_CHARACTERS_DICTIONARY_PATH: string = dict;

export const DEFAULT_MODEL_OPTIONS: ModelPathOptions = {
  detection: DEFAULT_DETECTION_MODEL_PATH,
  recognition: DEFAULT_RECOGNITION_MODEL_PATH,
  charactersDictionary: DEFAULT_CHARACTERS_DICTIONARY_PATH,
};

export const DEFAULT_DEBUGGING_OPTIONS: DebuggingOptions = {
  verbose: false,
  debug: false,
  debugFolder: "out",
};

export const DEFAULT_DETECTION_OPTIONS: DetectionOptions = {
  autoDeskew: true,
  mean: [0.485, 0.456, 0.406],
  stdDeviation: [0.229, 0.224, 0.225],
  maxSideLength: 960,
  minimumAreaThreshold: 20,
  paddingVertical: 0.4,
  paddingHorizontal: 0.6,
};

export const DEFAULT_RECOGNITION_OPTIONS: RecognitionOptions = {
  imageHeight: 48,
  charactersDictionary: [],
};

export const DEFAULT_PADDLE_OPTIONS: PaddleOptions = {
  model: DEFAULT_MODEL_OPTIONS,
  detection: DEFAULT_DETECTION_OPTIONS,
  recognition: DEFAULT_RECOGNITION_OPTIONS,
  debugging: DEFAULT_DEBUGGING_OPTIONS,
};
