import path from "path";
import type {
  DebuggingOptions,
  DetectionOptions,
  ModelPathOptions,
  PaddleOptions,
  RecognitionOptions,
} from "./interface";

import { fileURLToPath } from "url";

const getCurrentDirPath = () => {
  if (typeof import.meta !== "undefined" && import.meta.url) {
    const __filename = fileURLToPath(import.meta.url);
    return path.dirname(__filename);
  }
  return __dirname;
};

const PACKAGE_ROOT = getCurrentDirPath();

export const DEFAULT_DETECTION_MODEL_PATH: string = path.join(
  PACKAGE_ROOT,
  "models",
  "en_PP-OCRv3_det_infer.onnx"
);

export const DEFAULT_RECOGNITION_MODEL_PATH: string = path.join(
  PACKAGE_ROOT,
  "models",
  "en_PP-OCRv3_rec_infer.onnx"
);

export const DEFAULT_CHARACTERS_DICTIONARY: string[] = [
  "",
  "0",
  "1",
  "2",
  "3",
  "4",
  "5",
  "6",
  "7",
  "8",
  "9",
  ":",
  ";",
  "<",
  "=",
  ">",
  "?",
  "@",
  "A",
  "B",
  "C",
  "D",
  "E",
  "F",
  "G",
  "H",
  "I",
  "J",
  "K",
  "L",
  "M",
  "N",
  "O",
  "P",
  "Q",
  "R",
  "S",
  "T",
  "U",
  "V",
  "W",
  "X",
  "Y",
  "Z",
  "[",
  "\\",
  "]",
  "^",
  "_",
  "`",
  "a",
  "b",
  "c",
  "d",
  "e",
  "f",
  "g",
  "h",
  "i",
  "j",
  "k",
  "l",
  "m",
  "n",
  "o",
  "p",
  "q",
  "r",
  "s",
  "t",
  "u",
  "v",
  "w",
  "x",
  "y",
  "z",
  "{",
  "|",
  "}",
  "~",
  "!",
  '"',
  "#",
  "$",
  "%",
  "&",
  "'",
  "(",
  ")",
  "*",
  "+",
  ",",
  "-",
  ".",
  "/",
  " ",
  "",
];

export const DEFAULT_MODEL_OPTIONS: ModelPathOptions = {
  detection: DEFAULT_DETECTION_MODEL_PATH,
  recognition: DEFAULT_RECOGNITION_MODEL_PATH,
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
  paddingVertical: 0.4,
  paddingHorizontal: 0.6,
};

export const DEFAULT_RECOGNITION_OPTIONS: RecognitionOptions = {
  imageHeight: 48,
  charactersDictionary: DEFAULT_CHARACTERS_DICTIONARY,
};

export const DEFAULT_PADDLE_OPTIONS: PaddleOptions = {
  model: DEFAULT_MODEL_OPTIONS,
  detection: DEFAULT_DETECTION_OPTIONS,
  recognition: DEFAULT_RECOGNITION_OPTIONS,
  debugging: DEFAULT_DEBUGGING_OPTIONS,
};
