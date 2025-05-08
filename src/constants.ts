import path from "path";

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
export const DEFAULT_EN_DICT_PATH: string = path.join(
  __dirname,
  "models",
  "en_dict.txt"
);
