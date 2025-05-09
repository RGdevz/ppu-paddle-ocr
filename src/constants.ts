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

export const MAX_SIDE_LEN = 960;
export const DET_MEAN: [number, number, number] = [0.485, 0.456, 0.406];
export const DET_STD: [number, number, number] = [0.229, 0.224, 0.225];
