import * as ort from "onnxruntime-node";
import { Canvas, Contours, createCanvas, ImageProcessor } from "ppu-ocv";
import type { Box, DebuggingOptions, DetectionOptions } from "../interface";

export interface PreprocessDetectionResult {
  tensor: Float32Array<ArrayBufferLike>;
  width: number;
  height: number;
  resizeRatio: number;
  originalWidth: number;
  originalHeight: number;
}

export class DetectionService {
  private options: DetectionOptions;
  private debugging: DebuggingOptions;
  private session: ort.InferenceSession;

  constructor(
    session: ort.InferenceSession,
    options?: DetectionOptions,
    debugging?: DebuggingOptions
  ) {
    this.session = session;

    this.options = {
      ...options,
    };

    this.debugging = {
      ...debugging,
    };
  }

  protected log(text: string): void {
    if (this.debugging.verbose) {
      console.log(text);
    }
  }

  async run(image: ArrayBuffer): Promise<Box[]> {
    const input = await this.prepocessDetection(image);
    const detection = await this.runDetection(
      input.tensor,
      input.width,
      input.height
    );

    if (!detection) {
      console.error("Text detection failed (output map is null).");
      return [];
    }

    const detectedBoxes = this.postprocessDetection(detection, input);
    return detectedBoxes;
  }

  private async prepocessDetection(
    image: ArrayBuffer
  ): Promise<PreprocessDetectionResult> {
    const initialCanvas = await ImageProcessor.prepareCanvas(image);
    const MAX_SIDE_LEN = this.options.maxSideLength!;

    let resizeW = initialCanvas.width;
    let resizeH = initialCanvas.height;
    let resizeRatio = 1.0;

    if (Math.max(resizeH, resizeW) > MAX_SIDE_LEN) {
      if (resizeH > resizeW) {
        resizeRatio = MAX_SIDE_LEN / resizeH;
      } else {
        resizeRatio = MAX_SIDE_LEN / resizeW;
      }
    }

    resizeW = Math.round(resizeW * resizeRatio);
    resizeH = Math.round(resizeH * resizeRatio);

    const processor = new ImageProcessor(initialCanvas);
    const resizedCanvas: Canvas = processor
      .resize({
        width: resizeW,
        height: resizeH,
      })
      .toCanvas();

    processor.destroy();

    const width = Math.ceil(resizeW / 32) * 32;
    const height = Math.ceil(resizeH / 32) * 32;

    const paddedCanvas = createCanvas(width, height);
    const paddedCtx = paddedCanvas.getContext("2d");
    paddedCtx.drawImage(resizedCanvas, 0, 0, resizeW, resizeH);

    const imageDataFromPaddedCanvas = paddedCtx.getImageData(
      0,
      0,
      width,
      height
    );

    const rgbaPaddedData = imageDataFromPaddedCanvas.data;

    const numChannels = 3;
    const tensor = new Float32Array(numChannels * height * width);
    const DET_MEAN = this.options.mean!;
    const DET_STD = this.options.stdDeviation!;

    for (let h = 0; h < height; h++) {
      for (let w = 0; w < width; w++) {
        const R_idx_in_rgba = (h * width + w) * 4 + 0;
        const G_idx_in_rgba = (h * width + w) * 4 + 1;
        const B_idx_in_rgba = (h * width + w) * 4 + 2;

        let r = rgbaPaddedData[R_idx_in_rgba]! / 255.0;
        let g = rgbaPaddedData[G_idx_in_rgba]! / 255.0;
        let b = rgbaPaddedData[B_idx_in_rgba]! / 255.0;

        r = (r - DET_MEAN[0]) / DET_STD[0];
        g = (g - DET_MEAN[1]) / DET_STD[1];
        b = (b - DET_MEAN[2]) / DET_STD[2];

        tensor[0 * height * width + h * width + w] = r;
        tensor[1 * height * width + h * width + w] = g;
        tensor[2 * height * width + h * width + w] = b;
      }
    }

    this.log(
      `Detection preprocessed: original(${initialCanvas.width}x${
        initialCanvas.height
      }), model_input(${width}x${height}), resize_ratio_to_padded_input: ${resizeRatio.toFixed(
        4
      )}`
    );

    return {
      tensor,
      width,
      height,
      resizeRatio,
      originalWidth: initialCanvas.width,
      originalHeight: initialCanvas.height,
    };
  }

  private async runDetection(
    tensor: Float32Array,
    inputWidth: number,
    inputHeight: number
  ): Promise<Float32Array | null> {
    try {
      const inputOrtTensor = new ort.Tensor("float32", tensor, [
        1,
        3,
        inputHeight,
        inputWidth,
      ]);

      this.log("Running detection inference...");

      const feeds = { x: inputOrtTensor };
      const results = await this.session.run(feeds);
      const outputTensor = results["sigmoid_0.tmp_0"];

      this.log("Detection inference complete!");

      if (!outputTensor) {
        console.error(
          "Output tensor 'sigmoid_0.tmp_0' not found in det results."
        );
        return null;
      }

      return outputTensor.data as Float32Array;
    } catch (error) {
      console.error(
        "Error in detectText:",
        error instanceof Error ? error.message : String(error)
      );
      throw error;
    }
  }

  private tensorToCanvas(
    tensor: Float32Array,
    width: number,
    height: number
  ): Canvas {
    const canvas = createCanvas(width, height);
    const ctx = canvas.getContext("2d");

    const imageData = ctx.createImageData(width, height);
    const data = imageData.data;

    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const mapIndex = y * width + x;
        const probability = tensor[mapIndex] || 0;

        const grayValue = Math.round(probability * 255);
        const pixelStartIndex = (y * width + x) * 4;

        data[pixelStartIndex + 0] = grayValue;
        data[pixelStartIndex + 1] = grayValue;
        data[pixelStartIndex + 2] = grayValue;
        data[pixelStartIndex + 3] = 255;
      }
    }
    ctx.putImageData(imageData, 0, 0);

    return canvas;
  }

  private postprocessDetection(
    detection: Float32Array,
    input: PreprocessDetectionResult,
    minBoxAreaOnPadded: number = 20,
    paddingRatio: number = 0.6
  ): Box[] {
    this.log("Post-processing detection results...");

    const { width, height, resizeRatio, originalWidth, originalHeight } = input;
    const canvas = this.tensorToCanvas(detection, width, height);

    const processor = new ImageProcessor(canvas);
    processor.grayscale().convert({ rtype: cv.CV_8UC1 });

    const contours = new Contours(processor.toMat(), {
      mode: cv.RETR_LIST,
      method: cv.CHAIN_APPROX_SIMPLE,
    });
    const detectedBoxes: Box[] = [];

    contours.iterate((contour) => {
      let rect = contours.getRect(contour);
      if (rect.width * rect.height > minBoxAreaOnPadded) {
        const verticalPadding = Math.round(rect.height * paddingRatio);
        const horizontalPadding = Math.round(rect.height * paddingRatio * 2);

        let paddedRectX = rect.x - horizontalPadding;
        let paddedRectY = rect.y - verticalPadding;
        let paddedRectWidth = rect.width + 2 * horizontalPadding;
        let paddedRectHeight = rect.height + 2 * verticalPadding;

        paddedRectX = Math.max(0, paddedRectX);
        paddedRectY = Math.max(0, paddedRectY);
        paddedRectWidth =
          rect.x - paddedRectX + paddedRectWidth - (rect.x - paddedRectX);
        paddedRectHeight =
          rect.y - paddedRectY + paddedRectHeight - (rect.y - paddedRectY);

        paddedRectWidth = Math.min(width - paddedRectX, paddedRectWidth);
        paddedRectHeight = Math.min(height - paddedRectY, paddedRectHeight);

        const originalX = paddedRectX / resizeRatio;
        const originalY = paddedRectY / resizeRatio;
        const originalWidthBox = paddedRectWidth / resizeRatio;
        const originalHeightBox = paddedRectHeight / resizeRatio;

        const finalX = Math.max(0, Math.round(originalX));
        const finalY = Math.max(0, Math.round(originalY));
        const finalWidth = Math.min(
          originalWidth - finalX,
          Math.round(originalWidthBox)
        );
        const finalHeight = Math.min(
          originalHeight - finalY,
          Math.round(originalHeightBox)
        );

        if (finalWidth > 5 && finalHeight > 5) {
          detectedBoxes.push({
            x: finalX,
            y: finalY,
            width: finalWidth,
            height: finalHeight,
          });
        }
      }
    });

    processor.destroy();
    contours.destroy();

    detectedBoxes.sort((a, b) => {
      if (Math.abs(a.y - b.y) < (a.height + b.height) / 4) {
        return a.x - b.x;
      }
      return a.y - b.y;
    });

    this.log(`Found ${detectedBoxes.length} potential text boxes.`);

    return detectedBoxes;
  }

  async destroy(): Promise<void> {
    await this.session.release();
  }
}
