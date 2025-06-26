import * as ort from "onnxruntime-node";
import {
  Canvas,
  CanvasToolkit,
  Contours,
  createCanvas,
  cv,
  ImageProcessor,
} from "ppu-ocv";
import {
  DEFAULT_DEBUGGING_OPTIONS,
  DEFAULT_DETECTION_OPTIONS,
} from "../constants";
import type { Box, DebuggingOptions, DetectionOptions } from "../interface";

/**
 * Result of preprocessing an image for text detection
 */
export interface PreprocessDetectionResult {
  tensor: Float32Array;
  width: number;
  height: number;
  resizeRatio: number;
  originalWidth: number;
  originalHeight: number;
}

/**
 * Service for detecting text regions in images
 */
export class DetectionService {
  private readonly options: DetectionOptions;
  private readonly debugging: DebuggingOptions;
  private readonly session: ort.InferenceSession;

  private static readonly NUM_CHANNELS = 3;

  constructor(
    session: ort.InferenceSession,
    options: Partial<DetectionOptions> = {},
    debugging: Partial<DebuggingOptions> = {}
  ) {
    this.session = session;

    this.options = {
      ...DEFAULT_DETECTION_OPTIONS,
      ...options,
    };

    this.debugging = {
      ...DEFAULT_DEBUGGING_OPTIONS,
      ...debugging,
    };
  }

  /**
   * Logs a message if verbose debugging is enabled
   */
  private log(message: string): void {
    if (this.debugging.verbose) {
      console.log(`[DetectionService] ${message}`);
    }
  }

  /**
   * Main method to run text detection on an image
   * @param image ArrayBuffer of the image or Canvas
   */
  async run(image: ArrayBuffer | Canvas): Promise<Box[]> {
    this.log("Starting text detection process");

    try {
      let canvasToProcess =
        image instanceof Canvas
          ? image
          : await ImageProcessor.prepareCanvas(image);

      if (this.options.autoDeskew) {
        this.log(
          "Auto-deskew enabled. Performing initial pass for angle detection."
        );
        const angle = await this.calculateSkewAngle(canvasToProcess);

        this.log(
          `Detected skew angle: ${angle.toFixed(
            2
          )}°. Rotating image at ${-angle.toFixed(2)}° (to ${
            -angle > 1 ? "left" : "right"
          })...`
        );

        const processor = new ImageProcessor(canvasToProcess);
        const rotatedCanvas = processor.rotate({ angle }).toCanvas();
        processor.destroy();

        canvasToProcess = rotatedCanvas;

        if (this.debugging.debug) {
          await CanvasToolkit.getInstance().saveImage({
            canvas: canvasToProcess,
            filename: "deskewed-image-debug",
            path: this.debugging.debugFolder!,
          });
        }
      }

      const input = await this.preprocessDetection(canvasToProcess);
      const detection = await this.runInference(
        input.tensor,
        input.width,
        input.height
      );

      if (!detection) {
        console.error("Text detection failed (output tensor is null)");
        return [];
      }

      const detectedBoxes = this.postprocessDetection(detection, input);

      if (this.debugging.debug) {
        await this.debugDetectionCanvas(detection, input.width, input.height);
        await this.debugDetectedBoxes(canvasToProcess, detectedBoxes);
      }

      this.log(`Detected ${detectedBoxes.length} text boxes in image`);

      return detectedBoxes;
    } catch (error) {
      console.error(
        "Error during text detection:",
        error instanceof Error ? error.message : String(error)
      );
      return [];
    }
  }

  /**
   * Runs a lightweight detection pass to determine the average text skew angle.
   * Uses multiple methods to robustly calculate skew from all detected text regions.
   * @param canvas The input canvas.
   * @returns The calculated skew angle in degrees.
   */
  private async calculateSkewAngle(canvas: Canvas): Promise<number> {
    const input = await this.preprocessDetection(canvas);
    const detection = await this.runInference(
      input.tensor,
      input.width,
      input.height
    );

    if (!detection) {
      this.log("Skew calculation failed: no detection output from model.");
      return 0;
    }

    const { width, height } = input;
    const probabilityMapCanvas = this.tensorToCanvas(detection, width, height);

    if (this.debugging.debug) {
      await CanvasToolkit.getInstance().saveImage({
        canvas: probabilityMapCanvas,
        filename: "deskew-probability-map-debug.png",
        path: this.debugging.debugFolder!,
      });
    }

    const processor = new ImageProcessor(probabilityMapCanvas);
    const mat = processor.grayscale().threshold().toMat();

    const contours = new Contours(mat, {
      mode: cv.RETR_LIST,
      method: cv.CHAIN_APPROX_SIMPLE,
    });

    processor.destroy();

    const minAngle = -20;
    const maxAngle = 20;
    const minArea = this.options.minimumAreaThreshold || 20;

    const textRegions: Array<{
      rect: { x: number; y: number; width: number; height: number };
      contour: any;
      area: number;
      aspectRatio: number;
    }> = [];

    contours.iterate((contour) => {
      const rect = contours.getRect(contour);
      const area = rect.width * rect.height;

      if (area < minArea) return;

      const aspectRatio = rect.width / rect.height;

      if (aspectRatio > 0.2 && aspectRatio < 10) {
        textRegions.push({
          rect,
          contour,
          area,
          aspectRatio,
        });
      }
    });

    if (textRegions.length === 0) {
      this.log("No valid text regions found for skew calculation.");
      contours.destroy();
      return 0;
    }

    this.log(`Found ${textRegions.length} text regions for skew analysis.`);

    // Method 1: Analyze angles using minimum area rectangles
    const minRectAngles = this.calculateMinRectAngles(textRegions, contours);

    // Method 2: Analyze angles using line fitting on text baselines
    const baselineAngles = this.calculateBaselineAngles(textRegions);

    // Method 3: Analyze angles using Hough transform for dominant lines
    const houghAngles = this.calculateHoughAngles(mat, minAngle, maxAngle);

    contours.destroy();

    // Combine all angle measurements with weights
    const allAngles: Array<{ angle: number; weight: number; method: string }> =
      [
        ...minRectAngles.map((a) => ({ ...a, method: "minRect" })),
        ...baselineAngles.map((a) => ({ ...a, method: "baseline" })),
        ...houghAngles.map((a) => ({ ...a, method: "hough" })),
      ];

    if (allAngles.length === 0) {
      this.log("No angles detected from any method.");
      return 0;
    }

    // Calculate weighted consensus angle
    const consensusAngle = this.calculateConsensusAngle(
      allAngles,
      minAngle,
      maxAngle
    );

    this.log(
      `Calculated skew angle: ${consensusAngle.toFixed(3)}° (from ${
        allAngles.length
      } measurements)`
    );

    return consensusAngle;
  }

  /**
   * Calculate angles using minimum area rectangles around text regions
   */
  private calculateMinRectAngles(
    textRegions: any[],
    contours: Contours
  ): Array<{ angle: number; weight: number }> {
    const angles: Array<{ angle: number; weight: number }> = [];

    for (const region of textRegions) {
      try {
        const minRect = cv.minAreaRect(region.contour);
        if (!minRect) continue;

        let angle = minRect.angle;

        // Normalize angle to [-45, 45] range
        if (angle > 45) {
          angle -= 90;
        } else if (angle < -45) {
          angle += 90;
        }

        // Weight by area and aspect ratio (prefer larger, more text-like regions)
        const areaWeight = Math.log(region.area + 1);
        const aspectWeight =
          Math.min(region.aspectRatio, 1 / region.aspectRatio) * 2;
        const weight = areaWeight * aspectWeight;

        angles.push({ angle, weight });
      } catch (error) {
        // Skip regions that cause errors
        continue;
      }
    }

    return angles;
  }

  /**
   * Calculate angles by analyzing text baselines using contour points
   */
  private calculateBaselineAngles(
    textRegions: any[]
  ): Array<{ angle: number; weight: number }> {
    const angles: Array<{ angle: number; weight: number }> = [];

    for (const region of textRegions) {
      try {
        // Get contour points
        const points = region.contour.data32S;
        if (!points || points.length < 8) continue; // Need at least 4 points (x,y pairs)

        // Extract bottom edge points for baseline analysis
        const bottomPoints: Array<{ x: number; y: number }> = [];

        for (let i = 0; i < points.length; i += 2) {
          const x = points[i];
          const y = points[i + 1];
          if (x !== undefined && y !== undefined) {
            bottomPoints.push({ x, y });
          }
        }

        if (bottomPoints.length < 3) continue;

        // Sort points by x-coordinate
        bottomPoints.sort((a, b) => a.x - b.x);

        // Find bottom edge by selecting points with maximum y-values in segments
        const segments = 3;
        const segmentSize = Math.floor(bottomPoints.length / segments);
        const baselinePoints: Array<{ x: number; y: number }> = [];

        for (let seg = 0; seg < segments; seg++) {
          const start = seg * segmentSize;
          const end =
            seg === segments - 1
              ? bottomPoints.length
              : (seg + 1) * segmentSize;
          const segmentPoints = bottomPoints.slice(start, end);

          if (segmentPoints.length > 0) {
            const maxYPoint = segmentPoints.reduce((max, point) =>
              point.y > max.y ? point : max
            );
            baselinePoints.push(maxYPoint);
          }
        }

        if (baselinePoints.length >= 2) {
          // Calculate angle using linear regression
          const angle = this.calculateLineAngle(baselinePoints);
          const weight =
            region.area * Math.min(region.aspectRatio, 1 / region.aspectRatio);

          angles.push({ angle, weight });
        }
      } catch (error) {
        continue;
      }
    }

    return angles;
  }

  /**
   * Calculate angles using Hough line transform for dominant lines
   */
  private calculateHoughAngles(
    mat: any,
    minAngle: number,
    maxAngle: number
  ): Array<{ angle: number; weight: number }> {
    const angles: Array<{ angle: number; weight: number }> = [];

    try {
      // Apply morphological operations to connect text components
      const kernel = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(3, 1));
      const morphed = new cv.Mat();
      cv.morphologyEx(mat, morphed, cv.MORPH_CLOSE, kernel);

      // Apply Hough line transform
      const lines = new cv.Mat();
      cv.HoughLinesP(
        morphed,
        lines,
        1, // rho resolution
        Math.PI / 180, // theta resolution (1 degree)
        30, // threshold
        50, // min line length
        10 // max line gap
      );

      // Extract angles from detected lines
      for (let i = 0; i < lines.rows; i++) {
        const line = lines.data32S.subarray(i * 4, (i + 1) * 4);
        const [x1, y1, x2, y2] = line;

        if (
          x1 !== undefined &&
          y1 !== undefined &&
          x2 !== undefined &&
          y2 !== undefined
        ) {
          const dx = x2 - x1;
          const dy = y2 - y1;

          if (Math.abs(dx) > 1) {
            // Avoid vertical lines
            let angle = (Math.atan2(dy, dx) * 180) / Math.PI;

            // Normalize to expected range
            if (angle > 45) angle -= 90;
            if (angle < -45) angle += 90;

            if (angle >= minAngle && angle <= maxAngle) {
              const lineLength = Math.sqrt(dx * dx + dy * dy);
              angles.push({ angle, weight: lineLength });
            }
          }
        }
      }

      morphed.delete();
      lines.delete();
      kernel.delete();
    } catch (error) {
      this.log("Hough transform failed, skipping this method.");
    }

    return angles;
  }

  /**
   * Calculate angle from a set of points using linear regression
   */
  private calculateLineAngle(points: Array<{ x: number; y: number }>): number {
    if (points.length < 2) return 0;

    const n = points.length;
    const sumX = points.reduce((sum, p) => sum + p.x, 0);
    const sumY = points.reduce((sum, p) => sum + p.y, 0);
    const sumXY = points.reduce((sum, p) => sum + p.x * p.y, 0);
    const sumXX = points.reduce((sum, p) => sum + p.x * p.x, 0);

    const denominator = n * sumXX - sumX * sumX;

    if (Math.abs(denominator) < 1e-10) return 0;

    const slope = (n * sumXY - sumX * sumY) / denominator;
    let angle = (Math.atan(slope) * 180) / Math.PI;

    // Normalize to [-45, 45] range
    if (angle > 45) angle -= 90;
    if (angle < -45) angle += 90;

    return angle;
  }

  /**
   * Calculate consensus angle from multiple measurements using robust statistics
   */
  private calculateConsensusAngle(
    angles: Array<{ angle: number; weight: number; method: string }>,
    minAngle: number,
    maxAngle: number
  ): number {
    if (angles.length === 0) return 0;

    // Filter out outliers using IQR method
    const sortedAngles = [...angles].sort((a, b) => a.angle - b.angle);
    const q1Index = Math.floor(sortedAngles.length * 0.25);
    const q3Index = Math.floor(sortedAngles.length * 0.75);
    const q1 = sortedAngles[q1Index]?.angle || 0;
    const q3 = sortedAngles[q3Index]?.angle || 0;
    const iqr = q3 - q1;
    const lowerBound = q1 - 1.5 * iqr;
    const upperBound = q3 + 1.5 * iqr;

    const filteredAngles = angles.filter(
      (a) =>
        a.angle >= lowerBound &&
        a.angle <= upperBound &&
        a.angle >= minAngle &&
        a.angle <= maxAngle
    );

    if (filteredAngles.length === 0) {
      this.log(
        "All angles filtered out as outliers, using median of original set."
      );
      const medianIndex = Math.floor(sortedAngles.length / 2);
      return sortedAngles[medianIndex]?.angle || 0;
    }

    // Calculate weighted average
    const totalWeight = filteredAngles.reduce((sum, a) => sum + a.weight, 0);

    if (totalWeight === 0) {
      // Fallback to simple average
      const average =
        filteredAngles.reduce((sum, a) => sum + a.angle, 0) /
        filteredAngles.length;
      return average;
    }

    const weightedSum = filteredAngles.reduce(
      (sum, a) => sum + a.angle * a.weight,
      0
    );
    const weightedAverage = weightedSum / totalWeight;

    // Log method distribution for debugging
    const methodCounts = filteredAngles.reduce((counts, a) => {
      counts[a.method] = (counts[a.method] || 0) + 1;
      return counts;
    }, {} as Record<string, number>);

    this.log(
      `Angle methods used: ${Object.entries(methodCounts)
        .map(([method, count]) => `${method}:${count}`)
        .join(", ")}`
    );

    return Math.max(minAngle, Math.min(maxAngle, weightedAverage));
  }

  /**
   * Preprocess an image for text detection
   */
  private async preprocessDetection(
    canvas: Canvas
  ): Promise<PreprocessDetectionResult> {
    const { width: originalWidth, height: originalHeight } = canvas;

    const {
      width: resizeW,
      height: resizeH,
      ratio: resizeRatio,
    } = this.calculateResizeDimensions(originalWidth, originalHeight);

    const processor = new ImageProcessor(canvas);
    const resizedCanvas = processor
      .resize({ width: resizeW, height: resizeH })
      .toCanvas();
    processor.destroy();

    const width = Math.ceil(resizeW / 32) * 32;
    const height = Math.ceil(resizeH / 32) * 32;

    const paddedCanvas = this.createPaddedCanvas(
      resizedCanvas,
      resizeW,
      resizeH,
      width,
      height
    );

    const tensor = this.imageToTensor(paddedCanvas, width, height);

    this.log(
      `Detection preprocessed: original(${originalWidth}x${originalHeight}), ` +
        `model_input(${width}x${height}), resize_ratio: ${resizeRatio.toFixed(
          4
        )}`
    );

    return {
      tensor,
      width,
      height,
      resizeRatio,
      originalWidth,
      originalHeight,
    };
  }

  /**
   * Calculate dimensions for resizing the image
   */
  private calculateResizeDimensions(
    originalWidth: number,
    originalHeight: number
  ) {
    const MAX_SIDE_LEN = this.options.maxSideLength!;

    let resizeW = originalWidth;
    let resizeH = originalHeight;
    let ratio = 1.0;

    if (Math.max(resizeH, resizeW) > MAX_SIDE_LEN) {
      ratio = MAX_SIDE_LEN / (resizeH > resizeW ? resizeH : resizeW);
      resizeW = Math.round(resizeW * ratio);
      resizeH = Math.round(resizeH * ratio);
    }

    return { width: resizeW, height: resizeH, ratio };
  }

  /**
   * Create a padded canvas from the resized image
   */
  private createPaddedCanvas(
    resizedCanvas: Canvas,
    resizeW: number,
    resizeH: number,
    targetWidth: number,
    targetHeight: number
  ): Canvas {
    const paddedCanvas = createCanvas(targetWidth, targetHeight);
    const paddedCtx = paddedCanvas.getContext("2d");
    paddedCtx.drawImage(resizedCanvas, 0, 0, resizeW, resizeH);
    return paddedCanvas;
  }

  /**
   * Convert an image to a normalized tensor for model input
   */
  private imageToTensor(
    canvas: Canvas,
    width: number,
    height: number
  ): Float32Array {
    const ctx = canvas.getContext("2d");
    const imageData = ctx.getImageData(0, 0, width, height);
    const rgbaData = imageData.data;

    const tensor = new Float32Array(
      DetectionService.NUM_CHANNELS * height * width
    );
    const { mean, stdDeviation } = this.options;

    for (let h = 0; h < height; h++) {
      for (let w = 0; w < width; w++) {
        const rgbaIdx = (h * width + w) * 4;
        const tensorBaseIdx = h * width + w;

        // Normalize RGB values
        for (let c = 0; c < DetectionService.NUM_CHANNELS; c++) {
          const pixelValue = rgbaData[rgbaIdx + c]! / 255.0;
          const normalizedValue = (pixelValue - mean![c]) / stdDeviation![c];
          tensor[c * height * width + tensorBaseIdx] = normalizedValue;
        }
      }
    }

    return tensor;
  }

  /**
   * Run the detection model inference
   */
  private async runInference(
    tensor: Float32Array,
    width: number,
    height: number
  ): Promise<Float32Array | null> {
    try {
      this.log("Running detection inference...");

      const inputTensor = new ort.Tensor("float32", tensor, [
        1,
        3,
        height,
        width,
      ]);

      const feeds = { x: inputTensor };
      const results = await this.session.run(feeds);
      const outputTensor =
        results[this.session.outputNames[0] || "sigmoid_0.tmp_0"];

      this.log("Detection inference complete!");

      if (!outputTensor) {
        console.error(
          `Output tensor ${this.session.outputNames[0]}  not found in detection results`
        );
        return null;
      }

      return outputTensor.data as Float32Array;
    } catch (error) {
      console.error(
        "Error during model inference:",
        error instanceof Error ? error.message : String(error)
      );
      throw error;
    }
  }

  /**
   * Convert a tensor to a canvas for visualization and processing
   */
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

        const pixelIdx = (y * width + x) * 4;
        data[pixelIdx] = grayValue; // R
        data[pixelIdx + 1] = grayValue; // G
        data[pixelIdx + 2] = grayValue; // B
        data[pixelIdx + 3] = 255; // A
      }
    }

    ctx.putImageData(imageData, 0, 0);
    return canvas;
  }

  /**
   * Process detection results to extract bounding boxes
   */
  private postprocessDetection(
    detection: Float32Array,
    input: PreprocessDetectionResult,
    minBoxAreaOnPadded: number = this.options.minimumAreaThreshold || 20,
    paddingVertical: number = this.options.paddingVertical || 0.4,
    paddingHorizontal: number = this.options.paddingHorizontal || 0.6
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

    const boxes = this.extractBoxesFromContours(
      contours,
      width,
      height,
      resizeRatio,
      originalWidth,
      originalHeight,
      minBoxAreaOnPadded,
      paddingVertical,
      paddingHorizontal
    );

    processor.destroy();
    contours.destroy();

    this.log(`Found ${boxes.length} potential text boxes`);
    return boxes;
  }

  /**
   * Extract boxes from contours
   */
  private extractBoxesFromContours(
    contours: Contours,
    width: number,
    height: number,
    resizeRatio: number,
    originalWidth: number,
    originalHeight: number,
    minBoxArea: number,
    paddingVertical: number,
    paddingHorizontal: number
  ): Box[] {
    const boxes: Box[] = [];

    contours.iterate((contour) => {
      const rect = contours.getRect(contour);

      if (rect.width * rect.height <= minBoxArea) {
        return;
      }

      const paddedRect = this.applyPaddingToRect(
        rect,
        width,
        height,
        paddingVertical,
        paddingHorizontal
      );

      const finalBox = this.convertToOriginalCoordinates(
        paddedRect,
        resizeRatio,
        originalWidth,
        originalHeight
      );

      if (finalBox.width > 5 && finalBox.height > 5) {
        boxes.push(finalBox);
      }
    });

    return boxes;
  }

  /**
   * Apply padding to a rectangle
   */
  private applyPaddingToRect(
    rect: { x: number; y: number; width: number; height: number },
    maxWidth: number,
    maxHeight: number,
    paddingVertical: number,
    paddingHorizontal: number
  ) {
    const verticalPadding = Math.round(rect.height * paddingVertical);
    const horizontalPadding = Math.round(rect.height * paddingHorizontal);

    let x = rect.x - horizontalPadding;
    let y = rect.y - verticalPadding;
    let width = rect.width + 2 * horizontalPadding;
    let height = rect.height + 2 * verticalPadding;

    x = Math.max(0, x);
    y = Math.max(0, y);

    const rightEdge = Math.min(
      maxWidth,
      rect.x + rect.width + horizontalPadding
    );
    const bottomEdge = Math.min(
      maxHeight,
      rect.y + rect.height + verticalPadding
    );
    width = rightEdge - x;
    height = bottomEdge - y;

    return { x, y, width, height };
  }

  /**
   * Convert coordinates from resized image back to original image
   */
  private convertToOriginalCoordinates(
    rect: { x: number; y: number; width: number; height: number },
    resizeRatio: number,
    originalWidth: number,
    originalHeight: number
  ): Box {
    const scaledX = rect.x / resizeRatio;
    const scaledY = rect.y / resizeRatio;
    const scaledWidth = rect.width / resizeRatio;
    const scaledHeight = rect.height / resizeRatio;

    const x = Math.max(0, Math.round(scaledX));
    const y = Math.max(0, Math.round(scaledY));
    const width = Math.min(originalWidth - x, Math.round(scaledWidth));
    const height = Math.min(originalHeight - y, Math.round(scaledHeight));

    return { x, y, width, height };
  }

  /**
   * Debug the detection canvas in binary image format (thresholded)
   */
  private async debugDetectionCanvas(
    detection: Float32Array,
    width: number,
    height: number
  ): Promise<void> {
    const canvas = this.tensorToCanvas(detection, width, height);

    const dir = this.debugging.debugFolder!;
    await CanvasToolkit.getInstance().saveImage({
      canvas,
      filename: "detection-debug",
      path: dir,
    });

    this.log(`Probability map visualized and saved to: ${dir}`);
  }

  /**
   * Debug the bounding boxes by drawinga rectangle onto the original image
   */
  private async debugDetectedBoxes(image: ArrayBuffer | Canvas, boxes: Box[]) {
    const canvas =
      image instanceof Canvas
        ? image
        : await ImageProcessor.prepareCanvas(image);

    const ctx = canvas.getContext("2d");

    const toolkit = CanvasToolkit.getInstance();

    for (const box of boxes) {
      const { x, y, width, height } = box;
      toolkit.drawLine({
        ctx,
        x,
        y,
        width,
        height,
      });
    }

    const dir = this.debugging.debugFolder!;
    await CanvasToolkit.getInstance().saveImage({
      canvas,
      filename: "boxes-debug",
      path: dir,
    });

    this.log(`Boxes visualized and saved to: ${dir}`);
  }
}
