import { afterAll, beforeAll, describe, expect, test } from "bun:test";
import { PaddleOcrService } from "../src/processor/paddle-ocr.service";

describe("PaddleOcrService.recognize()", () => {
  let service: PaddleOcrService;
  let imageBuffer: ArrayBuffer;

  // Setup: Initialize the service and load the test image once
  beforeAll(async () => {
    service = await PaddleOcrService.getInstance();

    const imgFile = Bun.file("./assets/receipt.jpg");
    imageBuffer = await imgFile.arrayBuffer();
  });

  // Teardown: Release the ONNX session
  afterAll(async () => {
    await service.destroy();
  });

  test("should return grouped results by default (flatten: false)", async () => {
    const result = await service.recognize(imageBuffer);

    // Check the overall structure for grouped results
    expect(result).toBeObject();
    expect(result).toHaveProperty("text");
    expect(result).toHaveProperty("lines");
    expect(result).toHaveProperty("confidence");
    expect(result).not.toHaveProperty("results"); // The 'results' key should not exist

    // Validate the content types
    expect(result.text).toBeString();
    expect(result.confidence).toBeNumber();
    expect(result.confidence).toBeGreaterThan(0); // Sanity check
    expect(result.lines).toBeArray();
    expect(result.lines.length).toBeGreaterThan(0); // The test image should yield results

    // Validate the nested structure
    const firstLine = result.lines[0];
    expect(firstLine).toBeArray();
    expect(firstLine!.length).toBeGreaterThan(0);

    // Validate a single recognition item
    const firstItem = firstLine![0];
    expect(firstItem).toBeObject();
    expect(firstItem).toHaveProperty("text");
    expect(firstItem).toHaveProperty("box");
    expect(firstItem).toHaveProperty("confidence");
    expect(firstItem!.confidence).toBeNumber();
    expect(firstItem!.box).toHaveProperty("x");
  });

  test("should return flattened results when flatten option is true", async () => {
    const result = await service.recognize(imageBuffer, { flatten: true });

    // Check the overall structure for flattened results
    expect(result).toBeObject();
    expect(result).toHaveProperty("text");
    expect(result).toHaveProperty("results"); // The key should be 'results'
    expect(result).toHaveProperty("confidence");
    expect(result).not.toHaveProperty("lines"); // The 'lines' key should not exist

    // Validate the content types
    expect(result.text).toBeString();
    expect(result.confidence).toBeNumber();
    expect(result.confidence).toBeGreaterThan(0);
    expect(result.results).toBeArray();
    expect(result.results.length).toBeGreaterThan(0);

    // Validate that the results array is flat (contains objects, not arrays)
    if (result.results.length > 0) {
      expect(result.results[0]).not.toBeArray();
    }

    // Validate a single recognition item
    const firstItem = result.results[0];
    expect(firstItem).toBeObject();
    expect(firstItem).toHaveProperty("text");
    expect(firstItem).toHaveProperty("box");
    expect(firstItem).toHaveProperty("confidence");
    expect(firstItem!.confidence).toBeNumber();
  });

  test("should return consistent data between grouped and flattened modes", async () => {
    const groupedResult = await service.recognize(imageBuffer);
    const flattenedResult = await service.recognize(imageBuffer, {
      flatten: true,
    });

    // The overall confidence should be identical
    expect(flattenedResult.confidence).toBe(groupedResult.confidence);

    // The full text output should be identical
    expect(flattenedResult.text).toBe(groupedResult.text);

    // The total number of recognized text items should be the same
    const groupedItemCount = groupedResult.lines.flat().length;
    expect(flattenedResult.results.length).toBe(groupedItemCount);
  });
});
