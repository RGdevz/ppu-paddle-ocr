import type { PaddleUtilsOptions } from "./interface";

export class PaddleOcrUtils {
  private verbose: boolean;

  constructor(options: PaddleUtilsOptions) {
    this.verbose = options.verbose;
  }

  log(text: string): void {
    if (this.verbose) {
      console.log(text);
    }
  }
}
