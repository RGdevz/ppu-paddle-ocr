export function log(verbose: Boolean | undefined, text: string): void {
  if (verbose) {
    console.log(text);
  }
}
