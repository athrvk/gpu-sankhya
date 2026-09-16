import { gzipSync } from "node:zlib";
import { readFileSync } from "node:fs";

const buf = readFileSync(new URL("../dist/index.js", import.meta.url));
const gz = gzipSync(buf, { level: 9 });
const kb = (gz.length / 1024).toFixed(2);
console.log(`dist/index.js: ${buf.length} bytes raw, ${gz.length} bytes gzipped (${kb} KB)`);
if (gz.length > 60 * 1024) {
  console.log(`WARNING: exceeds 60 KB gzipped target`);
}
