// tsc's declaration emit keeps relative ".ts" import specifiers (we author
// src/*.ts importing each other with ".ts" extensions so plain `node
// --experimental-strip-types` can run the tests directly). Rewrite those to
// ".js" in the emitted .d.ts so consumers resolve against dist/index.js.
import { readdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

const dir = fileURLToPath(new URL("../dist/", import.meta.url));
for (const name of readdirSync(dir)) {
  if (!name.endsWith(".d.ts")) continue;
  const p = join(dir, name);
  const src = readFileSync(p, "utf-8");
  const fixed = src.replace(/(from\s+["']\.[^"']*?)\.ts(["'])/g, "$1.js$2");
  writeFileSync(p, fixed);
}
console.log("fixed .d.ts relative import extensions");
