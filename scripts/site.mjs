import { existsSync, mkdirSync, copyFileSync, rmSync, readdirSync, statSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";

const root = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const siteDir = path.join(root, "site");
const demoDir = path.join(root, "demo");

rmSync(siteDir, { recursive: true, force: true });
mkdirSync(siteDir, { recursive: true });

for (const name of readdirSync(demoDir)) {
  const from = path.join(demoDir, name);
  if (statSync(from).isFile()) {
    copyFileSync(from, path.join(siteDir, name));
  }
}

copyFileSync(path.join(root, "dist", "index.js"), path.join(siteDir, "index.js"));

const mapPath = path.join(root, "dist", "index.js.map");
if (existsSync(mapPath)) {
  copyFileSync(mapPath, path.join(siteDir, "index.js.map"));
}

console.log(`site/ assembled at ${siteDir}`);
