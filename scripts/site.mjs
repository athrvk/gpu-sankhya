import { existsSync, mkdirSync, copyFileSync, rmSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";

const root = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const siteDir = path.join(root, "site");

rmSync(siteDir, { recursive: true, force: true });
mkdirSync(siteDir, { recursive: true });

copyFileSync(path.join(root, "demo", "index.html"), path.join(siteDir, "index.html"));
copyFileSync(path.join(root, "dist", "index.js"), path.join(siteDir, "index.js"));

const mapPath = path.join(root, "dist", "index.js.map");
if (existsSync(mapPath)) {
  copyFileSync(mapPath, path.join(siteDir, "index.js.map"));
}

console.log(`site/ assembled at ${siteDir}`);
