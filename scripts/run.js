const { spawnSync } = require("child_process");
const fs = require("fs");
const path = require("path");

const isWindows = process.platform === "win32";
const repoRoot = path.resolve(__dirname, "..");
const venvDir = path.join(repoRoot, "venv");

const pythonPath = isWindows
  ? (fs.existsSync(path.join(venvDir, "Scripts", "python.exe"))
      ? path.join(venvDir, "Scripts", "python.exe")
      : path.join(venvDir, "bin", "python.exe"))
  : (fs.existsSync(path.join(venvDir, "bin", "python"))
      ? path.join(venvDir, "bin", "python")
      : path.join(venvDir, "bin", "python3"));

if (!fs.existsSync(pythonPath)) {
  console.error("❌ Virtual environment not found at: " + pythonPath);
  console.error("👉 Run 'npm install' first to set up the environment.\n");
  process.exit(1);
}

const rawArgs = process.argv.slice(2);
let commandArgs = [];

if (rawArgs.length === 0) {
  commandArgs = ["main.py", "--chat"];
} else if (rawArgs[0] === "test") {
  commandArgs = ["-m", "unittest", "discover", "-s", "tests", "-v"];
} else if (rawArgs[0] === "server" || rawArgs[0] === "api" || rawArgs[0] === "--server") {
  commandArgs = ["-m", "uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000", "--reload"];
} else {
  commandArgs = ["main.py", ...rawArgs];
}

const result = spawnSync(pythonPath, commandArgs, {
  cwd: repoRoot,
  stdio: "inherit",
});

if (result.error) {
  console.error("Execution error:", result.error.message);
  process.exit(1);
}

process.exit(result.status !== null ? result.status : 0);
