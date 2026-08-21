const { spawnSync } = require("child_process");
const fs = require("fs");
const path = require("path");

const isWindows = process.platform === "win32";
const repoRoot = path.resolve(__dirname, "..");
const venvDir = path.join(repoRoot, "venv");

// Cross-platform paths for venv executables
const pipPath = isWindows
  ? path.join(venvDir, "Scripts", "pip.exe")
  : path.join(venvDir, "bin", "pip");

const pythonPath = isWindows
  ? path.join(venvDir, "Scripts", "python.exe")
  : path.join(venvDir, "bin", "python");

function run(command, args, options = {}) {
  const result = spawnSync(command, args, {
    cwd: repoRoot,
    stdio: options.silent ? "pipe" : "inherit",
    shell: false,
  });

  if (result.error) {
    return { ok: false, error: result.error };
  }

  return {
    ok: result.status === 0,
    status: result.status,
    stdout: result.stdout ? result.stdout.toString() : "",
  };
}

console.log("\n" + "=".repeat(65));
console.log(" 🚀 Initializing Native RAG Pipeline Environment");
console.log(` 💻 Detected Platform: ${process.platform} (${isWindows ? "Windows" : process.platform === "darwin" ? "macOS" : "Linux"})`);
console.log("=".repeat(65) + "\n");

// 1. Create venv if missing
const venvExists = fs.existsSync(pythonPath);

if (!venvExists) {
  console.log("📦 Step 1/3: Creating Python 3 virtual environment (venv)...");

  const pythonCandidates = isWindows
    ? [
        ["py", ["-3", "-m", "venv", "venv"]],
        ["python", ["-m", "venv", "venv"]],
        ["python3", ["-m", "venv", "venv"]],
      ]
    : [
        ["python3", ["-m", "venv", "venv"]],
        ["python", ["-m", "venv", "venv"]],
      ];

  let created = false;

  for (const [cmd, args] of pythonCandidates) {
    const res = run(cmd, args, { silent: true });
    if (res.ok && (fs.existsSync(pipPath) || fs.existsSync(pythonPath))) {
      created = true;
      console.log(`   ✓ Created virtual environment using '${cmd}'`);
      break;
    }
  }

  if (!created) {
    console.error("\n❌ Failed to create virtual environment.");
    console.error("Please install Python 3.10+ and ensure 'python3' (or 'py' on Windows) is in your PATH.\n");
    process.exit(1);
  }
} else {
  console.log("📦 Step 1/3: Virtual environment (venv) already exists. Skipping creation.");
}

// 2. Upgrade pip, setuptools, wheel
console.log("\n⚡ Step 2/3: Upgrading pip, setuptools, and wheel...");
run(pipPath, ["install", "--upgrade", "pip", "setuptools", "wheel"], { silent: false });

// 3. Install requirements.txt
console.log("\n📥 Step 3/3: Installing dependencies from requirements.txt...");
const installResult = run(pipPath, ["install", "-r", "requirements.txt"]);

if (!installResult.ok) {
  console.error("\n❌ Dependency installation failed.");
  console.error("Please review pip error output above.\n");
  process.exit(installResult.status || 1);
}

// Success banner
console.log("\n" + "=".repeat(65));
console.log(" 🎉 RAG Pipeline Environment Ready!");
console.log("=".repeat(65));
console.log(" 📋 Quick Start (Next Steps):");
console.log("  1. Start OpenSearch:   docker compose up -d   (or docker start native-rag-pipeline)");
console.log("  2. Start Ollama:       ollama serve           (and 'ollama pull llama3.2')");
console.log("  3. Launch Pipeline:    npm start              (or './rag --chat' / 'rag.bat --chat')");
console.log("=".repeat(65) + "\n");
