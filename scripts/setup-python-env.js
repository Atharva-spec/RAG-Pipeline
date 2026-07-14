const { spawnSync } = require("child_process");
const path = require("path");

const isWindows = process.platform === "win32";
const repoRoot = path.resolve(__dirname, "..");
const venvDir = path.join(repoRoot, "venv");
const pipPath = isWindows
  ? path.join(venvDir, "Scripts", "pip.exe")
  : path.join(venvDir, "bin", "pip");

function run(command, args) {
  const result = spawnSync(command, args, {
    cwd: repoRoot,
    stdio: "inherit",
    shell: false,
  });

  if (result.error) {
    return { ok: false, error: result.error };
  }

  return { ok: result.status === 0, status: result.status };
}

function ensureSuccess(result, failureMessage) {
  if (!result.ok) {
    if (result.error) {
      console.error(failureMessage);
      console.error(result.error.message);
    }
    process.exit(result.status || 1);
  }
}

const pythonCandidates = isWindows
  ? [
      ["py", ["-3", "-m", "venv", "venv"]],
      ["python", ["-m", "venv", "venv"]],
    ]
  : [
      ["python3", ["-m", "venv", "venv"]],
      ["python", ["-m", "venv", "venv"]],
    ];

let created = false;

for (const [command, args] of pythonCandidates) {
  const result = run(command, args);
  if (result.ok) {
    created = true;
    break;
  }
}

if (!created) {
  console.error("Failed to create virtual environment.");
  console.error("Install Python 3 and ensure it is available in PATH.");
  process.exit(1);
}

const pipInstall = run(pipPath, ["install", "-r", "requirements.txt"]);
ensureSuccess(
  pipInstall,
  "Virtual environment was created, but dependency installation failed."
);
