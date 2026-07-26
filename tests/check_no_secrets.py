#!/usr/bin/env python3
"""
Fail if an API key is about to be committed. Run before pushing:

    python tests/check_no_secrets.py

Keys reach a repo by being pasted into a helper script rather than read from the
environment. Scanning tracked files catches that before it becomes permanent — a key in
git history survives deleting the file, so the only real remedy afterwards is revocation.
"""
import re
import subprocess
import sys

PATTERNS = {
    "OpenRouter": r"sk-or-v1-[A-Za-z0-9]{20,}",
    "HuggingFace": r"hf_[A-Za-z0-9]{30,}",
    "Anthropic": r"sk-ant-[A-Za-z0-9\-]{20,}",
    "OpenAI": r"sk-[A-Za-z0-9]{40,}",
    "RunPod": r"rpa_[A-Za-z0-9]{20,}",
}


def main():
    files = subprocess.run(["git", "ls-files"], capture_output=True, text=True,
                           check=True).stdout.split()
    hits = []
    for f in files:
        if f.endswith(".example"):
            continue
        try:
            text = open(f, encoding="utf-8", errors="ignore").read()
        except (IsADirectoryError, FileNotFoundError):
            continue
        for name, pat in PATTERNS.items():
            for m in re.finditer(pat, text):
                line = text[:m.start()].count("\n") + 1
                hits.append(f"  {f}:{line}  {name} key")
    if hits:
        print("SECRETS FOUND IN TRACKED FILES:")
        print("\n".join(hits))
        print("\nRemove them, then ROTATE the key — it is already in your working tree,\n"
              "and if it was ever committed it stays in history after deletion.")
        return 1
    print(f"clean — no API keys in {len(files)} tracked files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
