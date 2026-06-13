import zipfile
from pathlib import Path
from pathspec import PathSpec

PROJECT_ROOT = Path(".")
OUT_ZIP = "nijigp.zip"
TOP_DIR = "nijigp"

EXTRA_IGNORE = [
    "/scripts/",
    "/.git/",
    "/.github/",
    ".git*",
    "pyrightconfig.json",
    "README.md"
]

ignore_lines = []
gitignore = PROJECT_ROOT / ".gitignore"
if gitignore.exists():
    ignore_lines.extend(gitignore.read_text().splitlines())
ignore_lines.extend(EXTRA_IGNORE)
spec = PathSpec.from_lines("gitwildmatch", ignore_lines)

def should_ignore(path: Path):
    return spec.match_file(str(path))

with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED) as z:
    for path in PROJECT_ROOT.rglob("*"):
        if path == gitignore or ".git" in path.parts:
            continue

        if should_ignore(path):
            continue

        if path.is_file():
            arcname = f"{TOP_DIR}/{path.relative_to(PROJECT_ROOT)}"
            print(arcname)
            z.write(path, arcname)