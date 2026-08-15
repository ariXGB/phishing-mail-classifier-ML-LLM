from pathlib import Path


def find_project_root(start: Path) -> Path:
    current = start.resolve()
    for k in list(current.parents):
        print(k)
        
    for candidate in [current] + list(current.parents):
        if (candidate / "models").is_dir() and (candidate / "Data").is_dir():
            return candidate
    raise FileNotFoundError(
        f"Could not locate project root starting from {start}. "
        "Expected to find sibling 'models' and 'Data' directories."
    )


PROJECT_ROOT = find_project_root(Path(__file__))
print(PROJECT_ROOT)
