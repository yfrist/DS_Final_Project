from pathlib import Path

def print_tree(start_path: Path, prefix: str = ""):
    contents = sorted(start_path.iterdir())
    entries = [p for p in contents if not p.name.startswith('.')]  # ignore hidden files
    for i, path in enumerate(entries):
        connector = "└── " if i == len(entries) - 1 else "├── "
        print(prefix + connector + path.name)
        if path.is_dir():
            extension = "    " if i == len(entries) - 1 else "│   "
            print_tree(path, prefix + extension)

# Usage:
root = Path(".")
print(root.name + "/")
print_tree(root)
