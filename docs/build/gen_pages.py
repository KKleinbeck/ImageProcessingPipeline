"""Generate the code reference pages."""

from pathlib import Path


def main():  # noqa: D103
  root = Path(__file__).parents[2]
  doc_dir = root / "docs" / "src" / "reference"
  src_dir = root / "src"

  for file in src_dir.glob("**/*.py"):
    if file.name == "__init__.py":
      continue
    relative_file_path = file.relative_to(src_dir)

    doc_file = doc_dir / relative_file_path.parent / relative_file_path.name.replace(".py", ".md")
    doc_file.parent.mkdir(exist_ok=True, parents=True)

    module_name = str(relative_file_path).replace("/", ".").removesuffix(".py")
    with open(doc_file, "w") as fp:
      fp.write(f"# `{file.name}`\n\n::: {module_name}")


if __name__ == "__main__":
  main()
