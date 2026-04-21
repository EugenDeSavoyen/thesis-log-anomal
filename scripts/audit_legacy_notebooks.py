from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks" / "legacy"
OUTPUT_PATH = ROOT / "outputs" / "reports" / "legacy_notebook_inventory.json"


@dataclass
class NotebookInventory:
    name: str
    code_cells: int
    markdown_cells: int
    outputs: int
    file_size: int
    functions: list[str]
    classes: list[str]
    reads: list[str]
    writes: list[str]
    keywords: list[str]


KEYWORDS = [
    "drain",
    "window",
    "sliding",
    "gev",
    "genextreme",
    "pot",
    "threshold",
    "ollama",
    "openai",
    "llm",
]


def _cell_source(cell: dict) -> str:
    source = cell.get("source", "")
    if isinstance(source, list):
        return "".join(source)
    return str(source)


def inspect_notebook(path: Path) -> NotebookInventory:
    nb = json.loads(path.read_text(encoding="utf-8"))
    cells = nb.get("cells", [])
    code_cells = [c for c in cells if c.get("cell_type") == "code"]
    markdown_cells = [c for c in cells if c.get("cell_type") == "markdown"]
    text = "\n".join(_cell_source(c) for c in cells)
    lower = text.lower()

    functions = sorted(set(re.findall(r"^def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", text, flags=re.M)))
    classes = sorted(set(re.findall(r"^class\s+([A-Za-z_][A-Za-z0-9_]*)\s*[\(:]", text, flags=re.M)))
    reads = sorted(
        set(re.findall(r"""open\(['"]([^'"]+)['"]\s*,\s*['"]r""", text))
        | set(re.findall(r"""read_csv\(['"]([^'"]+)['"]""", text))
    )
    writes = sorted(set(re.findall(r"""open\(['"]([^'"]+)['"]\s*,\s*['"]w""", text)))
    outputs = sum(len(cell.get("outputs", [])) for cell in code_cells)
    keywords = [keyword for keyword in KEYWORDS if keyword in lower]

    return NotebookInventory(
        name=path.name,
        code_cells=len(code_cells),
        markdown_cells=len(markdown_cells),
        outputs=outputs,
        file_size=path.stat().st_size,
        functions=functions,
        classes=classes,
        reads=reads,
        writes=writes,
        keywords=keywords,
    )


def main() -> None:
    inventories = []
    for path in sorted(NOTEBOOK_DIR.glob("*.ipynb")):
        if path.stat().st_size == 0:
            inventories.append(
                NotebookInventory(
                    name=path.name,
                    code_cells=0,
                    markdown_cells=0,
                    outputs=0,
                    file_size=0,
                    functions=[],
                    classes=[],
                    reads=[],
                    writes=[],
                    keywords=[],
                )
            )
            continue
        inventories.append(inspect_notebook(path))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps([asdict(item) for item in inventories], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Wrote {len(inventories)} notebook records to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
