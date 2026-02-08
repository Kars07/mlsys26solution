"""
Pack solution source files into solution.json.
(Patched for Windows + generic Pydantic support)
"""

import sys
import os
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import tomllib
except ImportError:
    import tomli as tomllib

# Remove 'Source' from import to avoid ImportError
from flashinfer_bench import BuildSpec, Solution

def load_config() -> dict:
    """Load configuration from config.toml."""
    config_path = PROJECT_ROOT / "config.toml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "rb") as f:
        return tomllib.load(f)

def pack_solution(output_path: Path = None) -> Path:
    """Pack solution files into a Solution JSON manually."""
    config = load_config()

    solution_config = config["solution"]
    build_config = config["build"]

    language = build_config["language"]
    entry_point = build_config["entry_point"]

    # Determine source directory
    if language == "triton":
        source_dir = PROJECT_ROOT / "solution" / "triton"
    elif language == "cuda":
        source_dir = PROJECT_ROOT / "solution" / "cuda"
    else:
        raise ValueError(f"Unsupported language: {language}")

    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    # Create Build Spec
    spec = BuildSpec(
        language=language,
        target_hardware=["cuda"],
        entry_point=entry_point,
    )

    # Read Files into List of Dictionaries
    # Pydantic will convert these dicts to the internal Source model automatically
    sources = []
    for root, _, filenames in os.walk(source_dir):
        for filename in filenames:
            file_path = Path(root) / filename
            rel_path = file_path.relative_to(source_dir).as_posix()
            content = file_path.read_text(encoding="utf-8")

            # Use a dictionary instead of the Source object
            sources.append({
                "path": rel_path,
                "content": content
            })

    # Create Solution Object
    solution = Solution(
        name=solution_config["name"],
        definition=solution_config["definition"],
        author=solution_config["author"],
        spec=spec,
        sources=sources
    )

    # Write Output
    if output_path is None:
        output_path = PROJECT_ROOT / "solution.json"

    output_path.write_text(solution.model_dump_json(indent=2))
    print(f"Solution packed: {output_path}")

    return output_path

if __name__ == "__main__":
    pack_solution()
