"""Implements a platform-independent way of calling nbstripout (used in pyproject.toml)."""
import glob
import os
from pathlib import Path

if __name__ == "__main__":
    main_dir = Path(__file__).parent
    for path in glob.glob(str(main_dir / "docs" / "02_notebooks" / "*.ipynb")):
        cmd = f"nbstripout {path}"
        os.system(cmd)
    for path in glob.glob(str(main_dir / "notebooks" / "*.ipynb")):
        cmd = f"nbstripout {path}"
        os.system(cmd)
