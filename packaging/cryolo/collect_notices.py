"""Collect GUI notices without claiming to redistribute SAM or crYOLO."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from collect_licenses import collect

if __name__ == "__main__":
    target = Path(sys.argv[1])
    collect(target)
    overview = target / "THIRD_PARTY_NOTICES.txt"
    text = overview.read_text(encoding="utf-8")
    text = text.replace("The bundled original SAM ViT-B model is distributed under Apache-2.0.\n"
                        "Model source: https://github.com/facebookresearch/segment-anything", 
                        "This GUI bundle contains neither SAM nor the crYOLO software/runtime.\n"
                        "crYOLO is installed separately from its official distribution.\n"
                        "crYOLO license: https://cryolo-doc.readthedocs.io/en/stable/other/license.html")
    overview.write_text(text, encoding="utf-8")
