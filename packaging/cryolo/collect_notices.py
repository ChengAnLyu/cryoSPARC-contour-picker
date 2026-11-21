"""Collect bundled GUI/SAM notices and identify the external crYOLO dependency."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from collect_licenses import collect

if __name__ == "__main__":
    target = Path(sys.argv[1])
    collect(target)
    overview = target / "THIRD_PARTY_NOTICES.txt"
    text = overview.read_text(encoding="utf-8")
    text += "\ncrYOLO is installed separately from its official distribution; its software and\ngeneral pretrained weights are not redistributed in this bundle.\ncrYOLO license: https://cryolo-doc.readthedocs.io/en/stable/other/license.html\n"
    overview.write_text(text, encoding="utf-8")
