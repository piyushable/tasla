"""Replace the buggy EMBEDDED_3D_HTML string inside your Streamlit app with the fixed page.

Usage:  python embed_fix.py your_app.py            (expects vehicle_path_3d.html next to this script)
A backup is written to your_app.py.bak first.
"""
import pathlib
import shutil
import sys

app = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "app.py")
html = (pathlib.Path(__file__).parent / "vehicle_path_3d.html").read_text(encoding="utf-8")

lines = app.read_text(encoding="utf-8").split("\n")
hits = [i for i, l in enumerate(lines) if l.startswith("EMBEDDED_3D_HTML = ")]
if len(hits) != 1:
    sys.exit(f"Expected exactly one 'EMBEDDED_3D_HTML = ' line in {app}, found {len(hits)}.")

shutil.copy(app, str(app) + ".bak")
lines[hits[0]] = "EMBEDDED_3D_HTML = " + repr(html)
app.write_text("\n".join(lines), encoding="utf-8")
print(f"Updated {app} (backup: {app}.bak). The embedded page is now the fixed version.")
