"""Pack x.ifc into x.ifczip (one member x.ifc, deflate 9, fixed 1980-01-01 timestamp: the same bytes every time)."""
import sys, zipfile
from pathlib import Path
src, dst = Path(sys.argv[1]), Path(sys.argv[2])
info = zipfile.ZipInfo(src.name, date_time=(1980, 1, 1, 0, 0, 0))
info.compress_type = zipfile.ZIP_DEFLATED
info.external_attr = 0o644 << 16
with zipfile.ZipFile(dst, "w") as z:
    z.writestr(info, src.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
