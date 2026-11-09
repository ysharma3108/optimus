"""Extract the speed of sound of every material in the OptimUS database.

The output, assistant/data/material_speeds.json, lets the mesh check work
out the wavelength for `optimus.material.load_material("<name>")` without
pandas or the Excel file. Re-run it whenever the database changes:

    pip install xlrd
    python -m assistant.tools.extract_material_speeds

Despite its .xlsx name, Material_database.xlsx is an old .xls (OLE) file,
which xlrd reads. Names are lower-cased because load_material compares
lower-cased names (optimus/material/common.py).
"""

import json
from pathlib import Path

import xlrd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "optimus" / "material" / "Material_database.xlsx"
OUT = ROOT / "assistant" / "data" / "material_speeds.json"


def main():
    sheet = xlrd.open_workbook(str(SRC)).sheet_by_index(0)
    header = [str(v) for v in sheet.row_values(0)]
    col = header.index("Speed of Sound (m/s)")
    speeds = {}
    for r in range(2, sheet.nrows):  # two header rows, as in common.py
        row = sheet.row_values(r)
        name = str(row[0]).strip().lower()
        if name and isinstance(row[col], float) and row[col] > 0:  # 0 = no data
            speeds[name] = round(row[col], 4)
    data = {"source": "optimus/material/Material_database.xlsx", "speed_of_sound_m_per_s": speeds}
    OUT.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print("wrote %d materials to %s" % (len(speeds), OUT))


if __name__ == "__main__":
    main()
