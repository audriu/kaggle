"""Write blend/weights.json into kernels/submit/submit.py as the BLEND literal (Kaggle script kernels upload only the code file)."""
import json, pathlib, re
ROOT = pathlib.Path(__file__).resolve().parents[1]
w = json.load(open(ROOT / "blend" / "weights.json"))
lit = json.dumps(dict(members=w["members"], weights=[round(x, 4) for x in w["weights"]], temps=w["temps"]))
p = ROOT / "kernels" / "submit" / "submit.py"; s = p.read_text()
s2, n = re.subn(r"^BLEND = \{.*?\}(\s*#.*)?$", f"BLEND = {lit}   # from blend/weights.json (nested CV {w.get('cv_nested', float('nan')):.4f}); overwritten by blend.json if present", s, count=1, flags=re.M)
assert n == 1, "BLEND line not found"
tw_path = ROOT / "blend" / "twins.json"
if tw_path.exists():   # cluster-centred twin weights chosen by scripts/sim_twins.py
    tw = json.load(open(tw_path))["twin_w"]; tw = {k: round(v, 3) for k, v in tw.items() if v > 0}
    s2, n = re.subn(r"^TWIN_W = \{.*?\}(\s*#.*)?$", f"TWIN_W = {json.dumps(tw)}   # from blend/twins.json (scripts/sim_twins.py)", s2, count=1, flags=re.M)
    assert n == 1, "TWIN_W line not found"; print("twins:", tw)
p.write_text(s2); print("installed:", lit)
