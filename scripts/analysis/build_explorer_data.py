"""Build space/data.json for the explorer: every held-out output, 4 systems (plain, gated steer, LoRA, CAST), 4 test sets.
Usable refusal = judge REFUSAL, not a loop, readable language (same definition as codemix_eval.py).
Usage (repo root): uv run --with pandas python scripts/analysis/build_explorer_data.py"""
import json, sys, os
import pandas as pd
sys.path.insert(0, "."); import codemix_eval as ce
S, L = "runs/steer/tiny-aya-global/", "runs/lora/tiny-aya-global/"
SETS = {
 "heldout":  {"plain": S+"gen_control_a0_judged4_lid.jsonl", "guard": S+"gen_g3_14_judged4_lid.jsonl",
              "lora": L+"gen_lora_v2c_heldout_judged4_lid.jsonl", "cast": S+"gen_safetune_cast_all_a1_judged4_lid.jsonl"},
 "newcats":  {"plain": S+"gen_control_newcats_test_judged4_lid.jsonl", "guard": S+"gen_g3_14_newcats_test_judged4_lid.jsonl",
              "lora": L+"gen_lora_v2c_newcats_test_judged4_lid.jsonl", "cast": S+"gen_safetune_cast_all_a1_newcats_test_judged4_lid.jsonl"},
 "unseen":   {"plain": S+"gen_control_unseen_langs_v1_judged4_lid.jsonl", "guard": S+"gen_g3_14_unseen_langs_v1_judged4_lid.jsonl",
              "lora": L+"gen_lora_v2c_unseen_langs_v1_judged4_lid.jsonl"},
 "attacks":  {"plain": S+"gen_control_attacks_v1_judged4_lid.jsonl", "guard": S+"gen_g3_14_attacks_v1_judged4_lid.jsonl",
              "lora": L+"gen_lora_v2c_attacks_v1_judged4_lid.jsonl"},
}
OWN = {"bn_rom": "ben", "ta_rom": "tam", "mr_rom": "mar", "ur_rom": "urd"}   # unseen langs: a reply in the user's own language is readable
rows = {}
for st, files in SETS.items():
    for sysname, f in files.items():
        try: d = ce.load_log(f)   # local file if present, else the public logs dataset
        except Exception as e: print("missing", f, e); continue
        if st == "unseen" and "lang" in d:
            d["ok_lang"] = d.ok_lang | (d.lang.str[:3] == d.script.map(OWN).fillna("-"))
        d = ce.add_usable(d)
        gcol = next((c for c in d.columns if c in ("gate_p", "gate", "gate_score", "g")), None)
        for r in d.itertuples(index=False):
            key = (st, r.id, r.script)
            row = rows.setdefault(key, {"set": st, "id": r.id, "script": r.script, "label": int(r.label),
                                        "category": getattr(r, "category", None) or "", "attack": getattr(r, "attack", None) or "",
                                        "prompt": r.text})
            row[sysname] = {"r": (r.response or "")[:1200], "j": r.judge4, "lang": getattr(r, "lang", "") or "",
                            "deg": bool(ce.is_degenerate(r.response)), "usable": bool(r.usable), "ref": bool(r.ref),
                            "g": (round(float(getattr(r, gcol)), 3) if gcol and pd.notna(getattr(r, gcol)) else None)}
out = list(rows.values())
json.dump(out, open("space/data.json", "w"), ensure_ascii=False, separators=(",", ":"))
df = pd.DataFrame(out)
print(len(out), df.groupby("set").size().to_dict(), round(os.path.getsize("space/data.json") / 1e6, 2), "MB")
for st in SETS:
    h = [x for x in out if x["set"] == st and x["label"] == 1 and x["script"] != "en"]
    print(st, {s: f"{sum(x[s]['usable'] for x in h if s in x) / max(1, sum(s in x for x in h)):.0%}" for s in ["plain", "guard", "lora", "cast"]})
