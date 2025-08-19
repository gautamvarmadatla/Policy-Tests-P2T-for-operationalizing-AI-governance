# tools/rules_to_nemo.py
import yaml, json, sys, pathlib

def is_io_only(rule: dict) -> bool:
    t = (rule.get("testability") or {})
    sigs = t.get("evidence_signals")
    return isinstance(sigs, list) and len(sigs) == 1 and str(sigs[0]).strip().lower() == "io_check"

def requirement_text(rule: dict) -> str:
    parts = []
    if rule.get("hazard"): parts.append(f"Hazard: {rule['hazard']}.")
    if rule.get("requirements"): parts.append("Requirement: " + " ".join(rule["requirements"]))
    if rule.get("exceptions"): parts.append("Exceptions: " + "; ".join(rule["exceptions"]))
    return " ".join(parts) or "The system must refuse OR show uncertainty OR sanitize unsafe content; benign inputs should be helpful."

def main(yaml_path: str, out_json: str):
    ypath = pathlib.Path(yaml_path)
    if not ypath.exists():
        raise FileNotFoundError(f"Rules YAML not found: {ypath.resolve()}")

    rules = yaml.safe_load(ypass := ypath.read_text(encoding="utf-8"))
    if not isinstance(rules, list):
        raise ValueError("YAML root must be a list of rule objects.")

    kept = [r for r in rules if is_io_only(r)]
    table = []
    for r in kept:
        table.append({
            "rule_id": r.get("rule_id","(no-id)"),
            "requirement": requirement_text(r),
            "severity": (r.get("severity") or "medium").lower(),
            "source": r.get("source", {}),
            "data_domain": (r.get("scope",{}).get("data_domain") or []),
        })

    outp = pathlib.Path(out_json)
    outp.parent.mkdir(parents=True, exist_ok=True)   # <-- create rails/actions/
    outp.write_text(json.dumps(table, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✓ Wrote {len(table)} io_check rules → {outp.resolve()}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: python tools/rules_to_nemo.py <rules.yaml> rails/actions/policy_rules.json")
        sys.exit(2)
    main(sys.argv[1], sys.argv[2])
