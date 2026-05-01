# ...existing code...
import json, sys, os

def load(p):
    with open(p,'r') as f:
        return json.load(f)

def summarize(data):
    keys = list(data.keys())
    total = len(keys)
    stats = {
        'kernel_generation_success':0,
        'compilation_success':0,
        'execution_success':0,
        'verification_success':0
    }
    for k in keys:
        d = data[k] if isinstance(data[k], dict) else {}
        for fld in list(stats.keys()):
            if d.get(fld) is True:
                stats[fld] += 1
    return total, stats

def print_summary(label, total, stats):
    print(f"\n=== {label} (n={total}) ===")
    for fld, count in stats.items():
        pct = 100.0 * count / total if total > 0 else 0.0
        print(f"{fld:25s}: {count:5d} / {total:5d}  ({pct:5.1f}%)")

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python analyze_pass_rates.py spec.json nonspec.json")
        sys.exit(1)
    spec_p, nonspec_p = sys.argv[1], sys.argv[2]
    for p in (spec_p, nonspec_p):
        if not os.path.exists(p):
            print("Missing:", p); sys.exit(2)
    spec = load(spec_p); nonspec = load(nonspec_p)
    t_spec, s_spec = summarize(spec)
    t_nsp, s_nsp = summarize(nonspec)
    print_summary("SPEC", t_spec, s_spec)
    print_summary("NONSPEC", t_nsp, s_nsp)
    print("\n=== DIFF (SPEC - NONSPEC in percentage points) ===")
    for fld in s_spec:
        p_spec = 100.0 * s_spec[fld] / t_spec if t_spec > 0 else 0
        p_nsp  = 100.0 * s_nsp[fld] / t_nsp if t_nsp > 0 else 0
        print(f"{fld:25s}: {p_spec - p_nsp:6.2f} pp (SPEC {p_spec:5.1f}%  vs  NONSPEC {p_nsp:5.1f}%)")
# ...existing code...