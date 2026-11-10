"""Effect cards: an index of all effects plus one static HTML page per factor.

    python -m hlab.report traces/ --out report/
"""
import argparse
import html
import os
from pathlib import Path

import pandas as pd

from hlab import analysis
from hlab.middleware import FACTORS

CSS = """
:root{--bg:#fff;--fg:#1a1a1a;--muted:#666;--line:#ddd;--pos:#1a7f37;--neg:#c62828}
@media (prefers-color-scheme:dark){:root{--bg:#161616;--fg:#eee;--muted:#999;--line:#333;--pos:#5fd38d;--neg:#ff7b7b}}
body{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;max-width:960px;margin:0 auto;padding:16px}
table{border-collapse:collapse;width:100%;margin:8px 0 24px}th,td{border-bottom:1px solid var(--line);
padding:6px 8px;text-align:left;font-variant-numeric:tabular-nums}th{color:var(--muted);font-weight:600}
.pos{color:var(--pos)}.neg{color:var(--neg)}.muted{color:var(--muted)}a{color:inherit}
.wrap{overflow-x:auto}
"""


def _sig(row):
    return "pos" if row["pp_lo"] > 0 else "neg" if row["pp_hi"] < 0 else ""


def _pp(row):
    return f'{row["pp"]*100:+.1f} pts <span class="muted">[{row["pp_lo"]*100:+.1f}, {row["pp_hi"]*100:+.1f}]</span>'


def _effect_rows(effects):
    return "".join(f'<tr class="{_sig(r)}"><td>{html.escape(r["aliases"])}</td><td>{_pp(r)}</td>'
                   f'<td>{r["odds_ratio"]:.2f}</td><td>{r["p"]:.3f}</td></tr>' for _, r in effects.iterrows())


def _table(head, body):
    return f'<div class="wrap"><table><tr>{"".join(f"<th>{h}</th>" for h in head)}</tr>{body}</table></div>'


def _page(title, body):
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" '
            f'content="width=device-width,initial-scale=1"><title>{html.escape(title)}</title>'
            f"<style>{CSS}</style></head><body>{body}</body></html>")


def _link(path, out):
    if not path:
        return '<span class="muted">none</span>'
    return f'<a href="{html.escape(os.path.relpath(path, out))}">{html.escape(Path(path).name)}</a>'


def write(df, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    effects, info = analysis.fit_success(df)
    secondary = pd.concat([analysis.fit_secondary(df, m) for m in ("tokens", "wall_s", "cost")], ignore_index=True)
    head = ["Effect (aliases)", "High vs low, success [95% CI]", "Odds ratio", "p"]
    note = (f'<p class="muted">{len(df)} runs, {df["task"].nunique()} tasks, {df["seed"].nunique()} seeds, '
            f'model {", ".join(df["model"].unique())}. Fit used {info["tasks_used"]} tasks; '
            f'{info["tasks_dropped"]} all-pass/all-fail tasks carry no within-task information and were dropped. '
            "Effects are percentage points for a task of average difficulty. Green/red: CI excludes zero.</p>")
    if effects.empty:
        (out / "index.html").write_text(_page("Harness effects", "<h1>Harness effects</h1>" + note +
                                              "<p>No task has both passing and failing runs: nothing to estimate.</p>"))
        return out / "index.html"
    effects = effects.reindex(effects["z"].abs().sort_values(ascending=False).index)
    links = " · ".join(f'<a href="factor_{f}.html">{f} {html.escape(n)}</a>' for f, (n, _, _) in FACTORS.items())
    (out / "index.html").write_text(_page("Harness effects", f"<h1>Harness effects</h1>{note}<p>{links}</p>"
                                          + _table(head, _effect_rows(effects))))
    for f, (name, low, high) in FACTORS.items():
        main = effects[effects["term"] == f]
        inter = effects[(effects["kind"] == "2FI") & effects["aliases"].str.contains(f)]
        sec = secondary[secondary["factor"] == f] if not secondary.empty else secondary
        sec_rows = "".join(f'<tr><td>{r["metric"]}</td><td>{r["pct"]:+.0f}% <span class="muted">'
                           f'[{r["pct_lo"]:+.0f}, {r["pct_hi"]:+.0f}]</span></td></tr>' for _, r in sec.iterrows())
        cat = analysis.by_category(df, f)
        cat_rows = "".join(f'<tr><td>{html.escape(str(r["category"]))}</td><td>{r.get("low", float("nan")):.2f}</td>'
                           f'<td>{r.get("high", float("nan")):.2f}</td><td>{r["diff"]:+.2f}</td></tr>'
                           for _, r in cat.iterrows())
        helped, hurt = analysis.examples(df, f)
        ex = lambda items: "".join(f'<tr><td>{html.escape(e["task"])}</td><td>{e["delta"]:+.2f}</td>'
                                   f'<td>{_link(e["trace_high"], out)}</td><td>{_link(e["trace_low"], out)}</td></tr>'
                                   for e in items)
        body = (f'<p><a href="index.html">← all effects</a></p><h1>{f}: {html.escape(name)}</h1>'
                f'<p class="muted">low (−1): {html.escape(low)} · high (+1): {html.escape(high)}</p>'
                "<h2>Main effect</h2>" + _table(head, _effect_rows(main)) +
                "<h2>Interactions involving this factor</h2>"
                '<p class="muted">Each row is an alias chain: the estimate is the sum of all its interactions.</p>'
                + _table(head, _effect_rows(inter)) +
                "<h2>Cost delta (high vs low)</h2>" + _table(["Metric", "Change [95% CI]"], sec_rows) +
                "<h2>By task category</h2>" + _table(["Category", "Success low", "Success high", "Diff"], cat_rows) +
                "<h2>Where it helped</h2>" + _table(["Task", "Δ success", "Trace (high, passed)",
                                                     "Trace (low, failed)"], ex(helped)) +
                "<h2>Where it hurt</h2>" + _table(["Task", "Δ success", "Trace (high, failed)",
                                                   "Trace (low, passed)"], ex(hurt)))
        (out / f"factor_{f}.html").write_text(_page(f"Effect card {f}", body))
    effects.to_csv(out / "effects.csv", index=False)
    return out / "index.html"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("traces")
    p.add_argument("--out", default="report")
    a = p.parse_args(argv)
    df = analysis.load(a.traces)
    if df.empty:
        raise SystemExit(f"no finished runs in {a.traces}")
    print(f"wrote {write(df, a.out)}")


if __name__ == "__main__":
    main()
