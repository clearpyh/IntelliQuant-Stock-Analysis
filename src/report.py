from pathlib import Path
import json
import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

def summarize_fundamentals(sym_dir: Path):
    metrics = {}
    fi_fp = sym_dir / "fina_indicator.csv"
    if fi_fp.exists():
        fi = pd.read_csv(fi_fp)
        if not fi.empty:
            last = fi.sort_values(["end_date","ann_date"]).iloc[-1]
            for k in ["roe","roa","grossprofit_margin","debt_to_assets","oper_cash_flow","pe","pb"]:
                if k in fi.columns:
                    try:
                        metrics[k] = float(last[k])
                    except Exception:
                        metrics[k] = None
    return metrics

def render_industry_report(industry: str, symbols: list, fundamentals_dir: Path, export_dir: Path, charts: list, conclusions_path_list: list):
    rows = []
    for sym in symbols:
        sym_dir = fundamentals_dir / sym.replace('.', '_')
        m = summarize_fundamentals(sym_dir)
        rows.append({"symbol": sym, **m})
    conclusions = []
    for cp in conclusions_path_list:
        fp = Path(cp)
        if fp.exists():
            try:
                data = json.loads(fp.read_text(encoding="utf-8"))
                conclusions.extend(data)
            except Exception:
                pass
    env = Environment(loader=FileSystemLoader(str(Path(__file__).resolve().parents[1] / "templates")), autoescape=select_autoescape(["html"]))
    tpl = env.get_template("report.html.j2")
    html = tpl.render(industry=industry, rows=rows, charts=charts, conclusions=conclusions)
    out_dir = export_dir / industry
    out_dir.mkdir(parents=True, exist_ok=True)
    out_fp = out_dir / "industry_report.html"
    out_fp.write_text(html, encoding="utf-8")
    return out_fp

def render_single_report(symbol: str, industry: str, fundamentals_dir: Path, export_dir: Path, charts: list, conclusions: list, facts: dict):
    env = Environment(loader=FileSystemLoader(str(Path(__file__).resolve().parents[1] / "templates")), autoescape=select_autoescape(["html"]))
    tpl = env.get_template("single_report.html.j2")
    sym_dir = fundamentals_dir / symbol.replace('.', '_')
    m = summarize_fundamentals(sym_dir)
    html = tpl.render(symbol=symbol, industry=industry, charts=charts, conclusions=conclusions, metrics=m, facts=facts)
    out_dir = export_dir / industry
    out_dir.mkdir(parents=True, exist_ok=True)
    out_fp = out_dir / f"{symbol}_report.html"
    out_fp.write_text(html, encoding="utf-8")
    return out_fp
