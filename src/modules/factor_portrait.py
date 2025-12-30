from datetime import datetime
from pathlib import Path
import pandas as pd

def run(inputs: dict) -> dict:
    symbol: str = inputs["symbol"]
    fundamentals_dir: Path = inputs["fundamentals_dir"]
    sym_dir = fundamentals_dir / symbol.replace('.', '_')
    fi_fp = sym_dir / "fina_indicator.csv"
    rows = []
    if fi_fp.exists():
        fi = pd.read_csv(fi_fp)
        fi = fi.sort_values(["end_date","ann_date"]) if "end_date" in fi.columns and "ann_date" in fi.columns else fi
        metrics_cols = [c for c in ["roe","roa","grossprofit_margin","debt_to_assets","oper_cash_flow","pe","pb"] if c in fi.columns]
        for m in metrics_cols:
            ser = pd.to_numeric(fi[m], errors="coerce").dropna()
            val = float(ser.iloc[-1]) if not ser.empty else None
            trend = float((ser.diff().dropna().iloc[-1])) if ser.shape[0] >= 2 else 0.0
            rows.append({"metric": m, "value": val, "trend": trend, "percentile": 0.5})
    ts = datetime.now().isoformat()
    return {"timestamp": ts, "data": {"portrait": rows}}
