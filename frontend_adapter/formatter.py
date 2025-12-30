import pandas as pd

def to_dataframe_table(data: dict, key: str) -> pd.DataFrame:
    d = data.get(key)
    if not d:
        return pd.DataFrame()
    if "columns" in d and "data" in d and "index" in d:
        return pd.DataFrame(d["data"], index=pd.to_datetime(d["index"]), columns=d["columns"])
    return pd.DataFrame(d)
