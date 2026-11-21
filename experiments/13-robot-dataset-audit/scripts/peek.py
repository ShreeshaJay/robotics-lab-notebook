import sys, pandas as pd, pyarrow.parquet as pq, numpy as np
from pathlib import Path
root = Path(sys.argv[1])
ep = pd.read_parquet(sorted((root/"meta/episodes").rglob("*.parquet"))[0])
print("episode cols:", [c for c in ep.columns if not c.startswith("stats/")][:40], "n stats cols", sum(c.startswith("stats/") for c in ep.columns))
print(ep[[c for c in ep.columns if not c.startswith("stats/")]].head(3).T)
t = pd.read_parquet(root/"meta/tasks.parquet"); print("tasks:", t.shape, t.index.name, list(t.columns)); print(t.head(12))
f = sorted((root/"data").rglob("*.parquet"))[0]
pf = pq.ParquetFile(f); print(f, pf.metadata.num_rows, pf.schema_arrow.names)
