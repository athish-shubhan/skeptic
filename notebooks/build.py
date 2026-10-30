"""Build and execute the two Skeptic notebooks: python notebooks/build.py"""

from pathlib import Path

import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

HERE = Path(__file__).resolve().parent
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

canaries = [
    md("# Canaries: does Skeptic catch false discoveries?\n\nEach canary runs on simulated data where the truth "
       "is known. The full-size results and charts come from `studies/canaries.py`; this notebook re-runs the "
       "quick ones live and shows the saved charts for the rest."),
    code("import json\nfrom pathlib import Path\nfrom IPython.display import Image, display\nimport numpy as np\n"
         "from skeptic import canaries\nREPORTS = Path('..') / 'reports'"),
    md("## 1. Noise canary\n1,000 zero-skill strategies with fat-tailed returns. About 5% should pass an "
       "uncorrected 5% test; almost none should survive correction."),
    code("r = canaries.noise_canary(n_strategies=1000, n_obs=1260, seed=0)\n"
         "print({k: f'{v:.1%}' for k, v in r.pass_rate.items()})\n"
         "print(f'best of 1,000: annual Sharpe {r.best_sr * np.sqrt(252):.2f}, PSR {r.best_psr:.3f}, DSR {r.best_dsr:.3f}')"),
    code("display(Image(REPORTS / 'canary_noise.png', width=640))"),
    md("## 2. Leakage canary\nA lookahead feature and overlapping labels, both on a random walk where the "
       "true accuracy is 50%."),
    code("print(json.dumps(json.loads((REPORTS / 'canaries.json').read_text())['leakage'], indent=2))\n"
         "display(Image(REPORTS / 'canary_leakage.png', width=640))"),
    md("## 3. Planted signal: power\nHow often a real edge is detected, alone and as the best of 100 trials."),
    code("display(Image(REPORTS / 'canary_power.png', width=820))"),
    md("## 4. Best of N\nThe best in-sample Sharpe of N zero-skill strategies grows with N exactly as the "
       "False Strategy Theorem predicts; the deflated Sharpe of the winner stays near 0.5."),
    code("display(Image(REPORTS / 'canary_best_of_n.png', width=640))"),
]

zoo = [
    md("# Signal zoo: how many of the usual signals survive?\n\nPre-registered in `preregistration.md`. "
       "Results come from `studies/zoo.py`; this notebook reads its outputs."),
    code("import json\nfrom pathlib import Path\nimport numpy as np\nimport pandas as pd\n"
         "from IPython.display import Image, display\nREPORTS = Path('..') / 'reports'\n"
         "res = json.loads((REPORTS / 'zoo.json').read_text())"),
    md("## Funnel"),
    code("pd.DataFrame({k: v['counts'] for k, v in res.items()})"),
    code("display(Image(REPORTS / 'zoo_funnel.png', width=700))"),
    md("## Best configurations at the primary cost (10 bps per unit turnover)\nSharpe ratios annualised: "
       "8,760 periods a year for 60-minute bars and 35,040 for 15-minute bars (crypto trades 24/7)."),
    code("t = pd.read_csv(REPORTS / 'zoo_primary_trials.csv')\n"
         "ppy = np.where(t['params'].str.contains('60min'), 8760, 35040)\n"
         "t['gross_sr_ann'], t['net_sr_ann'] = t['gross_sr'] * np.sqrt(ppy), t['net_sr'] * np.sqrt(ppy)\n"
         "t.sort_values('net_sr', ascending=False).head(8)[['signal', 'params', 'gross_sr_ann', 'net_sr_ann', "
         "'p_net', 'p_adj', 'dsr', 'turnover']].round(3)"),
    md("## By signal family: gross edge versus what costs take"),
    code("t.groupby('signal')[['gross_sr_ann', 'net_sr_ann', 'turnover']].median().round(2).sort_values('net_sr_ann', ascending=False)"),
    md("## Probability of backtest overfitting by dataset"),
    code("pd.DataFrame({k: v['pbo'] for k, v in res.items()}).round(3)"),
]

for name, cells in (("canaries", canaries), ("signal_zoo", zoo)):
    nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"}})
    ExecutePreprocessor(timeout=600, kernel_name="python3").preprocess(nb, {"metadata": {"path": str(HERE)}})
    nbf.write(nb, HERE / f"{name}.ipynb")
    print("wrote", name)
