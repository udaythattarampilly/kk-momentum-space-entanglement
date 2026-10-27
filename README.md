# Momentum-space entanglement and mutual information in Kaluza–Klein reduction: numerical calculations

This repository contains the Python script that produces all numerical results of

> U. Thattarampilly and Y. Zheng, *Momentum-space entanglement and mutual information in Kaluza–Klein reduction* (2026).

The script computes every number in the paper that comes from a numerical integral, including the values in Table 1, and draws Figure 1.

## Requirements

Python 3.9 or later, with `numpy`, `scipy` and `matplotlib`:

```
pip install numpy scipy matplotlib
```

## How to get each result

Run the script with the names of the calculations you want. Add `figure` to also draw the corresponding panel of Figure 1.

| Result in the paper | Command |
|---|---|
| Cubic theory: J_n^(3) → A_3/(nR), eq. (6.3), Fig. 1(a) | `python kk_entanglement_numerics.py cubic figure` |
| Cubic theory: zero-mode coefficient D_3, eq. (6.7) | `python kk_entanglement_numerics.py d3` |
| Quartic (2,2) channel: J_n → A/(nR³), eq. (4.8) | `python kk_entanglement_numerics.py j22` |
| Quartic (2,2) + mixed: K_n and a, b, eq. (5.10), Fig. 1(b) | `python kk_entanglement_numerics.py kn beta figure` |
| Quartic: constants a′, β, a, b only, eq. (5.9) | `python kk_entanglement_numerics.py beta` |
| Quartic (1,3) channel: c and c′, eq. (5.11), Fig. 1(c) | `python kk_entanglement_numerics.py j13 figure` |
| Closed-form numbers of sections 4, 6 and 8 | `python kk_entanglement_numerics.py misc` |
| Everything, including the whole of Figure 1 | `python kk_entanglement_numerics.py` |

By default the script uses smaller Monte Carlo samples, which take minutes and reproduce the paper within the statistical errors. Add `--full` to use the sample sizes of the paper (a few hours for the whole run).

In a Jupyter notebook, use `%run kk_entanglement_numerics.py j13 figure`, or paste the script into a cell and call `main(["j13", "figure"])`.

## Output

- the numbers, printed on the screen with the equation they belong to;
- `results.json`, where every result is stored, so that a slow calculation only has to be run once;
- the figure: `fig_mc.pdf` when all three panels are available, otherwise a file with the panels computed so far (for example `fig_1c.pdf`).

The header of the script explains the conventions and the integration methods, and each calculation is documented where it is defined. Equation and section numbers refer to the submitted version of the paper.

## Citation

If you use this code, please cite the paper above.

## Licence

MIT licence; see `LICENSE`.
