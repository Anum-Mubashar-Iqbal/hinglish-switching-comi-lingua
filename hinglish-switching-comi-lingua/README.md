# Where Hinglish Switches: analysis code

This repository holds the Python code behind the article *Where Hinglish Switches: Matrix Language, Equivalence and Part-of-Speech Asymmetries in Devanagari and Roman Hindi–English Text* by Anum Mubashar Iqbal (University of Chenab, Gujrat, Pakistan).

The scripts reproduce every number, table and figure in the article from the public COMI-LINGUA corpus. Running them on the released files gives exactly the values reported in the paper.

## Data

The analysis uses the token-level language identification (LID) and part-of-speech (POS) subsets of COMI-LINGUA (Sheth, Beniwal & Singh, 2025), released under CC BY 4.0. The data are not included here. Download them from [Hugging Face](https://huggingface.co/datasets/LingoIITGN/COMI-LINGUA) and place these four files in the `data/` folder:

```
data/LID_train.csv
data/LID_test.csv
data/POS_train.csv
data/POS_test.csv
```

## How to run

Python 3.10 or later is recommended. The results in the article were produced with Python 3.13, pandas 3.0, NumPy 2.5, SciPy 1.18, statsmodels 0.15 and matplotlib 3.11.

```bash
pip install -r requirements.txt
./run_all.sh
```

On Windows, run the scripts one by one from the repository folder in the order shown below, for example `python scripts/load.py`. The whole run takes about three minutes on an ordinary laptop. The permutation test uses a fixed random seed (2026), so repeated runs give the same results.

## What each script does

| Script | What it does | Article |
|---|---|---|
| `scripts/load.py` | Reads the four CSV files and normalises the two tag formats used in them | Section 4.2 |
| `scripts/gold.py` | Combines the three annotators by majority vote, removes duplicate sentences, computes Fleiss' kappa | Table 1 |
| `scripts/align.py` | Aligns LID tokens with POS tags character by character | Section 4.3 |
| `scripts/analysis.py` | Corpus profile, CMI, I-index, M-index, length of English stretches | Table 2, Figure 3 |
| `scripts/analysis2.py` | Codes switch junctures and runs the 1,000-run permutation baseline | Table 3, Figure 1 |
| `scripts/analysis3.py` | Matrix-language checks, POS × language tests, odds ratios | Tables 4 and 5, Figure 2 |
| `scripts/analysis4.py` | Logistic regression and comparison of Roman and Devanagari text | Tables 6 and 7, Figure 4 |
| `scripts/figs.py` | Draws the four figures into `fig/` | Figures 1–4 |

## Output files

| File | Contents |
|---|---|
| `R1.json` | Corpus description, mixing indices, English stretch lengths |
| `R2.json`, `junctures.csv` | Juncture counts, expected values, O/E ratios |
| `R3.json`, `pos_table.csv` | Matrix-language checks and the POS × language table |
| `R4.json`, `logit.csv` | Regression results and script comparisons |
| `fig/fig2_junctures.png` | Article Figure 1 |
| `fig/fig1_pos.png` | Article Figure 2 |
| `fig/fig4_spans.png` | Article Figure 3 |
| `fig/fig3_script_pos.png` | Article Figure 4 |

## Citation

If you use this code, please cite the article and this repository. GitHub's "Cite this repository" button uses the details in `CITATION.cff`.

Please also cite the corpus:

Sheth, R., Beniwal, H., & Singh, M. (2025). COMI-LINGUA: Expert annotated large-scale dataset for multitask NLP in Hindi-English code-mixing. In *Findings of the Association for Computational Linguistics: EMNLP 2025* (pp. 7973–7992). Association for Computational Linguistics.

## Licence

The code is released under the MIT Licence (see `LICENSE`). The COMI-LINGUA data remain under their own CC BY 4.0 licence.
