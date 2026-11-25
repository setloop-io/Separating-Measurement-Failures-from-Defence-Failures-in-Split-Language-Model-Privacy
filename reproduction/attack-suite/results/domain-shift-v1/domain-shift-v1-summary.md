# domain-shift-v1: seam inversion across domains (GPT-2 split 6/12, vmi arm)

Gate: frozen-gate.v2 (z=2.5391848136513127, threshold +1.0 pp, >=3 seeds; control = label_free_majority).

| corpus | mode | docs | n tok | control % | recovery % (mean seeds) | upper-95 excess pp | floor pp | verdict | wall s |
|---|---|---|---|---|---|---|---|---|---|
| code | training/none | 96 | 768 | 20.05 | 68.88 | 52.898 | 3.911 | FAIL | 71.6 |
| code | inference/none | 96 | 3072 | 20.31 | 64.94 | 46.782 | 1.904 | FAIL | 71.6 |
| clinical | training/none | 96 | 768 | 6.25 | 100.00 | 93.750 | 2.603 | FAIL | 56.6 |
| clinical | inference/none | 96 | 3072 | 6.25 | 100.00 | 93.750 | 1.203 | FAIL | 56.6 |
| wikitext-reference | training/none | 96 | 768 | 4.69 | 65.76 | 65.268 | 2.342 | FAIL | 63.9 |
| wikitext-reference | inference/none | 96 | 3072 | 2.64 | 63.25 | 62.791 | 0.839 | FAIL | 63.9 |
