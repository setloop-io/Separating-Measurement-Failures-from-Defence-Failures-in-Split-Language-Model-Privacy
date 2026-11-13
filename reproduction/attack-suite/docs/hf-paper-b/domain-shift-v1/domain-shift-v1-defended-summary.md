# domain-shift-v1: seam inversion across domains (GPT-2 split 6/12, vmi arm)

Gate: frozen-gate.v2 (z=2.5391848136513127, threshold +1.0 pp, >=3 seeds; control = label_free_majority).

| corpus | mode | docs | n tok | control % | recovery % (mean seeds) | upper-95 excess pp | floor pp | verdict | wall s |
|---|---|---|---|---|---|---|---|---|---|
| code | training/gaussian | 96 | 768 | 20.05 | 68.75 | 52.774 | 3.911 | FAIL | 131.6 |
| code | training/none | 96 | 768 | 20.05 | 68.88 | 52.898 | 3.911 | FAIL | 131.6 |
| code | inference/gaussian | 96 | 3072 | 20.31 | 65.69 | 47.518 | 1.904 | FAIL | 131.6 |
| code | inference/none | 96 | 3072 | 20.31 | 64.94 | 46.782 | 1.904 | FAIL | 131.6 |
| code | inference: gaussian minus none (paired) | 96 | - | - | 0.75 pp [-9.15, 10.65] | - | - | - | 131.6 |
| code | training: gaussian minus none (paired) | 96 | - | - | -0.13 pp [-16.22, 15.96] | - | - | - | 131.6 |
| clinical | training/gaussian | 96 | 768 | 6.25 | 100.00 | 93.750 | 2.603 | FAIL | 127.6 |
| clinical | training/none | 96 | 768 | 6.25 | 100.00 | 93.750 | 2.603 | FAIL | 127.6 |
| clinical | inference/gaussian | 96 | 3072 | 6.25 | 100.00 | 93.750 | 1.203 | FAIL | 127.6 |
| clinical | inference/none | 96 | 3072 | 6.25 | 100.00 | 93.750 | 1.203 | FAIL | 127.6 |
| clinical | inference: gaussian minus none (paired) | 96 | - | - | 0.00 pp [0.00, 0.00] | - | - | - | 127.6 |
| clinical | training: gaussian minus none (paired) | 96 | - | - | 0.00 pp [0.00, 0.00] | - | - | - | 127.6 |
| wikitext-reference | training/gaussian | 96 | 768 | 4.69 | 66.02 | 65.519 | 2.342 | FAIL | 132.3 |
| wikitext-reference | training/none | 96 | 768 | 4.69 | 65.76 | 65.268 | 2.342 | FAIL | 132.3 |
| wikitext-reference | inference/gaussian | 96 | 3072 | 2.64 | 64.58 | 64.105 | 0.839 | FAIL | 132.3 |
| wikitext-reference | inference/none | 96 | 3072 | 2.64 | 63.25 | 62.791 | 0.839 | FAIL | 132.3 |
| wikitext-reference | inference: gaussian minus none (paired) | 96 | - | - | 1.33 pp [-9.76, 12.43] | - | - | - | 132.3 |
| wikitext-reference | training: gaussian minus none (paired) | 96 | - | - | 0.26 pp [-5.06, 5.58] | - | - | - | 132.3 |
