# Retrieval evaluation (k=5)

## all questions (n=185)

| Configuration | MRR | nDCG@5 | Recall@5 | Keyword cov. | ms/query |
|---|---|---|---|---|---|
| BM25 · fixed chunks | 0.976 | 0.957 | 0.966 | 0.989 | 0.0 |
| Dense (LSA) · fixed chunks | 0.932 | 0.932 | 0.972 | 0.986 | 1.3 |
| Hybrid RRF · fixed chunks | 0.974 | 0.957 | 0.966 | 0.986 | 1.4 |
| BM25 · markdown + context headers | 0.975 | 0.954 | 0.969 | 0.964 | 0.0 |
| Dense (LSA) · markdown + context headers | 0.945 | 0.930 | 0.962 | 0.937 | 1.8 |
| Hybrid RRF · markdown + context headers | 0.978 | 0.955 | 0.963 | 0.953 | 1.8 |
| Hybrid RRF · markdown · + MMR | 0.980 | 0.959 | 0.972 | 0.954 | 4.2 |

Per-category breakdown for best config (Hybrid RRF · markdown · + MMR):

| Category | n | MRR | nDCG | Recall | Keyword cov. |
|---|---|---|---|---|---|
| direct_fact | 71 | 1.000 | 1.000 | 1.000 | 1.000 |
| holistic | 7 | 1.000 | 0.656 | 0.552 | 1.000 |
| multi_hop | 13 | 1.000 | 0.865 | 0.846 | 0.808 |
| numerical | 21 | 0.827 | 0.870 | 1.000 | 0.810 |
| paraphrase | 53 | 1.000 | 1.000 | 1.000 | 0.962 |
| relationship | 16 | 1.000 | 0.975 | 1.000 | 1.000 |
| spanning | 2 | 1.000 | 0.960 | 1.000 | 1.000 |
| temporal | 2 | 1.000 | 0.926 | 1.000 | 1.000 |

## hard (paraphrase + multi-hop) (n=66)

| Configuration | MRR | nDCG@5 | Recall@5 | Keyword cov. | ms/query |
|---|---|---|---|---|---|
| BM25 · fixed chunks | 0.985 | 0.969 | 0.977 | 0.985 | 0.0 |
| Dense (LSA) · fixed chunks | 0.891 | 0.904 | 0.985 | 0.992 | 1.5 |
| Hybrid RRF · fixed chunks | 0.977 | 0.968 | 0.985 | 0.992 | 1.6 |
| BM25 · markdown + context headers | 0.992 | 0.971 | 0.985 | 0.924 | 0.0 |
| Dense (LSA) · markdown + context headers | 0.904 | 0.900 | 0.977 | 0.894 | 2.1 |
| Hybrid RRF · markdown + context headers | 1.000 | 0.979 | 0.985 | 0.939 | 2.2 |
| Hybrid RRF · markdown · + MMR | 1.000 | 0.973 | 0.970 | 0.932 | 4.0 |

Per-category breakdown for best config (Hybrid RRF · markdown + context headers):

| Category | n | MRR | nDCG | Recall | Keyword cov. |
|---|---|---|---|---|---|
| multi_hop | 13 | 1.000 | 0.893 | 0.923 | 0.846 |
| paraphrase | 53 | 1.000 | 1.000 | 1.000 | 0.962 |
