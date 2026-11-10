# Anonymization

This package was prepared for double-blind review. Author names, affiliations, e-mail addresses, local
paths, repository URLs and the project's working name were removed or replaced mechanically; no code
logic was changed. The literal substitutions, applied
in this order to every text file and path, were:

| What was replaced | Replacement | Occurrences |
|---|---|---|
| graph run-directory name | `graphs_v2` | 13 |
| method id of the typed pooled estimator | `typed_pooled` | 73 |
| horizon-family key of the typed pooled estimator | `"typed",` | 2 |
| concept-bottleneck class name | `ConceptCBM` | 10 |
| Colab driver directory name | `colab/` | 6 |
| project working name (prose) | `CLBench` | 94 |
| project working name (upper case) | `CLBENCH` | 0 |
| project working name (capitalized) | `CLBench` | 0 |
| package / module name | `clbench` | 423 |
| time-zone label of protocol time stamps | ` local time` | 15 |

Consequences:

* the Python package is `clbench` (`import clbench`), the method id of the typed pooled estimator is
  `typed_pooled` (the paper's VARX with a dagger in Table 1), and the graph run directories are
  `runs/iclr/graphs_v2` and `runs/iclr/graphs_v2_fast`;
* hashes of the shipped protocol texts and registered-code copies differ from the registered hashes where
  a file contained a substituted token (`PROTOCOLS.md` lists both);
* the license file was replaced by an MIT license held by "The Authors".

Not included (would identify the authors, or is outside the scope of the benchmark and graph construction):

* the repository LICENSE (Apache-2.0 with the copyright holder's name) - replaced by MIT, "The Authors"
* `data/vitaldb_manifest.csv` (produced by `scripts/14_vitaldb_ingest.py`; contains per-case fields copied from VitalDB's clinical_data.csv)
* run outputs under `runs/` (their `meta.json` / logs hold local absolute paths); regenerate with the commands in README.md
* concept-model training and faithfulness code (`scripts/03*`, `scripts/04*` except 04b, `clbench.eval.faithfulness_cf`, ...), the MIMIC-IV pipeline, the API and the front-end
* the Colab notebook that trained the concept models (it contains a notebook URL) and the zipped code bundle the Colab drivers unpack
* `.env`, `.git`, `.DS_Store`, `__pycache__`, editor and cache directories
* the outputs of the extended analyses (`runs/extended/**`) and the authors' result notes, which quote them; regenerate with the commands in README.md, section "Extended analyses"
