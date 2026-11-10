# Pre-registered protocols and their hashes

Every confirmatory analysis in the paper was specified in a written protocol that was hashed (SHA-256) before
its confirmatory run; the paper's appendix on pre-specified protocols lists the first eight hexadecimal
digits of each hash. The protocol texts are shipped here under the same relative paths the runners use
(`runs/iclr/<analysis>/protocol*.md`).

**Anonymization and hashes.** The shipped copies differ from the registered originals only by the literal
substitutions listed in `ANONYMIZATION.md` (project name, package name, method id, run-directory names).
Those substitutions change the bytes, so the SHA-256 of a shipped copy differs from the registered hash
whenever the original contained one of the substituted tokens. The table lists both. The `protocol.sha256`
files next to the shipped copies hold the hash of the *shipped* text so that the runners' integrity checks
pass; the registered originals (with the registered hashes) will be published with the de-anonymized
release, where the diff can be checked to be exactly the listed substitutions.

| Analysis | Registered SHA-256 (paper) | Shipped file | Shipped SHA-256 | Substitutions |
|---|---|---|---|---|
| ClosedLoopBench 1.1: 29 relations, 2 s, 3,442 cases (Table 1) | `418281f18a34c6c9c66111250b0d17d8f9296e8718c469a08515ba386793d10e` | `runs/iclr/closed_loop_v4/protocol.md` | `1a9da7cec9e31383b4fdde48afc4a3862544defffc2e610759c5fbe613e4fbea` | method id of the typed pooled estimator x2; package / module name x13; time-zone label of protocol time stamps x1 |
| ClosedLoopBench 1.0: 18 relations, 30 s (3,442 cases) and 2 s (500 cases) | `5e578794942fd1d78bedcf7d0e2fde1cbff9a430ca0917ffc9b86e5c342d47a9` | `runs/iclr/closed_loop_v3/protocol.md` | `5cfb4bdc3f4928d10c77d1e72b6e39d4213bf14e310cb45401dd10abd5fe72a7` | method id of the typed pooled estimator x6; project working name (prose) x2; package / module name x11; time-zone label of protocol time stamps x1 |
| Slow-response specificity (early/late split, 11 relations) | `482db32fca3503f80c391d7414819176cd8b3e768a70ee0f60d51818a27833d6` | `runs/iclr/closed_loop_v2/protocol.md` | `da0097f515303753d6b04027e1be95352a79c8998caa82f9a27feb4cba8b623a` | method id of the typed pooled estimator x2; project working name (prose) x2; package / module name x2; time-zone label of protocol time stamps x1 |
| Clinician-in-the-loop simulator (14 hypotheses) | `7b75ce95e99dee0afe05440c8aa0aaf9559ab0e83d608e0ed335dbb4443443b5` | `runs/iclr/sim_closed_loop/v2/protocol.md` | `554e1daaa34a0a1b02a27a22cab4d807d0e2193a136521f7a9d686d0616d3a3a` | package / module name x1; time-zone label of protocol time stamps x1 |
| Simulator amendment: descriptive phase-grid analysis | `e9e25964ae61cb2b9d75e4a481a0d4343e362af6d795190f4d05824e2191237c` | `runs/iclr/sim_closed_loop/v2/protocol_amendment1.md` | `7bb24781641fbfdc26ae5cd48971addb5ab42806237aeaa1c9158d933a9e10c7` | package / module name x1; time-zone label of protocol time stamps x1 |
| Concept-level typed relations (hazard model, nulls, sign checks) | `43947eaf29f6a2d026c163b7a37936cff42f2628d3e7174da093823231cfa5bc` | `runs/iclr/typed_edges/protocol.md` | `ed7ea19ea27e46f65afcbc1e41b0ded103849f49aee25e993d045c41c1956118` | package / module name x3 |
| Patient graphs, concept layer (pointer rule, false instances; Table 2) | `c767688d2e7b1dcc4f8897f7984c7e80ce39b09dc56e074ff8e218c0456156f3` | `runs/iclr/graphs_v2/protocol.md` | `eb32d159f997324b668c0606034c6f59725bfa9c579c87925766d22f61bd85a8` | project working name (prose) x1; package / module name x1; time-zone label of protocol time stamps x1 |
| Beat concepts and fast couplings in graphs | `eab2d58343b260510d51c39d0794fdb9d1baa0a070b498104ab18f46dc21d207` | `runs/iclr/graphs_v2_fast/protocol.md` | `e2ff4ead15ff4024b90222db299b7c2b02bf68e676626c2a37aea1003a146319` | graph run-directory name x1; project working name (prose) x1; package / module name x1; time-zone label of protocol time stamps x1 |
| Addendum: PAT-SBP edge, prolonged PAT | `20f873f53108df240e918a842af93ebae9cebc9ddf19060d6ebb0c662207fd51` | `runs/iclr/graphs_v2_fast/protocol_pat.md` | `714d38f5faf72ab741477954897898cb9cf85bf19d5dfa998824923f994276fe` | package / module name x1; time-zone label of protocol time stamps x1 |
| Fast couplings at scale (gates, epoch statistics, baroreflex bar) | `af14df0605c83ae6bfe10f47c687ca6b12b2ea53d30a84b71c66f9e79a75c4c5` | `runs/iclr/fast_vitaldb/protocol.md` | `0f8ee922d8b2df45893ec173045befd5b33c8266cea56172c04c437f41a80730` | package / module name x1; time-zone label of protocol time stamps x1 |
| PAT analyses (delay-aware PAT, slope sign bar, specificity rule) | `230b57c55e6a483bf9a056330046dee55eabacf453863bbef6c214ce56efff8d` | `runs/iclr/pat_vitaldb/protocol.md` | `7fdbeb3b00173be43803588e3005e299d51b34799395d67172c728ac09f54813` | package / module name x1; time-zone label of protocol time stamps x1 |
| Fast-coupling specificity, MIMIC-IV-WDB and VitalDB | `6d88096d2b1418517f342cfdd550a4a2a32dc2ae3a3403cca09ce81c8bb7d417` | `runs/iclr/fast_specificity/protocol.md` | `279511e0b3a6cf9405c8ba8123fbfd15bec564db34aa8cf4cf25fed37c07452a` | package / module name x3; time-zone label of protocol time stamps x1 |
| Concept heads, six models (not reproducible from this package: training code not included) | `ce7ed6fe59ad45d41c4a7a46294532481fefb8dbfad734ee52770daacf890fe5` | `runs/iclr/nodes_multiseed/protocol.md` | `8373454e2d92d50f1f60d186a128a5dd8ea50fe681de11024311e8893190f3b9` | package / module name x1; time-zone label of protocol time stamps x1 |
| Counterfactual faithfulness (not reproducible from this package: concept-model code not included) | `6037da7332c06e027a0c3bf9ea44878dc825df80863faa0639f3b2adff05cf9d` | `runs/iclr/faithfulness/protocol.md` | `cca0c880f27740f77c5f488526907174c50e99e0296a7854a9d7c3bce18aafde` | project working name (prose) x1; package / module name x2; time-zone label of protocol time stamps x1 |

## Registered code snapshot (ClosedLoopBench 1.1)

`runs/iclr/closed_loop_v4/registered_code/` holds byte copies of the benchmark code at registration
(listed with their hashes at the end of `runs/iclr/closed_loop_v4/protocol.md`). The copies shipped here
carry the same substitutions; `REGISTERED_CODE.md` in that directory lists the registered and the shipped
hash of each file.
