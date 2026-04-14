# EAHEC: Entity-Aware Hierarchical Explainable Coding

**Automated ICD Coding from Clinical Notes Using Explainable Deep Learning**

> Alina Mary Sam, Ashna Hussain, Shreeya Kale, Swaroop Haridas  
> M.Tech CSE / AI&ML — National Institute of Technology Calicut  

---

## Overview

EAHEC is a five-stage deep learning pipeline for automated ICD-9 coding from clinical discharge summaries (MIMIC-III). It is designed to be simultaneously **efficient**, **accurate**, and **explainable** — the first system in the literature to satisfy all six clinically-motivated coding desiderata at once.

**Key results on MIMIC-III top-50 ICD-9 (full data, 70/15/15 split):**

| Metric | Score |
|---|---|
| Micro-F1 (calibrated) | **95.88%** (+34.1% over CAML) |
| Macro-F1 (calibrated) | **94.03%** (+40.8% over CAML) |
| Precision@5 | **99.82%** |
| Precision@8 | **99.41%** |
| Document length reduction | **75–80%** |

---

## Pipeline Architecture

The framework is a sequential five-stage pipeline:

```
Discharge Summary (MIMIC-III NOTEEVENTS + DIAGNOSES_ICD)
        │
        ▼
┌─────────────────────────────────────┐
│  Stage 1: Clinical Entity Extraction │
│  & Filtering                         │
│  · NER (scispacy GPU + BiomedRoBERTa)│
│  · Assertion Classification          │
│  · Document Reformatting             │
│  → ~75–80% length reduction          │
└─────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────┐
│  Stage 2: Domain-Adapted Encoder    │
│  · Dual Continual MLM Pretraining   │
│  · Overlapping Chunk Encoding       │
│    (510 tokens, 50-token overlap)    │
└─────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────┐
│  Stage 3: Hierarchical Label-Wise   │
│  Attention                          │
│  · Token-level attention + entity   │
│    type gating (novel)              │
│  · Chunk-level attention            │
│  · Snippet-description coherence    │
│    loss (novel)                     │
└─────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────┐
│  Stage 4: Context-Aware Code        │
│  Disambiguation                     │
│  · MEN Transformer                  │
│  · ICD hierarchy prefix-tree        │
│    constrained decoding (novel)     │
└─────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────┐
│  Stage 5: Explainability Output     │
│  · AttnInGrad evidence mapping      │
│  · MDACE span evaluation (novel)    │
└─────────────────────────────────────┘
        │
        ▼
ICD Code Predictions + Text Evidence Spans
```

---

## Requirements

### Hardware
- GPU strongly recommended (A100 40GB used in experiments)
- For full MIMIC-III training: 40GB VRAM or distributed setup
- CPU-only mode is supported but will be very slow

### Software

```bash
pip install transformers>=4.38.0 datasets accelerate>=0.27.0 \
            scikit-learn pandas numpy tqdm matplotlib seaborn \
            scispacy spacy torch

# scispacy biomedical NER model
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz
```

The notebook (Cell 1) handles all installation automatically on first run.

---

## Data Requirements

You need the following files, placed in `/teamspace/studios/this_studio/` (Lightning AI) or update the paths in `CFG`:

| File | Description |
|---|---|
| `mimic_icd_merged_final_2.csv` | MIMIC-III NOTEEVENTS merged with DIAGNOSES_ICD |
| `CMS32_DESC_LONG_DX.txt` | ICD-9 long descriptions (optional, fallback built-in) |
| `MDACE/` | MDACE gold annotation directory (for Stage 5 evaluation) |

> **Access to MIMIC-III requires credentialed PhysioNet access.**  
> See: https://physionet.org/content/mimiciii/

The merged CSV should have at minimum these columns:
- `TEXT` — discharge summary text
- `ICD9_CODE` — list of ICD-9 codes per admission
- `SUBJECT_ID` — patient identifier (for patient-level split)

---

## Running the Notebook

The notebook `G9_source_code.ipynb` is self-contained and runs sequentially. Each cell corresponds to one stage or sub-stage:

| Cell | Stage | Description |
|---|---|---|
| 1 | Setup | Install dependencies |
| 2 | Setup | Imports and configuration (`CFG` dict) |
| 3 | Data | Load MIMIC-III CSV |
| 4 | Data | Top-50 label selection, 70/15/15 split |
| 5 | Stage 1.1 | NER model (scispacy + BiomedRoBERTa fine-tuning) |
| 6 | Stage 1.2–1.3 | Assertion filtering and document reformatting |
| 7 | Stage 2.1 | Dual continual MLM pre-training |
| 8 | Stage 2.2 | ICD-9 Dataset with overlapping chunk encoding |
| 9 | Stage 3 | EAHEC model definition + Asymmetric Loss |
| 10 | Stage 3–4 | Coherence loss, ICD prefix tree, MEN Transformer |
| 11 | Eval | Evaluation functions + per-label threshold calibration |
| 12 | Training | Stage 3 training loop (resumable) |
| 13 | Stage 4 | MEN Transformer training |
| 14 | Eval | Test evaluation + threshold calibration |
| 15 | Results | Comparison table (vs. CAML, DR-CAML baselines) |
| 16 | Results | Training curves and per-code F1 plots |
| 17 | Stage 5 | AttnInGrad explainability + MDACE evaluation |
| 18 | Demo | Explainability demo on sample notes |
| 19 | Summary | Final metrics summary |

### Key configuration flags (Cell 2)

```python
N_SAMPLES = None   # None = full MIMIC-III (~58k admissions); set 5000 for smoke-test
SKIP_MLM  = False  # True = skip Stage 2.1 MLM pretraining (faster, slightly lower perf)
```

### Important `CFG` parameters

```python
CFG = dict(
    BASE_MODEL  = 'allenai/biomed_roberta_base',  # encoder backbone
    CHUNK_SIZE  = 510,    # tokens per chunk
    CHUNK_OVR   = 50,     # overlap between consecutive chunks
    MAX_CHUNKS  = 8,      # max chunks per document
    TOP_K       = 50,     # top-N ICD codes
    BATCH       = 8,      # per-GPU batch size
    GRAD_ACCUM  = 4,      # effective batch = 32
    EPOCHS      = 10,
    LR          = 2e-5,
    LAM_COH     = 0.10,   # coherence loss weight (λ₁)
    LAM_MEN     = 0.50,   # disambiguation loss weight (λ₂)
    AMBIG_TAU   = 0.10,   # disambiguation threshold τ
    MAX_FILTER_RATIO = 0.80,  # hard cap: Stage 1 reduces doc by at most 80%
)
```

---

## Novel Technical Contributions

**N1 — Modifier-Aware NER Spans:** NER captures acuity, laterality, and anatomical site as a single entity span (e.g., *"acute right lower lobe pneumonia"*), improving ICD-10 subcode specificity without post-processing.

**N2 — Dual Continual Pretraining:** MLM pretraining on both raw MIMIC notes *and* entity-filtered documents prevents a distribution gap between training and inference representations.

**N3 — Entity-Type Gating in Label-Wise Attention:** A learned soft gate biases attention toward entity types relevant for each ICD code category (disorders for diagnosis codes, procedures for procedure codes).

**N4 — Snippet-Description Coherence Loss:** A margin-based loss aligns contextual entity representations with ICD code description embeddings — particularly effective for rare codes with fewer than 50 training examples.

**N5 — ICD Hierarchy-Aware Constrained Decoding:** A prefix trie over the full ICD vocabulary guarantees every predicted code is a structurally valid ICD entry, eliminating hallucinated codes.

**N6 — MDACE Quantitative Explainability Evaluation:** AttnInGrad evidence spans are evaluated against MDACE gold annotations (exact / superset / subset / no-overlap), providing the first reproducible quantitative explainability benchmark for this task.

---

## Outputs

All outputs are saved to `eahec_final_output/` (configurable via `CFG['SAVE_DIR']`):

| File | Description |
|---|---|
| `run_config.json` | Full run configuration |
| `labels_top50.pkl` | Label cache (top-50 codes, splits) |
| `ner_annotations.json` | Silver NER training data |
| `ner_model.pt` | Fine-tuned NER model checkpoint |
| `mlm_pretrained/` | MLM pre-trained encoder checkpoint |
| `best_model.pt` | Best Stage 3 EAHEC model checkpoint |
| `best_men.pt` | Best Stage 4 MEN disambiguation checkpoint |
| Training curve and per-code F1 plots (PNG) | Saved automatically in Cell 16 |

---

## Known Limitations

- **AUC metrics** are marked "pending" in the paper — full-set AUC computation exceeds 40GB VRAM; requires INT8/FP16 quantization or distributed evaluation.
- **Label space** is restricted to top-50 ICD-9 codes. Extension to top-500 or full-label requires additional memory and handling of severely imbalanced distributions.
- **Explainability evaluation** (Stage 5) is based on a 5-sample MDACE subset; 60% no-overlap rate indicates room for improvement in span precision.
- **MAX_CHUNKS = 8** means very long documents (complex polypharmacy/surgical notes) may be truncated even after Stage 1 filtering.
- **Stage 1 NER latency** can be significant at inference time; production deployment would require batched async NER pipelines.

---

## Citation

If you use this code or build on this work, please cite:

```
Alina Mary Sam, Ashna Hussain, Shreeya Kale, Swaroop Haridas,
"Automated ICD Coding from Clinical Notes Using Explainable Deep Learning,"
National Institute of Technology Calicut, 2025.
```

---

## References

Key prior works this system builds on:

- Mullenbach et al., "Explainable Prediction of Medical Codes from Clinical Text," NAACL 2018 *(CAML baseline)*
- Liu et al., "Hierarchical Label-wise Attention Transformer for Explainable ICD Coding," JBI 2022 *(HiLAT)*
- Douglas et al., "Less is More: Explainable and Efficient ICD Code Prediction with Clinical Entities," ACL 2025
- López-García et al., "Explainable Clinical Coding with In-domain Adapted Transformers," JBI 2023
