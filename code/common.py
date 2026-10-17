"""
Shared loading and preprocessing for the gene-expression part of the project.

Everything in this file is label-free (it never looks at Disease_State), except
`load_expression`, which attaches the labels so the scripts can use them later.
That matters because every step here is run on all samples before
cross-validation, so none of it should be able to leak outcome information.
"""
import re
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*keyword-only.*")
warnings.filterwarnings("ignore", message=".*penalty.*deprecated.*")
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Datasets"
RESULTS = ROOT / "results"
FIG = RESULTS / "figures"
TAB = RESULTS / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

ALL_GENES = ["MTHFR", "TYMS", "MTR", "MTRR", "CYP1A1", "GSTM1", "GSTT1",
             "GSTP1", "TLR2", "TLR4", "TLR7", "TLR9", "STAT4"]
RAW_SETS = ["GSE10325", "GSE38351"]          # raw intensities -> log2
SEED = 42
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)


def read_raw_database(path=DATA / "database.csv"):
    """database.csv has unquoted commas inside the title field, so parse by position."""
    rows = []
    with open(path, encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split(",")
        n_genes = len(header) - 4
        for line in f:
            parts = line.rstrip("\n").split(",")
            rows.append(parts[:3] + [",".join(parts[3:-n_genes])] + parts[-n_genes:])
    df = pd.DataFrame(rows, columns=header)
    df[header[4:]] = df[header[4:]].astype(float)
    return df


def load_expression(fix_labels=True):
    """
    Raw curated expression values plus the curated Disease_State labels.

    fix_labels: GSE72747 is a lupus nephritis cohort sampled at T0, T12 and T24.
    The curated file labels the T0 samples "Healthy Control", but every sample in
    that series comes from an LN patient, so we relabel them as SLE.
    """
    raw = read_raw_database()
    lab = pd.read_csv(DATA / "gene_expression_normalized.csv",
                      usecols=["GSM_ID", "Disease_State", "Data_Type"])
    df = raw.merge(lab, on="GSM_ID", how="left", validate="one_to_one")
    assert df["Disease_State"].notna().all()
    df["Disease_State_curated"] = df["Disease_State"]
    if fix_labels:
        m = df["GSE_ID"].eq("GSE72747")
        df.loc[m, "Disease_State"] = "SLE"
    df["batch"] = df["GSE_ID"] + "_" + df["GPL"]
    df["subject"] = subject_ids(df)
    return df


def subject_ids(df):
    """
    Map each GSM to the person it came from, using the GEO sample titles.

    Several series profile the same person more than once: GSE10325 (three cell
    types per donor), GSE38351 (monocytes from the same healthy donors under up to
    five culture conditions, plus technical replicates), GSE39088 and GSE72747
    (three time points per patient) and GSE32591 (tubulointerstitial and glomerular
    biopsies from the same kidney). Treating those as independent samples lets the
    same person sit in both the training and the test fold.
    Where the title does not identify a person we assume one sample per person.
    """
    out = []
    for gse, title, gsm in zip(df["GSE_ID"], df["title"], df["GSM_ID"]):
        t = title.lower()
        sid = gsm
        if gse == "GSE10325":
            m = re.match(r"(healthy|sle) .* (\d+)$", t)
            sid = f"{m.group(1)}_{m.group(2)}"
        elif gse == "GSE38351":
            m = re.search(r"(sle patient|ra patient|healthy donor) number (\d+)", t)
            sid = f"{m.group(1).split()[0]}_{m.group(2)}"
        elif gse == "GSE39088":
            m = re.search(r"sle patient (\d+)", t)
            if m:
                sid = f"sle_{m.group(1)}"
        elif gse == "GSE72747":
            m = re.search(r"\((sle\d+) t\d+\)", t)
            sid = m.group(1)
        elif gse == "GSE32591":
            m = re.match(r"(tub|glom)_(ln|ld)_(\d+)", t)
            num = int(m.group(3))
            # glomerular IDs are the tubular ID + 100 (LN) or + 105 (LD)
            if m.group(1) == "glom":
                num -= 100 if m.group(2) == "ln" else 105
            sid = f"{m.group(2)}_{num}"
        out.append(f"{gse}:{sid}")
    return np.array(out)


def harmonize_scale(df, genes=ALL_GENES):
    """
    Put all series on a log2-like scale (label-free, per sample).
    -1.00 in the raw-intensity series is a missing-value flag; values <= 0 there
    cannot be logged. The other four series are already log-scale or centred.
    """
    df = df.copy()
    m = df["GSE_ID"].isin(RAW_SETS)
    vals = df.loc[m, genes]
    df.loc[m, genes] = np.log2(vals.where(vals > 0))
    return df


def combat_ok_genes(df, genes=ALL_GENES):
    """Genes ComBat can model: no missing values and non-zero SD in every batch."""
    keep = []
    for g in genes:
        sd = df.groupby("batch")[g].std()
        if not (sd.isna().any() or (sd == 0).any() or df[g].isna().any()):
            keep.append(g)
    return keep


def disease_covariates(labels):
    """Dummy-coded disease design (Healthy Control is the reference)."""
    s = pd.Series(np.asarray(labels))
    return np.column_stack([(s == "SLE").values, (s == "Other/Unknown").values]).astype(float)


def savefig(fig, name):
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", dpi=200, bbox_inches="tight")
