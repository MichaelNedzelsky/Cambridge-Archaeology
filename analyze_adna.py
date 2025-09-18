
"""
Analyze Cambridgeshire aDNA dataset

Generates:
Table 2: Summary of Identified Kinship Pairs and Haplogroup Diversity by Site

Columns in output:
Site, Kinship Pairs (Degree), Y-chr Samples (N), Y-chr Diversity (H), mtDNA Samples (N), mtDNA Diversity (H)

Diversity uses Nei's haplotype diversity:
    H = (n/(n-1)) * (1 - sum(p_i^2)), for n > 1; else H = 0
"""

import argparse
import pandas as pd
import numpy as np
import re
from collections import Counter, defaultdict

def haplotype_diversity(haplos):
    """Nei's haplotype diversity: H = (n/(n-1)) * (1 - sum(p_i^2)) for n>1, else 0."""
    vals = [str(x).strip() for x in haplos if str(x).strip() and str(x).strip().lower() not in {"unknown", "nan"}]
    n = len(vals)
    if n <= 1:
        return 0.0, n
    counts = Counter(vals)
    freqs_sq_sum = sum((c / n) ** 2 for c in counts.values())
    H = (n / (n - 1)) * (1.0 - freqs_sq_sum)
    return H, n

REL_WORDS = {
    "father-son": r"\bfather[-\s]?son\b",
    "mother-son": r"\bmother[-\s]?son\b",
    "siblings": r"\bsiblings?\b|\bbrother[-\s]?sister\b|\bsisters?\b|\bbrothers?\b",
    "parent-offspring": r"\bparent[-\s]?offspring\b|\bPO\b",
    "cousins": r"\bcousins?\b",
    "avuncular": r"\bavuncular\b|\buncle[-\s]?nephew\b|\baunt[-\s]?nephew\b|\baunt[-\s]?niece\b",
    "grandparent-grandchild": r"\bgrand(parent|mother|father)[-\s]?(grand)?(child|son|daughter)\b|\bGP[-\s]?GC\b",
}

DEGREE_PAT = r"\b(1st|first)[-\s]?(degree)?\b|\b(2nd|second)[-\s]?(degree)?\b|\b(3rd|third)[-\s]?(degree)?\b|\b(4th|fourth)[-\s]?(degree)?\b"

def parse_kinship_text(text):
    """Return (relation_label(s), degree_labels) extracted from free text."""
    if not isinstance(text, str):
        return [], []
    t = text.lower()
    rels = []
    for label, pat in REL_WORDS.items():
        if re.search(pat, t, flags=re.IGNORECASE):
            rels.append(label)
    # Degrees
    degs = []
    for m in re.finditer(DEGREE_PAT, t, flags=re.IGNORECASE):
        seg = m.group(0)
        # Normalize
        seg = seg.replace("first", "1st").replace("second", "2nd").replace("third", "3rd").replace("fourth", "4th")
        seg = re.sub(r"\s*degree\s*", "", seg)
        seg = seg.strip()
        if seg:
            degs.append(seg)
    return rels, degs

def summarize_kin(df_site, kinship_data=None):
    """
    Build a compact kinship summary string like:
    'Father-Son (1st, x2), Mother-Son (1st), 2nd, 3rd'
    Uses 'Notes' and (if present) 'Kinship' columns to extract relations and degrees.
    Also uses kinship_data DataFrame if provided.
    """
    rel_counter = defaultdict(list)  # relation -> list of degrees
    degree_only = Counter()

    # Process kinship data if available
    if kinship_data is not None and not kinship_data.empty:
        for _, row in kinship_data.iterrows():
            # Extract from 'Degree' column
            degree_text = str(row.get('Degree', ''))
            if 'First' in degree_text or '1st' in degree_text:
                deg = '1st'
            elif 'Second' in degree_text or '2nd' in degree_text:
                deg = '2nd'
            elif 'Third' in degree_text or '3rd' in degree_text:
                deg = '3rd'
            elif 'Fourth' in degree_text or '4th' in degree_text:
                deg = '4th'
            else:
                deg = ''
            
            # Extract from 'Likely relationship' column
            rel_text = str(row.get('Likely relationship', ''))
            rels, _ = parse_kinship_text(rel_text)
            
            if rels:
                for r in set(rels):
                    if deg:
                        rel_counter[r].append(deg)
                    else:
                        rel_counter[r].append('')
            elif deg:
                degree_only[deg] += 1
    
    # Also process from main dataframe
    text_cols = [c for c in df_site.columns if c.lower() in {"kinship", "notes"}]

    for _, row in df_site.iterrows():
        text_blob = " ".join([str(row.get(c, "")) for c in text_cols if c in df_site.columns and pd.notna(row.get(c))])
        rels, degs = parse_kinship_text(text_blob)
        if rels:
            if degs:
                for r in set(rels):
                    rel_counter[r].extend(degs)
            else:
                for r in set(rels):
                    rel_counter[r].append("")
        else:
            for d in degs:
                degree_only[d] += 1

    parts = []
    for rlabel, deg_list in rel_counter.items():
        dcounts = Counter([d for d in deg_list if d])
        if dcounts:
            deg_part = ", ".join(sorted([f"{d}" + (f", x{cnt}" if cnt > 1 else "") for d, cnt in dcounts.items()]))
            parts.append(f"{rlabel.replace('-', ' ').title()} ({deg_part})")
        else:
            parts.append(f"{rlabel.replace('-', ' ').title()}")

    for d, cnt in degree_only.items():
        parts.append(f"{d}" + (f", x{cnt}" if cnt > 1 else ""))

    return ", ".join(parts) if parts else ""

def main():
    parser = argparse.ArgumentParser(description="Summarize aDNA kinship pairs and haplogroup diversity by site.")
    parser.add_argument("--input", required=True, help="Path to combined CSV (columns include 'Site' or 'Site Group', 'Y-chr Haplogroup', 'mtDNA Haplogroup', 'Notes').")
    parser.add_argument("--kinship", help="Path to kinship details CSV (optional).")
    parser.add_argument("--output", required=True, help="Path to write the summary CSV.")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    
    # Load kinship data if provided
    kinship_df = None
    if args.kinship:
        # The kinship CSV has multi-row headers, skip first row
        kinship_df = pd.read_csv(args.kinship, skiprows=1)
        # Rename the first column to 'Site'
        kinship_df.columns = ['Site', 'Individual 1', 'Individual 2', 'Y_chr_1', 'Y_chr_2', 'Y_identical',
                               'mtDNA_1', 'mtDNA_2', 'mt_identical', 'blank', 'Degree', 'Predicted', 'Likely relationship']

    # Harmonize site column name
    lower_cols = {c.lower(): c for c in df.columns}
    site_col = lower_cols.get("site group") or lower_cols.get("site")
    if not site_col:
        raise ValueError("No 'Site' or 'Site Group' column found in input")

    df["Site"] = df[site_col]

    y_col = lower_cols.get("y-chr haplogroup", "Y-chr Haplogroup")
    mt_col = lower_cols.get("mtdna haplogroup", "mtDNA Haplogroup")
    sex_col = lower_cols.get("sex")  # optional

    rows = []
    for site, g in df.groupby("Site", dropna=False):
        # Get kinship data for this site if available
        site_kinship = None
        if kinship_df is not None:
            site_kinship = kinship_df[kinship_df['Site'].str.lower() == site.lower()] if isinstance(site, str) else pd.DataFrame()
        # Y-chr: restrict to males if Sex available
        y_vals = g[y_col]
        if sex_col in df.columns:
            try:
                mask_male = g[sex_col].astype(str).str.lower().str.startswith("m")
                y_vals = g.loc[mask_male, y_col]
            except Exception:
                pass
        Hy, Ny = haplotype_diversity(y_vals.tolist())

        # mtDNA
        Hm, Nm = haplotype_diversity(g[mt_col].tolist())

        kin_summary = summarize_kin(g, site_kinship)

        rows.append({
            "Site": site,
            "Kinship Pairs (Degree)": kin_summary,
            "Y-chr Samples (N)": Ny,
            "Y-chr Diversity (H)": round(Hy, 2),
            "mtDNA Samples (N)": Nm,
            "mtDNA Diversity (H)": round(Hm, 2),
        })

    out = pd.DataFrame(rows).sort_values("Site")
    out.to_csv(args.output, index=False)
    print(out.to_string(index=False))

if __name__ == "__main__":
    main()
