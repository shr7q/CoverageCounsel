"""
Week 1 real-data swap: pull real CMS coverage determinations and Medicare
Benefit Policy Manual sections out of the bulk downloads in data/ and write
them as plain-text docs in data/real_docs/, alongside (not replacing) the
synthetic data/sample_docs/ set.

Sources (already present in data/):
  - ncd_csv.zip           -> extracted to data/_extract/ncd/ncd_trkg.csv
  - current_lcd_csv.zip   -> extracted to data/_extract/lcd/lcd.csv,
                             lcd_x_contractor.csv, contractor.csv
  - chapter 1.pdf, chapter15.pdf (Medicare Benefit Policy Manual)

Covers the same three procedure areas as the synthetic sample docs:
bariatric surgery, home oxygen therapy, knee replacement.

Run from the repo root:
    python scripts/extract_real_docs.py
"""

import csv
import html
import os
import re

from pypdf import PdfReader

csv.field_size_limit(10_000_000)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
EXTRACT_DIR = os.path.join(DATA_DIR, "_extract")
OUT_DIR = os.path.join(DATA_DIR, "real_docs")

NCD_IDS = {"57", "169"}  # Bariatric Surgery, Home Use of Oxygen
LCD_IDS = {"35022", "33797", "36575"}  # Bariatric, Oxygen/Oxygen Equipment, Total Knee Arthroplasty


def clean_html(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def slugify(text: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()
    return text[:50]


def load_lcd_jurisdictions() -> dict[str, list[str]]:
    contractors = {}
    with open(os.path.join(EXTRACT_DIR, "lcd", "contractor.csv"), encoding="latin-1") as f:
        for row in csv.DictReader(f):
            contractors[row["contractor_id"]] = row["contractor_bus_name"]

    jurisdictions: dict[str, set[str]] = {}
    with open(os.path.join(EXTRACT_DIR, "lcd", "lcd_x_contractor.csv"), encoding="latin-1") as f:
        for row in csv.DictReader(f):
            if row["lcd_id"] not in LCD_IDS:
                continue
            name = contractors.get(row["contractor_id"], row["contractor_id"])
            jurisdictions.setdefault(row["lcd_id"], set()).add(name)

    return {lcd_id: sorted(names) for lcd_id, names in jurisdictions.items()}


def extract_ncds() -> dict[str, str]:
    docs = {}
    path = os.path.join(EXTRACT_DIR, "ncd", "ncd_trkg.csv")
    with open(path, encoding="latin-1") as f:
        for row in csv.DictReader(f):
            if row["NCD_id"] not in NCD_IDS:
                continue

            title = row["NCD_mnl_sect_title"]
            lines = [
                "CMS NATIONAL COVERAGE DETERMINATION (NCD)",
                f"NCD ID: {row['NCD_id']} | Manual Section: {row['NCD_mnl_sect']} | {title}",
                f"Effective Date: {row['NCD_efctv_dt'] or 'not specified'}",
                f"Termination Date: {row['NCD_trmntn_dt'] or 'none (currently active)'}",
                "",
                "Indications and Limitations of Coverage",
                clean_html(row["indctn_lmtn"]),
            ]
            if row.get("itm_srvc_desc", "").strip():
                lines += ["", "Item/Service Description", clean_html(row["itm_srvc_desc"])]
            if row.get("xref_txt", "").strip():
                lines += ["", "Cross-References", clean_html(row["xref_txt"])]
            if row.get("othr_txt", "").strip():
                lines += ["", "Other", clean_html(row["othr_txt"])]

            fname = f"ncd_{row['NCD_id']}_{slugify(title)}.txt"
            docs[fname] = "\n".join(lines)
    return docs


def extract_lcds() -> dict[str, str]:
    jurisdictions = load_lcd_jurisdictions()
    docs = {}
    path = os.path.join(EXTRACT_DIR, "lcd", "lcd.csv")
    with open(path, encoding="latin-1") as f:
        for row in csv.DictReader(f):
            if row["lcd_id"] not in LCD_IDS:
                continue

            title = row["title"]
            juris = jurisdictions.get(row["lcd_id"], [])
            eff_date = row["rev_eff_date"] or row["orig_det_eff_date"] or "not specified"
            lines = [
                "CMS LOCAL COVERAGE DETERMINATION (LCD)",
                f"LCD ID: {row['lcd_id']} | {title}",
                f"Effective Date: {eff_date}",
                f"Contractor Jurisdiction(s) This Version Applies To: "
                f"{'; '.join(juris) if juris else 'not specified'}",
                "NOTE: this is one MAC jurisdiction's LCD. Coverage criteria, "
                "documentation requirements, or thresholds may differ in other "
                "jurisdictions -- check the corresponding LCD for the beneficiary's region.",
                "",
                "Indications and Limitations of Coverage",
                clean_html(row["indication"]),
            ]
            if row.get("doc_reqs", "").strip():
                lines += ["", "Documentation Requirements", clean_html(row["doc_reqs"])]
            if row.get("diagnoses_support", "").strip():
                lines += ["", "ICD-10 Codes That Support Medical Necessity", clean_html(row["diagnoses_support"])]
            if row.get("coding_guidelines", "").strip():
                lines += ["", "Coding Guidelines", clean_html(row["coding_guidelines"])]

            fname = f"lcd_{row['lcd_id']}_{slugify(title)}.txt"
            docs[fname] = "\n".join(lines)
    return docs


PDF_SECTIONS = [
    {
        "pdf": "chapter 1.pdf",
        "chapter_title": "Medicare Benefit Policy Manual, Chapter 1 - Inpatient Hospital Services Covered Under Part A",
        "section_label": "Section 10.2 - Hospital Inpatient Admission Order and Certification",
        "start_pattern": r"10\.2\s.\s*Hospital Inpatient Admission Order and Certification",
        "end_pattern": r"20 - Nursing and Other Services",
        "out": "bpm_ch1_sec10.2_admission_order_and_certification.txt",
        "skip_first_match": True,  # first hit is the table of contents entry
    },
    {
        "pdf": "chapter15.pdf",
        "chapter_title": "Medicare Benefit Policy Manual, Chapter 15 - Covered Medical and Other Health Services",
        "section_label": "Section 110-110.1 - Durable Medical Equipment, General and Definition",
        "start_pattern": r"110 - Durable Medical Equipment - General",
        "end_pattern": r"110\.2 - Repair",
        "out": "bpm_ch15_sec110_durable_medical_equipment_general.txt",
        "skip_first_match": True,  # first hit is the table of contents entry
    },
]


def extract_pdf_sections() -> dict[str, str]:
    docs = {}
    for spec in PDF_SECTIONS:
        pdf_path = os.path.join(DATA_DIR, spec["pdf"])
        reader = PdfReader(pdf_path)
        full_text = "\n".join(page.extract_text() for page in reader.pages)

        starts = [m.start() for m in re.finditer(spec["start_pattern"], full_text)]
        if len(starts) <= (1 if spec.get("skip_first_match") else 0):
            print(f"  WARNING: could not find body section in {spec['pdf']}, skipping")
            continue
        body_start = starts[1] if spec.get("skip_first_match") else starts[0]

        ends = [m.start() for m in re.finditer(spec["end_pattern"], full_text) if m.start() > body_start]
        section_text = full_text[body_start:ends[0]] if ends else full_text[body_start:body_start + 8000]

        section_text = re.sub(r"[ \t]+", " ", section_text)
        section_text = re.sub(r"\n\s*\n+", "\n\n", section_text).strip()

        lines = [spec["chapter_title"], spec["section_label"], "", section_text]
        docs[spec["out"]] = "\n".join(lines)
    return docs


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    all_docs = {}
    all_docs.update(extract_ncds())
    all_docs.update(extract_lcds())
    all_docs.update(extract_pdf_sections())

    for fname, text in sorted(all_docs.items()):
        out_path = os.path.join(OUT_DIR, fname)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"wrote {fname} ({len(text)} chars)")

    print(f"\n{len(all_docs)} real documents written to {os.path.abspath(OUT_DIR)}")


if __name__ == "__main__":
    main()
