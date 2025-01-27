import csv
from pathlib import Path
from PyPDF2 import PdfReader
from ai_utils import (
    extract_meta_info,
    extract_result_details,
    extract_user_items
)

# Replace `get_pdf_text` with a local PDF text extraction function
def extract_text_from_pdf(pdf_path: Path) -> str:
    """
    Extracts text from a PDF file using PyPDF2.

    Args:
        pdf_path (Path): Path to the PDF file.

    Returns:
        str: The extracted text.
    """
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        print(f"[ERROR] Failed to extract text from {pdf_path}: {e}")
        return ""

def run_three_pass_extraction(
    pdf_folder: str,
    output_csv: str = "extraction_results.csv",
    user_items: list[str] = None,
    model: str = "gpt-4o"
):
    """
    For each PDF in pdf_folder:
      1) Extract paper text
      2) Pass 1: Meta info => basic fields + # main results + semicolon list
      3) Pass 3: If user_items is non-empty => gather them in a single GPT call
      4) For each main result => Pass 2 => effect sizes, etc.
      5) Write each main result as a separate row
         (replicating the user_items data in each row).
    """
    if user_items is None:
        user_items = []

    pdf_path = Path(pdf_folder)
    if not pdf_path.exists():
        print(f"[ERROR] Folder not found: {pdf_folder}")
        return

    results_rows = []

    for pdf_file in pdf_path.glob("*.pdf"):
        print(f"Processing {pdf_file.name}")
        try:
            paper_text = extract_text_from_pdf(pdf_file)
        except Exception as e:
            print(f"[ERROR] Could not read PDF {pdf_file}: {e}")
            continue

        if not paper_text.strip():
            print(f"[WARNING] No text extracted from {pdf_file}, skipping.")
            continue

        # --- Pass 1: Meta info ---
        meta_data = extract_meta_info(
            paper_text=paper_text,
            model=model
        )

        # --- Pass 3: User items (once per study) ---
        user_data = {}
        if user_items:
            user_data = extract_user_items(
                paper_text=paper_text,
                user_items=user_items,
                model=model
            )

        # figure out how many main results
        try:
            num_results = int(meta_data["num_main_results"])
        except:
            num_results = 0

        main_results_str = meta_data["main_results_list"]  # e.g. "Result A; Result B"
        main_results_list = []
        if main_results_str and main_results_str != "NA":
            parts = [x.strip() for x in main_results_str.split(";")]
            main_results_list = [p for p in parts if p]

        if num_results != len(main_results_list):
            print(
                f"[INFO] Mismatch in # of main results: {num_results} vs semicolon-split {len(main_results_list)}. "
                f"Using semicolon-split length for {pdf_file.name}."
            )
            num_results = len(main_results_list)

        # For each main result => Pass 2
        # Merge everything into final row
        if num_results == 0:
            # No main results => add a single row with pass-2 fields = NA
            row_data = build_row(
                pdf_file=pdf_file.name,
                meta_data=meta_data,
                result_index=0,
                result_text="NA",
                detail_data={  # fill pass-2 fields as NA
                    "effect_size_type": "NA",
                    "effect_size": "NA",
                    "effect_size_uncertainty": "NA",
                    "p_value": "NA",
                    "total_sample_size": "NA",
                    "intervention_or_predictor_variable": "NA",
                    "outcome_variable": "NA"
                },
                user_data=user_data
            )
            results_rows.append(row_data)
        else:
            for i, result_text in enumerate(main_results_list, start=1):
                detail_data = extract_result_details(
                    paper_text=paper_text,
                    result_text=result_text,
                    model=model
                )
                row_data = build_row(
                    pdf_file=pdf_file.name,
                    meta_data=meta_data,
                    result_index=i,
                    result_text=result_text,
                    detail_data=detail_data,
                    user_data=user_data
                )
                results_rows.append(row_data)

    # Write final CSV
    # We define columns for meta data, pass-2 data, user_data
    fieldnames = [
        "filename",
        "study_title",
        "population_outcome_measured_in",
        "population_intervention_affected_or_predictor",
        "secondary_characteristics",
        "country",
        "study_type_letter",
        "num_main_results",
        "main_result_index",
        "main_result_text",
        "effect_size_type",
        "effect_size",
        "effect_size_uncertainty",
        "p_value",
        "total_sample_size",
        "intervention_or_predictor_variable",
        "outcome_variable",
    ]

    # Add columns for user_data
    if user_items:
        for i in range(len(user_items)):
            col = f"extra_{i}"
            fieldnames.append(col)

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results_rows:
            writer.writerow(row)

    print(f"[DONE] Three-pass extraction complete. Results saved to {output_csv}")


def build_row(
    pdf_file: str,
    meta_data: dict,
    result_index: int,
    result_text: str,
    detail_data: dict,
    user_data: dict
) -> dict:
    """
    Helper to merge all extracted info into a single row dict.
    """
    row = {
        "filename": pdf_file,
        # meta
        "study_title": meta_data.get("study_title", "NA"),
        "population_outcome_measured_in": meta_data.get("population_outcome_measured_in", "NA"),
        "population_intervention_affected_or_predictor": meta_data.get("population_intervention_affected_or_predictor", "NA"),
        "secondary_characteristics": meta_data.get("secondary_characteristics", "NA"),
        "country": meta_data.get("country", "NA"),
        "study_type_letter": meta_data.get("study_type_letter", "NA"),
        "num_main_results": meta_data.get("num_main_results", "0"),
        # pass-2
        "main_result_index": result_index,
        "main_result_text": result_text,
        "effect_size_type": detail_data.get("effect_size_type", "NA"),
        "effect_size": detail_data.get("effect_size", "NA"),
        "effect_size_uncertainty": detail_data.get("effect_size_uncertainty", "NA"),
        "p_value": detail_data.get("p_value", "NA"),
        "total_sample_size": detail_data.get("total_sample_size", "NA"),
        "intervention_or_predictor_variable": detail_data.get("intervention_or_predictor_variable", "NA"),
        "outcome_variable": detail_data.get("outcome_variable", "NA"),
    }

    # Add user_data => e.g. "extra_0", "extra_1", ...
    for k, v in user_data.items():
        row[k] = v

    return row
