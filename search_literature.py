# search_literature.py

import pandas as pd
from pathlib import Path
import requests
from openalex_utils import fetch_openalex_results
from ai_utils import filter_references_with_gpt

def download_pdf(pdf_url, pdf_folder, filename):
    """
    Downloads a PDF from the given URL and saves it to the specified folder.
    
    Args:
        pdf_url (str): The URL of the PDF to download.
        pdf_folder (Path): The folder to save the downloaded PDF.
        filename (str): The name to save the PDF file as.
    """
    try:
        response = requests.get(pdf_url, stream=True)
        response.raise_for_status()

        # Save PDF to folder
        pdf_path = pdf_folder / filename
        with open(pdf_path, "wb") as pdf_file:
            for chunk in response.iter_content(chunk_size=8192):
                pdf_file.write(chunk)

        print(f"Downloaded: {pdf_path}")
    except Exception as e:
        print(f"Failed to download {pdf_url}: {e}")

def run_literature_search(
    search_term: str,
    description: str,
    user_email: str,
    min_cites: str = ">4",
    n_works: int = 200
):
    """
    Fetch and filter relevant literature from OpenAlex using a single search term.
    Also uses GPT-based filtering if you want to use 'description' as a prompt.

    Returns:
        references_csv_path (str): path to a CSV with filtered references
        pdf_folder (str): path to a folder containing downloaded PDFs
    """
    # 1) fetch from openalex
    df = fetch_openalex_results(
        search_term=search_term,
        user_email=user_email,
        min_cites=min_cites,
        n_works=n_works
    )

    # 2) do GPT filtering
    if description:
        references = df.to_dict("records")
        filtered_refs = filter_references_with_gpt(references, user_description=description)  # or just pass a single item list
        df_filtered = pd.DataFrame(filtered_refs)
    else:
        # skip filtering
        df_filtered = df

    # 3) write CSV
    output_folder = Path("results/search_1")
    output_folder.mkdir(parents=True, exist_ok=True)

    # Add a unique ID for each study
    df_filtered.insert(0, "unique_id", [f"study_{i+1}" for i in range(len(df_filtered))])

    references_csv = output_folder / "filtered_references.csv"
    df_filtered.to_csv(references_csv, index=False)
    
    # 4) download PDFs
    pdf_folder = output_folder / "pdfs"
    pdf_folder.mkdir(parents=True, exist_ok=True)

    for _, row in df_filtered.iterrows():
        pdf_url = row.get("pdf_url")
        unique_id = row.get("unique_id")
        if pdf_url and unique_id and isinstance(pdf_url, str) and pdf_url.strip():
            sanitized_filename = f"{unique_id}.pdf"
            download_pdf(pdf_url, pdf_folder, sanitized_filename)

    return str(references_csv), str(pdf_folder)
