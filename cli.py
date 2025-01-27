import os
import sys
import openai

# Local modules (adjust the import paths as needed)
from search_literature import run_literature_search
from extract_data import run_three_pass_extraction#, run_data_extraction_pdf_only
from ontology_mapping import map_interventions_and_outcomes

def main():
    print("Welcome to the AI Pipeline CLI.")

    # 1. Ask for user’s OpenAI API key
    api_key = input("Please enter your OpenAI API key: ").strip()
    if not api_key:
        print("No API key provided. Exiting.")
        sys.exit(1)
    openai.api_key = api_key

    # 2. Ask for user’s email (for OpenAlex usage, if they run a search)
    user_email = input("Please enter your email (required for OpenAlex API usage): ").strip()
    if not user_email:
        print("No user email provided. You can still skip the search step if you prefer.")
        # We won't exit, but if they do try to search, we won't have an email to pass.

    # 3. Ask if user wants to run a literature search
    do_search = input("Do you want to run a literature search? (y/n): ").strip().lower()

    references_csv = None
    pdf_folder = None

    if do_search == 'y':
        if not user_email:
            print("Cannot run literature search without an email. Skipping search.")
        else:
            # Prompt for a single search term
            search_term = input("Enter your search term or query for OpenAlex: ").strip()

            # Prompt for free-text description (optional) for GPT filtering
            descriptions = input("Enter free-text descriptions for GPT-based filtering (optional): ").strip()

            print("\nRunning literature search...")
            # run_literature_search should return (references_csv, pdf_folder)
            references_csv, pdf_folder = run_literature_search(
                search_term=search_term,
                description=descriptions,
                user_email=user_email
            )
            print(f"Search complete. References CSV: {references_csv}, PDFs folder: {pdf_folder}")
    else:
        print("Skipping literature search.")

    # 4. Ask if user wants to run data extraction
    do_extract = input("Do you want to run data extraction? (y/n): ").strip().lower()

    extraction_csv_path = None  # We'll store path to the final extraction CSV

    if do_extract == 'y':
        # Ask if user wants extra items
        extra_items = []
        want_extras = input("Do you want to specify extra variables to extract? (y/n): ").strip().lower()
        if want_extras == "y":
            print("Enter each extra variable on a new line. Press Enter on an empty line to finish:")
            while True:
                line = input()
                if not line.strip():
                    break
                extra_items.append(line.strip())

        if references_csv and pdf_folder:
            # If the user ran a search, we have references + a PDF folder
            out_csv = os.path.join(os.path.dirname(references_csv), 'extraction_results.csv')
            
            # Extraction from references + PDFs, with abstract fallback
            run_three_pass_extraction(
                pdf_folder=pdf_folder,
                user_items=extra_items,
                output_csv=out_csv,
                model="gpt-4o")
            extraction_csv_path = out_csv

        else:
            # No search => user must have their own PDF folder
            pdf_folder = input("Enter path to folder containing PDFs: ").strip()
            out_csv = input("Enter path for extraction output CSV (default: 'extraction_results.csv'): ").strip() or "extraction_results.csv"

            run_data_extraction_pdf_only(
                pdf_folder=pdf_folder,
                extra_items=extra_items,
                output_csv_path=out_csv,
                model="gpt-4o"
            )
            extraction_csv_path = out_csv

    else:
        print("Skipping data extraction.")

    # 5. If we have a final extraction CSV, ask if user wants ontology mapping
    if extraction_csv_path:
        do_ontology = input("Do you want to map interventions and outcomes to an ontology? (y/n): ").strip().lower()
        if do_ontology == 'y':
            intervention_ont_path = input("Enter path to your intervention ontology file (CSV or JSON): ").strip()
            outcome_ont_path = input("Enter path to your outcome ontology file (CSV or JSON): ").strip()
            mapped_csv = input("Enter path for the mapped output CSV (default: 'extraction_mapped.csv'): ").strip() or "extraction_mapped.csv"

            map_interventions_and_outcomes(
                extracted_csv_path=extraction_csv_path,
                intervention_ontology_path=intervention_ont_path,
                outcome_ontology_path=outcome_ont_path,
                output_csv_path=mapped_csv,
                openai_model="text-embedding-ada-002"
            )
            print(f"Ontology mapping complete. Output: {mapped_csv}")
        else:
            print("Ontology mapping skipped.")
    else:
        print("No extraction CSV available for ontology mapping.")

    print("All done!")

if __name__ == "__main__":
    main()

