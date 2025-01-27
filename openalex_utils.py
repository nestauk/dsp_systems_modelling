import pandas as pd
import pyalex

def reconstruct_abstract(inverted_index):
    """
    Reconstructs a plain-text abstract from the OpenAlex inverted index.
    """
    if not inverted_index:
        return None
    
    max_index = max(pos for positions in inverted_index.values() for pos in positions)
    abstract = [""] * (max_index + 1)

    for word, positions in inverted_index.items():
        for pos in positions:
            abstract[pos] = word

    return " ".join(abstract)

def extract_oa_metadata(best_oa_location):
    """
    Extracts relevant Open Access fields from best_oa_location.
    """
    if not best_oa_location:
        return {
            "landing_page_url": None,
            "pdf_url": None,
            "is_oa": None,
        }
    return {
        "landing_page_url": best_oa_location.get("landing_page_url"),
        "pdf_url": best_oa_location.get("pdf_url"),
        "is_oa": best_oa_location.get("is_oa", False),
    }

def fetch_openalex_results(
    search_term: str,
    user_email: str,
    min_cites: str = ">4",
    n_works: int = 1000
) -> pd.DataFrame:
    """
    Fetch works from OpenAlex based on a search term and citation filter.

    Args:
        search_term (str): The main search term or question.
        user_email (str): User email for OpenAlex API config.
        min_cites (str): Minimum citation count filter (e.g., '>10'). Defaults to '>4'.
        n_works (int): Maximum number of works to fetch. Defaults to 1000.

    Returns:
        pd.DataFrame: A DataFrame containing key work metadata (title, doi, year, abstract, etc.).
    """
    pyalex.config["email"] = user_email

    # Build the query (we can add more .filter() calls as needed)
    query = pyalex.Works().search(search_term).filter(cited_by_count=min_cites)

    results = []
    # Paginate through OpenAlex results
    for page in query.paginate(per_page=200, n_max=n_works):
        for work in page:
            # Reconstruct abstract
            abstract = reconstruct_abstract(work.get("abstract_inverted_index"))
            # Extract OA metadata
            oa_metadata = extract_oa_metadata(work.get("best_oa_location"))

            # Build a minimal dict of relevant fields
            filtered_work = {
                "title": work.get("title"),
                "doi": work.get("doi"),
                "publication_year": work.get("publication_year"),
                "abstract": abstract,
                "landing_page_url": oa_metadata["landing_page_url"],
                "pdf_url": oa_metadata["pdf_url"],
                "is_oa": oa_metadata["is_oa"],
            }
            results.append(filtered_work)

    # Convert to DataFrame and do any basic cleaning
    df = pd.DataFrame(results).dropna(subset=["title", "abstract"], how="any")

    return df
