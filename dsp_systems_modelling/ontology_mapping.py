import numpy as np
import openai
import pandas as pd


def map_interventions_and_outcomes(
    extracted_csv_path: str,
    intervention_ontology_path: str,
    outcome_ontology_path: str,
    output_csv_path: str,
    openai_model: str = "text-embedding-ada-002",
):
    """
    Reads the extracted CSV, loads user-supplied ontologies, and uses OpenAI embeddings.

    Finds the closest ontology terms for interventions and outcomes.

    :param extracted_csv_path: Path to the CSV output from data extraction
    :param intervention_ontology_path: CSV or JSON file with intervention terms
    :param outcome_ontology_path: CSV or JSON file with outcome terms
    :param output_csv_path: Where to write the mapped CSV
    :param openai_model: The OpenAI model used for embeddings (default: "text-embedding-ada-002")
    """
    # 1. Load extraction data
    df_extracted = pd.read_csv(extracted_csv_path)

    # Identify which columns in your extraction contain the interventions and outcomes:
    # For example, from your enumerated instructions:
    #   "4" => "intervention variable"
    #   "7" => "outcome variable"
    # Adjust if your extraction CSV is labeled differently.
    intervention_col = "4"  # or "4: Intervention" if you renamed
    outcome_col = "7"  # or "7: Outcome"

    # 2. Load ontologies
    #    We'll define a helper to load CSV or JSON into a list of terms
    intervention_terms = load_ontology(intervention_ontology_path)
    outcome_terms = load_ontology(outcome_ontology_path)

    # 3. Pre-embed ontology terms
    #    We'll store each term plus an embedding vector
    intervention_onto_df = create_ontology_embeddings(intervention_terms, openai_model)
    outcome_onto_df = create_ontology_embeddings(outcome_terms, openai_model)

    # 4. For each row, embed the extracted intervention and outcome, find best match
    mapped_interventions = []
    mapped_outcomes = []

    for _, row in df_extracted.iterrows():
        extracted_interv_text = str(row.get(intervention_col, "NA"))
        extracted_outcome_text = str(row.get(outcome_col, "NA"))

        # If either is "NA" or empty, we won't do matching
        best_interv_term = "NA"
        best_out_term = "NA"

        if extracted_interv_text and extracted_interv_text.upper() != "NA":
            best_interv_term = find_best_match_in_ontology(
                item_text=extracted_interv_text, onto_df=intervention_onto_df, model=openai_model
            )

        if extracted_outcome_text and extracted_outcome_text.upper() != "NA":
            best_out_term = find_best_match_in_ontology(
                item_text=extracted_outcome_text, onto_df=outcome_onto_df, model=openai_model
            )

        mapped_interventions.append(best_interv_term)
        mapped_outcomes.append(best_out_term)

    # 5. Add new columns to df_extracted
    df_extracted["mapped_intervention"] = mapped_interventions
    df_extracted["mapped_outcome"] = mapped_outcomes

    # 6. Write updated CSV
    df_extracted.to_csv(output_csv_path, index=False)
    print(f"Ontology mapping complete. Saved to {output_csv_path}")


def load_ontology(file_path: str):
    """
    Loads an ontology from CSV or JSON into a list of terms (strings).

    If CSV, expects a column 'term' with the label (and optional columns).
    If JSON, expects a list of objects each containing a 'term' field, or a list of strings.
    """
    import json
    import os

    ext = os.path.splitext(file_path)[1].lower()  # noqa: PTH122
    if ext in [".csv"]:
        df = pd.read_csv(file_path)
        # Expect a 'term' column
        if "term" in df.columns:
            return df["term"].tolist()
        else:
            # fallback: just take the first column
            return df.iloc[:, 0].dropna().tolist()

    elif ext in [".json"]:
        with open(file_path, encoding="utf-8") as f:  # noqa: PTH123
            data = json.load(f)
        # If it's a list of dicts
        if isinstance(data, list) and len(data) > 0:
            if isinstance(data[0], dict) and "term" in data[0]:
                return [item["term"] for item in data if "term" in item]
            else:
                # maybe it's just a list of strings
                if all(isinstance(x, str) for x in data):
                    return data
        # fallback
        return []
    else:
        print(f"[WARNING] Unsupported ontology file format: {file_path}")
        return []


def create_ontology_embeddings(terms: list[str], model: str) -> pd.DataFrame:
    """
    Create ontology embeddings.

    Given a list of ontology terms, compute embeddings for each
    and return a DataFrame with columns: ['term', 'embedding'].
    """
    if not terms:
        return pd.DataFrame(columns=["term", "embedding"])

    # We'll do a batch approach if you have many terms, but for simplicity:
    embeddings = []
    for term in terms:
        emb = compute_embedding(term, model=model)
        embeddings.append(emb)

    df = pd.DataFrame({"term": terms, "embedding": embeddings})
    return df


def find_best_match_in_ontology(item_text: str, onto_df: pd.DataFrame, model: str) -> str:
    """
    Finds best matche in ontology for a given item text.

    Embeds item_text, compares with each row in onto_df (which has 'term' and 'embedding'),
    returns the 'term' with highest cosine similarity.
    """
    if onto_df.empty:
        return "NA"

    item_embedding = compute_embedding(item_text, model=model)

    # Compute cosine similarity with each ontology embedding
    sims = []
    for _, row in onto_df.iterrows():
        onto_emb = row["embedding"]
        sim = cosine_similarity(item_embedding, onto_emb)
        sims.append(sim)

    # Argmax
    best_index = int(np.argmax(sims))
    best_term = onto_df.iloc[best_index]["term"]
    return best_term


def compute_embedding(text: str, model: str = "text-embedding-ada-002"):
    """
    Calls the OpenAI Embeddings API and returns the embedding vector (list of floats).
    """
    try:
        response = openai.Embeddings.create(model=model, input=[text])  # Ensure input is a list
        emb = response.data[0].embedding
        return emb  # noqa: TRY300
    except Exception as e:
        print(f"[ERROR] Embedding failed for text='{text[:30]}...': {e}")
        return []  # fallback


def cosine_similarity(vec_a, vec_b):
    """
    Computes the cosine similarity between two embedding vectors.
    """
    if not vec_a or not vec_b:
        return 0.0
    a = np.array(vec_a)
    b = np.array(vec_b)
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
