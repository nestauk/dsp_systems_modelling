import openai
import re

def extract_meta_info(paper_text: str, model: str = "gpt-4o", temperature: float = 0.0) -> dict:
    """
    Pass 1: Extracts basic study info, number of main results, and the semicolon-separated list of them.
    """
    meta_prompt = (
        "Extract the following information from the scientific paper:\n\n"
        "1: The study title.\n"
        "2: The population outcome was measured in (e.g., if the intervention educated parents of children aged 2-4...).\n"
        "3: The population any intervention directly affected or the predictors were measured in (if not available, return 'NA').\n"
        "4: Secondary characteristics of the population context (e.g. families of low socioeconomic status).\n"
        "5: Country the study was carried out in.\n"
        "6: Identify the type of study. Provide only the letter:\n"
        "   a) purely cross-sectional study...\n"
        "   b) Study measures outcome pre and post...\n"
        "   c) purely cross-sectional study, uses control variables...\n"
        "   d) Study measures outcome pre and post...\n"
        "   e) Comparison of outcomes in treated group...\n"
        "   f) Quasi-experimental study\n"
        "   g) Randomised controlled trial\n"
        "   h) Meta-analysis.\n"
        "7: How many main results does this study report? Focus only on main results. Return only an integer.\n"
        "8: List each of the main results of the study (e.g. parenting education decreased child mental health problems), "
        "   separated by semi-colons.\n\n"
        "If any item is not available, return 'NA'. "
        "Number your answers exactly: '1: ...', '2: ...', etc.\n\n"
        "Example output:\n"
        "1: Study on Parenting Strategies\n"
        "2: Children aged 2-4\n"
        "3: Parents of children aged 2-4\n"
        "4: Families from urban areas\n"
        "5: USA\n"
        "6: g\n"
        "7: 3\n"
        "8: Parenting education improved child mental health; Parenting education increased school readiness; Parenting education reduced parental stress."
    )

    user_prompt = f"{meta_prompt}\n\nPaper text:\n{paper_text}\n\n"

    try:
        messages = [
            {"role": "user", "content": user_prompt}
        ]
        
        response = openai.chat.completions.create(
            model=model,
            messages=messages
        )
        content = response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[ERROR in meta extraction]: {e}")
        return {
            "study_title": "NA",
            "population_outcome_measured_in": "NA",
            "population_intervention_affected_or_predictor": "NA",
            "secondary_characteristics": "NA",
            "country": "NA",
            "study_type_letter": "NA",
            "num_main_results": "0",
            "main_results_list": "NA"
        }

    return parse_meta_extraction(content)


def parse_meta_extraction(content: str) -> dict:
    """
    Parses 8 enumerated lines from extract_meta_info.
    """
    data = ["NA"] * 8
    for i in range(1, 9):
        if i < 8:
            pattern = re.compile(rf"(?s)(?<=\b{i}:)(.*?)(?=\b{i+1}:|$)")
        else:
            pattern = re.compile(rf"(?s)(?<=\b{i}:)(.*)$")

        match = pattern.search(content)
        if match:
            val = match.group(1).strip()
            data[i - 1] = val if val else "NA"
        else:
            data[i - 1] = "NA"

    meta_dict = {
        "study_title": data[0],
        "population_outcome_measured_in": data[1],
        "population_intervention_affected_or_predictor": data[2],
        "secondary_characteristics": data[3],
        "country": data[4],
        "study_type_letter": data[5],
        "num_main_results": data[6],
        "main_results_list": data[7],
    }
    return meta_dict


def extract_result_details(paper_text: str, result_text: str, model: str = "gpt-4o", temperature: float = 0.0) -> dict:
    """
    Pass 2: For a single main result, extract effect size, p-value, sample size, etc.
    """
    detail_prompt = (
        "We have a specific main result from the study:\n"
        f"'{result_text}'\n\n"
        "Extract the following information:\n"
        "1: The effect size type for this main result (e.g. odds ratio, difference of means).\n"
        "2: The effect size for this main result.\n"
        "3: The estimate of uncertainty in the effect size (e.g. s.e., 95% CI).\n"
        "4: The P-value for this main result.\n"
        "5: The total sample size for the study.\n"
        "6: The intervention or predictor variable (i.e., what was manipulated or used as a predictor).\n"
        "7: The outcome variable.\n\n"
        "If any of the information is not available, return 'NA'. "
        "Number your answers exactly as '1: ...', '2: ...', etc.\n\n"
        "Example output:\n"
        "1: Odds ratio\n"
        "2: 1.8\n"
        "3: 95% CI [1.2, 2.4]\n"
        "4: 0.03\n"
        "5: 250\n"
        "6: Parenting education\n"
        "7: Child mental health problems"
    )

    user_prompt = detail_prompt + "\n\nFull paper text:\n" + paper_text + "\n\n"

    try:
        messages = [
            {"role": "user", "content": user_prompt}
        ]
        
        response = openai.chat.completions.create(
            model=model,
            messages=messages
        )
        content = response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[ERROR in details extraction]: {e}")
        return {
            "effect_size_type": "NA",
            "effect_size": "NA",
            "effect_size_uncertainty": "NA",
            "p_value": "NA",
            "total_sample_size": "NA",
            "intervention_or_predictor_variable": "NA",
            "outcome_variable": "NA"
        }

    return parse_detail_extraction(content)


def parse_detail_extraction(content: str) -> dict:
    """
    Parses lines 1..7 for the detail pass.
    """
    data = ["NA"] * 7
    for i in range(1, 8):
        if i < 7:
            pattern = re.compile(rf"(?s)(?<=\b{i}:)(.*?)(?=\b{i+1}:|$)")
        else:
            pattern = re.compile(rf"(?s)(?<=\b{i}:)(.*)$")

        match = pattern.search(content)
        if match:
            val = match.group(1).strip()
            data[i - 1] = val if val else "NA"
        else:
            data[i - 1] = "NA"

    out = {
        "effect_size_type": data[0],
        "effect_size": data[1],
        "effect_size_uncertainty": data[2],
        "p_value": data[3],
        "total_sample_size": data[4],
        "intervention_or_predictor_variable": data[5],
        "outcome_variable": data[6],
    }
    return out


# ----------------------------------------------------------------------------
# Pass 3: User-Supplied Extra Items
# ----------------------------------------------------------------------------

def extract_user_items(paper_text: str, user_items: list[str], model: str = "gpt-4o", temperature: float = 0.0) -> dict:
    """
    Takes a list of user-specified extra items (strings). 
    We prompt GPT with an enumerated list of these items, 
    asking it to provide '1: answer', '2: answer', etc.
    
    Returns a dict like:
      {
         "extra_1": "...",
         "extra_2": "...",
         ...
      }
    If any item is missing, we store "NA".
    """
    if not user_items:
        return {}

    # Build an enumerated prompt
    lines = []
    for i, item in enumerate(user_items, start=1):
        lines.append(f"{i}: {item}")

    # We instruct GPT to return each item enumerated as "1: <answer>", "2: <answer>", etc.
    prompt_header = (
        "The user has additional items they want extracted from this paper.\n"
        "Please respond with the answers to each item, enumerated exactly as '1: ...', '2: ...', etc.\n"
        "If the information is not available, return 'NA'.\n\n"
        "Example output:\n"
        "1: This is the first answer\n"
        "2: This is the second answer\n"
        "3: NA"
    )
    enumerated_instructions = "\n".join(lines)

    user_prompt = f"{prompt_header}{enumerated_instructions}\n\nPaper text:\n{paper_text}\n\n"

    try:
        messages = [
            {"role": "user", "content": user_prompt}
        ]
        
        response = openai.chat.completions.create(
            model=model,
            messages=messages
        )
        content = response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[ERROR in user items extraction]: {e}")
        # fallback: everything "NA"
        return {f"extra_{i}": "NA" for i in range(1, len(user_items) + 1)}

    # Parse the enumerated answers
    out_dict = parse_user_items_response(content, len(user_items))
    return out_dict


def parse_user_items_response(content: str, num_items: int) -> dict:
    """
    We look for lines '1:', '2:', ..., up to 'num_items:'.
    Return a dict: {"extra_1": answer1, "extra_2": answer2, ...}
    """
    results = ["NA"] * num_items
    for i in range(1, num_items + 1):
        if i < num_items:
            pattern = re.compile(rf"(?s)(?<=\b{i}:)(.*?)(?=\b{i+1}:|$)")
        else:
            # last item matches until the end
            pattern = re.compile(rf"(?s)(?<=\b{i}:)(.*)$")

        match = pattern.search(content)
        if match:
            val = match.group(1).strip()
            results[i - 1] = val if val else "NA"
        else:
            results[i - 1] = "NA"

    out = {f"extra_{i}": results[i - 1] for i in range(num_items)}
    return out

def filter_references_with_gpt(
    references: list[dict],
    user_description: str,
    model: str = "gpt-4o",
    temperature: float = 0.0
) -> list[dict]:
    """
    Filters a list of references (with 'title' and 'abstract') according to
    the user description of what studies they want included.

    For each reference:
      - We provide GPT with the user description, the reference title, and abstract.
      - We ask GPT to answer ONLY "include" or "exclude".
      - If GPT says "include", we keep the reference; otherwise, we discard it.

    Returns:
        A filtered list of reference dicts (subset of the original list).
    """

    filtered = []
    for ref in references:
        title = ref.get("title", "")
        abstract = ref.get("abstract", "")

        if not title and not abstract:
            # If there's no title or abstract, skip or handle as exclude by default
            continue

        # Build the prompt
        system_message = {
            "role": "system",
            "content": (
                "You are an expert research assistant. Your task is to determine "
                "if a given study is relevant to the user's description. "
                "Respond ONLY with 'include' or 'exclude'."
            )
        }
        user_message = {
            "role": "user", "content": (
                f"User's description of relevant studies:\n{user_description}\n\n"
                f"Study Title: {title}\n"
                f"Study Abstract: {abstract}\n\n"
                "Is this study relevant? Respond ONLY with 'include' or 'exclude'."
            )
        }

        try:
            messages = [
                system_message,
                user_message
            ]
            
            response = openai.chat.completions.create(
                model=model,
                messages=messages
            )
            content = response.choices[0].message.content.strip().lower()
            if "include" in content and "exclude" not in content:
                filtered.append(ref)
        except Exception as e:
            print(f"[ERROR in GPT filtering]: {e}")
            continue

    return filtered
