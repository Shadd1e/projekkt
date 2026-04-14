"""
processor.py
============
Core document processing pipeline:
  1. Extract paragraphs
  2. Perplexity scoring (AI detection)
  3. Brave + OpenAlex plagiarism check
  4. DeepSeek paraphrase + humanization
  5. Write corrected .docx
"""

import os
import re
import time
import math
import requests
from collections import Counter
from openai import OpenAI
from docx import Document
from docx.shared import RGBColor

BRAVE_API_KEY    = os.getenv("BRAVE_API_KEY", "")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")

SIMILARITY_THRESHOLD  = 0.60   # plagiarism word-overlap threshold
PERPLEXITY_THRESHOLD  = 60.0   # below this = likely AI-written
MIN_WORDS             = 15     # skip very short paragraphs
MAX_PARAPHRASE_PASSES = 2      # second pass for high-confidence matches

# ── Helpers ───────────────────────────────────────────────────────────────────

def is_reference_entry(text: str) -> bool:
    """Skip bibliography / reference list entries."""
    patterns = [
        r'^[A-Z][a-z]+,\s+[A-Z]',
        r'^\w.+\(\d{4}\)',
        r'^\w.+\d{4}\.',
        r'^https?://',
        r'^\[?\d+\]',
        r'^doi:',
        r'^\d+\.\s+[A-Z]',
        r'^[A-Z][a-z]+\s+[A-Z][a-z]+.*\d{4}',
        r'.*journal.*vol.*',
        r'.*pp\.\s*\d+',
    ]
    text_lower = text.lower()
    for pattern in patterns:
        if re.match(pattern, text.strip(), re.IGNORECASE):
            return True
    ref_keywords = ["doi:", "isbn", "issn", "vol.", "pp.", "et al.", "retrieved from"]
    return any(kw in text_lower for kw in ref_keywords)


# ── Step 1: Extract paragraphs ────────────────────────────────────────────────

def extract_paragraphs(filepath: str):
    """Return list of (doc_index, text) for paragraphs worth checking."""
    doc = Document(filepath)
    result = []
    for i, para in enumerate(doc.paragraphs):
        text = para.text.strip()
        if text and len(text.split()) >= MIN_WORDS:
            result.append((i, text))
    return result


# ── Step 2: Perplexity scorer (AI detection) ──────────────────────────────────

def _bigram_perplexity(text: str) -> float:
    """
    Estimate text perplexity using bigram model on the paragraph itself.
    Lower perplexity = more predictable = more likely AI-written.
    This is a lightweight proxy for GPTZero-style detection.
    """
    words = re.findall(r'\b\w+\b', text.lower())
    if len(words) < 4:
        return 999.0  # too short to score, assume human

    unigrams = Counter(words)
    bigrams  = Counter(zip(words[:-1], words[1:]))
    total    = sum(unigrams.values())

    log_prob = 0.0
    for w1, w2 in zip(words[:-1], words[1:]):
        p_w1  = unigrams[w1] / total
        p_w2_given_w1 = (bigrams[(w1, w2)] + 1) / (unigrams[w1] + len(unigrams))  # Laplace
        log_prob += math.log(p_w2_given_w1 + 1e-10)

    avg_log_prob = log_prob / max(len(words) - 1, 1)
    perplexity   = math.exp(-avg_log_prob)
    return round(perplexity, 2)


def score_ai_likelihood(paragraphs: list) -> dict:
    """
    Returns {list_position: perplexity_score} for all paragraphs.
    Flags those below PERPLEXITY_THRESHOLD as likely AI-written.
    """
    scores = {}
    for pos, (_, text) in enumerate(paragraphs):
        scores[pos] = _bigram_perplexity(text)
    return scores


# ── Step 3: Plagiarism checks ─────────────────────────────────────────────────

def check_brave(text: str) -> tuple:
    """Check paragraph against live web via Brave Search."""
    query = text[:120].strip()
    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": BRAVE_API_KEY,
    }
    params = {"q": f'"{query}"', "count": 5}

    try:
        r = requests.get(
            "https://api.search.brave.com/res/v1/web/search",
            headers=headers, params=params, timeout=10,
        )
        if r.status_code != 200:
            return False, None, 0.0

        results = r.json().get("web", {}).get("results", [])
        if not results:
            return False, None, 0.0

        para_words = set(text.lower().split())
        best_score, best_url = 0.0, None

        for result in results:
            snippet = result.get("description", "") + " " + result.get("title", "")
            s_words = set(snippet.lower().split())
            if not s_words:
                continue
            overlap = len(para_words & s_words) / len(para_words)
            if overlap > best_score:
                best_score = overlap
                best_url   = result.get("url", "Unknown source")

        return best_score >= SIMILARITY_THRESHOLD, best_url, round(best_score, 2)

    except Exception:
        return False, None, 0.0


def check_openalex(text: str) -> tuple:
    """Check paragraph against OpenAlex academic paper database (free)."""
    query = text[:100].strip()
    try:
        r = requests.get(
            "https://api.openalex.org/works",
            params={"search": query, "per-page": 5},
            timeout=10,
        )
        if r.status_code != 200:
            return False, None, 0.0

        results = r.json().get("results", [])
        if not results:
            return False, None, 0.0

        para_words = set(text.lower().split())
        best_score, best_url = 0.0, None

        for work in results:
            title   = work.get("title", "") or ""
            abstract_inv = work.get("abstract_inverted_index", {}) or {}
            abstract = " ".join(abstract_inv.keys())
            snippet  = (title + " " + abstract).lower()
            s_words  = set(snippet.split())
            if not s_words:
                continue
            overlap = len(para_words & s_words) / len(para_words)
            if overlap > best_score:
                best_score = overlap
                best_url   = work.get("id", "OpenAlex record")

        return best_score >= SIMILARITY_THRESHOLD, best_url, round(best_score, 2)

    except Exception:
        return False, None, 0.0


def check_internal_similarity(paragraphs: list) -> set:
    """Flag paragraphs that are too similar to each other within the doc."""
    flagged = set()
    texts   = [t for _, t in paragraphs]
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            wi = set(texts[i].lower().split())
            wj = set(texts[j].lower().split())
            if not wi or not wj:
                continue
            overlap = len(wi & wj) / min(len(wi), len(wj))
            if overlap >= SIMILARITY_THRESHOLD:
                flagged.add(i)
                flagged.add(j)
    return flagged


# ── Step 4: DeepSeek paraphrase + humanization ────────────────────────────────

def _get_deepseek_client():
    return OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")


def paraphrase(text: str, client, pass_number: int = 1) -> str:
    """Paraphrase a paragraph with DeepSeek."""
    if pass_number == 1:
        prompt = (
            "You are an expert academic writer. Rewrite the following paragraph "
            "so it is completely original and undetectable as plagiarism. Apply ALL of these:\n\n"
            "1. Completely restructure every sentence\n"
            "2. Replace key terms with high-quality synonyms\n"
            "3. Mix sentence lengths — some short, some complex\n"
            "4. Alternate active and passive voice\n"
            "5. Add natural transitional phrases\n"
            "6. Reorder supporting ideas where logical\n"
            "7. Preserve all technical terms, numbers, and facts\n"
            "8. Maintain academic tone\n\n"
            "Return ONLY the rewritten paragraph. No commentary.\n\n"
            f"Original:\n{text}"
        )
    else:
        prompt = (
            "This paragraph still resembles its source too closely. "
            "Rewrite it completely from scratch using entirely different sentence structures, "
            "vocabulary, and logical flow. Imagine explaining this to a colleague in your own words. "
            "Preserve all facts and technical terms but change everything else.\n\n"
            "Return ONLY the rewritten paragraph. No commentary.\n\n"
            f"Paragraph:\n{text}"
        )

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a professional academic editor specialising in making "
                        "written content completely original while preserving meaning. "
                        "Return only the requested output with no extra text."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=1024,
            temperature=0.85,
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return text


def humanize(text: str, client) -> str:
    """
    Second-layer humanization pass.
    Makes AI-paraphrased text sound more naturally human-written
    by introducing rhythm variation and natural imperfections.
    """
    prompt = (
        "The paragraph below was rewritten by an AI tool and may still sound artificial. "
        "Your job is to make it sound like it was written by a real human student or researcher. "
        "Apply these techniques:\n\n"
        "1. Vary sentence rhythm — mix very short sentences with longer ones\n"
        "2. Add one or two natural filler phrases humans use (e.g. 'In other words', "
        "'Put simply', 'It is worth noting')\n"
        "3. Slightly loosen overly formal phrasing in 1-2 places\n"
        "4. Introduce minor stylistic asymmetry — not every sentence should be the same length\n"
        "5. Do NOT change facts, technical terms, or overall meaning\n"
        "6. Keep academic register — this is still a formal document\n\n"
        "Return ONLY the humanized paragraph. No commentary.\n\n"
        f"Paragraph:\n{text}"
    )

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a human writing coach who makes AI-generated academic text "
                        "sound authentically human. Return only the output with no extra text."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=1024,
            temperature=0.9,
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return text


# ── Step 5: Replace paragraph text in .docx ──────────────────────────────────

def replace_paragraph_text(para, new_text: str):
    """Replace paragraph text preserving original formatting, highlight green."""
    if para.runs:
        first = para.runs[0]
        bold, italic, underline = first.bold, first.italic, first.underline
        font_name, font_size    = first.font.name, first.font.size
    else:
        bold = italic = underline = False
        font_name = font_size = None

    for run in para.runs:
        run._element.getparent().remove(run._element)

    new_run           = para.add_run(new_text)
    new_run.bold      = bold
    new_run.italic    = italic
    new_run.underline = underline
    if font_name:
        new_run.font.name = font_name
    if font_size:
        new_run.font.size = font_size
    new_run.font.color.rgb = RGBColor(0, 130, 0)   # green = edited


# ── Main pipeline ─────────────────────────────────────────────────────────────

def process_document(input_path: str, output_path: str) -> dict:
    """
    Full pipeline. Returns a report dict for the API response.
    """
    paragraphs = extract_paragraphs(input_path)
    if not paragraphs:
        raise ValueError("No processable paragraphs found in document.")

    total_paragraphs = len(paragraphs)

    # ── AI detection ──────────────────────────────────────────────────────────
    perplexity_scores  = score_ai_likelihood(paragraphs)
    ai_flagged_pos     = {
        pos for pos, score in perplexity_scores.items()
        if score < PERPLEXITY_THRESHOLD
    }

    # ── Internal similarity ───────────────────────────────────────────────────
    internally_flagged = check_internal_similarity(paragraphs)

    # ── Web + academic plagiarism ─────────────────────────────────────────────
    web_flagged      = {}   # {para_doc_index: (url, score)}
    academic_flagged = {}   # {para_doc_index: (url, score)}

    for idx, (para_idx, text) in enumerate(paragraphs):
        if is_reference_entry(text):
            continue

        # Brave web check
        is_web, url, score = check_brave(text)
        if is_web:
            web_flagged[para_idx] = (url, score)
        time.sleep(0.4)

        # OpenAlex academic check
        is_acad, acad_url, acad_score = check_openalex(text)
        if is_acad:
            academic_flagged[para_idx] = (acad_url, acad_score)
        time.sleep(0.3)

    # ── Build master flag set (skip references) ───────────────────────────────
    all_flagged = set()
    skipped_refs = []

    for list_pos, (para_idx, text) in enumerate(paragraphs):
        reasons = []
        if list_pos in ai_flagged_pos:
            reasons.append("ai")
        if list_pos in internally_flagged:
            reasons.append("internal")
        if para_idx in web_flagged:
            reasons.append("web")
        if para_idx in academic_flagged:
            reasons.append("academic")

        if reasons:
            if is_reference_entry(text):
                skipped_refs.append(para_idx)
            else:
                all_flagged.add(para_idx)

    # ── Paraphrase + humanize ─────────────────────────────────────────────────
    client          = _get_deepseek_client()
    doc             = Document(input_path)
    paraphrased_count = 0
    report_items    = []

    for i, para in enumerate(doc.paragraphs):
        if i not in all_flagged or not para.text.strip():
            continue

        original = para.text.strip()

        # First paraphrase pass
        rewritten = paraphrase(original, client, pass_number=1)
        time.sleep(1)

        # Second pass for high web similarity
        if i in web_flagged and web_flagged[i][1] >= 0.75 and MAX_PARAPHRASE_PASSES >= 2:
            rewritten = paraphrase(rewritten, client, pass_number=2)
            time.sleep(1)

        # Humanization layer
        rewritten = humanize(rewritten, client)
        time.sleep(1)

        replace_paragraph_text(para, rewritten)
        paraphrased_count += 1

        # Build report entry
        flags = []
        list_pos = next(
            (lp for lp, (pi, _) in enumerate(paragraphs) if pi == i), None
        )
        if list_pos is not None and list_pos in ai_flagged_pos:
            flags.append(f"AI-written (perplexity: {perplexity_scores.get(list_pos, '?')})")
        if list_pos is not None and list_pos in internally_flagged:
            flags.append("Internal repetition")
        if i in web_flagged:
            url, score = web_flagged[i]
            flags.append(f"Web match {int(score*100)}% — {url}")
        if i in academic_flagged:
            url, score = academic_flagged[i]
            flags.append(f"Academic match {int(score*100)}% — {url}")

        report_items.append({
            "paragraph": i + 1,
            "preview": original[:100] + ("..." if len(original) > 100 else ""),
            "flags": flags,
            "action": "paraphrased",
        })

    # References that were flagged but skipped
    for para_idx in skipped_refs:
        report_items.append({
            "paragraph": para_idx + 1,
            "preview": "",
            "flags": ["Matched source — skipped (reference entry)"],
            "action": "skipped",
        })

    doc.save(output_path)

    return {
        "total_paragraphs_checked": total_paragraphs,
        "paragraphs_paraphrased":   paraphrased_count,
        "references_skipped":       len(skipped_refs),
        "items": report_items,
    }
