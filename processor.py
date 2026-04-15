"""
processor.py
============
Core document processing pipeline:
  1. Extract paragraphs
  2. Claude AI detection (replaces local bigram scorer)
  3. Brave + OpenAlex plagiarism check
  4. DeepSeek paraphrase + humanization
  5. Write corrected .docx
"""

import os
import re
import time
import requests
import anthropic
from openai import OpenAI
from docx import Document
from docx.shared import RGBColor

BRAVE_API_KEY     = os.getenv("BRAVE_API_KEY", "")
DEEPSEEK_API_KEY  = os.getenv("DEEPSEEK_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

SIMILARITY_THRESHOLD  = 0.60   # plagiarism word-overlap threshold
AI_CONFIDENCE_MIN     = 0.75   # Claude must be >= 75% confident to flag as AI
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


# ── Step 2: Claude AI detection ───────────────────────────────────────────────

def _score_paragraph_with_claude(text: str, client: anthropic.Anthropic) -> float:
    """
    Ask Claude to score how likely a paragraph was AI-written.
    Returns a confidence float between 0.0 (human) and 1.0 (AI).
    Falls back to 0.0 on any error so processing continues.
    """
    prompt = (
        "You are an expert AI content detector. Analyse the following paragraph and "
        "estimate the probability (0.0 to 1.0) that it was written by an AI rather than a human.\n\n"
        "Consider these signals:\n"
        "- Overly uniform sentence length and structure\n"
        "- Excessive use of transitional phrases like 'Furthermore', 'Moreover', 'In conclusion'\n"
        "- Unnaturally precise or comprehensive coverage of a topic\n"
        "- Absence of personal voice, hedging, or natural imperfection\n"
        "- Generic phrasing that could apply to any document on the topic\n\n"
        "Respond with ONLY a decimal number between 0.0 and 1.0. Nothing else.\n\n"
        f"Paragraph:\n{text}"
    )
    try:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=10,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()
        score = float(raw)
        return max(0.0, min(1.0, score))  # clamp to [0, 1]
    except Exception:
        return 0.0  # safe default — don't flag if detection fails


def score_ai_likelihood(paragraphs: list) -> dict:
    """
    Uses Claude to score each paragraph for AI likelihood.
    Returns {list_position: confidence_score}.
    Batches with a small delay to respect rate limits.
    """
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    scores = {}
    for pos, (_, text) in enumerate(paragraphs):
        scores[pos] = _score_paragraph_with_claude(text, client)
        time.sleep(0.3)  # small delay between Claude calls
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
            title        = work.get("title", "") or ""
            abstract_inv = work.get("abstract_inverted_index", {}) or {}
            abstract     = " ".join(abstract_inv.keys())
            snippet      = (title + " " + abstract).lower()
            s_words      = set(snippet.split())
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
    Makes AI-paraphrased text sound more naturally human-written.
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
    new_run.font.color.rgb = RGBColor(0, 130, 0)  # green = edited


# ── Main pipeline ─────────────────────────────────────────────────────────────

def process_document(input_path: str, output_path: str) -> dict:
    """
    Full pipeline. Returns a report dict for the API response.
    """
    paragraphs = extract_paragraphs(input_path)
    if not paragraphs:
        raise ValueError("No processable paragraphs found in document.")

    total_paragraphs = len(paragraphs)

    # ── Claude AI detection ───────────────────────────────────────────────────
    ai_scores     = score_ai_likelihood(paragraphs)
    ai_flagged_pos = {
        pos for pos, score in ai_scores.items()
        if score >= AI_CONFIDENCE_MIN
    }

    # ── Internal similarity ───────────────────────────────────────────────────
    internally_flagged = check_internal_similarity(paragraphs)

    # ── Web + academic plagiarism ─────────────────────────────────────────────
    web_flagged      = {}
    academic_flagged = {}

    for idx, (para_idx, text) in enumerate(paragraphs):
        if is_reference_entry(text):
            continue

        is_web, url, score = check_brave(text)
        if is_web:
            web_flagged[para_idx] = (url, score)
        time.sleep(0.4)

        is_acad, acad_url, acad_score = check_openalex(text)
        if is_acad:
            academic_flagged[para_idx] = (acad_url, acad_score)
        time.sleep(0.3)

    # ── Build master flag set ─────────────────────────────────────────────────
    all_flagged  = set()
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
    client            = _get_deepseek_client()
    doc               = Document(input_path)
    paraphrased_count = 0
    report_items      = []

    for i, para in enumerate(doc.paragraphs):
        if i not in all_flagged or not para.text.strip():
            continue

        original  = para.text.strip()
        rewritten = paraphrase(original, client, pass_number=1)
        time.sleep(1)

        if i in web_flagged and web_flagged[i][1] >= 0.75 and MAX_PARAPHRASE_PASSES >= 2:
            rewritten = paraphrase(rewritten, client, pass_number=2)
            time.sleep(1)

        rewritten = humanize(rewritten, client)
        time.sleep(1)

        replace_paragraph_text(para, rewritten)
        paraphrased_count += 1

        flags    = []
        list_pos = next(
            (lp for lp, (pi, _) in enumerate(paragraphs) if pi == i), None
        )
        if list_pos is not None and list_pos in ai_flagged_pos:
            confidence = int(ai_scores.get(list_pos, 0) * 100)
            flags.append(f"AI-written (Claude confidence: {confidence}%)")
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
            "preview":   original[:100] + ("..." if len(original) > 100 else ""),
            "flags":     flags,
            "action":    "paraphrased",
        })

    for para_idx in skipped_refs:
        report_items.append({
            "paragraph": para_idx + 1,
            "preview":   "",
            "flags":     ["Matched source — skipped (reference entry)"],
            "action":    "skipped",
        })

    # ── Process table cells ───────────────────────────────────────────────────
    tables_processed = 0
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                for para in cell.paragraphs:
                    text = para.text.strip()
                    if not text or len(text.split()) < MIN_WORDS:
                        continue
                    if is_reference_entry(text):
                        continue
                    # Run plagiarism + AI check on cell text
                    is_web, _, web_score   = check_brave(text)
                    is_acad, _, acad_score = check_openalex(text)
                    time.sleep(0.3)

                    if is_web or is_acad or web_score >= SIMILARITY_THRESHOLD:
                        rewritten = paraphrase(text, client, pass_number=1)
                        time.sleep(1)
                        rewritten = humanize(rewritten, client)
                        time.sleep(1)
                        replace_paragraph_text(para, rewritten)
                        paraphrased_count += 1
                        tables_processed  += 1

    doc.save(output_path)

    return {
        "total_paragraphs_checked": total_paragraphs,
        "paragraphs_paraphrased":   paraphrased_count,
        "tables_cells_rewritten":   tables_processed,
        "references_skipped":       len(skipped_refs),
        "items":                    report_items,
    }


# ── Quick analysis (no AI calls) ──────────────────────────────────────────────

def analyse_document(filepath: str) -> dict:
    """
    Fast scan of a .docx file.
    Returns word count, paragraph count, table count, image count.
    No AI calls — runs in under a second.
    """
    doc = Document(filepath)

    word_count      = 0
    paragraph_count = 0
    table_count     = len(doc.tables)
    image_count     = 0

    # Count words and paragraphs in body text
    for para in doc.paragraphs:
        text = para.text.strip()
        if text:
            words = len(text.split())
            word_count      += words
            paragraph_count += 1

    # Count words inside table cells too
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                for para in cell.paragraphs:
                    text = para.text.strip()
                    if text:
                        word_count += len(text.split())

    # Count images (inline shapes stored as relationships)
    try:
        for rel in doc.part.rels.values():
            if "image" in rel.reltype:
                image_count += 1
    except Exception:
        pass

    return {
        "word_count":      word_count,
        "paragraph_count": paragraph_count,
        "table_count":     table_count,
        "image_count":     image_count,
    }
