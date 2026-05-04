

import os
import re
import json
import time
from typing import Optional, Dict, List

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI


# =========================================================
# CONFIG
# =========================================================
INPUT_JSONL = os.getenv("INPUT_JSONL", "tests_overall_EN_120.JSONL")
OUTPUT_XLSX = os.getenv("OUTPUT_XLSX", "answersGPTmini_overall_EN_120.xlsx")

# Model
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Generation params
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
TOP_P = float(os.getenv("TOP_P", "1.0"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "500"))
FREQUENCY_PENALTY = float(os.getenv("FREQUENCY_PENALTY", "0.0"))
PRESENCE_PENALTY = float(os.getenv("PRESENCE_PENALTY", "0.0"))

# Retry / pacing
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
SLEEP_BETWEEN_CALLS = float(os.getenv("SLEEP_BETWEEN_CALLS", "0.3"))


# =========================================================
# LANGUAGE GUARDRAIL
# =========================================================
_TR_CHARS = set("çğıöşüÇĞİÖŞÜ")
_TR_HINT_WORDS = {
    "ve", "veya", "ama", "de", "da", "ile", "için", "nasıl", "neden",
    "çözüm", "hata", "sorun", "ayar", "oyun", "bilgisayar", "lütfen",
}

def detect_text_lang(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return "en"
    if any(ch in _TR_CHARS for ch in t):
        return "tr"
    low = t.lower()
    tokens = re.findall(r"[a-zA-ZçğıöşüÇĞİÖŞÜ]+", low)
    if tokens:
        hit = sum(1 for tok in tokens if tok in _TR_HINT_WORDS)
        if hit >= 2:
            return "tr"
    return "en"


def language_system_rule(lang: str) -> str:
    lang = (lang or "en").lower()
    if lang.startswith("tr"):
        return "Yanıtı kesinlikle TÜRKÇE ver. İngilizce kelime veya cümle kullanma."
    return "Answer strictly in ENGLISH. Do not use Turkish words or Turkish characters."


# =========================================================
# PROMPT
# =========================================================
def build_llm_only_prompt(user_query: str, target_lang: Optional[str] = None) -> str:
    lang = (target_lang or detect_text_lang(user_query)).lower()

    if lang.startswith("tr"):
        assistant_role = "Bilgisayar oyunları için teknik sorun giderme asistanısın."
        rules = "\n".join([
            "Kurallar:",
            "- Yalnızca bilgisayar oyunlarıyla ilgili teknik sorunları yanıtla.",
            "- Kullanıcı oyunu açıkça belirtmemişse ve sorunun hangi oyuna ait olduğu net değilse bunu söyle.",
            "- Eksik bilgi varsa bunu açıkça belirt.",
            "- Çözümü kısa, net ve adım adım ver.",
            "- Emin olmadığın yerde kesin konuşma.",
            "- Resmî belgeye baktığını iddia etme.",
            "- Kullanıcının cihazına veya sistemine erişimin varmış gibi konuşma.",
            "- System-level context kullanma.",
        ])
        return f"""{assistant_role}

{rules}

Kullanıcı sorusu:
{user_query}

En iyi cevabı yaz:
"""
    else:
        assistant_role = "You are a technical troubleshooting assistant for computer games."
        rules = "\n".join([
            "Rules:",
            "- Only answer technical troubleshooting questions related to computer games.",
            "- If the game is unclear, say that clearly.",
            "- If key details are missing, state what is missing.",
            "- Give short, clear, step-by-step actions.",
            "- Do not sound overly certain when information is incomplete.",
            "- Do not claim that you checked official documents.",
            "- Do not act as if you have access to the user's device or environment.",
            "- Do not use system-level context.",
        ])
        return f"""{assistant_role}

{rules}

User question:
{user_query}

Write the best possible answer:
"""

# =========================================================
# HELPERS
# =========================================================
def one_line(text: str) -> str:
    """Collapse answer into a single-line cell for Excel."""
    return re.sub(r"\s+", " ", (text or "").strip())


def load_jsonl(path: str) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[WARN] Skipping bad JSON at line {line_no}: {e}")
    return rows


# =========================================================
# LLM CALL
# =========================================================
def call_llm(client: OpenAI, query: str, lang: str) -> str:
    prompt = build_llm_only_prompt(query, target_lang=lang)
    system_rule = language_system_rule(lang)

    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system_rule},
                    {"role": "user", "content": prompt},
                ],
                temperature=TEMPERATURE,
                top_p=TOP_P,
                max_tokens=MAX_TOKENS,
                frequency_penalty=FREQUENCY_PENALTY,
                presence_penalty=PRESENCE_PENALTY,
            )
            return (resp.choices[0].message.content or "").strip()

        except Exception as e:
            last_err = e
            wait_s = 2 * attempt
            print(f"[WARN] Attempt {attempt}/{MAX_RETRIES} failed: {type(e).__name__}: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(wait_s)

    return f"[ERROR] {type(last_err).__name__}: {last_err}"


# =========================================================
# MAIN BATCH PROCESS
# =========================================================
def main():
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not found in environment or .env")

    if not os.path.exists(INPUT_JSONL):
        raise FileNotFoundError(f"Input file not found: {INPUT_JSONL}")

    client = OpenAI(api_key=api_key)

    print(f"[INFO] Reading: {INPUT_JSONL}")
    data = load_jsonl(INPUT_JSONL)
    print(f"[INFO] Loaded {len(data)} rows")

    results = []

    for idx, item in enumerate(data, start=1):
        qid = item.get("id", f"row_{idx}")
        lang = item.get("lang", "").lower() or detect_text_lang(item.get("query", ""))
        query = (item.get("query") or "").strip()
        game = item.get("game", "")

        print(f"[{idx}/{len(data)}] {qid}")

        if not query:
            answer = "[ERROR] Empty query"
        else:
            raw_answer = call_llm(client, query, lang)
            answer = one_line(raw_answer)

        results.append({
            "id": qid,
            "lang": lang,
            "game": game,
            "query": query,
            "answer": answer,
        })

        time.sleep(SLEEP_BETWEEN_CALLS)

    df = pd.DataFrame(results, columns=["id", "lang", "game", "query", "answer"])
    df.to_excel(OUTPUT_XLSX, index=False)

    print(f"[DONE] Saved: {OUTPUT_XLSX}")


if __name__ == "__main__":
    main()
