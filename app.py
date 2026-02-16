import os, re, json, base64, sqlite3
import textwrap
from datetime import date, timedelta
from typing import List, Tuple, Optional
import hashlib

import streamlit as st
import pandas as pd
from openai import OpenAI

# -------------------------
# DB path (stable)
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "greenreceipt.db")

MODEL_TEXT = os.getenv("OPENAI_MODEL_TEXT", "gpt-4o-mini")
MODEL_VISION = os.getenv("OPENAI_MODEL_VISION", "gpt-4o-mini")

# -------------------------
# Session state
# -------------------------
for k, v in {
    "receipt_text_raw": "",
    "items_final_text": "",
    "items_editor": "",
    "last_run": None,
    "pack_overrides": {},
    "receipt_fingerprint": "",  # NEW: detect new receipt so items aren't overwritten
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# -------------------------
# UI styling
# -------------------------
st.set_page_config(page_title="GreenReceipt", page_icon="🌿", layout="wide")
st.markdown(
    """
    <style>
      .stApp { background: linear-gradient(135deg, #f3fff6 0%, #f6fbff 45%, #fff7fb 100%); }
      .card { background: rgba(255,255,255,0.90); border: 1px solid rgba(0,0,0,0.06);
              border-radius: 16px; padding: 16px 18px; box-shadow: 0 6px 18px rgba(0,0,0,0.04); }
      .title { font-size: 40px; font-weight: 850; letter-spacing: -0.5px; margin-bottom: 4px; }
      .subtitle { color: rgba(0,0,0,0.65); font-size: 14px; margin-top: 0px; }
      .affirm { margin-top: 8px; padding: 10px 12px; border-radius: 14px;
                background: rgba(34,197,94,0.10); border: 1px solid rgba(34,197,94,0.18);
                color: rgba(0,0,0,0.70); font-size: 14px; }
      .pill { display:inline-block; padding:6px 10px; border-radius:999px;
              background: rgba(34,197,94,0.12); border:1px solid rgba(34,197,94,0.25); font-weight:700; }
      .muted { color: rgba(0,0,0,0.65); }
      .item { background: rgba(255,255,255,0.88); border: 1px solid rgba(0,0,0,0.06);
              border-radius: 14px; padding: 14px 16px; margin-bottom: 10px; }
      .tag { display:inline-block; padding:3px 8px; border-radius:999px;
             background: rgba(0,0,0,0.06); font-size: 12px; margin-left: 6px; }
      .reason { font-size: 13px; color: rgba(0,0,0,0.70); margin-top: 8px; }
      .pickcard { background: rgba(255,255,255,0.88); border: 1px solid rgba(0,0,0,0.06);
                  border-radius: 14px; padding: 12px 12px; margin-bottom: 8px; }
      .picktitle { font-weight: 800; margin-bottom: 6px; }
      div.stButton > button[kind="primary"] { background-color:#e74c3c !important; border-color:#e74c3c !important; }
      div.stButton > button[kind="primary"]:hover { background-color:#c0392b !important; border-color:#c0392b !important; }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------
# Helpers
# -------------------------
def oai() -> OpenAI:
    return OpenAI()

def clean_text(t: str) -> str:
    if not t:
        return ""
    return t.replace("```text", "").replace("```json", "").replace("```", "").strip()

def looks_like_docs_or_code(t: str) -> bool:
    if not t:
        return True
    bad = ["DeltaGenerator", "add_rows", "altair_chart", "Traceback", "ModuleNotFoundError", "def ", "class ", "import "]
    return sum(1 for m in bad if m.lower() in t.lower()) >= 2

def fingerprint_text(t: str) -> str:
    return hashlib.sha256((t or "").encode("utf-8", errors="ignore")).hexdigest()

def strip_prices_counts(text: str) -> str:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    out = []
    for ln in lines:
        ln = re.sub(r"\$?\s*\d+\.\d{2}\b", "", ln)
        ln = re.sub(r"\b\d+(\.\d+)?\s*/\s*(pc|ea|each|lb|kg|g|oz)\b", "", ln, flags=re.I)
        ln = re.sub(r"\b(x\s*\d+|\d+\s*x)\b", "", ln, flags=re.I)
        ln = re.sub(r"\(\s*\d+\s*(ct|count|pk|pack)?\s*\)", "", ln, flags=re.I)
        ln = re.sub(r"^\s*\d+[\.\)]\s*", "", ln)
        ln = re.sub(r"\s{2,}", " ", ln).strip()
        if ln and len(ln) > 2:
            out.append(ln)

    seen, deduped = set(), []
    for x in out:
        k = x.lower()
        if k not in seen:
            seen.add(k)
            deduped.append(x)
    return "\n".join(deduped)

@st.cache_data(show_spinner=False)
def openai_ocr_receipt(file_bytes: bytes, mime: str) -> str:
    c = oai()
    b64 = base64.b64encode(file_bytes).decode("utf-8")
    data_url = f"data:{mime};base64,{b64}"
    prompt = "Read this shopping receipt image and output ONLY the receipt text as plain text. Preserve line breaks."
    resp = c.chat.completions.create(
        model=MODEL_VISION,
        messages=[{
            "role": "user",
            "content": [{"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_url}}]
        }],
        temperature=0
    )
    return clean_text(resp.choices[0].message.content or "")

@st.cache_data(show_spinner=False)
def openai_affirmation() -> str:
    c = oai()
    prompt = "Write ONE short affirmation (max 14 words) about small shopping choices helping the environment. No emojis."
    resp = c.chat.completions.create(
        model=MODEL_TEXT,
        messages=[{"role":"system","content":"You write short affirmations."},
                  {"role":"user","content":prompt}],
        temperature=0.6
    )
    return clean_text(resp.choices[0].message.content or "")

def openai_debrand_lines(items_text: str) -> List[str]:
    c = oai()
    schema = {"items":[{"name":"string"}]}
    prompt = f"""
Return ONLY JSON.

Schema:
{json.dumps(schema, indent=2)}

Task:
For each line, remove brand names and keep only the product type.
Keep it short and generic.

Input lines:
{items_text}
""".strip()

    resp = c.chat.completions.create(
        model=MODEL_TEXT,
        messages=[{"role":"system","content":"Remove brand names. Output JSON only."},
                  {"role":"user","content":prompt}],
        temperature=0
    )
    data = json.loads(clean_text(resp.choices[0].message.content or "{}"))
    lines = [x.get("name","").strip() for x in data.get("items", []) if x.get("name","").strip()]

    seen, out = set(), []
    for ln in lines:
        k = ln.lower()
        if k not in seen:
            seen.add(k)
            out.append(ln)
    return out

def openai_group_names(names: List[str]) -> List[str]:
    c = oai()
    schema = {"final_items":[{"name":"string"}]}
    prompt = f"""
Return ONLY JSON.

Schema:
{json.dumps(schema, indent=2)}

Rules:
- Combine duplicates / similar items into ONE:
  - all chips -> "chips"
  - all eggs -> "eggs"
- Keep names generic and short.
- Output one final item name per entry.

Input items:
{json.dumps(names, ensure_ascii=False)}
""".strip()

    resp = c.chat.completions.create(
        model=MODEL_TEXT,
        messages=[{"role":"system","content":"Group similar items into one. Output JSON only."},
                  {"role":"user","content":prompt}],
        temperature=0.2
    )
    data = json.loads(clean_text(resp.choices[0].message.content or "{}"))
    out = [x.get("name","").strip() for x in data.get("final_items", []) if x.get("name","").strip()]

    seen, deduped = set(), []
    for x in out:
        k = x.lower()
        if k not in seen:
            seen.add(k)
            deduped.append(x)
    return deduped

def categorize(name: str) -> str:
    n = name.lower()
    if any(w in n for w in ["egg","yogurt","milk","cheese","butter"]): return "dairy"
    if any(w in n for w in ["chicken","beef","pork","turkey","fish"]): return "meat"
    if any(w in n for w in ["chip","chips","cracker","cookies","candy","bar"]): return "snack"
    if any(w in n for w in ["coriander","pepper","apple","banana","lettuce","tomato","onion","herb"]): return "produce"
    if any(w in n for w in ["soap","shampoo","toothpaste","deodorant"]): return "personal_care"
    if any(w in n for w in ["detergent","cleaner","paper towel","garbage bag"]): return "household"
    if any(w in n for w in ["water","juice","soda","coffee","tea"]): return "beverage"
    return "other"

def packaging_default(category: str, name: str) -> str:
    n = name.lower()
    if "bread" in n: return "plastic_bag"
    if category == "snack" and "chip" in n: return "plastic_bag"
    if category == "dairy" and "egg" in n: return "paper_carton"
    if category == "dairy" and "yogurt" in n: return "plastic_tub"
    if category == "meat": return "plastic_wrap_tray"
    if category == "produce": return "mixed"
    return "unknown"

def eco_from_pack(pack: str, category: str, name: str) -> bool:
    n = name.lower()
    if "bread" in n:
        return False
    if "egg" in n and pack == "paper_carton":
        return True
    if category == "produce":
        return True
    if pack in ["plastic_bag","plastic_tub","plastic_wrap_tray"]:
        return False
    return False

def item_score(pack: str, category: str) -> int:
    cat_w = {"meat":5,"dairy":4,"snack":3,"beverage":3,"produce":2,"household":3,"personal_care":2,"other":2}
    base = cat_w.get(category, 2)
    pack_pen = {"paper_carton":1,"glass_jar":1,"mixed":2,"unknown":2,"plastic_bag":3,"plastic_tub":3,"plastic_wrap_tray":4}.get(pack, 2)
    penalty = (base * 10) + (pack_pen * 12)
    return max(0, min(100, 100 - penalty))

# --- NEW: hard guard against "health food swaps" ---
FORBIDDEN_SWAP_WORDS = [
    "popcorn", "nuts", "salad", "fruit", "healthy", "healthier", "calorie", "protein", "diet", "low-fat", "low fat"
]

def fallback_swap(name: str, category: str, pack: str) -> str:
    n = name.lower()
    if "chip" in n or n == "chips" or category == "snack":
        return "chips in a larger bag or bulk-bin (fewer bags)"
    if "bread" in n:
        return "same bread as bakery loaf in paper bag"
    if category == "meat" or "chicken" in n:
        return "same cut wrapped in butcher paper (no tray)"
    if "yogurt" in n:
        return "same yogurt in a larger tub / refillable container"
    if "egg" in n:
        return "eggs in paper carton"
    return "same item with minimal packaging"

def sanitize_suggestion(name: str, category: str, pack: str, eco: bool, sug: dict) -> dict:
    if eco:
        sug["swap"] = None
        return sug

    swap = (sug.get("swap") or "").lower()
    option = (sug.get("option") or "").lower()

    if any(w in swap for w in FORBIDDEN_SWAP_WORDS) or any(w in option for w in FORBIDDEN_SWAP_WORDS):
        sug["swap"] = fallback_swap(name, category, pack)
        sug["option"] = "Quick win: choose the same item with less packaging or fewer single-use parts."
        if not sug.get("why"):
            sug["why"] = "Less packaging generally means less waste and fewer hard-to-recycle materials."
    else:
        if len((sug.get("swap") or "")) > 0 and len((sug.get("swap") or "")) > 40:
            sug["swap"] = fallback_swap(name, category, pack)

    return sug

def openai_suggest_one(name: str, category: str, pack: str, eco: bool) -> dict:
    c = oai()
    schema = {"option":"string","swap":"string|null","why":"string"}
    prompt = f"""
Return ONLY JSON.

Schema:
{json.dumps(schema, indent=2)}

Item: "{name}" (category: {category})
Packaging: {pack}
Eco-friendly: {eco}

STRICT RULES:
- Environmental only (packaging/waste/materials/transport). NO health talk.
- DO NOT suggest replacing the food with a different food (e.g., popcorn, nuts, salad).
- Swaps must stay the SAME product type (chips->chips, bread->bread, yogurt->yogurt, chicken->chicken).
- No new brand names.

If eco-friendly:
- option: compliment (<= 14 words)
- swap: null
- why: short (<= 14 words)

If could be greener:
- option: friendly 2–3 short sentences (max 45 words) describing packaging/waste impact.
- swap: short plain packaging-focused alternative (<= 14 words), SAME product type
- why: short reason (<= 18 words)

Packaging-focused examples:
- chips: "larger bag / bulk-bin chips"
- bread: "bakery loaf in paper bag"
- meat/chicken: "butcher-paper wrap (no tray)"
- yogurt: "larger tub / refillable"
""".strip()

    resp = c.chat.completions.create(
        model=MODEL_TEXT,
        messages=[{"role":"system","content":"You write packaging-only eco tips. Output JSON only."},
                  {"role":"user","content":prompt}],
        temperature=0.15
    )
    data = json.loads(clean_text(resp.choices[0].message.content or "{}"))
    return sanitize_suggestion(name, category, pack, eco, data)

def openai_write_insights(greenscore: int, items: List[dict], streaks: Tuple[int,int]) -> dict:
    c = oai()
    schema = {"score_explanation":"string","why_recommendations":"string","streak_explanation":"string"}
    current_streak, best_streak = streaks
    prompt = f"""
Return ONLY JSON.

Schema:
{json.dumps(schema, indent=2)}

GreenScore: {greenscore}/100
Items: {json.dumps([{"name":i["name"],"eco":i["eco"],"pack":i["pack"]} for i in items], ensure_ascii=False)}

Explain:
- score_explanation (2–3 sentences): score = average of ItemScores; biggest drivers are packaging/waste.
- why_recommendations (2–3 sentences): focus on packaging/waste.
- streak_explanation (2 sentences): current={current_streak}, best={best_streak}.
  Define: "Current streak" = consecutive days you analyzed at least one receipt.
  Define: "Best streak" = your longest run so far.
  Mention: this is a habit-builder for packaging awareness.

Environmental only. Friendly. No brands.
""".strip()

    resp = c.chat.completions.create(
        model=MODEL_TEXT,
        messages=[{"role":"system","content":"Write short UI copy. Output JSON only."},
                  {"role":"user","content":prompt}],
        temperature=0.25
    )
    return json.loads(clean_text(resp.choices[0].message.content or "{}"))

# -------------------------
# DB
# -------------------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
      CREATE TABLE IF NOT EXISTS stats (
        id INTEGER PRIMARY KEY CHECK (id=1),
        last_active TEXT,
        current_streak INTEGER,
        best_streak INTEGER
      )
    """)
    cur.execute("INSERT OR IGNORE INTO stats (id,last_active,current_streak,best_streak) VALUES (1,NULL,0,0)")
    cur.execute("""
      CREATE TABLE IF NOT EXISTS receipts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at TEXT,
        score INTEGER
      )
    """)
    conn.commit(); conn.close()

def get_stats():
    conn = sqlite3.connect(DB_PATH); cur = conn.cursor()
    cur.execute("SELECT last_active,current_streak,best_streak FROM stats WHERE id=1")
    row = cur.fetchone()
    conn.close()
    return row

def update_streak_on_analyze():
    today = date.today()
    last_active, current_streak, best_streak = get_stats()

    if last_active is None:
        new_streak = 1
    else:
        last = date.fromisoformat(last_active)
        if last == today:
            new_streak = current_streak
        elif last == today - timedelta(days=1):
            new_streak = current_streak + 1
        else:
            new_streak = 1

    new_best = max(best_streak, new_streak)
    conn = sqlite3.connect(DB_PATH); cur = conn.cursor()
    cur.execute("UPDATE stats SET last_active=?, current_streak=?, best_streak=? WHERE id=1",
                (today.isoformat(), new_streak, new_best))
    conn.commit(); conn.close()
    return new_streak, new_best

def save_receipt(score: int):
    conn = sqlite3.connect(DB_PATH); cur = conn.cursor()
    cur.execute("INSERT INTO receipts (created_at, score) VALUES (?, ?)", (date.today().isoformat(), score))
    conn.commit(); conn.close()

def recent_scores(limit=10) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH); cur = conn.cursor()
    cur.execute("SELECT created_at AS Date, score AS GreenScore FROM receipts ORDER BY id DESC LIMIT ?", (limit,))
    rows = cur.fetchall(); conn.close()
    return pd.DataFrame(rows, columns=["Date","GreenScore"])

def reset_all():
    conn = sqlite3.connect(DB_PATH); cur = conn.cursor()
    cur.execute("DELETE FROM receipts")
    cur.execute("UPDATE stats SET last_active=NULL,current_streak=0,best_streak=0 WHERE id=1")
    conn.commit(); conn.close()

def canada_disposal_tip(item_name: str, category: str, pack: str) -> str:
    if category in ["produce", "meat", "dairy"]:
        return "Food scraps → Green bin. Packaging (clean/dry) → usually Blue bin; otherwise Garbage. (Rules vary by city.)"
    if pack in ["plastic_bag", "plastic_tub"]:
        return "Rinse/empty first → usually Blue bin. If dirty/greasy → Garbage. (Rules vary by city.)"
    if pack == "plastic_wrap_tray":
        return "Foam/tray + cling wrap → often Garbage. Any clean paper label/cardboard → Blue bin. (Rules vary by city.)"
    if pack == "paper_carton":
        return "Flatten carton → usually Blue bin. (Rules vary by city.)"
    return "Separate parts. Clean paper/plastic → usually Blue bin; food scraps → Green bin; dirty/foam → Garbage. (Rules vary by city.)"

def plural(n: int, word: str) -> str:
    n = int(n)
    return f"{n} {word}" + ("" if n == 1 else "s")

# -------------------------
# Packaging check (NO rerun on each selection -> use st.form)
# -------------------------
PACK_PICK_DEFS = {
    "eggs": {
        "title": "Eggs packaging",
        "a_label": "Paper carton (greener)",
        "a_pack": "paper_carton",
        "b_label": "Plastic clamshell (less green)",
        "b_pack": "plastic_tub",
    },
    "bread": {
        "title": "Bread packaging",
        "a_label": "Paper bakery bag (greener)",
        "a_pack": "paper_carton",
        "b_label": "Plastic bag (less green)",
        "b_pack": "plastic_bag",
    },
    "yogurt": {
        "title": "Yogurt packaging",
        "a_label": "Large tub / refillable (greener)",
        "a_pack": "glass_jar",
        "b_label": "Small plastic cups/tub (less green)",
        "b_pack": "plastic_tub",
    },
    "chicken": {
        "title": "Chicken / meat packaging",
        "a_label": "Butcher paper / minimal wrap (greener)",
        "a_pack": "paper_carton",
        "b_label": "Foam tray + cling wrap (less green)",
        "b_pack": "plastic_wrap_tray",
    },
    "chips": {
        "title": "Chips packaging",
        "a_label": "Larger bag / fewer bags (greener)",
        "a_pack": "mixed",
        "b_label": "Small single-use bag (less green)",
        "b_pack": "plastic_bag",
    },
}

def get_override_pack(item_name: str) -> Optional[str]:
    n = (item_name or "").lower().strip()
    po = st.session_state.get("pack_overrides", {})
    if "egg" in n or n == "eggs":
        return po.get("eggs")
    if "bread" in n:
        return po.get("bread")
    if "yogurt" in n:
        return po.get("yogurt")
    if "chicken" in n or n == "meat":
        return po.get("chicken")
    if "chip" in n or n == "chips":
        return po.get("chips")
    return None

# -------------------------
# App
# -------------------------
init_db()
last_active, current_streak, best_streak = get_stats()

st.markdown('<div class="title">🌿 GreenReceipt</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload a receipt → get packaging-focused eco suggestions + streaks.</div>', unsafe_allow_html=True)

if os.getenv("OPENAI_API_KEY"):
    st.markdown(f'<div class="affirm">{openai_affirmation()}</div>', unsafe_allow_html=True)

# Better streak UI (PLACEHOLDERS so we can refresh after Analyze without rerun)
top1, top2, top3 = st.columns([1.05, 1.05, 1.2])
streak_ph_1 = top1.empty()
streak_ph_2 = top2.empty()
streak_prog_ph = top2.empty()

def render_streak_ui(cur: int, best: int):
    streak_ph_1.markdown(
        textwrap.dedent(f"""
        <div class="card">
          <div class="pill">🔥 Current streak</div>
          <h2 style="margin:10px 0 0 0;">{plural(cur, "day")}</h2>
          <div class="muted">Consecutive days you analyzed at least one receipt.</div>
        </div>
        """),
        unsafe_allow_html=True
    )

    streak_ph_2.markdown(
        textwrap.dedent(f"""
        <div class="card">
          <div class="pill">🏆 Best streak</div>
          <h2 style="margin:10px 0 0 0;">{plural(best, "day")}</h2>
          <div class="muted">Your longest run of consecutive analysis days.</div>
        </div>
        """),
        unsafe_allow_html=True
    )

    denom = max(1, best)
    streak_prog_ph.progress(min(1.0, cur / denom))

# initial render
render_streak_ui(current_streak, best_streak)

with top3:
    st.markdown('<div class="card"><div class="pill">🧹 Reset</div>', unsafe_allow_html=True)
    cols = st.columns([1, 1])
    with cols[0]:
        if st.button("Reset history"):
            reset_all()
            st.session_state["pack_overrides"] = {}
            st.success("Reset complete ✅")
            st.rerun()
    with cols[1]:
        if st.button("Clear current receipt"):
            st.session_state["receipt_text_raw"] = ""
            st.session_state["items_final_text"] = ""
            st.session_state["items_editor"] = ""
            st.session_state["last_run"] = None
            st.session_state["pack_overrides"] = {}
            st.session_state["receipt_fingerprint"] = ""
            st.success("Cleared ✅")
            st.rerun()

st.divider()

left, right = st.columns([1.15, 1.85])

with left:
    st.subheader("1) Upload receipt")
    uploaded = st.file_uploader("Upload TXT/PNG/JPG", type=["txt", "png", "jpg", "jpeg"])

    if uploaded is not None:
        if uploaded.type == "text/plain" or uploaded.name.lower().endswith(".txt"):
            raw = uploaded.getvalue().decode("utf-8", errors="replace")
            raw = clean_text(raw)
        else:
            st.image(uploaded, caption="Receipt uploaded ✅", width=55)
            with st.spinner("Reading receipt text automatically…"):
                mime = uploaded.type or "image/jpeg"
                raw = openai_ocr_receipt(uploaded.getvalue(), mime)

        st.session_state["receipt_text_raw"] = raw

        fp = fingerprint_text(raw)
        if fp != st.session_state.get("receipt_fingerprint", ""):
            st.session_state["receipt_fingerprint"] = fp
            st.session_state["items_final_text"] = ""
            st.session_state["items_editor"] = ""
            st.session_state["pack_overrides"] = {}
            st.session_state["last_run"] = None

    raw_receipt = st.session_state.get("receipt_text_raw", "")
    if raw_receipt and looks_like_docs_or_code(raw_receipt):
        st.error("This doesn’t look like a receipt (it looks like docs/code text). Please upload again.")
        st.stop()

    st.subheader("2) Items from receipt (edit if needed)")

    if raw_receipt.strip() and os.getenv("OPENAI_API_KEY") and not st.session_state.get("items_editor", "").strip():
        items_only = strip_prices_counts(raw_receipt)
        if items_only.strip():
            with st.spinner("Cleaning item names…"):
                debranded_list = openai_debrand_lines(items_only)
                grouped_list = openai_group_names(debranded_list)
                st.session_state["items_final_text"] = "\n".join(grouped_list)
                st.session_state["items_editor"] = st.session_state["items_final_text"]

    st.text_area(
        "Items from receipt (edit if needed).",
        key="items_editor",
        height=220
    )

    names_preview = [ln.strip() for ln in st.session_state.get("items_editor", "").splitlines() if ln.strip()]
    names_preview_l = [n.lower() for n in names_preview]

    st.subheader("3) Quick packaging check (optional)")
    st.markdown('<div class="muted">Pick the packaging you actually bought (no brands). This avoids wrong assumptions.</div>', unsafe_allow_html=True)

    with st.form("packaging_form", clear_on_submit=False):
        if any("egg" in n for n in names_preview_l) or "eggs" in names_preview_l:
            d = PACK_PICK_DEFS["eggs"]
            st.markdown(f'<div class="pickcard"><div class="picktitle">{d["title"]}</div></div>', unsafe_allow_html=True)
            egg_choice = st.radio("eggs_pack", [d["a_label"], d["b_label"]], horizontal=True, label_visibility="collapsed")
        else:
            egg_choice = None

        if any("bread" in n for n in names_preview_l):
            d = PACK_PICK_DEFS["bread"]
            st.markdown(f'<div class="pickcard"><div class="picktitle">{d["title"]}</div></div>', unsafe_allow_html=True)
            bread_choice = st.radio("bread_pack", [d["a_label"], d["b_label"]], horizontal=True, label_visibility="collapsed")
        else:
            bread_choice = None

        if any("yogurt" in n for n in names_preview_l):
            d = PACK_PICK_DEFS["yogurt"]
            st.markdown(f'<div class="pickcard"><div class="picktitle">{d["title"]}</div></div>', unsafe_allow_html=True)
            yogurt_choice = st.radio("yogurt_pack", [d["a_label"], d["b_label"]], horizontal=True, label_visibility="collapsed")
        else:
            yogurt_choice = None

        if any("chicken" in n for n in names_preview_l) or "meat" in names_preview_l:
            d = PACK_PICK_DEFS["chicken"]
            st.markdown(f'<div class="pickcard"><div class="picktitle">{d["title"]}</div></div>', unsafe_allow_html=True)
            chicken_choice = st.radio("chicken_pack", [d["a_label"], d["b_label"]], horizontal=True, label_visibility="collapsed")
        else:
            chicken_choice = None

        if any("chip" in n for n in names_preview_l) or "chips" in names_preview_l:
            d = PACK_PICK_DEFS["chips"]
            st.markdown(f'<div class="pickcard"><div class="picktitle">{d["title"]}</div></div>', unsafe_allow_html=True)
            chips_choice = st.radio("chips_pack", [d["a_label"], d["b_label"]], horizontal=True, label_visibility="collapsed")
        else:
            chips_choice = None

        analyze_clicked = st.form_submit_button("Analyze ✨")

        if analyze_clicked:
            overrides = {}

            def set_override(key: str, choice_label: Optional[str]):
                if not choice_label:
                    return
                dd = PACK_PICK_DEFS[key]
                overrides[key] = dd["a_pack"] if choice_label == dd["a_label"] else dd["b_pack"]

            set_override("eggs", egg_choice)
            set_override("bread", bread_choice)
            set_override("yogurt", yogurt_choice)
            set_override("chicken", chicken_choice)
            set_override("chips", chips_choice)

            st.session_state["pack_overrides"] = overrides
            st.session_state["last_run"] = {
                "items_final_text": st.session_state.get("items_editor", "").strip(),
                "pack_overrides": dict(overrides),
            }

with right:
    st.subheader("Results")

    if st.session_state.get("last_run") is None:
        st.info("Upload a receipt and click **Analyze ✨**.")
        st.stop()

    if not os.getenv("OPENAI_API_KEY"):
        st.error("Missing OPENAI_API_KEY. Set it and restart Streamlit.")
        st.stop()

    items_text_run = st.session_state["last_run"]["items_final_text"]
    if not items_text_run:
        st.error("No items found. Please upload a receipt again.")
        st.stop()

    st.session_state["pack_overrides"] = st.session_state["last_run"].get("pack_overrides", {})

    st.toast("Analyzing… 🍃", icon="🌿")
    step_line = st.empty()
    progress = st.progress(0)

    step_line.info("Step 1/3: Preparing items…")
    progress.progress(30)

    names = [ln.strip() for ln in items_text_run.splitlines() if ln.strip()]
    normalized = []
    for nm in names:
        cat = categorize(nm)

        override_pack = get_override_pack(nm)
        pack = override_pack if override_pack else packaging_default(cat, nm)

        eco = eco_from_pack(pack, cat, nm)
        score = item_score(pack, cat)
        normalized.append({"name": nm, "category": cat, "pack": pack, "eco": eco, "item_score": score})

    step_line.info("Step 2/3: Writing suggestions…")
    progress.progress(70)

    for it in normalized:
        sug = openai_suggest_one(it["name"], it["category"], it["pack"], it["eco"])
        it["option"] = sug.get("option", "")
        it["swap"] = sug.get("swap")
        it["why"] = sug.get("why", "")

    greenscore = int(round(sum(i["item_score"] for i in normalized) / max(1, len(normalized))))

    step_line.info("Step 3/3: Saving + explaining…")
    progress.progress(90)

    new_streak, new_best = update_streak_on_analyze()
    render_streak_ui(new_streak, new_best)  # <-- FIX: refresh the top cards immediately (no rerun)
    insights = openai_write_insights(greenscore, normalized, (new_streak, new_best))
    save_receipt(greenscore)

    progress.progress(100)
    step_line.success("Done ✅ Results are ready!")

    st.markdown(
        textwrap.dedent(f"""
        <div class="card">
          <div class="pill">🌱 GreenScore</div>
          <h1 style="margin: 8px 0 0 0;">{greenscore}/100</h1>
          <div class="muted" style="margin-top: 6px;">{insights.get("score_explanation","")}</div>
        </div>
        """),
        unsafe_allow_html=True
    )
    st.progress(greenscore / 100)

    tab1, tab2, tab3 = st.tabs(["Why these recommendations?", "About streaks", "Recent scores"])
    with tab1:
        st.markdown(f"""<div class="card">{insights.get("why_recommendations","")}</div>""", unsafe_allow_html=True)
    with tab2:
        st.markdown(f"""<div class="card">{insights.get("streak_explanation","")}</div>""", unsafe_allow_html=True)
    with tab3:
        df = recent_scores(limit=10)
        if df.empty:
            st.info("No scores yet.")
        else:
            st.dataframe(df, use_container_width=True, hide_index=True)

    st.subheader("Item-by-item suggestions")
    for it in normalized:
        badge = "✅ Eco-friendly" if it["eco"] else "— Could be greener"
        st.markdown(
            textwrap.dedent(f"""
            <div class="item">
              <b>{it["name"]}</b>
              <span class="tag">{it["category"]}</span>
              <span class="tag">{badge}</span>

              <div class="muted" style="margin-top:8px;">{it.get("option","")}</div>
              {("<div style='margin-top:8px;'><b>👉 Swap:</b> " + str(it.get("swap")) + "</div>") if (it.get("swap") and not it["eco"]) else ""}
              {("<div class='reason'><b>Why:</b> " + it.get("why","") + "</div>") if (it.get("why") and not it["eco"]) else ""}

              <div class="reason"><b>If you can’t swap:</b> {canada_disposal_tip(it["name"], it["category"], it["pack"])}</div>
            </div>
            """),
            unsafe_allow_html=True
        )
