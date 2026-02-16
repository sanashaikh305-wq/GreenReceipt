# GreenReceipt

GreenReceipt is a Streamlit app that turns a shopping receipt into **packaging-focused** eco suggestions.
It extracts items, lets you confirm packaging (optional), computes a GreenScore, and tracks daily streaks.

## Features
- Upload receipt (JPG)
- Extract item names (brand removed)
- Optional packaging check (egg/bread/yogurt/chicken/chips)
- Packaging-only eco suggestions (no health advice)
- GreenScore + streak tracking (current + best)
- Recent score history

---

## Requirements
- Python 3.10+ recommended
- An OpenAI API key

---

## Quick Start (Mac)
### 1) Download / clone the repo
If you downloaded ZIP: unzip and open the folder in Terminal.

### 2) Create a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
## Setup Instructions

1. Install dependencies:
pip install -r requirements.txt

2. Set your OpenAI API key:
export OPENAI_API_KEY="YOUR_KEY"

3. Run the app:
streamlit run app.py

