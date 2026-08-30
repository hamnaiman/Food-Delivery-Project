# Food Delivery Analytics Challenge

A small end-to-end analytics pipeline that digs into a real food delivery
dataset (~39,000 orders) to understand what actually slows deliveries down,
and uses an AI model to explain the findings in plain English.

## What this project answers

1. **How does road traffic density affect delivery time?**
2. **How does delivery distance affect delivery time?**
3. **What happens when bad weather and heavy traffic combine?**

Every number in the final report is computed directly from the dataset at
run time — nothing is hardcoded.

## Project structure

```
data/food_delivery_dataset.csv   Raw dataset
src/analysis.py                  Data loading, cleaning, and the core analysis
src/visualization.py             Chart generation (matplotlib)
src/ai_explanation.py            Sends results to Gemini for a plain-English write-up
tests/test_analysis.py           Unit tests for the analysis logic
outputs/charts/                  Generated PNG charts (created on run)
outputs/report.txt               Full text report (created on run)
main.py                          Runs the whole pipeline end to end
```

## Setup

1. Create and activate a virtual environment (skip if you already have one):

   ```bash
   python -m venv .venv
   source .venv/bin/activate      # Windows: .venv\Scripts\activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. (Optional but recommended) Add your Gemini API key so Step 5 uses real
   AI-generated explanations instead of the local fallback summary:

   ```bash
   cp .env.example .env
   # then edit .env and paste in your key
   ```

   A free key is available at https://aistudio.google.com/apikey. If no key
   is set, the pipeline still runs completely — it just falls back to a
   summary built from the computed insights instead of calling Gemini.

## Running it

```bash
python main.py
```

This will print a full report to the console, save four charts to
`outputs/charts/`, and write the complete report to `outputs/report.txt`.

## Running the tests

```bash
pytest tests/ -v
```

## Key findings (from this dataset)

- **Traffic matters a lot.** Deliveries made during Jam-level traffic take
  roughly 10 minutes longer on average than deliveries made when traffic is
  Low — the single biggest lever on delivery time in this dataset.
- **Distance has a moderate, not dominant, effect.** Delivery time and
  distance are positively correlated, but the relationship flattens out
  past roughly 10-12 km — beyond that point, other factors (traffic,
  weather) matter more than the extra distance itself.
- **Weather and traffic compound each other.** The worst combination in
  the data is foggy/cloudy weather paired with jammed traffic, averaging
  close to 37 minutes per delivery — well above the ~27 minute overall
  average. This is the scenario operations teams should build the most
  buffer time around.

## Design notes

- `analysis.py` never touches plotting or the AI layer — it can be unit
  tested with a plain in-memory DataFrame, no charts or API keys required.
- Invalid values (negative times, out-of-range ages/ratings, zero
  distances) are nulled out rather than dropping the whole row, so a bad
  value in one column doesn't waste an otherwise-good record.
- `ai_explanation.py` fails soft: if there's no API key, no internet, or
  the API call errors out for any reason, the pipeline still completes
  using a locally-built summary instead of crashing.
