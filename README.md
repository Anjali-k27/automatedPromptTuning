# Newton SkillUP KPI Extractor (DSPy Optimized)

A local Streamlit dashboard that uses DSPy's `BootstrapFewShot` optimizer and Google Gemini 2.5 Flash to extract structured KPIs from raw, unstructured meeting transcripts — no manual prompt engineering required.

---

## Project Structure

```
.
├── .env                                  # Your Gemini API key (never commit)
├── .gitignore
├── requirements.txt
├── extractor_engine.py                   # Core DSPy engine (LMs, Signature, Module, Metric, Training data)
├── compile_and_test.py                   # One-time optimizer — runs BootstrapFewShot, saves state
├── app.py                                # Streamlit dashboard UI
└── gemini_optimized_dashboard_state.json # Generated after running compile_and_test.py
```

---

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Add your Gemini API key
Open `.env` and paste your key:
```
GEMINI_API_KEY=your_api_key_here
```

### 3. Compile and test the pipeline (run once)
```bash
python compile_and_test.py
```
This runs the DSPy optimizer, saves the compiled state to `gemini_optimized_dashboard_state.json`, and verifies the pipeline passes the metric test.

### 4. Launch the dashboard
```bash
streamlit run app.py
```
Open http://localhost:8501 in your browser.

---

## Deep Dive: How It All Works

### Step 1 — Define the Task as a DSPy Signature (`extractor_engine.py`)

DSPy replaces hand-written prompts with **Signatures** — declarative specs that describe inputs and outputs. Instead of crafting a prompt, you describe what you want:

```python
class DashboardDataExtraction(dspy.Signature):
    """Extract structured key performance indicators from raw meeting transcripts."""

    raw_transcript = dspy.InputField(desc="The raw text of the daily operations meeting.")
    json_kpis = dspy.OutputField(desc="Valid JSON containing exactly: 'revenue', 'churn_rate', and 'new_signups'. Output ONLY JSON.")
```

DSPy compiles this into an actual prompt at runtime — and crucially, **re-compiles it with few-shot examples** after optimization.

---

### Step 2 — Wrap it in a Module with ChainOfThought (`extractor_engine.py`)

A DSPy **Module** is like a PyTorch `nn.Module` for LLM pipelines. Instead of `nn.Linear`, you use `dspy.ChainOfThought` — a predictor that forces the model to reason step-by-step before producing its output:

```python
class DashboardExtractor(dspy.Module):
    def __init__(self):
        super().__init__()
        self.extract = dspy.ChainOfThought(DashboardDataExtraction)

    def forward(self, raw_transcript):
        return self.extract(raw_transcript=raw_transcript)
```

`ChainOfThought` adds a hidden `reasoning` field to the output, which is what you see in the "View DSPy Reasoning" expander in the UI.

---

### Step 3 — Define a Metric (`extractor_engine.py`)

The metric is a Python function that tells the optimizer what "correct" looks like. It must return `True` or `False`:

```python
def dashboard_json_metric(example, pred, trace=None):
    try:
        cleaned = pred.json_kpis.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(cleaned)
        return {"revenue", "churn_rate", "new_signups"} == set(parsed.keys())
    except Exception:
        return False
```

Key design decisions:
- **Markdown stripping** — models often wrap JSON in triple-backtick code blocks; this handles that.
- **Exact key match** — `set(parsed.keys()) == required_keys` ensures no extra or missing fields, not just a subset check.
- **Exception catch** — any malformed output (not valid JSON, wrong type) returns `False` cleanly.

---

### Step 4 — Provide Training Data (`extractor_engine.py`)

DSPy needs a small `trainset` of input examples. Crucially, **no labels are required** — `BootstrapFewShot` generates its own labels by running the teacher model and checking them against the metric.

The 5 examples intentionally cover a range of difficulty:

| Example | Challenge |
|---|---|
| Q3 earnings call | Clean, structured |
| Weekly standup | Different ordering |
| Casual meeting notes | Noisy, informal speech |
| No metrics reported | Edge case — missing data |
| End of month | Negative scenario, low numbers |

---

### Step 5 — Compile with BootstrapFewShot (`compile_and_test.py`)

`BootstrapFewShot` is DSPy's core optimizer. Here's what it does step by step:

1. Takes the teacher module (same architecture, runs with `gemini-2.5-flash`)
2. Runs the teacher on each training example
3. Checks each output against `dashboard_json_metric`
4. Keeps only the examples where the teacher **succeeded** — these become few-shot demonstrations
5. Injects those demonstrations into the student's prompt automatically
6. The student now has proven examples baked in, without you writing a single prompt

```python
with dspy.context(lm=teacher_lm):
    compiled_pipeline = optimizer.compile(
        student=unoptimized_pipeline,
        teacher=teacher_module,
        trainset=trainset,
    )
```

`dspy.context(lm=teacher_lm)` is a thread-safe context manager that temporarily overrides the default LM for the teacher's execution only — so the student and teacher can use different models.

The compiled state (few-shot demonstrations + prompt instructions) is saved to disk:
```bash
compiled_pipeline.save('gemini_optimized_dashboard_state.json')
```

---

### Step 6 — Test the Pipeline (`compile_and_test.py`)

After compiling, the script immediately runs a verification test:

**Test input:**
```
Final quarter push. We hit $900k in revenue! Signups are booming at 12,000. Churn is incredibly low at 0.5%.
```

**Expected output:**
```json
{
  "revenue": 900000,
  "churn_rate": 0.5,
  "new_signups": 12000
}
```

**Extraction logic (performed by the model):**
- `$900k` → `900000` (k suffix expanded ×1000, dollar sign stripped)
- `12,000` → `12000` (comma stripped, parsed as integer)
- `0.5%` → `0.5` (percent sign stripped, stored as float)

The assertion `assert dashboard_json_metric(None, result)` guarantees the pipeline passes the same metric used during optimization before the state is trusted for production use.

---

### Step 7 — Streamlit Dashboard (`app.py`)

The UI loads the compiled state at startup:

```python
pipeline = DashboardExtractor()
pipeline.load('gemini_optimized_dashboard_state.json')
```

When the user clicks **Extract KPIs**:
1. The transcript is passed to `pipeline.forward(raw_transcript=...)`
2. The result's `json_kpis` field is parsed and rendered with `st.json()`
3. The result's `reasoning` field (from `ChainOfThought`) is shown in the expander

If `gemini_optimized_dashboard_state.json` doesn't exist (i.e., `compile_and_test.py` hasn't been run), the app shows a warning instead of crashing.

---

## Environment Variables

| Variable | Description |
|---|---|
| `GEMINI_API_KEY` | Your Google Gemini API key |

The engine maps it internally: `os.environ["GOOGLE_API_KEY"] = os.environ.get("GEMINI_API_KEY", "")` so both LiteLLM and any Google SDK can find it.

---

## Models Used

| Role | Model | Purpose |
|---|---|---|
| Student | `gemini/gemini-2.5-flash` | Default inference LM, runs in production |
| Teacher | `gemini/gemini-2.5-flash` | Generates few-shot demonstrations during compilation only |
