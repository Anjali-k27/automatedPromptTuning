import dspy
import os
import json
from dotenv import load_dotenv

load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.environ.get("GEMINI_API_KEY","")

student_lm = dspy.LM(model='gemini/gemini-2.5-flash', max_tokens=500)
teacher_lm = dspy.LM(model='gemini/gemini-2.5-flash', max_tokens=500)

dspy.configure(lm=student_lm)

class DashboardDataExtraction(dspy.Signature):
    """Extract structured key performance indicators from raw meeting transcripts."""

    raw_transcript = dspy.InputField(desc="The raw text of the daily operations meeting.")
    json_kpis = dspy.OutputField(desc="Valid JSON containing exactly: 'revenue', 'churn_rate', and 'new_signups'. Output ONLY JSON.")

class DashboardExtractor(dspy.Module):
    def __init__(self):
        super().__init__()
        self.extract = dspy.ChainOfThought(DashboardDataExtraction)

    def forward(self, raw_transcript):
        return self.extract(raw_transcript=raw_transcript)
    
def dashboard_json_metric(example, pred, trace=None):
    try:
        cleaned = pred.json_kpis.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(cleaned)
        required_keys = {"revenue", "churn_rate", "new_signups"}
        return required_keys == set(parsed.keys())
    except Exception:
        return False
    
trainset = [
    dspy.Example(
        raw_transcript="Q3 earnings call. Revenue is $4.2M. Churn decreased to 1.1%. New signups reached 8,000."
    ).with_inputs("raw_transcript"),
    dspy.Example(
        raw_transcript="Weekly standup. We saw 400 new signups. Revenue flat at $50k. Churn is 5%."
    ).with_inputs("raw_transcript"),
    dspy.Example(
        raw_transcript="Hey guys so yeah the meeting was good um I think revenue was like 120 thousand. And John said churn is bad right now, hitting 8 percent. Signups? Oh yeah, 50."
    ).with_inputs("raw_transcript"),
    dspy.Example(
        raw_transcript="Data review. No metrics reported today regarding financials or user base."
    ).with_inputs("raw_transcript"),
    dspy.Example(
        raw_transcript="End of month. Churn spiked to 12%. Only 5 signups. Revenue tanked to $2k."
    ).with_inputs("raw_transcript"),
]