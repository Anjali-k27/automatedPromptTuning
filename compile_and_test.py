import dspy 
from extractor_engine import (DashboardExtractor, dashboard_json_metric, student_lm, teacher_lm, trainset)

TEST_TRANSCRIPT = (
                   "Final quarter push. We hit $900k in revenue! "
                   "Signups are booming at 12,000. Churn is incredibly low at 0.5%."
                   )

unoptimized_pipeline = DashboardExtractor()
teacher_module = DashboardExtractor()

optimizer = dspy.BootstrapFewShot(
    metric=dashboard_json_metric,
    max_bootstrapped_demos=3,
    max_labeled_demos=0)

# At the time of compilation, we should use teacher LM.
print("Compiling pipeline with Gemini 2.5 Flash as Teacher...")

with dspy.context(lm=teacher_lm):
    compiled_pipeline = optimizer.compile(
        student=unoptimized_pipeline,
        teacher=teacher_module,
        trainset=trainset,
    )

compiled_pipeline.save('gemini_optimized_dashboard_state.json')
print("Pipeline saved to gemini_optimized_dashboard_state.json")

production_pipeline = DashboardExtractor()
production_pipeline.load('gemini_optimized_dashboard_state.json')

result = production_pipeline(raw_transcript=TEST_TRANSCRIPT)

# assert will act like "if not"
assert dashboard_json_metric(None, result), (
    f"Metric failed. Raw output: {result.json_kpis}"
)

print("\nTest passed! Extracted KPIs:")
print(result.json_kpis)