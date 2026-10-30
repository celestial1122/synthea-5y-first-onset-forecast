"""Build an evidence-based first-iteration PDF using saved artifacts only.

Run with the bundled PDF runtime (reportlab), or install the report extra.
"""

from pathlib import Path
import json
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image


ROOT = Path(__file__).resolve().parents[1]


def load(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def main():
    out = ROOT / "output/pdf/initial_results.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    bag = load("artifacts/runs/bag-final-seed42/validation_metrics.json")
    metrics = load("artifacts/runs/transformer-seed42/validation_metrics.json")
    prior = load("artifacts/runs/transformer-seed42/prevalence_baseline_metrics.json")
    config = load("artifacts/runs/transformer-seed42/config.json")
    env = load("artifacts/runs/transformer-seed42/environment.json")
    summary = load("artifacts/runs/transformer-seed42/summary.json")
    audit = load("artifacts/prepared/audit.json")
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="ReportTitle", fontName="Helvetica-Bold", fontSize=23, leading=27, spaceAfter=12, textColor=colors.HexColor("#142B43")))
    styles.add(ParagraphStyle(name="ReportBody", fontName="Helvetica", fontSize=10, leading=14, spaceAfter=9))
    styles.add(ParagraphStyle(name="ReportSmall", fontName="Helvetica", fontSize=8.2, leading=11, spaceAfter=6))
    styles.add(ParagraphStyle(name="ReportHeading", fontName="Helvetica-Bold", fontSize=13, leading=17, spaceBefore=12, spaceAfter=7, textColor=colors.HexColor("#142B43")))
    styles.add(ParagraphStyle(name="ReportCode", fontName="Courier", fontSize=8, leading=11, spaceAfter=6))
    content = []

    def p(text, style="ReportBody"):
        content.append(Paragraph(text, styles[style]))

    def h(text):
        p(text, "ReportHeading")

    def table(rows, widths, font=9, repeat=1):
        result = Table(rows, colWidths=widths, repeatRows=repeat, hAlign="LEFT")
        result.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E9EFF5")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#142B43")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), font),
            ("LEADING", (0, 0), (-1, -1), font + 2),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 5 if font >= 9 else 2),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5 if font >= 9 else 2),
            ("LINEBELOW", (0, 0), (-1, 0), .6, colors.HexColor("#B6C3D0")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7F9FB")]),
        ]))
        content.append(result)
        content.append(Spacer(1, 7))

    p("Forecasting incident conditions<br/>from patient timelines", "ReportTitle")
    p("INITIAL TECHNICAL REPORT | 28 OCTOBER 2026", "ReportSmall")
    p("Local implementation and first measured experiments. This draft has no candidate identity or public experiment/repository URLs and is not a final submission.")
    h("Task and official evaluation population")
    p("Predict 40 independent probabilities per patient: whether the first-ever diagnosis of each target condition falls in the five-year prediction window. Official patient splits are retained: 2,791 training, 365 validation and 358 test patients. No external dataset, tokenizer or pretrained weights are used.")
    h("Time boundaries and leakage controls")
    p("For train/validation, the anchor is the latest encounter endpoint (STOP, or START if STOP is blank), minus five calendar years. Test anchors come directly from test_anchors.csv. Timestamps use recorded calendar dates: history date &lt; anchor; first diagnosis labels cover [anchor, final encounter date]. The anchor day is assigned to the outcome window.")
    p("Labels use each patient's earliest diagnosis date for a CODE across the complete condition history. Recurrences are negative for incident disease. Future rows are used only for labels. Death date, final costs, medication totals, STOP fields, identity fields and reason codes are excluded from model inputs. Previously diagnosed target conditions are masked from the loss and receive zero future-incidence probability using complete pre-anchor history.")
    h("Measured validation results")
    rows = [["Model", "mAP", "Macro AUROC", "Micro AP", "Brier"]]
    for name, result in [("Train incidence reference", prior), ("BagOfEvents", bag), ("Causal Transformer", metrics)]:
        rows.append([name, *[f"{result[key]:.4f}" for key in ["mAP", "macro_auroc", "micro_ap", "brier"]]])
    table(rows, [185, 65, 95, 65, 65])
    p("Source: saved validation_metrics.json and prevalence_baseline_metrics.json; official 365-patient validation split, all 40 targets. No test metric is available because test outcomes are withheld.", "ReportSmall")
    p("Validation mAP was the declared checkpoint and model selection criterion. The Transformer is selected for predictions.csv, but BagOfEvents has higher macro AUROC and lower Brier loss. The 0.0055 mAP difference does not establish a reliable general advantage.")

    content.append(PageBreak())
    p("Representation and training", "ReportTitle")
    h("One input unit")
    p("Each token is a dated clinical event. Nine structured modalities are used: conditions, medications, procedures, observations, encounters, immunizations, allergies, care plans and devices. Tokens combine modality and CODE; observations additionally encode the original unit and a categorical value or train-fitted quantile bin. Up to eight numeric bins are fitted using training pre-anchor observations; tied quantiles can reduce the actual number of bins.")
    p("The 1,260-token vocabulary is fitted only on retained training histories. PAD=0, UNK=1, BOS=2 and MASK=3; MASK is reserved and unused. Histories are deduplicated by day/code/value, sorted deterministically, then truncated to BOS plus the latest 255 events. BOS time is fixed at zero. Time features are log(1+days before anchor)/10 and capped years before anchor/120. Five static features encode standardized age, sex indicators and missing age; fitting uses training patients only.")
    h("Architecture and objectives")
    p(f"The Transformer has {config['n_layers']} causal layers, {config['n_heads']} heads, hidden width {config['d_model']}, 4x feed-forward width and dropout {config['dropout']}. Token, position and time embeddings enter the shared backbone. The last valid event state and static features feed a 40-logit head. Parameter count: {env['parameter_count']:,}.")
    p("Four next-event cross-entropy epochs use only training pre-anchor timelines and ignore PAD targets. Fine-tuning uses unweighted at-risk BCE, train-only incidence bias initialization, AdamW, cosine learning-rate decay, gradient norm clipping at 1.0, batch size 32 and CUDA mixed precision. Pretraining LR=0.0005; supervised LR=0.0003; weight decay=0.01. Maximum fine-tuning epochs=35, early-stopping patience=8. The selected checkpoint is epoch 8. Seed=42.")
    h("Observed learning curves")
    content.append(Image(str(ROOT / "artifacts/runs/transformer-seed42/training_curves.png"), width=475, height=139))
    p("Source: this run's history.jsonl. Each point is one fine-tuning epoch. Best validation mAP occurs before training BCE stops decreasing, consistent with a generalization limit on this small cohort.", "ReportSmall")
    p(f"Runtime: Python {env['python']}, PyTorch {escape(env['torch'])}, CUDA {env['cuda_runtime']}, {escape(env['gpu'])}. Saved run time: {summary['elapsed_seconds']:.1f} seconds (excludes data preparation and dependency setup).", "ReportSmall")

    content.append(PageBreak())
    p("Per-condition validation", "ReportTitle")
    p("Official 365-patient validation split; selected Transformer checkpoint. All 40 classes have positive and negative examples. AP is average precision; support is the number of positive patients. Descriptions below are shortened; complete names and values are in per_condition.csv.", "ReportSmall")
    rows = [["CODE", "Condition (shortened)", "+", "AUROC", "AP"]]
    for row in metrics["per_condition"]:
        name = row["description"].replace(" (disorder)", "").replace(" (finding)", "").replace(" (situation)", "")
        if len(name) > 49:
            name = name[:46] + "..."
        rows.append([row["code"], name, str(row["n_positive"]), f"{row['auroc']:.3f}", f"{row['average_precision']:.3f}"])
    table(rows, [67, 282, 26, 50, 50], font=8)
    p("Prostate-related conditions have high observed AP; chronic disease context, age and sex are plausible contributors, but no causal attribution was measured. Several acute infections and injuries have low AP, consistent with limited predictability from the retained history. Fourteen classes have fewer than ten validation positives, so individual patients can substantially change these scores.", "ReportSmall")

    content.append(PageBreak())
    p("Limits, reproducibility and workflow", "ReportTitle")
    h("Known limits")
    p("Only one seed and one fixed validation cohort have been evaluated. The same validation split selected epochs and the model, so its score is not an unbiased estimate of future test performance. No confidence intervals, calibration procedure or multi-seed comparison has yet been completed. The test labels are hidden; test AUROC/mAP must come from the evaluator.")
    truncated = audit["splits"]["train"]["truncated_histories"]
    p(f"Sequence length truncates {truncated}/2,791 training histories (35.5%); remote risk factors may be lost. Same-day order is an encoding convention, not measured clinical order, and may make next-event prediction artificially easy. Original units are kept separate rather than medically harmonized. No imaging/genomic enrichment is used. Two train/validation target first diagnoses occur exactly on the anchor day and follow the documented boundary rule.")
    p("Seeds, environment versions, configuration, source-data hashes and prepared-input hashes are saved. CUDA's memory-efficient attention backward kernel emitted a nondeterminism warning under warn-only deterministic mode; bitwise-identical retraining is not promised. Reloading the saved checkpoint is separately checked against saved predictions. The model is for this synthetic benchmark; clinical effectiveness has not been tested.")
    h("Reproduce locally")
    for command in [
        "python -m pytest -q",
        "python -m ehr_forecast prepare --data-dir research_intern_coding_test",
        "python -m ehr_forecast train --config configs/bag.json",
        "python -m ehr_forecast train --config configs/transformer.json",
        "python -m ehr_forecast predict --checkpoint<br/>  artifacts/runs/transformer-seed42/best.pt --output predictions.csv",
        "python -m ehr_forecast validate-submission --predictions predictions.csv",
    ]:
        p(command, "ReportCode")
    p("Use .venv/Scripts/python.exe on this machine. Existing run outputs are protected; choose a fresh output_dir to retrain. Tests cover calendar arithmetic, first-diagnosis labels, recurrence, future/validation contamination, empty histories, truncation, causal attention, padding, split isolation, checkpoint reloading and malformed submissions.", "ReportSmall")
    h("AI coding workflow and final submission status")
    p("AI assistance was used to interpret the task, divide implementation into preprocessing/model/metric modules, implement tests, and independently review the integrated pipeline. Review found the BOS timestamp initially copied the first event's timestamp; this was fixed, caches rebuilt and formal runs trained afterward. Human review of the scientific choices and final conclusions remains necessary.")
    p("wandb recorded real experiments in offline mode; cloud synchronization and a public run URL are still required. Public GitHub code and Hugging Face model publication have not been performed. Candidate identity, public links and externally returned test scores must be added before the final named PDF is submitted.")
    p("Data source: Synthea / Coherent Synthetic Data Set. CC BY 4.0, Copyright 2022 MITRE, Public Release Case 21-1917, as provided in the original ReadMe. This report does not redistribute the raw data.", "ReportSmall")

    def footer(canvas, document):
        canvas.setStrokeColor(colors.HexColor("#CED7E0"))
        canvas.line(43, 37, 552, 37)
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#5B6874"))
        canvas.drawString(43, 24, "Patient timeline forecasting | Initial local report | 2026-10-28")
        canvas.drawRightString(552, 24, str(document.page))

    document = SimpleDocTemplate(str(out), pagesize=(595.28, 841.89), rightMargin=48, leftMargin=48,
                                 topMargin=43, bottomMargin=52, title="Forecasting incident conditions - Initial report",
                                 author="", subject="First local PyTorch implementation and measured validation results")
    document.build(content, onFirstPage=footer, onLaterPages=footer)
    print(out)


if __name__ == "__main__":
    main()
