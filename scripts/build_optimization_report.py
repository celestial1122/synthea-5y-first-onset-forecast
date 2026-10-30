"""Render the completed optimization study from measured, saved results."""

from pathlib import Path
import json
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def main():
    a = load("artifacts/optimization/analysis.json")
    d = load("artifacts/optimization/history_diagnostics.json")
    selected = a["selected_recipe"]
    metrics = selected["metrics"]
    original = a["families"]["original"]["equal_weight_ensemble"]["metrics"]
    uncertainty = a["paired_comparisons"]["selected__vs__original"]
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="TitleOpt", fontName="Helvetica-Bold", fontSize=22, leading=26, spaceAfter=12, textColor=colors.HexColor("#163C64")))
    styles.add(ParagraphStyle(name="BodyOpt", fontName="Helvetica", fontSize=9.7, leading=13.4, spaceAfter=8))
    styles.add(ParagraphStyle(name="SmallOpt", fontName="Helvetica", fontSize=8.1, leading=10.5, spaceAfter=6))
    styles.add(ParagraphStyle(name="HeadOpt", fontName="Helvetica-Bold", fontSize=12.5, leading=16, spaceBefore=10, spaceAfter=6, textColor=colors.HexColor("#163C64")))
    styles.add(ParagraphStyle(name="CodeOpt", fontName="Courier", fontSize=7.4, leading=10.5, spaceAfter=7))
    story = []

    def p(value, style="BodyOpt"):
        story.append(Paragraph(value, styles[style]))

    def h(value):
        p(value, "HeadOpt")

    def table(rows, widths, font=8.4, compact=False):
        t = Table(rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E9EFF5")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), font), ("LEADING", (0, 0), (-1, -1), font + 2),
            ("TOPPADDING", (0, 0), (-1, -1), 2 if compact else 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2 if compact else 5),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7F9FB")]),
            ("LINEBELOW", (0, 0), (-1, 0), .5, colors.HexColor("#B6C3D0")),
        ]))
        story.append(t)
        story.append(Spacer(1, 7))

    p("Long histories and seed stability", "TitleOpt")
    p("OPTIMIZATION STUDY | 28 OCTOBER 2026 | SYNTHETIC EHR BENCHMARK", "SmallOpt")
    p("Ten experiment families were evaluated with seeds 17, 42 and 73. The original seed-42 fit is reused; 29 additional fits were run. All experiments retain the official 2,791/365/358 training/validation/test patient split. Test outcomes remain hidden.")
    h("Current local prediction model")
    p(f"Six equally weighted members combine three supervised-only 512-token Transformers with three clipped full-history summary MLPs. Measured validation mAP is <b>{metrics['mAP']:.4f}</b> and macro AUROC is <b>{metrics['macro_auroc']:.4f}</b>. Brier score is {metrics['brier']:.4f}; all 40 targets have positive and negative validation examples.")
    p(f"The first single-model mAP was 0.2195. A fairer seed-ensemble reference is the original three-seed ensemble at {original['mAP']:.4f}; the new predictor improves this validation score by {metrics['mAP'] - original['mAP']:.4f}. The bootstrap interval crosses zero, so stable improvement on new patients has not been demonstrated.")
    h("All families, including unsuccessful changes")
    order = ["original", "length-l256", "length-l512", "length-l1024", "no-pretrain-l512", "pooling-l512", "weighted-l512", "summary-linear", "summary-mlp", "summary-clip5"]
    rows = [["Family", "Single-fit mAP mean +/- SD", "3-seed mAP", "3-seed AUROC"]]
    for key in order:
        item = a["families"][key]
        stats = item["across_seeds"]["mAP"]
        ensemble = item["equal_weight_ensemble"]["metrics"]
        rows.append([key, f"{stats['mean']:.4f} +/- {stats['sample_std']:.4f}", f"{ensemble['mAP']:.4f}", f"{ensemble['macro_auroc']:.4f}"])
    table(rows, [145, 150, 85, 95])
    p("Source: saved validation predictions aligned by official patient IDs and target CODE. SD is sample standard deviation across three fits, not a patient confidence interval. Each ensemble includes every declared seed with equal probability weight.", "SmallOpt")
    h("Selection is adaptive")
    p("The initial screening rule used mean single-fit mAP. After seeing that the deployed predictor is an ensemble, selection was revised to compare every complete three-seed ensemble and one fixed 50:50 blend of the best sequence and summary ensembles. This uses validation results adaptively and adds selection bias. No individual best seed or test outcome was selected.")

    story.append(PageBreak())
    p("What truncation actually removed", "TitleOpt")
    train = d["splits"]["train"]
    val = d["splits"]["val"]
    rows = [["Sequence length", "Train patients truncated", "Validation truncated", "Train events retained"]]
    for length in (256, 512, 1024):
        t, v = train["lengths"][str(length)], val["lengths"][str(length)]
        retention = t.get("event_retention_fraction", t.get("retained_event_fraction"))
        # The diagnostics include exact retained and dropped counts under stable keys.
        if retention is None:
            retention = t["retained_events"] / (t["retained_events"] + t["dropped_events"])
        rows.append([str(length), f"{100*t['truncated_fraction']:.2f}%", f"{100*v['truncated_fraction']:.2f}%", f"{100*retention:.1f}%"])
    table(rows, [100, 135, 125, 115])
    p("A sequence includes BOS and up to L-1 recent events. Training event count median=188, 95th percentile=1,031 and maximum=8,026. Among patients truncated at 256, the lost lookback span has a median of about 42 years, with 6 entirely missing diagnosis codes in training and 5 in validation. This is the span between oldest full and oldest retained observations; it does not mean 42 years of continuously recorded care.")
    story.append(Image(str(ROOT / "artifacts/optimization/plots/length_ablation.png"), width=475, height=139))
    p("Source: history_diagnostics.json and matched 3-seed runs. Left: single-fit means and sample SD. Right: equal-seed ensemble scores within fixed short/long cohorts, whose prevalences differ; compare models within each cohort.", "SmallOpt")
    h("Controlled result")
    p("All three length variants share the same 1,276-token train-only encoder, numeric bins, static scaling, labels and patient order. The model position capacity is fixed at 1,024 for identical initial parameters under each seed; batch size is 16 with the same training schedule. Dropout draws naturally differ with input shape, so training paths are not bitwise paired.")
    p("Mean single-fit mAP was 0.2278, 0.2281 and 0.2252 at 256, 512 and 1,024. The 512-token ensemble improves the matched 256-token ensemble by only 0.0010 (conditional 95% interval -0.0154 to 0.0324). Longer input alone has not produced a stable gain.")
    p("For the final selected model versus the original three-seed ensemble, the long-history cohort has delta mAP -0.0209 (95% interval -0.0630 to 0.0428). The current optimization must not be described as solving the long-history subgroup problem.")

    story.append(PageBreak())
    p("Representation and objective tests", "TitleOpt")
    story.append(Image(str(ROOT / "artifacts/optimization/plots/family_comparison.png"), width=475, height=178))
    p("Source: ten complete seed families on the fixed 365-patient validation split. Error bars show between-seed sample SD; crosses show full-seed ensembles. Every checkpoint was selected on this same validation cohort.", "SmallOpt")
    h("Full-history summaries")
    p("The summary input has 3,821 dimensions: five demographic features plus three channels for each of 1,272 clinical tokens/bins. The channels are log1p lifetime count, log1p count in the preceding two calendar years, and exp(-days since latest occurrence / 3650). All history is strictly pre-anchor; vocabulary, numerical bins and feature scaling are fitted on training patients only.")
    p("Rare summary columns amplify noise when divided by tiny standard deviations: the original standardized maximum absolute value is 52.82 in training and 305.53 in validation. The fixed clip5 variant clips only the 3,816 summary dimensions to [-5,5], retaining the original five demographic features. This addresses a concrete numerical weakness, but its three individual mAPs still span 0.2127 to 0.2388; stability is limited.")
    h("Changes that were not adopted")
    p("Pooled mean-plus-last history representations with nonlinear static fusion and square-root positive weighting did not improve their three-seed means. The positive weights use training at-risk negatives/positives and are capped at 8. Inference subtracts log(weight) from weighted logits to undo the ideal weighted-BCE posterior shift; this does not itself change per-class AP/AUROC or guarantee finite-model calibration. The weighted branch is not in the final predictor.")
    p("Removing next-event pretraining did not improve the mean individual fit, but the resulting three-seed predictions were complementary and their ensemble scored 0.2578. Its fixed 50:50 blend with the clip5 summary ensemble scored 0.2677. No custom mixture weights were fitted.")
    h("Why higher capacity was not the first fix")
    p("The initial model already overfit: training BCE decreased while validation mAP declined. Dense observations occupy about 62% of retained tokens, while diagnosis tokens account for about 2.6%. More raw tokens and larger attention blocks do not automatically provide a better risk representation. Sequence compression or hierarchical aggregation remains a future hypothesis, rather than an established benefit here.")

    story.append(PageBreak())
    p("Per-condition validation", "TitleOpt")
    p("All 365 validation patients. Old AP is the original three-seed ensemble; new AP is the selected six-member predictor. Descriptions are shortened. The comparison is selected on validation and is descriptive, not an independent clinical finding.", "SmallOpt")
    old_by_code = {row["code"]: row for row in original["per_condition"]}
    rows = [["CODE", "Condition (shortened)", "+", "Old AP", "New AP", "Delta"]]
    for row in metrics["per_condition"]:
        name = row["description"].replace(" (disorder)", "").replace(" (finding)", "").replace(" (situation)", "")
        if len(name) > 41:
            name = name[:38] + "..."
        old = old_by_code[row["code"]]["average_precision"]
        rows.append([row["code"], name, str(row["n_positive"]), f"{old:.3f}", f"{row['average_precision']:.3f}", f"{row['average_precision']-old:+.3f}"])
    table(rows, [64, 245, 22, 48, 48, 48], font=7.7, compact=True)
    p("About 69% of the net macro-AP gain comes from three related prostate targets. Their positive sets contain 11, 8 and 6 patients and jointly cover only 11 unique validation patients. These correlated target improvements are not three independent replications, which further limits the strength of the overall gain claim.", "SmallOpt")
    p("Fourteen conditions have fewer than ten positive validation patients. A small number of patients can materially alter average precision and AUROC. Primary mAP always averages all 40 classes; a class without positives in a bootstrap resample contributes AP=0. AUROC excludes undefined classes and reports their number.", "SmallOpt")

    story.append(PageBreak())
    p("Uncertainty, Swin and reproduction", "TitleOpt")
    h("Uncertainty is not a hidden-test score")
    delta = uncertainty["overall"]["delta_mAP"]
    p(f"The final versus original three-seed ensemble delta mAP is {delta['estimate']:+.4f}. A 500-resample paired patient bootstrap gives [{delta['ci_lower']:+.4f}, {delta['ci_upper']:+.4f}]. The interval crosses zero. It describes conditional sampling variability for these fitted, selected models on this cohort; it does not undo epoch/architecture/ensemble selection or establish test-set or clinical generalization.")
    p("Seed variability measures fitting randomness, while patient bootstrap variability measures the validation sample. They answer different questions. Three seeds are a useful initial check, but do not establish a precise estimate of training variability. GPU attention training can also be nondeterministic under the recorded warn-only configuration.")
    h("Swin: useful design idea, different immediate task")
    p("Swin uses hierarchical shifted-window attention for vision; Swin UNETR applies a 3D Swin encoder to MRI tumor segmentation. The present task predicts first diagnoses from event histories. Local files cover only 196 training, 2 validation and 2 test patients with DICOMs; acquisition dates would also need verification. This offers little validation evidence for an image branch, and external pretrained weights are prohibited by the task.")
    p("The transferable idea is local-to-global aggregation of long inputs. Hi-BEHRT applies hierarchical Transformer modeling directly to longitudinal EHR. If preserving long history later yields consistent benefit, encounter/window encoders with a global patient aggregator are a better-matched next experiment than directly substituting an image backbone.")
    p('Primary sources: <link href="https://arxiv.org/abs/2103.14030" color="#163C64">Swin Transformer (2021)</link>; <link href="https://arxiv.org/abs/2201.01266" color="#163C64">Swin UNETR (2022)</link>; <link href="https://pmc.ncbi.nlm.nih.gov/articles/PMC7615082/" color="#163C64">Hi-BEHRT (2023)</link>. These motivate architectures; their reported gains are not transferred to this dataset.', "SmallOpt")
    h("Reproduce the selected artifact")
    for text in [
        "python -m ehr_forecast prepare-ensemble<br/>  --manifest artifacts/optimized_model_bundle/ensemble.json<br/>  --data-dir research_intern_coding_test<br/>  --output-root artifacts/optimization/reproduced_inputs",
        "python -m ehr_forecast predict-ensemble<br/>  --manifest artifacts/optimized_model_bundle/ensemble.json<br/>  --prepared-root artifacts/optimization/reproduced_inputs<br/>  --output predictions_reproduced.csv",
        "python -m ehr_forecast validate-submission<br/>  --predictions predictions_reproduced.csv",
    ]:
        p(text, "CodeOpt")
    p("Use the project Python 3.12 environment. The bundle contains all six weights, separate frozen encoders, source code, training configurations, dependency versions and SHA256 manifest; it contains no patient arrays. Prediction checks both encoder and metadata hashes, including the intended history length. The previous predictions are backed up before the optimized predictions replace the root submission file.", "SmallOpt")
    p("AI assistance supported hypothesis diagnosis, modular implementation, independent leakage/method review, automated tests and result interpretation. Current experiments are tracked in offline wandb; no online run URL, GitHub publication or Hugging Face upload is claimed. This report is an optimization draft; candidate identity and final publication/test-result fields remain to be added.", "SmallOpt")
    p("Data attribution: Coherent Synthetic Data Set / Synthea, CC BY 4.0, Copyright 2022 MITRE, Public Release Case 21-1917, as provided with the task.", "SmallOpt")

    def footer(canvas, document):
        canvas.setStrokeColor(colors.HexColor("#CED7E0"))
        canvas.line(43, 37, 552, 37)
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#5B6874"))
        canvas.drawString(43, 24, "Patient timeline forecasting | Optimization study | 2026-10-28")
        canvas.drawRightString(552, 24, str(document.page))

    destination = ROOT / "output/pdf/optimization_results.pdf"
    destination.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(str(destination), pagesize=(595.28, 841.89), leftMargin=48, rightMargin=48,
                                 topMargin=42, bottomMargin=51, title="Long histories and seed stability - Optimization study",
                                 author="", subject="Measured synthetic EHR optimization experiments")
    document.build(story, onFirstPage=footer, onLaterPages=footer)
    print(destination)


if __name__ == "__main__":
    main()
