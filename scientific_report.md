# Scientific Evaluation of Speech-to-Text Performance for Medical SOAP Note Generation

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [Methodology](#methodology)
3. [Overall Model Performance Rankings](#overall-model-performance-rankings)
4. [Detailed Section Performance Analysis](#detailed-section-performance-analysis)
5. [Noise Impact Analysis](#noise-impact-analysis)
6. [Language Analysis](#language-analysis)
7. [Medical Term Extraction Performance](#medical-term-extraction-performance)
8. [Specialty Analysis](#specialty-analysis)
9. [Clinical Implications](#clinical-implications)
10. [Recommendations](#recommendations)
11. [Visual Analysis](#visual-analysis)
12. [Conclusion](#conclusion)
13. [Image Gallery](#image-gallery)
    - [Performance Comparisons](#performance-comparisons)
    - [Section Analysis](#section-analysis)
    - [Model and Language Evaluations](#model-and-language-evaluations)
    - [Statistical Distributions and Correlations](#statistical-distributions-and-correlations)

## Executive Summary

This report presents the results of a comprehensive scientific evaluation of various speech-to-text models for generating medical SOAP notes. The evaluation covers two language models (Azure and Nova) across two languages (English and French) under different noise conditions (No noise, Semi-noise, Noisy). The analysis uses multiple established metrics including Word Error Rate (WER), BLEU score, semantic similarity, medical term extraction accuracy (F1), and key phrase matching.

**Key Findings:**
- English transcription/generation (0.44) substantially outperforms French (0.20) across all models and noise conditions
- Azure consistently demonstrates superior performance for English content (0.47 average score)
- All models perform poorly on Plan and Assessment sections compared to Subjective and Objective sections
- Noise impacts all models proportionally, with gradual degradation (3-4% per noise level)
- Medical term extraction accuracy varies significantly by language, with English models achieving ~68% F1 compared to ~33% for French

## Methodology

The evaluation was conducted across 6 different datasets:
1. Without-noise-Azure for English-Nova-2- for French
2. Without-noise-Azure for French-Nova-3-medical- for English
3. Semi-noise - Azure for English-Nova-2 for French
4. Semi-noise - Azure for French-Nova-3-medical- for English
5. Noisy-Azure for English-Nova-2- for French
6. Noisy-Azure for French-Nova-3-medical- for English

Each dataset was evaluated using:
- **Semantic Similarity**: TF-IDF vectorization with cosine similarity
- **Word Error Rate (WER)**: Levenshtein distance at word level
- **BLEU Score**: N-gram precision with smoothing
- **Medical Term Extraction**: F1 score for accuracy of medical terms
- **Key Phrase Matching**: Extraction and matching of clinically relevant phrases
- **SOAP Section Analysis**: Separate metrics for each SOAP note section

Total evaluation corpus: 480 medical conversations across cardiology and general practice specialties.

## Overall Model Performance Rankings

| Rank | Model | Language | Overall Score | WER (lower is better) | BLEU | Term F1 | Files |
|------|-------|----------|--------------|------------------------|------|---------|-------|
| 1 | Azure | en | 0.47 | 0.77 | 0.39 | 0.68 | 120 |
| 2 | Nova-3-medical | en | 0.44 | 0.81 | 0.37 | 0.67 | 80 |
| 3 | Nova 3 medical | en | 0.35 | 0.96 | 0.30 | 0.59 | 40 |
| 4 | Nova 2 | fr | 0.22 | 0.98 | 0.21 | 0.34 | 120 |
| 5 | Azure | fr | 0.18 | 1.09 | 0.19 | 0.32 | 120 |

## Detailed Section Performance Analysis

| Model | Language | Subjective | Objective | Assessment | Plan |
|-------|----------|------------|-----------|------------|------|
| Azure | en | 0.53 | 0.56 | 0.44 | 0.39 |
| Nova-3-medical | en | 0.52 | 0.49 | 0.41 | 0.35 |
| Nova 3 medical | en | 0.47 | 0.34 | 0.33 | 0.31 |
| Nova 2 | fr | 0.28 | 0.27 | 0.22 | 0.22 |
| Azure | fr | 0.32 | 0.19 | 0.15 | 0.15 |

### SOAP Section Analysis Across All Models

| Section | Semantic Similarity | WER (lower is better) | BLEU Score | Phrase Match |
|---------|---------------------|------------------------|------------|--------------|
| Subjective | 0.41 | 0.80 | 0.35 | 0.31 |
| Objective | 0.37 | 0.87 | 0.29 | 0.27 |
| Assessment | 0.30 | 1.14 | 0.23 | 0.26 |
| Plan | 0.27 | 0.90 | 0.26 | 0.19 |

## Noise Impact Analysis

The evaluation assessed performance across three noise conditions:

| Noise Level | Overall Score | Degradation |
|-------------|---------------|-------------|
| No Noise | 0.34 | Baseline |
| Semi-Noise | 0.32 | -5.9% |
| Full Noise | 0.30 | -11.4% |

Noise degradation is consistent across models and languages, with Azure showing slightly better resilience to noise.

## Language Analysis

The performance gap between English and French is substantial and consistent across all metrics:

| Language | Overall Score | WER | BLEU | Term F1 |
|----------|---------------|-----|------|---------|
| English | 0.44 | 0.84 | 0.35 | 0.65 |
| French | 0.20 | 1.04 | 0.20 | 0.33 |

This represents a 120% better performance for English compared to French.

## Medical Term Extraction Performance

Medical term extraction is critical for healthcare applications. The analysis shows significant variation in performance:

### Medical Term Extraction (F1) Rankings

| Rank | Model | Language | Term F1 Score |
|------|-------|----------|---------------|
| 1 | Azure | en | 0.68 |
| 2 | Nova-3-medical | en | 0.67 |
| 3 | Nova 3 medical | en | 0.59 |
| 4 | Nova 2 | fr | 0.34 |
| 5 | Azure | fr | 0.32 |

## Specialty Analysis

Performance across specialties shows minimal variation:

| Specialty | Overall Score |
|-----------|---------------|
| General Practice | 0.32 |
| Cardiology | 0.32 |

## Clinical Implications

The evaluation findings have important clinical implications:

1. **Clinical Documentation Accuracy**: With overall scores between 0.18-0.47, none of the models achieve the reliability typically needed for clinical documentation without human verification.

2. **Treatment Plan Risk**: The Plan section consistently scores lowest (0.15-0.39), indicating potential clinical risk if using automated generation for treatment plans without thorough review.

3. **Language Disparity**: The substantial performance gap between languages raises concerns about health equity if these systems were deployed in multilingual healthcare settings.

4. **Medical Term Accuracy**: Term extraction F1 scores suggest acceptable performance for English (65-68%) but potentially dangerous inaccuracy for French (32-34%).

5. **Noise Sensitivity**: Performance degradation in noisy environments may limit practical applications in busy clinical settings.

## Recommendations

Based on the scientific evaluation, we recommend:

1. **Model Selection**: Azure is the preferred model for English content, achieving the best performance across all metrics.

2. **Application Limitations**: Current models are best suited for first-draft documentation requiring thorough clinical review, not autonomous clinical documentation.

3. **French Language Development**: Significant improvement is needed for French language models before considering clinical deployment.

4. **Focused Improvements**: Special attention should be given to improving Assessment and Plan sections, which showed poorest performance but have highest clinical importance.

5. **Noise Mitigation**: Implement environmental controls or noise-reduction preprocessing in clinical settings to maintain higher performance.

## Visual Analysis

This report includes the following visualizations to support the findings:

- Section-by-section performance metrics
- Model comparison across languages
- Noise level impact analysis
- Word Error Rate distributions
- Medical term extraction accuracy
- Correlation heatmap of evaluation metrics

## Conclusion

This comprehensive scientific evaluation reveals that while speech-to-text models show promise for medical SOAP note generation, particularly for English content using Azure, significant improvements are needed before they can be reliably deployed in clinical settings without thorough human verification. The performance gap between languages and the poor accuracy in critical sections like Assessment and Plan present both technical challenges and potential clinical risks.

Further research should focus on:
1. Improving French language models specifically for medical context
2. Enhancing performance on Assessment and Plan sections
3. Developing more noise-resistant models for practical clinical environments
4. Increasing medical term extraction accuracy, especially for non-English languages

# Image Gallery

## Performance Comparisons

### Combined model scores

![combined model scores](custom_viz/combined_model_scores.png)

### Combined dataset scores

![combined dataset scores](custom_viz/combined_dataset_scores.png)

### Combined language scores

![combined language scores](custom_viz/combined_language_scores.png)

### Combined noise type scores

![combined noise type scores](custom_viz/combined_noise_type_scores.png)

### Combined specialty scores

![combined specialty scores](custom_viz/combined_specialty_scores.png)

## Section Analysis

### Combined section scores

![combined section scores](custom_viz/combined_section_scores.png)

### Multiple metrics by section

![multiple metrics by section](custom_viz/multiple_metrics_by_section.png)

## Model and Language Evaluations

### WER by model 

![wer by model](custom_viz/wer_by_model.png)

### WER by language

![wer by language](custom_viz/wer_by_language.png)

### Medical term accuracy

![medical term accuracy](custom_viz/medical_term_accuracy.png)

### Language noise heatmap

![language noise heatmap](custom_viz/language_noise_heatmap.png)

## Statistical Distributions and Correlations

### Combined score distribution

![combined score distribution](custom_viz/combined_score_distribution.png)

### Additional metrics boxplot

![additional metrics boxplot](custom_viz/additional_metrics_boxplot.png)

### Metrics correlation heatmap

![metrics correlation heatmap](custom_viz/metrics_correlation_heatmap.png)
