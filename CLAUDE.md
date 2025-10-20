# CAFA-6 Protein Function Prediction - Analysis & Model Design

This document contains detailed EDA findings and recommended modeling approaches for the CAFA-6 competition.

## Competition Overview

**Goal**: Predict protein functions using Gene Ontology (GO) terms based on amino acid sequences

**Competition Link**: https://www.kaggle.com/competitions/cafa-6-protein-function-prediction/overview

## Dataset Summary

### Basic Statistics
- **Total Proteins**: 82,404
- **Total GO Terms**: 26,125 unique labels
- **Total Annotations**: 537,027
- **Average Labels per Protein**: 6.52 (min: 1, max: 233)
- **Protein Sequence Length**: 3 to 35,213 amino acids (avg: 526 AA)

### Input & Output
- **INPUT**: Protein amino acid sequence (e.g., "MAQTVQNVTLSL...")
- **OUTPUT**: Multiple GO terms with confidence scores (multi-label classification)

### GO Term Aspects

Each GO term belongs to exactly ONE aspect (fixed, not predicted):
- **C** = Cellular Component Ontology (where protein is located)
- **F** = Molecular Function Ontology (what protein does molecularly)
- **P** = Biological Process Ontology (what biological processes protein participates in)

## Key EDA Findings

### 1. Aspect Distribution

| Aspect | Annotations | Percentage | Unique GO Terms | Unique Proteins | Avg Terms/Protein |
|--------|-------------|------------|-----------------|-----------------|-------------------|
| **P** (Process) | 250,805 | 46.7% | 16,858 | 59,958 | 4.18 |
| **C** (Component) | 157,770 | 29.4% | 2,651 | 60,292 | 2.62 |
| **F** (Function) | 128,452 | 23.9% | 6,616 | 58,001 | 2.21 |

**Key Insights**:
- Biological Process (P) is the most complex with 16,858 different labels
- Cellular Component (C) is simplest with only 2,651 labels
- Most proteins are labeled by processes they participate in

### 2. Multi-Label Characteristics

**Proteins by Aspect Combination**:
- All three (C+F+P): 44.6% - Most common pattern
- Only C: 12.1%
- F+P: 10.6%
- C+P: 8.9%
- Only P: 8.7%
- Only F: 7.6%
- C+F: 7.6%

**Insight**: Most proteins have annotations across all three aspects, suggesting models should predict them together.

### 3. Extreme Class Imbalance (CRITICAL!)

**Label Frequency Distribution**:
- **13,491 labels (51.6%)** appear in fewer than 10 proteins (ultra-rare)
- **5,715 labels** appear in 10-49 proteins
- **Only 4 labels** appear in 10,000+ proteins

**Top 5 Most Common Labels**:
1. GO:0005515 (protein binding, F): 33,713 proteins (40.9%)
2. GO:0005634 (nucleus, C): 13,283 proteins (16.1%)
3. GO:0005829 (cytosol, C): 13,040 proteins (15.8%)
4. GO:0005886 (plasma membrane, C): 10,150 proteins (12.3%)
5. GO:0005737 (cytoplasm, C): 9,442 proteins (11.5%)

**Coverage Analysis**:
- **4,395 labels (16.8%)** needed for 80% coverage
- **8,714 labels (33.4%)** needed for 90% coverage

**Insight**: This is an extremely imbalanced dataset. Special techniques required to handle rare labels.

### 4. Aspect-Specific Insights

#### Cellular Component (C) - EASIEST
- Rare terms: 61.9%
- Max labels per protein: 41
- Most proteins labeled with basic locations (nucleus, cytoplasm, membrane)

#### Molecular Function (F) - MEDIUM
- Rare terms: 76.9% (most imbalanced!)
- Max labels per protein: 34
- Dominated by GO:0005515 (protein binding) - 58.1% of all F annotations

#### Biological Process (P) - HARDEST
- Rare terms: 68.4%
- Max labels per protein: 188 (extreme!)
- Most diverse and complex label space

### 5. Sequence Characteristics

**Length Distribution**:
- Min: 3 AA
- Max: 35,213 AA
- Mean: 526 AA
- Median: 409 AA
- 25th percentile: 250 AA
- 75th percentile: 630 AA
- 99th percentile: 2,375 AA

**Amino Acid Composition**:
- 23 unique amino acids found
- Most common: L (9.65%), S (8.30%), A (7.03%), E (6.78%), G (6.40%)
- Standard distribution, no unusual patterns

## Modeling Challenges

1. **Multi-label classification**: Each protein needs multiple predictions
2. **Large label space**: 26,125 possible labels
3. **Extreme class imbalance**: 51.6% of labels are ultra-rare
4. **Variable sequence lengths**: 3 to 35,213 amino acids
5. **Different complexity per aspect**: P >> F > C in terms of difficulty

## Recommended Model Approach

### 1. Protein Language Models (PLMs) - PRIMARY RECOMMENDATION

**Best Options**:
- **ESM-2** (Meta) - State-of-the-art protein understanding
- **ProtBERT** - BERT adapted for proteins
- **ProtT5** - T5 model for proteins
- **Ankh** - Recent protein language model

**Why PLMs**:
- Pre-trained on millions of protein sequences
- Understand protein "language" and evolutionary patterns
- Handle variable sequence lengths naturally
- Transfer learning from massive datasets

### 2. Recommended Architecture

```
Input: Protein Sequence (variable length)
    ↓
ESM-2 Pre-trained Encoder (frozen or fine-tuned)
    ↓
[CLS] token embedding (fixed-size representation)
    ↓
Dropout(0.3)
    ↓
Dense(2048, ReLU)
    ↓
Dropout(0.3)
    ↓
Dense(1024, ReLU)
    ↓
Three parallel heads (Multi-Task Learning):
    - C_head: Dense(2651, sigmoid) for Cellular Component
    - F_head: Dense(6616, sigmoid) for Molecular Function
    - P_head: Dense(16858, sigmoid) for Biological Process
    ↓
Multi-label predictions with focal loss
```

### 3. Alternative Architectures

**CNN-based**:
- Good for finding local motifs/patterns
- Fast and efficient
- 1D convolutions on amino acid sequences
- Example: DeepGOPlus architecture

**Hybrid CNN + RNN**:
- CNNs for local patterns
- BiLSTM/GRU for long-range dependencies
- Best of both worlds

**Ensemble Approach** (Highly Recommended):
- Model 1: ESM-2 fine-tuned
- Model 2: CNN for motif detection
- Model 3: Separate models per aspect
- Weighted ensemble of predictions

### 4. Critical Techniques for Class Imbalance

**Loss Functions**:
1. **Focal Loss** - Focus on hard-to-learn rare labels
2. **Asymmetric Loss** - Different weights for positive/negative samples
3. **Class Weighted BCE** - Weight rare labels higher

**Training Strategies**:
1. **Label-wise Attention** - Pay more attention to rare labels
2. **Hierarchical Classification** - Use GO term parent-child relationships
3. **Few-Shot Learning** - Specialized techniques for rare labels
4. **Data Augmentation** - Augment sequences with rare labels

**Post-Processing**:
1. **Threshold Optimization** - Different thresholds per label
2. **Top-K Selection** - Select top K predictions
3. **GO Hierarchy Constraints** - Ensure parent-child consistency

### 5. Multi-Task Learning Strategy

Train separate heads for C, F, P aspects because:
- Different label space sizes (2,651 vs 6,616 vs 16,858)
- Different imbalance characteristics
- Different complexity levels
- Shared encoder learns general protein features

### 6. Baseline Model (Quick Start)

For rapid prototyping:
1. Use **pre-computed ESM-2 embeddings** (don't train encoder)
2. Simple **Dense NN** (2-3 layers) on top
3. **Binary Cross-Entropy** loss with **class weights**
4. Predict top-K most common labels first
5. Iterate and add complexity

### 7. Advanced Techniques (For Top Performance)

1. **Pseudo-labeling** on test set (semi-supervised learning)
2. **GO hierarchy** enforcement (parent-child relationships)
3. **Ensemble** 5-10 models with different architectures
4. **Separate models** for common vs rare labels
5. **Meta-learning** for few-shot labels
6. **Knowledge distillation** from larger models

## Implementation Roadmap

### Phase 1: Baseline (Week 1)
- [ ] Load and preprocess data
- [ ] Generate ESM-2 embeddings (or use pre-computed)
- [ ] Build simple Dense NN classifier
- [ ] Train with BCE loss + class weights
- [ ] Evaluate on validation set
- [ ] Establish baseline score

### Phase 2: Enhanced Model (Week 2)
- [ ] Implement multi-task architecture (C/F/P heads)
- [ ] Add focal loss for imbalance
- [ ] Implement threshold optimization
- [ ] Add dropout and regularization
- [ ] Cross-validation setup

### Phase 3: Advanced Features (Week 3)
- [ ] Fine-tune ESM-2 encoder
- [ ] Implement GO hierarchy constraints
- [ ] Add data augmentation
- [ ] Build ensemble of models
- [ ] Hyperparameter tuning

### Phase 4: Optimization (Week 4)
- [ ] Pseudo-labeling on test set
- [ ] Advanced threshold tuning per label
- [ ] Ensemble weight optimization
- [ ] Final submission preparation

## Evaluation Metrics

The competition likely uses:
- **F1-score** (macro or weighted)
- **Precision-Recall** curves
- **Coverage** (how many labels predicted correctly)

Focus on balancing precision and recall, especially for rare labels.

## Key References

- ESM-2: https://github.com/facebookresearch/esm
- DeepGOPlus: https://github.com/bio-ontology-research-group/deepgoplus
- GO Database: http://geneontology.org/

## Notes

- GO term aspects are FIXED - each GO term always belongs to the same aspect
- The "protein binding" problem: GO:0005515 appears in 40% of proteins - avoid over-predicting
- Most proteins have all three aspects - model should predict them jointly
- Rare labels are the key challenge - special handling required

---

*This analysis was generated through exploratory data analysis and domain knowledge about protein function prediction.*
