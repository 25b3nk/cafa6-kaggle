import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict

# Define data paths
DATA_PATH = Path('cafa-6-protein-function-prediction')
TRAIN_PATH = DATA_PATH / 'Train'

print("=== Advanced CAFA 6 Data Analysis ===\n")

# Load train terms
train_terms = pd.read_csv(TRAIN_PATH / 'train_terms.tsv', sep='\t')

# ====== ASPECT-SPECIFIC ANALYSIS ======
print("="*60)
print("ASPECT-SPECIFIC DETAILED ANALYSIS")
print("="*60 + "\n")

for aspect in sorted(train_terms['aspect'].unique()):
    print(f"\n{'='*60}")
    print(f"ASPECT: {aspect}")
    print(f"{'='*60}\n")
    
    aspect_df = train_terms[train_terms['aspect'] == aspect]
    
    # Basic stats
    unique_proteins = aspect_df['EntryID'].nunique()
    unique_terms = aspect_df['term'].nunique()
    total_annotations = len(aspect_df)
    
    print(f"Total annotations: {total_annotations:,}")
    print(f"Unique proteins: {unique_proteins:,}")
    print(f"Unique GO terms: {unique_terms:,}")
    print(f"Avg annotations per protein: {total_annotations / unique_proteins:.2f}")
    print(f"Avg proteins per term: {total_annotations / unique_terms:.2f}")
    
    # Distribution of terms per protein
    terms_per_protein = aspect_df.groupby('EntryID').size()
    print(f"\nTerms per protein:")
    print(f"  Min: {terms_per_protein.min()}")
    print(f"  Max: {terms_per_protein.max()}")
    print(f"  Mean: {terms_per_protein.mean():.2f}")
    print(f"  Median: {terms_per_protein.median():.2f}")
    print(f"  Std: {terms_per_protein.std():.2f}")
    
    # Distribution of proteins per term
    proteins_per_term = aspect_df.groupby('term').size()
    print(f"\nProteins per term:")
    print(f"  Min: {proteins_per_term.min()}")
    print(f"  Max: {proteins_per_term.max()}")
    print(f"  Mean: {proteins_per_term.mean():.2f}")
    print(f"  Median: {proteins_per_term.median():.2f}")
    print(f"  Std: {proteins_per_term.std():.2f}")
    
    # Top 10 most common terms
    print(f"\nTop 10 most common GO terms in {aspect}:")
    top_terms = proteins_per_term.sort_values(ascending=False).head(10)
    for i, (term, count) in enumerate(top_terms.items(), 1):
        print(f"  {i}. {term}: {count} proteins ({count/unique_proteins*100:.1f}%)")
    
    # Rare terms analysis
    rare_threshold = 10
    rare_terms = proteins_per_term[proteins_per_term < rare_threshold]
    print(f"\nRare terms (< {rare_threshold} proteins): {len(rare_terms)} ({len(rare_terms)/unique_terms*100:.1f}%)")

# ====== MULTI-LABEL ANALYSIS ======
print("\n" + "="*60)
print("MULTI-LABEL CHARACTERISTICS")
print("="*60 + "\n")

# Proteins annotated in multiple aspects
protein_aspects = train_terms.groupby('EntryID')['aspect'].apply(set)
aspect_combinations = protein_aspects.apply(lambda x: tuple(sorted(x)))
aspect_combo_counts = aspect_combinations.value_counts()

print("Proteins by aspect combination:")
for combo, count in aspect_combo_counts.items():
    print(f"  {combo}: {count:,} proteins ({count/len(protein_aspects)*100:.1f}%)")

# ====== CLASS IMBALANCE ANALYSIS ======
print("\n" + "="*60)
print("CLASS IMBALANCE ANALYSIS")
print("="*60 + "\n")

term_frequencies = train_terms['term'].value_counts()

print("Term frequency distribution:")
bins = [1, 10, 50, 100, 500, 1000, 5000, 10000, float('inf')]
labels = ['1-9', '10-49', '50-99', '100-499', '500-999', '1K-5K', '5K-10K', '10K+']

term_freq_binned = pd.cut(term_frequencies, bins=bins, labels=labels)
print(term_freq_binned.value_counts().sort_index())

# Plot class imbalance
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Overall term frequency distribution
axes[0, 0].hist(term_frequencies, bins=100, color='steelblue', edgecolor='black', alpha=0.7)
axes[0, 0].set_xlabel('Number of Proteins per Term', fontsize=12)
axes[0, 0].set_ylabel('Number of Terms', fontsize=12)
axes[0, 0].set_title('Term Frequency Distribution', fontsize=14, fontweight='bold')
axes[0, 0].set_yscale('log')
axes[0, 0].grid(axis='y', alpha=0.3)

# Log-log plot
axes[0, 1].scatter(range(len(term_frequencies)), term_frequencies, alpha=0.5, s=10)
axes[0, 1].set_xlabel('Term Rank', fontsize=12)
axes[0, 1].set_ylabel('Frequency', fontsize=12)
axes[0, 1].set_title('Term Frequency (Ranked)', fontsize=14, fontweight='bold')
axes[0, 1].set_xscale('log')
axes[0, 1].set_yscale('log')
axes[0, 1].grid(alpha=0.3)

# Annotations per protein distribution
protein_annotation_counts = train_terms.groupby('EntryID').size()
axes[1, 0].hist(protein_annotation_counts, bins=50, color='coral', edgecolor='black', alpha=0.7)
axes[1, 0].set_xlabel('Annotations per Protein', fontsize=12)
axes[1, 0].set_ylabel('Number of Proteins', fontsize=12)
axes[1, 0].set_title('Annotations per Protein Distribution', fontsize=14, fontweight='bold')
axes[1, 0].grid(axis='y', alpha=0.3)

# Cumulative coverage
sorted_freqs = term_frequencies.sort_values(ascending=False)
cumulative = np.cumsum(sorted_freqs) / sorted_freqs.sum()
axes[1, 1].plot(range(len(cumulative)), cumulative, linewidth=2, color='darkgreen')
axes[1, 1].axhline(y=0.8, color='red', linestyle='--', label='80% coverage')
axes[1, 1].axhline(y=0.9, color='orange', linestyle='--', label='90% coverage')
axes[1, 1].set_xlabel('Number of Top Terms', fontsize=12)
axes[1, 1].set_ylabel('Cumulative Coverage', fontsize=12)
axes[1, 1].set_title('Cumulative Annotation Coverage', fontsize=14, fontweight='bold')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Find how many terms needed for 80% and 90% coverage
terms_for_80 = np.argmax(cumulative >= 0.8) + 1
terms_for_90 = np.argmax(cumulative >= 0.9) + 1
print(f"\nTerms needed for 80% coverage: {terms_for_80} ({terms_for_80/len(term_frequencies)*100:.1f}% of all terms)")
print(f"Terms needed for 90% coverage: {terms_for_90} ({terms_for_90/len(term_frequencies)*100:.1f}% of all terms)")

# ====== SEQUENCE LENGTH VS ANNOTATIONS ======
print("\n" + "="*60)
print("SEQUENCE LENGTH vs ANNOTATIONS ANALYSIS")
print("="*60 + "\n")

fasta_file = TRAIN_PATH / 'train_sequences.fasta'
if fasta_file.exists():
    # Load sequences
    sequences = {}
    current_id = None
    current_seq = []
    
    with open(fasta_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_id:
                    sequences[current_id] = ''.join(current_seq)
                current_id = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line)
        if current_id:
            sequences[current_id] = ''.join(current_seq)
    
    # Merge with annotations
    seq_lengths = pd.Series({k: len(v) for k, v in sequences.items()}, name='seq_length')
    protein_data = pd.DataFrame(seq_lengths)
    protein_data['num_annotations'] = protein_annotation_counts
    protein_data = protein_data.dropna()
    
    # Correlation
    correlation = protein_data['seq_length'].corr(protein_data['num_annotations'])
    print(f"Correlation between sequence length and number of annotations: {correlation:.3f}")
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.scatter(protein_data['seq_length'], protein_data['num_annotations'], 
               alpha=0.3, s=10, color='purple')
    ax.set_xlabel('Sequence Length (amino acids)', fontsize=12)
    ax.set_ylabel('Number of Annotations', fontsize=12)
    ax.set_title(f'Sequence Length vs Number of Annotations (corr={correlation:.3f})', 
                 fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Binned analysis
    protein_data['length_bin'] = pd.cut(protein_data['seq_length'], 
                                         bins=[0, 200, 400, 600, 800, 1000, 2000, 10000],
                                         labels=['0-200', '200-400', '400-600', '600-800', 
                                                '800-1000', '1000-2000', '2000+'])
    
    print("\nAverage annotations by sequence length bin:")
    print(protein_data.groupby('length_bin')['num_annotations'].agg(['mean', 'median', 'count']))

print("\n" + "="*60)
print("Advanced EDA Complete!")
print("="*60)