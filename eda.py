import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
from collections import Counter
warnings.filterwarnings('ignore')

# Set up plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Define data paths
DATA_PATH = Path('cafa-6-protein-function-prediction')
TRAIN_PATH = DATA_PATH / 'Train'

print("=== CAFA 6 Data Exploration ===\n")

# List all available files
print("Available files in competition:")
for file in sorted(DATA_PATH.rglob('*')):
    if file.is_file():
        print(f"  - {file.relative_to(DATA_PATH)}")
    
print("\n" + "="*50 + "\n")

# Load train terms
print("Loading training terms...")
train_terms = pd.read_csv(TRAIN_PATH / 'train_terms.tsv', sep='\t')

print(f"Training terms shape: {train_terms.shape}")
print(f"Columns: {train_terms.columns.tolist()}\n")

# Display first few rows
print("First 10 rows:")
print(train_terms.head(10))
print("\n")

# Basic statistics
print("Data Info:")
print(train_terms.info())
print("\n")

# Check for missing values
print("Missing values:")
print(train_terms.isnull().sum())
print("\n")

# Unique values
print("=== Unique Values ===")
print(f"Unique proteins (EntryID): {train_terms['EntryID'].nunique()}")
print(f"Unique GO terms: {train_terms['term'].nunique()}")
print(f"Total annotations: {len(train_terms)}")
print(f"Aspects: {train_terms['aspect'].unique()}")
print("\n")

# Analyze aspects distribution
print("=== Distribution by Aspect ===")
aspect_counts = train_terms['aspect'].value_counts()
print(aspect_counts)
print(f"\nPercentages:")
print((aspect_counts / len(train_terms) * 100).round(2))
print("\n")

# Plot aspect distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Count plot
aspect_counts.plot(kind='bar', ax=axes[0], color=['#1f77b4', '#ff7f0e', '#2ca02c'])
axes[0].set_title('Distribution of Annotations by Aspect', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Aspect', fontsize=12)
axes[0].set_ylabel('Count', fontsize=12)
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=0)

# Pie chart
axes[1].pie(aspect_counts.values, labels=aspect_counts.index, autopct='%1.1f%%',
            colors=['#1f77b4', '#ff7f0e', '#2ca02c'], startangle=90)
axes[1].set_title('Percentage Distribution by Aspect', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()
print("\n")

# Analyze GO terms per aspect
print("=== GO Terms per Aspect ===")
for aspect in train_terms['aspect'].unique():
    aspect_df = train_terms[train_terms['aspect'] == aspect]
    unique_terms = aspect_df['term'].nunique()
    unique_proteins = aspect_df['EntryID'].nunique()
    print(f"{aspect}:")
    print(f"  - Unique GO terms: {unique_terms}")
    print(f"  - Unique proteins: {unique_proteins}")
    print(f"  - Avg terms per protein: {len(aspect_df) / unique_proteins:.2f}")
print("\n")

# Analyze proteins per GO term
print("=== Proteins per GO Term (Top 20 most common terms) ===")
term_counts = train_terms['term'].value_counts()
print(term_counts.head(20))
print("\n")

# Plot most common GO terms
fig, ax = plt.subplots(figsize=(12, 8))
term_counts.head(30).plot(kind='barh', ax=ax, color='steelblue')
ax.set_title('Top 30 Most Frequent GO Terms', fontsize=14, fontweight='bold')
ax.set_xlabel('Number of Proteins', fontsize=12)
ax.set_ylabel('GO Term', fontsize=12)
ax.invert_yaxis()
plt.tight_layout()
plt.show()
print("\n")

# Analyze annotations per protein
print("=== Annotations per Protein ===")
protein_annotation_counts = train_terms.groupby('EntryID').size()
print(f"Min annotations per protein: {protein_annotation_counts.min()}")
print(f"Max annotations per protein: {protein_annotation_counts.max()}")
print(f"Mean annotations per protein: {protein_annotation_counts.mean():.2f}")
print(f"Median annotations per protein: {protein_annotation_counts.median():.2f}")
print("\n")

# Distribution of annotations per protein
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(protein_annotation_counts, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
axes[0].set_xlabel('Number of Annotations', fontsize=12)
axes[0].set_ylabel('Number of Proteins', fontsize=12)
axes[0].set_title('Distribution of Annotations per Protein', fontsize=14, fontweight='bold')
axes[0].grid(axis='y', alpha=0.3)

# Log scale for better visualization
axes[1].hist(protein_annotation_counts, bins=50, color='lightcoral', edgecolor='black', alpha=0.7)
axes[1].set_xlabel('Number of Annotations', fontsize=12)
axes[1].set_ylabel('Number of Proteins', fontsize=12)
axes[1].set_title('Distribution of Annotations per Protein (Log Scale)', fontsize=14, fontweight='bold')
axes[1].set_yscale('log')
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()
print("\n")

# Load protein sequences
print("=== Loading Protein Sequences ===")
fasta_file = TRAIN_PATH / 'train_sequences.fasta'

if fasta_file.exists():
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
    
    print(f"Loaded {len(sequences)} protein sequences")
    
    # Analyze sequence lengths
    seq_lengths = [len(seq) for seq in sequences.values()]
    
    print(f"\n=== Sequence Length Statistics ===")
    print(f"Min length: {min(seq_lengths)}")
    print(f"Max length: {max(seq_lengths)}")
    print(f"Mean length: {np.mean(seq_lengths):.2f}")
    print(f"Median length: {np.median(seq_lengths):.2f}")
    print(f"Std deviation: {np.std(seq_lengths):.2f}")
    print(f"25th percentile: {np.percentile(seq_lengths, 25):.2f}")
    print(f"75th percentile: {np.percentile(seq_lengths, 75):.2f}")
    print(f"99th percentile: {np.percentile(seq_lengths, 99):.2f}")
    print("\n")
    
    # Plot sequence length distribution
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Full distribution
    axes[0, 0].hist(seq_lengths, bins=100, color='skyblue', edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('Sequence Length', fontsize=12)
    axes[0, 0].set_ylabel('Frequency', fontsize=12)
    axes[0, 0].set_title('Distribution of Protein Sequence Lengths', fontsize=14, fontweight='bold')
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # Sequences < 2000
    filtered_lengths = [l for l in seq_lengths if l < 2000]
    axes[0, 1].hist(filtered_lengths, bins=100, color='lightcoral', edgecolor='black', alpha=0.7)
    axes[0, 1].set_xlabel('Sequence Length', fontsize=12)
    axes[0, 1].set_ylabel('Frequency', fontsize=12)
    axes[0, 1].set_title('Sequence Lengths (< 2000 AA)', fontsize=14, fontweight='bold')
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Log scale
    axes[1, 0].hist(seq_lengths, bins=100, color='lightgreen', edgecolor='black', alpha=0.7)
    axes[1, 0].set_xlabel('Sequence Length', fontsize=12)
    axes[1, 0].set_ylabel('Frequency (Log Scale)', fontsize=12)
    axes[1, 0].set_title('Distribution (Log Scale)', fontsize=14, fontweight='bold')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # Box plot
    axes[1, 1].boxplot(seq_lengths, vert=True)
    axes[1, 1].set_ylabel('Sequence Length', fontsize=12)
    axes[1, 1].set_title('Box Plot of Sequence Lengths', fontsize=14, fontweight='bold')
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    print("\n")
    
    # Amino acid composition
    print("=== Amino Acid Analysis ===")
    all_aas = ''.join(sequences.values())
    aa_counts = Counter(all_aas)
    aa_df = pd.DataFrame.from_dict(aa_counts, orient='index', columns=['Count'])
    aa_df = aa_df.sort_values('Count', ascending=False)
    aa_df['Percentage'] = (aa_df['Count'] / aa_df['Count'].sum() * 100).round(2)
    
    print(f"Unique amino acids found: {len(aa_counts)}")
    print(f"\nAmino acid distribution:")
    print(aa_df)
    print("\n")
    
    # Plot amino acid distribution
    fig, ax = plt.subplots(figsize=(12, 6))
    aa_df['Percentage'].plot(kind='bar', ax=ax, color='teal', alpha=0.7)
    ax.set_title('Amino Acid Distribution', fontsize=14, fontweight='bold')
    ax.set_xlabel('Amino Acid', fontsize=12)
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Sample sequences
    print(f"\n=== Sample Sequences (first 5) ===")
    for i, (seq_id, seq) in enumerate(list(sequences.items())[:5]):
        print(f"\n{i+1}. EntryID: {seq_id}")
        print(f"   Length: {len(seq)}")
        print(f"   Sequence: {seq[:100]}...")
        
        # Check if this protein has annotations
        protein_annotations = train_terms[train_terms['EntryID'] == seq_id]
        if len(protein_annotations) > 0:
            print(f"   Annotations: {len(protein_annotations)}")
            print(f"   Aspects: {protein_annotations['aspect'].unique()}")
            print(f"   Sample terms: {protein_annotations['term'].head(3).tolist()}")
else:
    print(f"FASTA file not found at {fasta_file}")

print("\n" + "="*50)
print("=== EDA Summary ===")
print(f"Total proteins: {train_terms['EntryID'].nunique()}")
print(f"Total GO terms: {train_terms['term'].nunique()}")
print(f"Total annotations: {len(train_terms)}")
print(f"Aspects: {', '.join(train_terms['aspect'].unique())}")
if 'sequences' in locals():
    print(f"Sequences loaded: {len(sequences)}")
    print(f"Avg sequence length: {np.mean(seq_lengths):.0f} AA")
print("="*50)
print("\nEDA Complete!")