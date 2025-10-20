"""
Memory-efficient submission creation for Kaggle notebooks
Copy-paste this into a Kaggle notebook cell

This version:
- Writes predictions incrementally to CSV
- Processes in small batches
- Clears memory aggressively
- Avoids RAM overload
"""

import torch
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
import gc


def create_submission_memory_efficient(
    model,
    test_embeddings,
    protein_ids,
    idx_to_go_term,
    output_path='/kaggle/working/submission.csv',
    threshold=0.5,
    batch_size=16  # Small batch to avoid OOM
):
    """
    Create submission file without loading all predictions into RAM
    """
    print(f"Creating submission for {len(protein_ids)} proteins...")
    print(f"Threshold: {threshold}")
    print(f"Batch size: {batch_size}")

    model.eval()
    device = next(model.parameters()).device

    # Open file and write header
    with open(output_path, 'w') as f:
        f.write("Protein ID,GO Term,Confidence\n")

    total_predictions = 0
    no_pred_count = 0

    # Process in small batches
    with torch.no_grad():
        for start_idx in tqdm(range(0, len(protein_ids), batch_size)):
            end_idx = min(start_idx + batch_size, len(protein_ids))

            # Get batch
            batch_emb = test_embeddings[start_idx:end_idx]
            batch_ids = protein_ids[start_idx:end_idx]

            # Predict
            batch_tensor = torch.FloatTensor(batch_emb).to(device)
            outputs = model(batch_tensor)
            probs = torch.sigmoid(outputs).cpu().numpy()

            # Process each protein and write immediately
            rows = []
            for i, pid in enumerate(batch_ids):
                protein_probs = probs[i]

                # Get predictions above threshold
                pred_idx = np.where(protein_probs > threshold)[0]

                # If no predictions, use top 3
                if len(pred_idx) == 0:
                    no_pred_count += 1
                    pred_idx = np.argsort(protein_probs)[-3:]

                # Create rows
                for idx in pred_idx:
                    go_term = idx_to_go_term[idx]
                    conf = float(protein_probs[idx])
                    rows.append(f"{pid},{go_term},{conf:.6f}\n")
                    total_predictions += 1

            # Write batch to file immediately
            with open(output_path, 'a') as f:
                f.writelines(rows)

            # Aggressively clear memory
            del batch_tensor, outputs, probs, rows
            gc.collect()
            torch.cuda.empty_cache()

    print(f"\n✓ Submission created!")
    print(f"  File: {output_path}")
    print(f"  Total predictions: {total_predictions}")
    print(f"  Proteins with no predictions: {no_pred_count}")


# ============================================================
# Example usage in Kaggle notebook
# ============================================================

"""
# Cell 1: Load everything

import torch
import torch.nn as nn
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import gc

# Define model class
class SimpleClassifier(nn.Module):
    def __init__(self, input_dim=640, hidden_dim=512, num_classes=26125):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)


# Load test embeddings
print("Loading test embeddings...")
with open('/kaggle/working/test_embeddings.pkl', 'rb') as f:
    test_data = pickle.load(f)

test_embeddings = test_data['embeddings']
protein_ids = test_data['protein_ids']
print(f"✓ Loaded {len(protein_ids)} test proteins")

# Load labels
print("Loading labels...")
with open('/kaggle/working/preprocessed_labels.pkl', 'rb') as f:
    label_data = pickle.load(f)

idx_to_go_term = label_data['idx_to_go_term']
print(f"✓ Loaded {len(idx_to_go_term)} GO terms")

# Load model
print("Loading model...")
checkpoint = torch.load('/kaggle/working/baseline_model.pt')
model = SimpleClassifier(
    input_dim=checkpoint['embedding_dim'],
    num_classes=checkpoint['num_classes']
).cuda()
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print(f"✓ Model loaded")


# Cell 2: Create submission (memory-efficient)

# Copy the create_submission_memory_efficient function here
# Then call it:

create_submission_memory_efficient(
    model=model,
    test_embeddings=test_embeddings,
    protein_ids=protein_ids,
    idx_to_go_term=idx_to_go_term,
    output_path='/kaggle/working/submission.csv',
    threshold=0.5,
    batch_size=16  # Small batch size to avoid OOM
)

# Cell 3: Validate and preview

submission = pd.read_csv('/kaggle/working/submission.csv')
print(f"Submission shape: {submission.shape}")
print(f"\\nFirst 20 rows:")
print(submission.head(20))

print(f"\\nUnique proteins: {submission['Protein ID'].nunique()}")
print(f"Unique GO terms: {submission['GO Term'].nunique()}")
print(f"Total predictions: {len(submission)}")

# Check for required proteins
required_proteins = len(protein_ids)
submitted_proteins = submission['Protein ID'].nunique()
if submitted_proteins < required_proteins:
    print(f"⚠ Warning: {required_proteins - submitted_proteins} proteins missing!")
else:
    print(f"✓ All {required_proteins} proteins have predictions")

# Cell 4: Submit
# Download submission.csv and upload to competition
# Or use Kaggle API if available
"""
