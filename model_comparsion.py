import pandas as pd
from cancer_drug_predictor import CancerDrugDiscoveryPredictor
# Load your dataset
df = pd.read_csv("jak2_data")  # Make sure it has correct columns

# Instantiate the class
predictor = CancerDrugDiscoveryPredictor()

# Set the feature columns (descriptor columns, excluding target variables)
predictor.feature_columns = [col for col in df.columns if col not in ['overall_cancer_potency', 'is_toxic', 'selectivity_index']]

# Run the comparison and print the output
results_df = predictor.compare_models(df)
print(results_df)
