"""
Export Trained Models to Pickle Files
====================================
This script should be run after training models in the Jupyter notebook
to save them to pickle files for use in the Streamlit app.

To use this script:
1. Run the model training cells in the notebook first
2. Then run this script: python export_models.py

The models will be saved to the 'models/' directory.
"""

import pickle
import os
import glob

print("💾 Exporting trained models to pickle files...")
print("="*60)

# Note: This script assumes you have already trained the models in the notebook
# You need to run the notebook cells that create these variables:
# - lr_model, scaler_lr
# - dt_model
# - rf_model
# - svm_model, scaler_svm
# - xgb_model, le (label_encoder)
# - nn_model, scaler_nn
# - X_train (with columns)

print("\n⚠️  IMPORTANT: Run the notebook first to train the models!")
print("Then execute this script from the notebook with: exec(open('export_models.py').read())")
print("\nAlternatively, you can copy the model export code from this file")
print("and paste it into a new notebook cell after the model training cells.")

# The actual export code that should be added to the notebook:
print("\n" + "="*60)
print("COPY THIS CODE TO YOUR NOTEBOOK AFTER TRAINING MODELS:")
print("="*60)
print("""
import pickle
import os

# Create models directory
os.makedirs('models', exist_ok=True)

# Save all models
pickle.dump(lr_model, open('models/logistic_regression_model.pkl', 'wb'))
pickle.dump(scaler_lr, open('models/logistic_regression_scaler.pkl', 'wb'))

pickle.dump(dt_model, open('models/decision_tree_model.pkl', 'wb'))

pickle.dump(rf_model, open('models/random_forest_model.pkl', 'wb'))

pickle.dump(svm_model, open('models/svm_model.pkl', 'wb'))
pickle.dump(scaler_svm, open('models/svm_scaler.pkl', 'wb'))

pickle.dump(xgb_model, open('models/xgboost_model.pkl', 'wb'))
pickle.dump(le, open('models/label_encoder.pkl', 'wb'))

pickle.dump(nn_model, open('models/neural_network_model.pkl', 'wb'))
pickle.dump(scaler_nn, open('models/neural_network_scaler.pkl', 'wb'))

# Save feature names
pickle.dump(list(X_train.columns), open('models/feature_names.pkl', 'wb'))

print("✅ All models exported to 'models/' directory!")
""")
print("="*60)

