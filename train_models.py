#!/usr/bin/env python3
"""
Train models with test set as last 5 years (2020-2024).
Maximum training data + robust test set with many winners.
"""

import joblib
from f1_predictor import F1Predictor
import pandas as pd

print("="*70)
print("TRAINING WITH LAST 5 YEARS TEST SET (2020-2024)")
print("="*70)
print("\nUsing: Train 1950-2019, Val 2020, Test 2021-2024")
print("  - Training: 1950-2019 (70 years)")
print("  - Validation: 2020 (1 year)")
print("  - Test: 2021-2024 (4 years - ROBUST)")
print("  - Test samples: 1,359 with ~68 winners\n")

models = ['gradient_boosting', 'random_forest', 'xgboost']
results = {}

for model_type in models:
    print(f"\n{'='*70}")
    print(f"Training {model_type.upper()}")
    print(f"{'='*70}")
    
    predictor = F1Predictor(
        model_type=model_type,
        data_path='f1data'
    )
    
    try:
        train_metrics, val_metrics, test_metrics, optimal_threshold = predictor.train_model(
            use_smote=True,
            optimize_threshold=True,
            train_years=(1950, 2019),
            val_years=(2020, 2020),
            test_years=(2021, 2024)
        )
        
        print(f"\n✓ Training completed successfully")
        print(f"  Optimal threshold: {optimal_threshold:.2f}")
        print(f"\n  Test Metrics:")
        print(f"    - F1 Score: {test_metrics['f1']:.4f}")
        print(f"    - Precision: {test_metrics['precision']:.4f}")
        print(f"    - Recall: {test_metrics['recall']:.4f}")
        print(f"    - ROC AUC: {test_metrics['roc_auc']:.4f}")
        
        # Save model
        filename = f'f1_model_{model_type}_5yr_test.joblib'
        predictor.save_model(filename)
        print(f"\n  Saved to: {filename}")
        
        results[model_type] = {
            'f1': test_metrics['f1'],
            'precision': test_metrics['precision'],
            'recall': test_metrics['recall'],
            'roc_auc': test_metrics['roc_auc'],
            'threshold': optimal_threshold
        }
        
    except Exception as e:
        print(f"\n✗ Error training {model_type}: {e}")
        import traceback
        traceback.print_exc()

# Summary comparison
print(f"\n{'='*70}")
print("FINAL COMPARISON (5-Year Test Set: 2021-2024)")
print(f"{'='*70}")

if results:
    df_results = pd.DataFrame(results).T
    print(df_results.to_string())
    
    print(f"\n{'='*70}")
    print("KEY METRICS")
    print(f"{'='*70}")
    
    best_f1 = max(results.items(), key=lambda x: x[1]['f1'])
    print(f"\n✓ Best Model: {best_f1[0].upper()} (F1={best_f1[1]['f1']:.4f})")
    
    print(f"\nTraining Data:")
    print(f"  - Years: 1950-2019 (70 years)")
    print(f"  - Samples: 24,620")
    print(f"  - Winners: 1,021")
    
    print(f"\nTest Data (4 years):")
    print(f"  - Years: 2021-2024")
    print(f"  - Samples: 1,359")
    print(f"  - Expected winners: ~68")
    print(f"  - MUCH more robust than 24 or 46 winners!")
    
    print(f"\n✓ Models ready for robust evaluation")
    print(f"✓ File: f1_model_{best_f1[0]}_5yr_test.joblib")
