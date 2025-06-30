# Modular Results Evaluation
# Functions to analyze results with great-tables visualizations

import pandas as pd
from typing import Dict
from pathlib import Path
from great_tables import GT

def load_results(results_path: str) -> pd.DataFrame:
    """
    Load results from CSV file
    
    Args:
        results_path: Path to CSV results file
    
    Returns:
        DataFrame with results
        
    Markdown comment:
    ## Step 1: Load Results
    Load your saved CSV results for analysis.
    """
    df = pd.read_csv(results_path)
    print(f"✓ Loaded {len(df)} results from {results_path}")
    return df

def calculate_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate accuracy, precision, recall, and F1 score for a subset of data
    
    Args:
        df: DataFrame subset with predictions and true labels
    
    Returns:
        Dictionary with calculated metrics
        
    Markdown comment:
    ## Step 2: Calculate Metrics
    Computes standard classification metrics.
    
    **Metrics calculated:**
    - Accuracy: Overall correctness
    - Precision: True positives / (True positives + False positives)  
    - Recall: True positives / (True positives + False negatives)
    - F1 Score: Harmonic mean of precision and recall
    """
    # Filter out error predictions
    valid_df = df[~df['prediction'].str.startswith('error', na=False)]
    
    if len(valid_df) == 0:
        return {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0, 'valid_predictions': 0, 'total_predictions': len(df)}
    
    # Calculate metrics
    correct = (valid_df['prediction'] == valid_df['true_label']).sum()
    total = len(valid_df)
    accuracy = correct / total
    
    # Get unique labels for multi-class support
    labels = valid_df['true_label'].unique()
    if len(labels) == 2:
        # Binary classification - calculate for positive class
        positive_label = labels[0]  # Assumes first label is "positive" class
        
        tp = ((valid_df['prediction'] == positive_label) & (valid_df['true_label'] == positive_label)).sum()
        fp = ((valid_df['prediction'] == positive_label) & (valid_df['true_label'] != positive_label)).sum()
        fn = ((valid_df['prediction'] != positive_label) & (valid_df['true_label'] == positive_label)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    else:
        # Multi-class - use macro averaging
        precisions, recalls = [], []
        for label in labels:
            tp = ((valid_df['prediction'] == label) & (valid_df['true_label'] == label)).sum()
            fp = ((valid_df['prediction'] == label) & (valid_df['true_label'] != label)).sum()
            fn = ((valid_df['prediction'] != label) & (valid_df['true_label'] == label)).sum()
            
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            precisions.append(prec)
            recalls.append(rec)
        
        precision = sum(precisions) / len(precisions)
        recall = sum(recalls) / len(recalls)
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'valid_predictions': len(valid_df),
        'total_predictions': len(df)
    }

def analyze_by_configuration(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze results by prompt, model, and temperature combinations
    
    Args:
        df: Results DataFrame
    
    Returns:
        DataFrame with metrics for each configuration
        
    Markdown comment:
    ## Step 3: Analyze by Configuration
    Breaks down performance by each prompt/model/temperature combination.
    
    **Output:** Detailed metrics for every tested combination
    **Use for:** Finding the best performing configurations
    """
    results = []
    
    for prompt_id in df['prompt_id'].unique():
        for model in df['model'].unique():
            for temp in df['temperature'].unique():
                subset = df[(df['prompt_id'] == prompt_id) & 
                           (df['model'] == model) & 
                           (df['temperature'] == temp)]
                
                metrics = calculate_metrics(subset)
                results.append({
                    'prompt_id': prompt_id,
                    'model': model,
                    'temperature': temp,
                    **metrics
                })
    
    analysis_df = pd.DataFrame(results)
    print(f"✓ Analyzed {len(analysis_df)} configurations")
    return analysis_df

def create_summary_tables(analysis_df: pd.DataFrame, output_dir: str = "evaluation_tables"):
    """
    Create great-tables HTML visualizations
    
    Args:
        analysis_df: Analysis DataFrame from analyze_by_configuration()
        output_dir: Directory to save HTML tables
        
    Markdown comment:
    ## Step 4: Create Visual Tables
    Generates beautiful HTML tables showing your results.
    
    **Tables created:**
    1. **Model Summary:** Average performance by model
    2. **Temperature Impact:** How temperature affects accuracy  
    3. **Top Configurations:** Best performing combinations
    4. **Prompt Comparison:** Head-to-head prompt performance
    
    **Output:** HTML files you can open in your browser or embed in reports
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # 1. Model Summary Table (averaged across prompts at temp=0)
    temp0_data = analysis_df[analysis_df['temperature'] == 0.0]
    if len(temp0_data) > 0:
        model_summary = temp0_data.groupby('model').agg({
            'accuracy': ['mean', 'std'],
            'precision': 'mean',
            'recall': 'mean',
            'f1': 'mean'
        }).round(3)
        model_summary.columns = ['Accuracy Mean', 'Accuracy Std', 'Precision', 'Recall', 'F1 Score']
        model_summary = model_summary.reset_index()
        
        gt_model = (
            GT(model_summary)
            .tab_header(
                title="Model Performance Summary",
                subtitle="Averaged across all prompts at temperature=0"
            )
            .fmt_percent(
                columns=["Accuracy Mean", "Accuracy Std"],
                decimals=1
            )
            .fmt_number(
                columns=["Precision", "Recall", "F1 Score"],
                decimals=3
            )
            .cols_label(
                model="Model",
                **{"Accuracy Mean": "Avg Accuracy", "Accuracy Std": "Std Dev"}
            )
        )
        gt_model.save(str(output_path / "model_summary.html"))
    
    # 2. Temperature Impact Table
    temp_impact = analysis_df.groupby(['model', 'temperature'])['accuracy'].mean().reset_index()
    temp_pivot = temp_impact.pivot(index='model', columns='temperature', values='accuracy')
    temp_pivot = temp_pivot.round(3).reset_index()
    
    # Get temperature columns dynamically
    temp_cols = [col for col in temp_pivot.columns if isinstance(col, (int, float))]
    temp_col_labels = {col: f"Temp {col}" for col in temp_cols}
    
    gt_temp = (
        GT(temp_pivot)
        .tab_header(
            title="Temperature Impact on Accuracy",
            subtitle="Accuracy by model and temperature setting"
        )
        .fmt_percent(
            columns=temp_cols,
            decimals=1
        )
        .cols_label(
            model="Model",
            **temp_col_labels
        )
    )
    gt_temp.save(str(output_path / "temperature_impact.html"))
    
    # 3. Top Configurations Table
    top_configs = analysis_df.nlargest(10, 'accuracy')[
        ['prompt_id', 'model', 'temperature', 'accuracy', 'precision', 'recall', 'f1']
    ].round(3)
    
    gt_top = (
        GT(top_configs)
        .tab_header(
            title="Top 10 Configurations",
            subtitle="Best performing prompt/model/temperature combinations"
        )
        .fmt_percent(
            columns=["accuracy"],
            decimals=1
        )
        .fmt_number(
            columns=["precision", "recall", "f1"],
            decimals=3
        )
        .cols_label(
            prompt_id="Prompt",
            model="Model",
            temperature="Temp",
            accuracy="Accuracy",
            precision="Precision",
            recall="Recall",
            f1="F1 Score"
        )
    )
    gt_top.save(str(output_path / "top_configurations.html"))
    
    # 4. Prompt Comparison Table (at temp=0)
    if len(temp0_data) > 0:
        prompt_comparison = temp0_data.pivot(index='prompt_id', columns='model', values='accuracy')
        prompt_comparison = prompt_comparison.round(3).reset_index()
        
        model_cols = [col for col in prompt_comparison.columns if col != 'prompt_id']
        
        gt_prompt = (
            GT(prompt_comparison)
            .tab_header(
                title="Prompt Performance Comparison",
                subtitle="Accuracy by prompt and model at temperature=0"
            )
            .fmt_percent(
                columns=model_cols,
                decimals=1
            )
            .cols_label(
                prompt_id="Prompt ID"
            )
        )
        gt_prompt.save(str(output_path / "prompt_comparison.html"))
    
    print(f"✓ Tables saved to {output_dir}/")
    print("  - model_summary.html")
    print("  - temperature_impact.html") 
    print("  - top_configurations.html")
    print("  - prompt_comparison.html")

def generate_text_report(df: pd.DataFrame, analysis_df: pd.DataFrame, output_path: str = "evaluation_report.txt"):
    """
    Generate a comprehensive text report
    
    Args:
        df: Original results DataFrame
        analysis_df: Analysis DataFrame from analyze_by_configuration()
        output_path: Path for output text file
        
    Markdown comment:
    ## Step 5: Generate Text Report
    Creates a detailed text summary of your results.
    
    **Report sections:**
    - Overall statistics
    - Best performing configuration
    - Detailed results by configuration
    - Temperature impact analysis
    
    **Use for:** Sharing results or including in documentation
    """
    with open(output_path, 'w') as f:
        f.write("# Prompt Testing Results Report\n")
        f.write("=" * 50 + "\n\n")
        
        # Overall statistics
        f.write("## Overall Statistics\n")
        f.write(f"Total predictions: {len(df)}\n")
        f.write(f"Unique prompts tested: {df['prompt_id'].nunique()}\n")
        f.write(f"Models evaluated: {', '.join(sorted(df['model'].unique()))}\n")
        f.write(f"Temperatures tested: {sorted(df['temperature'].unique())}\n\n")
        
        # Best performing configuration
        if len(analysis_df) > 0:
            best_config = analysis_df.loc[analysis_df['accuracy'].idxmax()]
            f.write("## Best Performing Configuration\n")
            f.write(f"Prompt: {best_config['prompt_id']}\n")
            f.write(f"Model: {best_config['model']}\n")
            f.write(f"Temperature: {best_config['temperature']}\n")
            f.write(f"Accuracy: {best_config['accuracy']:.2%}\n")
            f.write(f"F1 Score: {best_config['f1']:.3f}\n\n")
            
            # Detailed results by configuration
            f.write("## Detailed Results\n")
            f.write("-" * 50 + "\n")
            
            for _, row in analysis_df.sort_values(['accuracy'], ascending=False).iterrows():
                f.write(f"\n### {row['prompt_id']} | {row['model']} | temp={row['temperature']}\n")
                f.write(f"Accuracy: {row['accuracy']:.2%}\n")
                f.write(f"Precision: {row['precision']:.3f}\n")
                f.write(f"Recall: {row['recall']:.3f}\n")
                f.write(f"F1 Score: {row['f1']:.3f}\n")
                f.write(f"Valid predictions: {row['valid_predictions']}/{row['total_predictions']}\n")
            
            # Temperature analysis
            f.write("\n## Temperature Impact Analysis\n")
            f.write("-" * 50 + "\n")
            temp_analysis = analysis_df.groupby('temperature')['accuracy'].agg(['mean', 'std'])
            for temp, row in temp_analysis.iterrows():
                f.write(f"Temperature {temp}: {row['mean']:.2%} (±{row['std']:.2%})\n")
    
    print(f"✓ Text report saved to {output_path}")

def evaluate_results(results_path: str, output_dir: str = "evaluation_output") -> pd.DataFrame:
    """
    Complete evaluation pipeline for results file
    
    Args:
        results_path: Path to CSV results file
        output_dir: Directory for output files
    
    Returns:
        Analysis DataFrame with metrics
        
    Markdown comment:
    ## Complete Evaluation Pipeline
    This function runs the full evaluation process.
    
    **Usage:**
    ```python
    analysis = evaluate_results("results_20240101_120000.csv")
    ```
    """
    print("📊 Starting results evaluation")
    print("=" * 40)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Run evaluation steps
    df = load_results(results_path)
    analysis_df = analyze_by_configuration(df)
    
    tables_dir = output_path / "tables"
    create_summary_tables(analysis_df, str(tables_dir))
    
    report_path = output_path / "report.txt"
    generate_text_report(df, analysis_df, str(report_path))
    
    print(f"\n✓ Evaluation complete! Check {output_dir}/ for results")
    print("\nTop 3 configurations by accuracy:")
    if len(analysis_df) > 0:
        top3 = analysis_df.nlargest(3, 'accuracy')[['prompt_id', 'model', 'temperature', 'accuracy']]
        for _, row in top3.iterrows():
            print(f"  {row['prompt_id']} | {row['model']} | temp={row['temperature']}: {row['accuracy']:.1%}")
    
    return analysis_df