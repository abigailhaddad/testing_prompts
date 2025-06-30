# Modular Prompt Testing Framework
# Each function takes explicit paths and configurations for maximum flexibility

import json
import pandas as pd
from typing import Dict, List
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from litellm import completion
from pydantic import create_model
from enum import Enum
from datetime import datetime

def load_environment(env_path: str = ".env"):
    """
    Load environment variables from specified .env file
    
    Args:
        env_path: Path to .env file (default: ".env")
        
    Markdown comment:
    ## Step 1: Setup Environment
    This loads your API keys from the .env file. 
    
    **To use different env file:** Change env_path parameter
    **Required variables:** OPENAI_API_KEY (and others based on your models)
    """
    import os
    from pathlib import Path
    
    # Check if .env file exists
    if not Path(env_path).exists():
        print(f"❌ ERROR: {env_path} file not found!")
        print(f"📝 Please create a {env_path} file with your API keys:")
        print("   OPENAI_API_KEY=your_key_here")
        print("   ANTHROPIC_API_KEY=your_key_here")
        raise FileNotFoundError(f"Missing {env_path} file")
    
    load_dotenv(env_path)
    
    # Check for API keys and show what we found
    api_keys_to_check = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY"]
    found_keys = []
    missing_keys = []
    
    for key in api_keys_to_check:
        if os.getenv(key):
            found_keys.append(key)
        else:
            missing_keys.append(key)
    
    if found_keys:
        print(f"✓ Found {len(found_keys)} API key(s) in {env_path}:")
        for key in found_keys:
            print(f"   - {key}")
    
    if missing_keys:
        print(f"⚠️  Missing {len(missing_keys)} API key(s):")
        for key in missing_keys:
            print(f"   - {key}")
    
    print(f"✓ Environment variables loaded from {env_path}")

def load_configuration(config_path: str) -> Dict:
    """
    Load prompts, models, and output format configuration
    
    Args:
        config_path: Path to JSON config file
    
    Returns:
        Dictionary with full configuration
        
    Markdown comment:
    ## Step 2: Load Configuration
    This reads your complete setup from a JSON config file.
    
    **Config structure:**
    - `data_settings`: Column names and sample size
    - `output_format`: Response field name and valid enum values  
    - `prompts`: List of prompt templates to test
    - `models`: List of model names to use
    - `temperatures`: List of temperature values to test
    
    **To modify:** Edit your config JSON file
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print(f"✓ {len(config['prompts'])} prompts, {len(config['models'])} models, {len(config['temperatures'])} temperatures")
    
    print(f"\n📝 Prompts:")
    for i, prompt in enumerate(config['prompts'], 1):
        print(f"   {i}. {prompt['id']}: {prompt['prompt']}")
        print()
    
    return config

def create_response_model(config: Dict):
    """
    Dynamically create Pydantic model based on config
    
    Args:
        config: Configuration dictionary with output_format section
    
    Returns:
        Pydantic model class for structured output
        
    Markdown comment:
    ## Step 3: Create Response Model
    This creates a Pydantic model based on your config's enum values.
    
    **Dynamic creation:** The enum and model are built from your config file
    **Structured output:** Ensures LLMs return exactly the format you want
    """
    # Create enum from config
    enum_values = config['output_format']['enum_values']
    ResponseEnum = Enum('ResponseEnum', {val: val for val in enum_values})
    
    # Create Pydantic model
    field_name = config['output_format']['response_field']
    ResponseModel = create_model(
        'ResponseModel',
        **{field_name: (ResponseEnum, ...)}
    )
    
    return ResponseModel

def load_dataset(data_path: str, config: Dict) -> pd.DataFrame:
    """
    Load and sample dataset based on configuration
    
    Args:
        data_path: Path to CSV file with your data
        config: Configuration dictionary with data_settings
    
    Returns:
        DataFrame with sampled data
        
    Markdown comment:
    ## Step 4: Load Dataset
    This loads your labeled data using the column names from config.
    
    **Required:** CSV file with text and label columns (names from config)
    **Sampling:** Evenly samples from each class based on config sample_size
    **To use different data:** Just change the data_path parameter
    """
    df = pd.read_csv(data_path)
    
    # Get column names from config
    text_col = config['data_settings']['text_column']
    label_col = config['data_settings']['label_column']
    sample_size = config['data_settings']['sample_size']
    
    # Validate columns exist
    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(f"Required columns '{text_col}' and '{label_col}' not found in {data_path}")
    
    # Sample evenly from each class
    unique_labels = df[label_col].unique()
    samples_per_class = sample_size // len(unique_labels)
    
    sampled_dfs = []
    for label in unique_labels:
        class_df = df[df[label_col] == label].sample(n=samples_per_class, random_state=42)
        sampled_dfs.append(class_df)
    
    samples = pd.concat(sampled_dfs).reset_index(drop=True)
    
    print(f"✓ {len(samples)} samples loaded")
    
    return samples

def create_tasks(config: Dict, samples: pd.DataFrame) -> List[Dict]:
    """
    Create all combinations of prompts, models, temperatures, and samples
    
    Args:
        config: Configuration dictionary
        samples: Sample data DataFrame
    
    Returns:
        List of task dictionaries ready for execution
        
    Markdown comment:
    ## Step 5: Create Task List
    This creates every combination of:
    - Each prompt × Each model × Each temperature × Each sample
    
    **Total tasks:** prompts × models × temperatures × samples
    **Each task contains:** All info needed to run one LLM call
    """
    tasks = []
    text_col = config['data_settings']['text_column']
    label_col = config['data_settings']['label_column']
    
    for prompt_config in config['prompts']:
        for model in config['models']:
            for temperature in config['temperatures']:
                for idx, row in samples.iterrows():
                    tasks.append({
                        'sample_id': idx,
                        'text_snippet': str(row[text_col])[:100] + "...",
                        'true_label': row[label_col],
                        'prompt_id': prompt_config['id'],
                        'prompt_template': prompt_config['prompt'],
                        'full_text': row[text_col],
                        'model': model,
                        'temperature': temperature
                    })
    
    return tasks

def tasks_to_dataframe(config: Dict, samples: pd.DataFrame) -> pd.DataFrame:
    """
    Create tasks DataFrame with all combinations
    
    Args:
        config: Configuration dictionary
        samples: Sample data DataFrame
    
    Returns:
        DataFrame with all task combinations that will be executed
        
    Markdown comment:
    ## Step 5: Create Tasks DataFrame
    This creates a DataFrame showing all task combinations that will be run.
    
    **Clean output:** Returns structured data for display in notebook
    """
    tasks = create_tasks(config, samples)
    tasks_df = pd.DataFrame(tasks)
    
    # Add summary stats as attributes for easy access
    tasks_df.attrs['num_prompts'] = len(config['prompts'])
    tasks_df.attrs['num_models'] = len(config['models'])
    tasks_df.attrs['num_temps'] = len(config['temperatures'])
    tasks_df.attrs['num_samples'] = len(samples)
    tasks_df.attrs['estimated_time_min'] = len(tasks) / 4 * 2 / 60  # 4 workers, 2 sec per task
    
    return tasks_df

def run_single_prompt(task: Dict, response_model, response_field: str) -> Dict:
    """
    Execute a single prompt task with structured output
    
    Args:
        task: Task dictionary with prompt, model, and data
        response_model: Pydantic model for structured output
        response_field: Name of the response field
    
    Returns:
        Result dictionary with prediction and metadata
        
    Markdown comment:
    ## Step 6: Run Individual Prompts
    This function:
    1. Fills in the prompt template with the text
    2. Calls the LLM with structured output
    3. Extracts the prediction from the structured response
    
    **Key feature:** Uses Pydantic structured output for clean results!
    """
    # Truncate very long text to avoid token limits
    text = str(task['full_text'])
    if len(text) > 2000:
        text = text[:2000] + "..."
    
    prompt = task['prompt_template'].replace('{text}', text)
    
    try:
        response = completion(
            model=task['model'],
            messages=[{"role": "user", "content": prompt}],
            temperature=task['temperature'],
            max_tokens=50,
            response_format=response_model
        )
        
        # Extract structured response
        prediction = response.choices[0].message.content
        parsed = json.loads(prediction)
        prediction = parsed.get(response_field, 'unknown')
        
    except Exception as e:
        prediction = f"error: {str(e)}"
    
    return {
        'sample_id': task['sample_id'],
        'text_snippet': task['text_snippet'],
        'true_label': task['true_label'],
        'prompt_id': task['prompt_id'],
        'model': task['model'],
        'temperature': task['temperature'],
        'prediction': prediction,
        'timestamp': datetime.now().isoformat()
    }

def run_all_tasks(tasks: List[Dict], config: Dict, max_workers: int = 4) -> List[Dict]:
    """
    Execute all tasks in parallel using configuration
    
    Args:
        tasks: List of task dictionaries
        config: Configuration with output format settings
        max_workers: Number of parallel workers
    
    Returns:
        List of results from all tasks
        
    Markdown comment:
    ## Step 7: Run All Tasks in Parallel
    This runs all your prompt combinations using parallel processing.
    
    **Performance:** Adjust max_workers based on API rate limits
    **Progress:** Shows real-time completion status
    **Error handling:** Failed tasks are captured with error messages
    """
    # Create response model from config
    response_model = create_response_model(config)
    response_field = config['output_format']['response_field']
    
    results = []
    completed = 0
    total = len(tasks)
    
    print(f"Running {total} tasks with {max_workers} parallel workers...")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_task = {
            executor.submit(run_single_prompt, task, response_model, response_field): task 
            for task in tasks
        }
        
        # Process completed tasks
        for future in as_completed(future_to_task):
            try:
                result = future.result()
                results.append(result)
                completed += 1
                print(f"Progress: {completed}/{total} ({completed*100//total}%)", end='\r')
            except Exception as exc:
                task = future_to_task[future]
                print(f"\nTask failed: {exc}")
                # Add error result
                results.append({
                    'sample_id': task['sample_id'],
                    'text_snippet': task['text_snippet'],
                    'true_label': task['true_label'],
                    'prompt_id': task['prompt_id'],
                    'model': task['model'],
                    'temperature': task['temperature'],
                    'prediction': f"error: {str(exc)}",
                    'timestamp': datetime.now().isoformat()
                })
    
    print(f"\n✓ Completed all {len(results)} tasks")
    return results

def save_results(results: List[Dict], output_path: str = "results") -> str:
    """
    Save results to CSV file
    
    Args:
        results: List of result dictionaries
        output_path: Base path for output files (without extension)
    
    Returns:
        Path to the saved CSV file
        
    Markdown comment:
    ## Step 8: Save Results
    Saves results as CSV with timestamp.
    
    **CSV:** Human-readable, works with Excel/Google Sheets
    **Timestamped:** Won't overwrite previous runs
    """
    df = pd.DataFrame(results)
    
    # Create timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f"{output_path}_{timestamp}.csv"
    
    # Save file
    df.to_csv(csv_path, index=False)
    
    return csv_path

def results_summary_dataframe(results: List[Dict]) -> pd.DataFrame:
    """
    Create a summary DataFrame of results with key metrics
    
    Args:
        results: List of result dictionaries
    
    Returns:
        DataFrame with success/failure summary and sample results
    """
    results_df = pd.DataFrame(results)
    
    # Calculate summary stats
    total_results = len(results)
    successful = len([r for r in results if not str(r['prediction']).startswith('error')])
    failed = total_results - successful
    success_rate = (successful / total_results * 100) if total_results > 0 else 0
    
    # Add summary stats as attributes
    results_df.attrs['total_results'] = total_results
    results_df.attrs['successful'] = successful
    results_df.attrs['failed'] = failed
    results_df.attrs['success_rate'] = success_rate
    
    return results_df

def create_results_table(results: List[Dict]):
    """
    Create and return a great-tables GT object for notebook display
    
    Args:
        results: List of result dictionaries
    
    Returns:
        GT table object that displays nicely in notebooks
    """
    from great_tables import GT
    
    df = pd.DataFrame(results)
    summary_data = []
    
    for prompt_id in sorted(df['prompt_id'].unique()):
        for model in sorted(df['model'].unique()):
            for temp in sorted(df['temperature'].unique()):
                subset = df[(df['prompt_id'] == prompt_id) & 
                           (df['model'] == model) & 
                           (df['temperature'] == temp)]
                
                # Count correct predictions (excluding errors)
                valid_subset = subset[~subset['prediction'].str.startswith('error')]
                if len(valid_subset) > 0:
                    correct = (valid_subset['prediction'] == valid_subset['true_label']).sum()
                    accuracy = correct / len(valid_subset)
                    
                    summary_data.append({
                        'Prompt': prompt_id,
                        'Model': model,
                        'Temperature': temp,
                        'Accuracy': accuracy,
                        'Correct': correct,
                        'Total': len(valid_subset)
                    })
    
    summary_df = pd.DataFrame(summary_data)
    
    if len(summary_df) > 0:
        # Create great-tables display
        gt_table = (
            GT(summary_df)
            .tab_header(
                title="Prompt Testing Results",
                subtitle=f"{len(summary_df)} combinations tested"
            )
            .fmt_percent(
                columns=["Accuracy"],
                decimals=1
            )
            .cols_align(
                align="center",
                columns=["Temperature", "Correct", "Total"]
            )
        )
        
        return gt_table
    else:
        return "No valid results to display"

def quick_analysis(results: List[Dict]) -> pd.DataFrame:
    """
    Generate a quick accuracy summary
    
    Args:
        results: List of result dictionaries
    
    Returns:
        DataFrame with accuracy by prompt/model/temperature
        
    Markdown comment:
    ## Step 9: Quick Analysis
    Shows which combinations work best.
    
    **Look for:**
    - Which prompts perform better?
    - How does temperature affect accuracy?
    - Are there significant model differences?
    """
    from great_tables import GT
    
    df = pd.DataFrame(results)
    summary_data = []
    
    for prompt_id in sorted(df['prompt_id'].unique()):
        for model in sorted(df['model'].unique()):
            for temp in sorted(df['temperature'].unique()):
                subset = df[(df['prompt_id'] == prompt_id) & 
                           (df['model'] == model) & 
                           (df['temperature'] == temp)]
                
                # Count correct predictions (excluding errors)
                valid_subset = subset[~subset['prediction'].str.startswith('error')]
                if len(valid_subset) > 0:
                    correct = (valid_subset['prediction'] == valid_subset['true_label']).sum()
                    accuracy = correct / len(valid_subset)
                    
                    summary_data.append({
                        'Prompt': prompt_id,
                        'Model': model,
                        'Temperature': temp,
                        'Accuracy': accuracy,
                        'Correct': correct,
                        'Total': len(valid_subset)
                    })
    
    summary_df = pd.DataFrame(summary_data)
    
    if len(summary_df) > 0:
        # Create a great-tables display
        gt_table = (
            GT(summary_df)
            .tab_header(
                title="Prompt Testing Results",
                subtitle="Accuracy by prompt, model, and temperature"
            )
            .fmt_percent(
                columns=["Accuracy"],
                decimals=1
            )
            .cols_align(
                align="center",
                columns=["Temperature", "Correct", "Total"]
            )
            .tab_style(
                style="background-color: #f0f8ff;",
                locations="body"
            )
        )
        
        # Save as HTML
        gt_table.save("quick_results.html")
        print("💾 Results table saved as: quick_results.html")
        
        # Show text summary
        for _, row in summary_df.iterrows():
            print(f"{row['Prompt']:20} | {row['Model']:15} | temp={row['Temperature']:3} | {row['Accuracy']:6.1%} ({row['Correct']}/{row['Total']})")
    
    return summary_df

def run_experiment(config_path: str, data_path: str, 
                  env_path: str = ".env", output_path: str = "results",
                  max_workers: int = 4) -> tuple:
    """
    Run complete experiment with specified paths
    
    Args:
        config_path: Path to JSON configuration file
        data_path: Path to CSV data file
        env_path: Path to .env file (default: ".env")
        output_path: Base path for output files (default: "results")
        max_workers: Number of parallel workers (default: 4)
    
    Returns:
        Tuple of (results_list, summary_dataframe, saved_file_path)
        
    Markdown comment:
    ## Run Complete Experiment
    This function chains all steps together with custom paths.
    
    **Usage:**
    ```python
    results, summary, file_path = run_experiment(
        config_path="my_config.json",
        data_path="my_data.csv",
        output_path="my_results"
    )
    ```
    
    **For notebooks:** Use individual functions above for step-by-step control
    """
    print("🚀 Starting modular prompt testing experiment")
    print("=" * 50)
    
    # Run all steps with provided paths
    load_environment(env_path)
    config = load_configuration(config_path)
    samples = load_dataset(data_path, config)
    tasks = create_tasks(config, samples)
    results = run_all_tasks(tasks, config, max_workers)
    file_path = save_results(results, output_path)
    summary = quick_analysis(results)
    
    print("\n🎉 Experiment complete!")
    return results, summary, file_path