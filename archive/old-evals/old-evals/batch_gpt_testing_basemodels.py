import openai
import pandas as pd
import os

# Make sure your API key is set
# personal API key 
openai.api_key = os.getenv("OPENAI_API_KEY")  # Ensure your API key is set in the environment variable

def get_available_base_models():
    """Return a list of available base OpenAI models for testing"""
    try:
        print("Retrieving available models from OpenAI API...")
        response = openai.models.list()
        
        # Filter for base models (exclude fine-tuned models)
        base_models = []
        for model in response.data:
            model_id = model.id
            # Include common base models and exclude fine-tuned ones
            if (model_id.startswith('gpt-') and 
                not model_id.startswith('ft:') and 
                not ':' in model_id and
                not model_id.endswith('-instruct') and
                not model_id.endswith('-beta')):
                base_models.append(model_id)
        
        # Sort models for consistent ordering
        base_models.sort()
        
        print(f"Found {len(base_models)} available base models")
        return base_models
        
    except Exception as e:
        print(f"Error retrieving models from API: {e}")
        print("Falling back to hardcoded list...")
        # Fallback to hardcoded list if API call fails
        fallback_models = [
            "gpt-4o",
            "gpt-4o-mini", 
            "gpt-4-turbo",
            "gpt-4",
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-16k"
        ]
        return fallback_models

def main():
    base_models = get_available_base_models()
    
    print("Available base models:")
    for idx, model_id in enumerate(base_models):
        print(f"{idx+1}. {model_id}")

    # Allow user to select multiple models
    print("\nYou can select one or multiple models to test.")
    print("Enter model numbers separated by commas (e.g., 1,3,5) or just one number:")
    
    try:
        choices = input("Enter your choice(s): ").strip()
        selected_indices = [int(x.strip()) - 1 for x in choices.split(',')]
        
        # Validate selections
        selected_models = []
        for idx in selected_indices:
            if 0 <= idx < len(base_models):
                selected_models.append(base_models[idx])
            else:
                print(f"Invalid selection: {idx + 1}")
                return
        
        if not selected_models:
            print("No valid models selected.")
            return
            
        print(f"\nSelected model(s): {', '.join(selected_models)}")
        
    except ValueError:
        print("Invalid input. Please enter numbers separated by commas.")
        return

    input_csv_path = input("Enter the path to the input CSV file: ")
    output_csv_path = input("Enter the name for the output CSV file: ")

    if not output_csv_path.lower().endswith('.csv'):
        output_csv_path += '.csv'

    print(f"Output will be saved to: {output_csv_path}")

    try:
        df = pd.read_csv(input_csv_path)
    except Exception as e:
        print(f"Error reading input CSV file: {e}")
        return

    if 'Text Input' not in df.columns:
        print("Error: CSV must contain a 'Text Input' column.")
        return

    total_prompts = len(df)
    total_runs = len(selected_models) * 5  # 5 runs per model
    print(f"\nProcessing {total_prompts} prompts with {len(selected_models)} model(s) - {total_runs} total runs...")

    # Process each model
    for model_idx, model_name in enumerate(selected_models):
        print(f"\n{'='*50}")
        print(f"Testing Model {model_idx + 1}/{len(selected_models)}: {model_name}")
        print(f"{'='*50}")
        
        # Test each model 5 times
        for run_num in range(1, 6):
            print(f"\n--- Run {run_num}/5 ---")
            
            responses = []
            for i, prompt in enumerate(df['Text Input'], 1):
                print(f"Processing prompt {i}/{total_prompts} ({(i/total_prompts)*100:.1f}%)")
                try:
                    response = openai.chat.completions.create(
                        model=model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=1
                    )
                    message = response.choices[0].message.content
                    print(f"  ✓ Success")
                except Exception as e:
                    message = f"Error: {str(e)}"
                    print(f"  ✗ Error: {str(e)}")

                responses.append(message)

            # Add responses to dataframe with model name and run number
            column_name = f'Response_{model_name.replace("-", "_")}_Run{run_num}'
            df[column_name] = responses

    # Save results
    try:
        abs_output_path = os.path.abspath(output_csv_path)
        print(f"\nAttempting to save to: {abs_output_path}")

        df.to_csv(abs_output_path, index=False)
        print(f"Output successfully saved to: {abs_output_path}")

        if os.path.exists(abs_output_path):
            file_size = os.path.getsize(abs_output_path)
            print(f"File created successfully. Size: {file_size} bytes")
        else:
            print("Warning: File was not created at the specified location")

    except Exception as e:
        print(f"\nError saving output file: {e}")
        print("Attempting to save to current directory...")
        try:
            current_dir_output = os.path.basename(output_csv_path)
            df.to_csv(current_dir_output, index=False)
            print(f"Output saved to current directory as: {current_dir_output}")
        except Exception as e2:
            print(f"Failed to save to current directory: {e2}")

if __name__ == "__main__":
    main()
