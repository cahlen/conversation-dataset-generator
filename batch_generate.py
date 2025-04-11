#!/usr/bin/env python
# coding=utf-8

import argparse
import yaml
import subprocess
import os
import sys
import logging
import time

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [BatchScript] - %(message)s')

def construct_generate_command(run_config: dict, global_config: dict, run_index: int) -> list[str]:
    """Constructs the command line arguments for generate.py based on a run's config."""
    # Define the assumed path to the virtual environment's Python
    venv_python_path = "venv/bin/python" 
    # Check if it exists, otherwise fall back to sys.executable with a warning
    python_executable = venv_python_path if os.path.exists(venv_python_path) else sys.executable
    if python_executable != venv_python_path:
        logging.warning(f"Could not find python at {venv_python_path}. Falling back to {sys.executable}. Ensure dependencies are installed globally or adjust the path.")
    
    command = [python_executable, "generate.py"] # Use the determined python interpreter
    
    # --- Determine Mode and Extract Args ---
    
    # Mode 1: Creative Brief
    if 'creative_brief' in run_config:
        logging.info(f"  Run {run_index+1}: Detected Creative Brief Mode.")
        command.extend(["--creative-brief", run_config['creative_brief']])
        if run_config.get('persona1_search_term'):
            command.extend(["--persona1-search-term", run_config['persona1_search_term']])
        if run_config.get('persona2_search_term'):
            command.extend(["--persona2-search-term", run_config['persona2_search_term']])
            
    # Mode 2: Fixed Persona with Variation
    elif run_config.get('enable_variation') and 'fixed_personas' in run_config and 'initial_context' in run_config:
        logging.info(f"  Run {run_index+1}: Detected Fixed Persona + Variation Mode.")
        command.append("--enable-variation")
        
        fp = run_config['fixed_personas']
        ic = run_config['initial_context']
        
        required_fp = ['persona1', 'persona1_desc', 'persona2', 'persona2_desc']
        required_ic = ['topic', 'scenario', 'style']
        
        missing_fp = [k for k in required_fp if k not in fp]
        missing_ic = [k for k in required_ic if k not in ic]
        
        if missing_fp:
            logging.error(f"  Run {run_index+1}: Missing required keys in 'fixed_personas': {missing_fp}. Skipping run.")
            return None
        if missing_ic:
            logging.error(f"  Run {run_index+1}: Missing required keys in 'initial_context': {missing_ic}. Skipping run.")
            return None
            
        command.extend([
            "--fixed-persona1", fp['persona1'],
            "--fixed-persona1-desc", fp['persona1_desc'],
            "--fixed-persona2", fp['persona2'],
            "--fixed-persona2-desc", fp['persona2_desc'],
            "--initial-topic", ic['topic'],
            "--initial-scenario", ic['scenario'],
            "--initial-style", ic['style']
        ])
        # Add include_points if present in initial_context
        if ic.get('include_points'):
             command.extend(["--include-points", ic['include_points']])
             
    # Mode 3: Manual Mode
    elif 'manual_args' in run_config:
        logging.info(f"  Run {run_index+1}: Detected Manual Mode.")
        ma = run_config['manual_args']
        required_ma = ['persona1', 'persona1_desc', 'persona2', 'persona2_desc', 'topic', 'scenario', 'style']
        missing_ma = [k for k in required_ma if k not in ma]
        if missing_ma:
            logging.error(f"  Run {run_index+1}: Missing required keys in 'manual_args': {missing_ma}. Skipping run.")
            return None
            
        command.extend([
            "--persona1", ma['persona1'],
            "--persona1-desc", ma['persona1_desc'],
            "--persona2", ma['persona2'],
            "--persona2-desc", ma['persona2_desc'],
            "--topic", ma['topic'],
            "--scenario", ma['scenario'],
            "--style", ma['style']
        ])
        if ma.get('include_points'):
            command.extend(["--include-points", ma['include_points']])
            
    else:
        logging.error(f"  Run {run_index+1}: Cannot determine generation mode from config keys: {list(run_config.keys())}. Skipping run.")
        return None
        
    # --- Add Common Arguments ---
    
    # Output file (REQUIRED per run now)
    if 'output_file' not in run_config:
        logging.error(f"  Run {run_index+1}: Missing required 'output_file' argument. Skipping run.")
        return None
        
    # Prepend the global output directory path
    output_path = os.path.join(global_config.get('output_directory', '.'), run_config['output_file'])
    command.extend(["--output-file", output_path])
    
    # Optional args (num_examples, model_id, max_new_tokens, load_in_4bit, upload_to_hub, force_upload)
    if 'num_examples' in run_config:
        command.extend(["--num-examples", str(run_config['num_examples'])])
    if 'model_id' in run_config:
        command.extend(["--model-id", run_config['model_id']])
    if 'max_new_tokens' in run_config:
         command.extend(["--max-new-tokens", str(run_config['max_new_tokens'])])
    if run_config.get('load_in_4bit'):
        command.append("--load-in-4bit")
    if 'upload_repo' in run_config: # Check per-run upload first
        command.extend(["--upload-to-hub", run_config['upload_repo']])
        # Check for force upload at run level, then global
        if run_config.get('force_upload') or global_config.get('force_upload'):
             command.append("--force-upload")
    elif 'upload_repo' in global_config: # Fallback to global upload repo
        command.extend(["--upload-to-hub", global_config['upload_repo']])
        # Check for force upload at global level if using global repo
        if global_config.get('force_upload'):
            command.append("--force-upload")
            
    return command

def main():
    parser = argparse.ArgumentParser(description='Run multiple generate.py processes based on a YAML configuration file.')
    parser.add_argument('config_file', type=str, help='Path to the YAML configuration file.')
    args = parser.parse_args()

    # --- Load YAML Config ---
    try:
        with open(args.config_file, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logging.error(f"Configuration file not found: {args.config_file}")
        sys.exit(1)
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML file {args.config_file}: {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"An unexpected error occurred loading the config: {e}")
        sys.exit(1)

    # --- Validate Basic Config Structure ---
    if not isinstance(config, dict):
         logging.error("YAML config root must be a dictionary.")
         sys.exit(1)
    if 'runs' not in config or not isinstance(config['runs'], list):
        logging.error("YAML config must contain a 'runs' key with a list of run configurations.")
        sys.exit(1)
    if 'output_directory' not in config:
        logging.warning("YAML config does not contain 'output_directory'. Output files will be saved relative to the current directory.")
        config['output_directory'] = '.' # Default to current directory

    # --- Create Output Directory ---
    output_dir = config['output_directory']
    try:
        os.makedirs(output_dir, exist_ok=True)
        logging.info(f"Ensured output directory exists: {output_dir}")
    except OSError as e:
        logging.error(f"Could not create output directory {output_dir}: {e}")
        sys.exit(1)

    # --- Execute Runs ---
    num_runs = len(config['runs'])
    logging.info(f"Found {num_runs} run(s) defined in {args.config_file}.")
    
    overall_start_time = time.monotonic()
    success_count = 0
    fail_count = 0

    for i, run_conf in enumerate(config['runs']):
        run_id = run_conf.get('id', f'run_{i+1}')
        logging.info(f"--- Starting Run {i+1}/{num_runs} (ID: {run_id}) ---")
        run_start_time = time.monotonic()

        if not isinstance(run_conf, dict):
            logging.error(f"  Run {i+1}: Configuration is not a dictionary. Skipping.")
            fail_count += 1
            continue

        command_args = construct_generate_command(run_conf, config, i)

        if command_args:
            logging.info(f"  Executing command: {' '.join(command_args)}")
            try:
                # Execute generate.py as a subprocess
                # Pipe stdout and stderr to capture logs from generate.py
                process = subprocess.Popen(
                    command_args, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.PIPE, 
                    text=True, 
                    encoding='utf-8'
                )
                
                # Stream output for real-time feedback
                # Log stdout as INFO and stderr as WARNING from the batch script's perspective
                while True:
                    stdout_line = process.stdout.readline()
                    if stdout_line:
                        logging.info(f"  [generate.py] {stdout_line.strip()}")
                    stderr_line = process.stderr.readline()
                    if stderr_line:
                        logging.warning(f"  [generate.py] {stderr_line.strip()}")
                        
                    # Break if process finished and no more output
                    if process.poll() is not None and not stdout_line and not stderr_line:
                        break
                        
                return_code = process.returncode
                run_end_time = time.monotonic()
                run_duration = run_end_time - run_start_time

                if return_code == 0:
                    logging.info(f"  Run {i+1} (ID: {run_id}) completed successfully in {run_duration:.2f}s.")
                    success_count += 1
                else:
                    logging.error(f"  Run {i+1} (ID: {run_id}) failed with return code {return_code} after {run_duration:.2f}s.")
                    fail_count += 1
                    
            except FileNotFoundError:
                 logging.error(f"  Error executing run {i+1}: 'generate.py' script not found or Python interpreter '{sys.executable}' is invalid.")
                 fail_count += 1
            except Exception as e:
                 run_end_time = time.monotonic()
                 run_duration = run_end_time - run_start_time
                 logging.error(f"  An unexpected error occurred running generate.py for run {i+1} (ID: {run_id}): {e} (failed after {run_duration:.2f}s)")
                 fail_count += 1
        else:
            # construct_generate_command already logged the error
            logging.warning(f"  Skipping run {i+1} (ID: {run_id}) due to configuration errors.")
            fail_count += 1
            
        logging.info(f"--- Finished Run {i+1}/{num_runs} (ID: {run_id}) ---")

    overall_end_time = time.monotonic()
    overall_duration = overall_end_time - overall_start_time
    logging.info("=" * 40)
    logging.info("Batch Processing Summary")
    logging.info(f"Total Runs: {num_runs}")
    logging.info(f"Successful Runs: {success_count}")
    logging.info(f"Failed Runs: {fail_count}")
    logging.info(f"Total Batch Time: {overall_duration:.2f}s")
    logging.info("=" * 40)

if __name__ == "__main__":
    main() 