#!/usr/bin/env python
# coding=utf-8

import argparse
import json
import re # Import regex module
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig # Use HF pipeline and add model/quantization imports
import logging
import os # Import os for path manipulation
from huggingface_hub import login, HfApi, HfFolder # Import necessary hub utilities
import io # For creating README in memory for upload
from datasets import load_dataset, DatasetDict, DatasetInfo, Features, Value
import sys # For exiting script gracefully
import time # For performance metrics
from duckduckgo_search import DDGS # Import for web search
import pandas as pd # Added for pandas import
from datasets.utils.logging import set_verbosity_error as datasets_set_verbosity_error
from transformers.utils import logging as transformers_logging

# Setup basic logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constants ---
ARG_GENERATION_SYSTEM_PROMPT = """You are an expert creative assistant specializing in setting up parameters for a conversational dialogue generation script. Your goal is to take a user's request, which typically names two entities (characters, concepts, etc.), and generate a complete set of *realistic* and *grounded* arguments suitable for the `generate.py` script.

The `generate.py` script requires the following arguments:
--persona1 "<name>"
--persona1-desc "<detailed description, including potential conversational tics or speech patterns>"
--persona2 "<name>"
--persona2-desc "<detailed description, including potential conversational tics or speech patterns>"
--topic "<plausible conversation topic>"
--scenario "<realistic setting/context>"
--style "<dialogue style/tone - aim for natural interaction>"
--include-points "<comma-separated keywords>" (Optional, but helpful)

Your Task:
1.  Analyze the user's request (e.g., "Generate a conversation between Character A and Character B").
2.  Identify the two main personas.
3.  Write detailed descriptions (`--persona1-desc`, `--persona2-desc`) that capture the essence of each persona. **Crucially, consider the context or era implied by the request (e.g., 'Snoop Dogg debating rap' implies his 90s G-Funk persona). Focus on how they might *actually speak*, incorporating characteristic slang, attitude, and speech patterns specific to that persona/era.** Include potential conversational tics (e.g., uses metaphors often, tends to interrupt, speaks formally, uses slang).
4.  Determine a *plausible* `--topic` for their conversation, grounded in their personas, a shared interest, recent event, or a potential point of disagreement. Avoid overly abstract philosophical debates unless specifically requested.
5.  Define a *realistic* `--scenario` (setting/context) where this conversation might naturally occur (e.g., backstage at an event, a chance meeting at an airport, during a shared meal).
6.  Describe the desired `--style` of the dialogue, focusing on natural interaction (e.g., 'casual chat', 'polite disagreement', 'awkward small talk', 'technical discussion', 'friendly debate').
7.  (Optional but Recommended) List relevant keywords or concepts as `--include-points` that should ideally appear *naturally* in the conversation.
8.  Format your entire output *strictly* as a list of key-value pairs, matching the script arguments exactly as shown above, with each argument on a new line. Use double quotes around the values. Do not include any other explanatory text, preamble, or markdown formatting in your final output.

Example Input (User Request):
"Generate a conversation between a grumpy cat and an overly enthusiastic golden retriever."

Example Output (Your Response - updated for realism):
--persona1 "Barnaby"
--persona1-desc "An older tabby cat. Values quiet. Communicates with sighs and minimal meows. Often stares blankly when annoyed. Might occasionally twitch his tail."
--persona2 "Sunshine"
--persona2-desc "A young golden retriever. High energy. Often whines or yips excitedly. Tail always wagging. Speaks in enthusiastic barks, sometimes interrupting. Easily distracted by potential play."
--topic "Trying to share the same small patch of sun"
--scenario "Both attempting to lie down on a rug near a sunny window"
--style "Comedic contrast, annoyed brevity vs. cheerful rambling, focus on actions and sounds"
--include-points "sunbeam, nap, tail wag, sigh, bark, space"

Now, analyze the user's request and generate the arguments for `generate.py`.
"""

# New prompt for generating topic/scenario variations
TOPIC_VARIATION_SYSTEM_PROMPT = """You are an expert creative assistant helping refine parameters for dialogue generation. You will be given an original creative brief, the fixed personas (names and descriptions) derived from it, and the initial topic/scenario/style.

Your Task:
1.  Review the original brief and the fixed personas.
2.  Generate a *new*, *related but distinct* `--topic` and `--scenario` for a conversation between these specific personas. The new topic/scenario should fit the spirit of the original brief but offer variety.
3.  Optionally, suggest a slightly adjusted `--style` if it makes sense for the new topic/scenario, otherwise reuse the original style.
4.  Format your output *strictly* as key-value pairs for `--topic`, `--scenario`, and `--style` ONLY. Use double quotes around the values. Each on a new line. Do not include any other text.

Example Input Context:
Original Brief: "A classic Seinfeldian conversation between Jerry and George... Jerry makes an absurd observation... George escalates..."
Persona 1: Jerry Seinfeld (observational comedian...)
Persona 2: George Costanza (neurotic, anxious...)
Initial Topic: "The annoyance of people who don't use turn signals"
Initial Scenario: "At Monk's diner..."
Initial Style: "Comedic banter..."

Example Output (Your Response - generating variation):
--topic "The strategic placement of condiments at a diner table"
--scenario "Still at Monk's diner, waiting for food"
--style "Observational comedy, escalating neuroticism about minor details"

Now, analyze the provided context and generate a new topic, scenario, and style variation.
"""

# Helper function for web search
def get_persona_context_from_web(persona_name: str, max_results: int = 3) -> str:
    """Performs a web search for a persona name and returns concatenated snippets."""
    logging.info(f"  Performing web search for: {persona_name}")
    context = "No relevant web context found." # Default
    try:
        with DDGS() as ddgs:
            query = f"Who is {persona_name}? Background, personality, notable traits."
            results = list(ddgs.text(query, max_results=max_results))
            
        if results:
            snippets = [f"- {r['body']}" for r in results if r.get('body')]
            if snippets:
                context = "\n".join(snippets)
                logging.info(f"    Found {len(snippets)} snippets for {persona_name}.")
            else:
                logging.warning(f"    Web search for {persona_name} yielded results but no usable text snippets.")
        else:
            logging.warning(f"    No web search results found for {persona_name}.")
            
    except Exception as e:
        logging.error(f"    Error during web search for {persona_name}: {e}")
        
    # Log the actual context being returned
    logging.debug(f"    Context returned for {persona_name}:\n{context}") 
    return context

# NEW Helper function for image search
def get_persona_image_url(persona_name: str) -> str | None:
    """Performs an image search for a persona name and returns the URL of the first result."""
    logging.info(f"  Performing image search for: {persona_name}")
    try:
        with DDGS() as ddgs:
            # Use DDGS().images()
            results = list(ddgs.images(
                keywords=persona_name,
                region="wt-wt",
                safesearch="moderate",
                size=None,
                color=None,
                type_image=None,
                layout=None,
                license_image=None,
                max_results=1  # We only need the first good result
            ))
        
        if results and results[0].get('image'):
            image_url = results[0]['image']
            logging.info(f"    Found image URL for {persona_name}: {image_url}")
            return image_url
        else:
            logging.warning(f"    No image results found for {persona_name}.")
            return None
            
    except Exception as e:
        logging.error(f"    Error during image search for {persona_name}: {e}")
        return None

def generate_args_from_brief(brief: str, generator_pipeline, tokenizer, p1_search_term: str | None, p2_search_term: str | None) -> dict | None:
    """Uses the LLM to generate detailed arguments from a high-level brief, optionally using user-provided search terms for context."""
    logging.info(f"Generating detailed args for brief: \"{brief}\"")
    
    persona1_context = "No search term provided."
    persona2_context = "No search term provided."
    search_term1 = p1_search_term
    search_term2 = p2_search_term

    # --- Perform Web Search if terms provided --- 
    if p1_search_term:
        logging.info(f"  Persona 1 Search Term provided: '{p1_search_term}'")
        persona1_context = get_persona_context_from_web(p1_search_term)
    else:
        logging.info("  No search term provided for Persona 1.")
        search_term1 = "Persona 1" # Placeholder for prompt

    if p2_search_term:
        logging.info(f"  Persona 2 Search Term provided: '{p2_search_term}'")
        persona2_context = get_persona_context_from_web(p2_search_term)
    else:
        logging.info("  No search term provided for Persona 2.")
        search_term2 = "Persona 2" # Placeholder for prompt

    # --- Generate Full Arguments (with or without web context) ---
    logging.info("  Generating full arguments using LLM...")
    system_prompt_content = ARG_GENERATION_SYSTEM_PROMPT # Base prompt
    user_prompt_content = brief # Original user brief
    
    # Inject context if web search was performed (i.e., terms were provided)
    if p1_search_term or p2_search_term:
        system_prompt_content += "\n\n--- Additional Context from Web Search ---"
        # Use the actual search term in the prompt header for clarity
        system_prompt_content += f"\nContext for '{search_term1}':\n{persona1_context}"
        system_prompt_content += f"\nContext for '{search_term2}':\n{persona2_context}"
        system_prompt_content += "\n--- End of Context ---"
        system_prompt_content += "\n\nNow, analyze the user's request and generate the arguments for generate.py, using the provided web context to inform the --persona1-desc and --persona2-desc where appropriate, ensuring they are realistic and grounded."

    messages = [
        {"role": "system", "content": system_prompt_content},
        {"role": "user", "content": user_prompt_content}
    ]
    
    max_retries = 3
    initial_delay = 1 # seconds

    for attempt in range(max_retries):
        try:
            prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception as e:
            # If template fails, unlikely to succeed on retry, log and exit
            logging.error(f"Failed to apply chat template for argument generation (attempt {attempt+1}/{max_retries}): {e}")
            return None

        try:
            logging.info(f"  LLM Arg Generation - Attempt {attempt+1}/{max_retries}...")
            # Increased max_new_tokens slightly to account for potentially longer context-informed descriptions
            outputs = generator_pipeline(
                prompt_text,
                max_new_tokens=600, 
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
                num_return_sequences=1
            )
            
            if outputs and isinstance(outputs, list) and 'generated_text' in outputs[0]:
                if prompt_text in outputs[0]['generated_text']:
                    raw_args_text = outputs[0]['generated_text'][len(prompt_text):].strip()
                else:
                    logging.warning("Prompt text not found in arg generation response. Using full output.")
                    raw_args_text = outputs[0]['generated_text'].strip()
                
                logging.debug(f"Raw arguments generated by LLM (attempt {attempt+1}):\n{raw_args_text}")
                
                generated_args = {}
                arg_pattern = re.compile(r'^--(persona1|persona1-desc|persona2|persona2-desc|topic|scenario|style|include-points)\s+"(.*?)"$', re.MULTILINE | re.DOTALL)
                matches = arg_pattern.findall(raw_args_text)
                
                if not matches:
                    logging.warning(f"Could not parse any arguments from LLM response (attempt {attempt+1}). Response:\n{raw_args_text}")
                    # On final attempt, continue anyway - we'll handle missing keys later
                    if attempt >= max_retries - 1:
                        logging.warning(f"Final attempt has missing keys: {raw_args_text}. Proceeding anyway.")
                        return generated_args
                    # Go to retry logic
                    raise ValueError("Parsing failed: No arguments found")
                    
                for key, value in matches:
                    cleaned_value = value.strip()
                    generated_args[key.replace('-', '_')] = cleaned_value
                
                required_keys = ['persona1', 'persona1_desc', 'persona2', 'persona2_desc', 'topic', 'scenario', 'style']
                missing_keys = [key for key in required_keys if key not in generated_args]
                if missing_keys:
                    logging.warning(f"LLM response missing required arguments (attempt {attempt+1}): {missing_keys}. Parsed args: {generated_args}")
                    # On final attempt, continue anyway - we'll handle missing keys later
                    if attempt >= max_retries - 1:
                        logging.warning(f"Final attempt has missing keys: {missing_keys}. Proceeding anyway.")
                        return generated_args
                    # Go to retry logic
                    raise ValueError(f"Parsing failed: Missing keys {missing_keys}")
                    
                logging.info("Successfully generated and parsed detailed arguments from brief.")
                return generated_args # Success!
                
            else:
                logging.warning(f"Unexpected output format from LLM during argument generation (attempt {attempt+1}).")
                # Go to retry logic
                raise ValueError("LLM Error: Unexpected output format")

        except Exception as e:
            logging.warning(f"Error during LLM call or parsing for argument generation (attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                delay = initial_delay * (2 ** attempt)
                logging.info(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                logging.error(f"Argument generation failed after {max_retries} attempts.")
                return None # Max retries reached

    # Should not be reached if loop completes, but return None as a fallback
    return None

def generate_args_from_brief_safe(brief: str, generator_pipeline, tokenizer, p1_search_term: str | None, p2_search_term: str | None) -> dict | None:
    """A wrapper around generate_args_from_brief that provides sensible defaults for missing fields."""
    generated_args = generate_args_from_brief(brief, generator_pipeline, tokenizer, p1_search_term, p2_search_term)
    
    # If we couldn't generate any args at all, return None
    if generated_args is None:
        return None
    
    # Check for required fields and provide defaults if missing
    if 'persona1' not in generated_args:
        logging.error("Failed to generate persona1 name. Cannot proceed.")
        return None
        
    if 'persona2' not in generated_args:
        logging.error("Failed to generate persona2 name. Cannot proceed.")
        return None
    
    # Handle missing persona descriptions with sensible defaults
    if 'persona1_desc' not in generated_args:
        persona1 = generated_args.get('persona1')
        generated_args['persona1_desc'] = f"A character named {persona1}. Speaks naturally with a distinct voice."
        logging.warning(f"persona1_desc was missing. Using default description: '{generated_args['persona1_desc']}'")
    
    if 'persona2_desc' not in generated_args:
        persona2 = generated_args.get('persona2')
        generated_args['persona2_desc'] = f"A character named {persona2}. Speaks naturally with a distinct voice."
        logging.warning(f"persona2_desc was missing. Using default description: '{generated_args['persona2_desc']}'")
    
    # Check for other required fields with simple defaults
    if 'topic' not in generated_args:
        generated_args['topic'] = "A casual conversation"
        logging.warning(f"topic was missing. Using default: '{generated_args['topic']}'")
    
    if 'scenario' not in generated_args:
        generated_args['scenario'] = "A neutral setting where the two personas meet"
        logging.warning(f"scenario was missing. Using default: '{generated_args['scenario']}'")
    
    if 'style' not in generated_args:
        generated_args['style'] = "Natural, casual conversation"
        logging.warning(f"style was missing. Using default: '{generated_args['style']}'")
    
    return generated_args

def create_generation_prompt(
    topic: str, 
    persona1: str, 
    persona2: str, 
    persona1_desc: str, 
    persona2_desc: str, 
    scenario: str, 
    style: str,
    include_points: str | None
) -> list[dict]:
    """Creates prompt messages for the LLM, emphasizing naturalness."""
    system_message = f"""You are a creative assistant skilled at generating *realistic* and *natural-sounding* conversational dialogue between two described personas in a specific scenario.
The conversation should be between {persona1} ({persona1_desc}) and {persona2} ({persona2_desc}).
The scenario is: '{scenario}'.
The central topic is: '{topic}'.
The requested interaction style is: '{style}'.

**IMPORTANT: Aim for a realistic, spontaneous conversation.** Avoid overly formal, dramatic, philosophical, or scripted-sounding language unless truly fitting for the specific personas and situation. Incorporate natural conversational elements like brief pauses, slight hesitations (use '...' sparingly), agreeing/disagreeing naturally, or occasional minor topic shifts. Focus on realism over perfect grammatical structure or constant back-and-forth argument. The conversation should have a **natural length, perhaps 5-15 turns.**

Start the output directly with the first turn (e.g., '{persona1}: ...'). Do not include any preamble or explanatory text outside the dialogue turns.""" # Updated goals and turn count

    # User message focuses on the task + optional points
    user_request = f"""Generate the conversation now, following all the instructions in the system message, especially the emphasis on naturalness and realism. Make sure each turn starts with either '{persona1}:' or '{persona2}:'."""

    # Add instruction for included points if provided
    if include_points:
        points_list = [p.strip() for p in include_points.split(',') if p.strip()]
        if points_list:
            points_instruction = f" Try to naturally incorporate discussion of the following points/keywords if possible: {', '.join(points_list)}."
            user_request += points_instruction # Add to user request

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_request}
    ]
    logging.debug(f"Generated messages: {messages}")
    return messages

def parse_conversation_to_sharegpt(conversation_text: str, persona1: str, persona2: str) -> tuple[list[dict] | None, str | None, str | None]:
    """Parses raw conversation text into ShareGPT turn structure.

    Args:
        conversation_text: The raw text output from the LLM.
        persona1: The name of the first speaker (maps to 'human').
        persona2: The name of the second speaker (maps to 'gpt').

    Returns:
        A tuple containing:
        - A list of dictionaries with turns, e.g., [{'from': 'human', 'value': '...'}, {'from': 'gpt', 'value': '...'}], or None if parsing fails
        - The name of the first speaker (persona1) or None if parsing fails
        - The name of the second speaker (persona2) or None if parsing fails
    """
    logging.debug(f"Parsing text for ShareGPT structure (length {len(conversation_text)}): {conversation_text[:100]}...")
    
    conversations = []
    # Regex to capture speaker and their dialogue, ignoring leading/trailing whitespace
    # Handles potential variations in spacing after the colon
    turn_pattern = re.compile(rf"^\s*({re.escape(persona1)}|{re.escape(persona2)})\s*:\s*(.*)", re.IGNORECASE | re.MULTILINE)
    
    current_turn_data = None
    processed_text = ""

    for line in conversation_text.strip().split('\n'):
        line_stripped = line.strip()
        if not line_stripped:
            continue

        match = turn_pattern.match(line_stripped)
        if match:
            # If we were accumulating text for a previous turn, add it now
            if current_turn_data:
                current_turn_data['value'] = processed_text.strip()
                # Only add if value is not empty
                if current_turn_data['value']:
                     conversations.append(current_turn_data)
                else:
                    logging.warning(f"Skipping empty turn for speaker {current_turn_data.get('from')}")
            
            # Start processing the new turn
            speaker = match.group(1).strip()
            value_start = match.group(2).strip()
            
            # Map speaker to role
            role = None
            if speaker.lower() == persona1.lower():
                role = "human"
            elif speaker.lower() == persona2.lower():
                role = "gpt"
            else:
                logging.warning(f"Unknown speaker found: {speaker} (Expected {persona1} or {persona2}) - Skipping turn")
                current_turn_data = None # Reset
                processed_text = "" # Reset
                continue
                
            current_turn_data = {"from": role, "value": None} # Value will be filled
            processed_text = value_start # Start accumulating text for this turn
            
        elif current_turn_data: 
            # This line is a continuation of the previous speaker's turn
            processed_text += "\n" + line_stripped # Preserve potential line breaks within a turn
        else:
             # This line doesn't match the start pattern and we aren't currently processing a turn
             logging.warning(f"Skipping unmatched line (not part of a turn): {line_stripped}")

    # Add the last captured turn if it exists and has content
    if current_turn_data:
        current_turn_data['value'] = processed_text.strip()
        if current_turn_data['value']:
            conversations.append(current_turn_data)
        else:
            logging.warning(f"Skipping empty final turn for speaker {current_turn_data.get('from')}")


    if not conversations:
        logging.warning("Could not parse any valid turns from the conversation text.")
        return None, None, None # Return None for turns and names
        
    # Basic validation: Ensure we have at least one human and one gpt turn if possible
    roles_present = {turn['from'] for turn in conversations}
    if len(conversations) > 1 and not ('human' in roles_present and 'gpt' in roles_present):
         logging.warning(f"Conversation parsed but missing expected roles (human/gpt). Roles found: {roles_present}")

    # Return turns and the actual names used for parsing
    # Assume persona1 maps to human, persona2 maps to gpt consistently as defined
    return conversations, persona1, persona2

# New function to generate topic/scenario variations
def generate_topic_variation(original_brief: str, persona1: str, persona1_desc: str, persona2: str, persona2_desc: str, initial_topic: str, initial_scenario: str, initial_style: str, generator_pipeline, tokenizer) -> dict | None:
    """Uses the LLM to generate a variation on topic/scenario/style given fixed personas."""
    logging.info("  Attempting to generate topic/scenario variation...")
    context = f"Original Brief: \"{original_brief}\"\nPersona 1: {persona1} ({persona1_desc})\nPersona 2: {persona2} ({persona2_desc})\nInitial Topic: \"{initial_topic}\"\nInitial Scenario: \"{initial_scenario}\"\nInitial Style: \"{initial_style}\""
    
    messages = [
        {"role": "system", "content": TOPIC_VARIATION_SYSTEM_PROMPT},
        {"role": "user", "content": context} # Provide context as user message
    ]
    
    try:
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception as e:
        logging.error(f"    Failed to apply chat template for topic variation: {e}")
        return None

    try:
        outputs = generator_pipeline(
            prompt_text,
            max_new_tokens=256, # Shorter max length needed
            do_sample=True,
            temperature=0.7, # Keep slightly creative but focused
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=1
        )
        
        if outputs and isinstance(outputs, list) and 'generated_text' in outputs[0]:
            raw_variation_text = outputs[0]['generated_text'][len(prompt_text):].strip()
            logging.debug(f"    Raw variation generated by LLM:\n{raw_variation_text}")
            
            # Parse the generated variation string (topic, scenario, style)
            variation_args = {}
            arg_pattern = re.compile(r'^--(topic|scenario|style)\s+"(.*?)"$', re.MULTILINE | re.DOTALL)
            matches = arg_pattern.findall(raw_variation_text)
            
            if not matches or len(matches) < 2: # Require at least topic and scenario
                logging.error(f"    Could not parse topic/scenario variation from LLM response. Response:\n{raw_variation_text}")
                return None
                
            for key, value in matches:
                variation_args[key] = value.strip()
            
            # Ensure essential keys are present
            if 'topic' not in variation_args or 'scenario' not in variation_args:
                 logging.error(f"    LLM variation response missing topic or scenario. Parsed: {variation_args}")
                 return None
                 
            # Use initial style if not generated
            if 'style' not in variation_args:
                variation_args['style'] = initial_style
                logging.debug("    Variation LLM did not provide style, reusing initial style.")
                
            logging.info("    Successfully generated and parsed topic/scenario variation.")
            return variation_args
            
        else:
            logging.error("    Unexpected output format from LLM during topic variation generation.")
            return None
            
    except Exception as e:
        logging.error(f"    Error during LLM call for topic variation: {e}")
        return None

if __name__ == "__main__":
    script_start_time = time.monotonic() # Time the whole script
    
    # --- Argument Parsing --- 
    parser = argparse.ArgumentParser(
        description='Generate synthetic conversational data or manage Hugging Face dataset repositories. Provide EITHER --creative-brief OR detailed arguments for generation, OR --delete-repo for deletion.',
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # --- Mode Selection Arguments (Optional - logic determines mode) ---
    # Creative Brief Mode trigger
    parser.add_argument(
        '--creative-brief', 
        type=str, 
        help='High-level creative brief (e.g., "Predator talks to Jerry Seinfeld"). Triggers automatic argument generation.'
    )
    
    # Deletion Mode trigger
    parser.add_argument(
        '--delete-repo', 
        type=str, 
        nargs='+', # Accept one or more arguments
        metavar='USERNAME/REPO_ID', 
        help='Specify one or more Hugging Face Hub dataset repository IDs to delete permanently. \nExample: --delete-repo user/repo1 user/repo2\nTHIS ACTION IS IRREVERSIBLE.'
    )

    # --- Argument Groups (Used based on mode) ---
    # Group for detailed manual arguments (Used if --creative-brief and --delete-repo are NOT provided)
    detailed_group = parser.add_argument_group('Detailed Arguments (Manual Generation Mode)')
    detailed_group.add_argument('--topic', type=str, help='Topic for the conversations.')
    detailed_group.add_argument('--persona1', type=str, help='Name of the first speaker (maps to human role).')
    detailed_group.add_argument('--persona1-desc', type=str, help='Description of the first persona.')
    detailed_group.add_argument('--persona2', type=str, help='Name of the second speaker (maps to gpt role).')
    detailed_group.add_argument('--persona2-desc', type=str, help='Description of the second persona.')
    detailed_group.add_argument('--scenario', type=str, help='Scenario/context for the conversation.')
    detailed_group.add_argument('--style', type=str, help='Style/tone of the conversation.')
    detailed_group.add_argument('--include-points', type=str, default=None, help='Comma-separated list of points/keywords to include.')

    # NEW group for Brief Mode context searching
    brief_context_group = parser.add_argument_group('Creative Brief Web Context (Optional)')
    brief_context_group.add_argument('--persona1-search-term', type=str, default=None, help='Optional web search term to gather context for Persona 1 when using --creative-brief.')
    brief_context_group.add_argument('--persona2-search-term', type=str, default=None, help='Optional web search term to gather context for Persona 2 when using --creative-brief.')

    # --- General Arguments (Applicable to Generation Modes) ---
    general_group = parser.add_argument_group('General Generation Arguments')
    general_group.add_argument('--num-examples', type=int, default=3, help='Number of conversation examples to generate.')
    general_group.add_argument('--output-file', type=str, default='generated_data.jsonl', help='Output file path (JSON Lines format).')
    general_group.add_argument('--model-id', type=str, default='meta-llama/Meta-Llama-3-8B-Instruct', help='Hugging Face model ID for ALL generation tasks.')
    general_group.add_argument('--max-new-tokens', type=int, default=768, help='Max new tokens for the conversation generation step.')
    general_group.add_argument('--upload-to-hub', type=str, default=None, metavar='USERNAME/REPO_ID', help='Optional: Repository ID to upload the dataset to on Hugging Face Hub. Requires prior login.')
    general_group.add_argument('--load-in-4bit', action='store_true', help='Enable 4-bit quantization (NF4) for model loading to reduce memory and potentially speed up inference. Requires `bitsandbytes`.')
    general_group.add_argument('--force-upload', action='store_true', help='Force upload to Hugging Face Hub without interactive confirmation.')
    general_group.add_argument('--validate-local-save', action='store_true', help='Load dataset back from local disk after saving for basic validation.')

    args = parser.parse_args()

    # --- Mode Execution ---
    # 1. Deletion Mode
    if args.delete_repo:
        delete_start_time = time.monotonic()
        repo_ids_to_delete = args.delete_repo
        logging.warning(f"\n*** WARNING: You are about to permanently delete the Hugging Face Hub repositories: {', '.join(repo_ids_to_delete)} ***")
        logging.warning("*** This action CANNOT be undone. All data in the repositories will be lost. ***")
        try:
            confirm = input(f"Type 'yes' to confirm deletion of {', '.join(repo_ids_to_delete)}, or anything else to cancel: ")
            if confirm.lower() == 'yes':
                logging.info(f"Attempting to delete repositories: {', '.join(repo_ids_to_delete)}")
                token = HfFolder.get_token()
                if not token:
                    logging.error("Hugging Face token not found. Please login using `huggingface-cli login`.")
                    sys.exit(1)
                api = HfApi()
                deleted_count = 0
                failed_repos = []
                repo_delete_start_time = time.monotonic()
                for repo_id in repo_ids_to_delete:
                    single_repo_start = time.monotonic()
                    try:
                        api.delete_repo(repo_id=repo_id, token=token, repo_type='dataset')
                        single_repo_end = time.monotonic()
                        logging.info(f"  Successfully deleted repository: {repo_id} (took {single_repo_end - single_repo_start:.2f}s)")
                        deleted_count += 1
                    except Exception as e:
                        single_repo_end = time.monotonic()
                        logging.error(f"  Failed to delete repository {repo_id}: {e} (attempt took {single_repo_end - single_repo_start:.2f}s)")
                        failed_repos.append(repo_id)
                repo_delete_end_time = time.monotonic()
                logging.info(f"Batch deletion process took {repo_delete_end_time - repo_delete_start_time:.2f}s total.")
                if failed_repos:
                    logging.warning(f"Completed deletion attempts. Success: {deleted_count}, Failed: {len(failed_repos)} ({', '.join(failed_repos)}) ")
                else:
                    logging.info(f"Successfully deleted all {deleted_count} specified repositories.")
            else:
                logging.info("Deletion cancelled by user.")
        except EOFError:
             logging.warning("Could not get user input (EOFError). Deletion cancelled.")
        except Exception as e:
             logging.warning(f"An error occurred during confirmation or deletion process: {e}. Deletion may be incomplete.")
        delete_end_time = time.monotonic()
        logging.info(f"Deletion mode finished in {delete_end_time - delete_start_time:.2f}s.")
        sys.exit(0)

    # 2. Generation Modes (Check if delete was not performed)
    text_generator = None
    tokenizer = None
    model_load_start_time = time.monotonic()
    model_loaded = False
    model = None # Initialize model variable

    # --- Load Model --- (Load once for all generation modes)
    logging.info(f"Loading tokenizer: {args.model_id}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_id)
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

        if args.load_in_4bit:
            logging.info(f"Loading model: {args.model_id} with 4-bit quantization (NF4)")
            # Define quantization config
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                bnb_4bit_use_double_quant=False # Optional, can be True for slightly more accuracy at cost of VRAM
            )
            model = AutoModelForCausalLM.from_pretrained(
                args.model_id,
                quantization_config=quantization_config,
                device_map="auto", # Accelerate handles placement
                trust_remote_code=True
            )
            logging.info("Model loaded with 4-bit quantization.")
        else:
            logging.info(f"Loading model: {args.model_id} with default precision (bf16/fp16)")
            model = AutoModelForCausalLM.from_pretrained(
                args.model_id,
                torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            logging.info("Model loaded with default precision.")

        # Now create the pipeline using the pre-loaded model and tokenizer
        logging.info("Creating text-generation pipeline...")
        text_generator = pipeline(
            "text-generation",
            model=model, # Pass the loaded model object
            tokenizer=tokenizer,
            # device_map is implicitly handled by the loaded model's device map
            # torch_dtype is implicitly handled by the loaded model's dtype
        )

        model_load_end_time = time.monotonic()
        logging.info(f"Tokenizer and model pipeline created successfully (took {model_load_end_time - model_load_start_time:.2f}s). Using 4-bit: {args.load_in_4bit}")
        model_loaded = True
    except ImportError as ie:
        # Specific error if bitsandbytes is missing and 4bit was requested
        if args.load_in_4bit and 'bitsandbytes' in str(ie).lower():
            logging.error("Failed to load model: The --load-in-4bit flag requires the 'bitsandbytes' library.")
            logging.error("Please install it: pip install bitsandbytes")
        else:
            logging.error(f"Failed to load tokenizer/model due to missing import: {ie}")
        sys.exit(1)
    except Exception as e:
        model_load_end_time = time.monotonic()
        logging.error(f"Failed to load tokenizer/model: {e} (attempt took {model_load_end_time - model_load_start_time:.2f}s)")
        # Print traceback for detailed debugging
        import traceback
        logging.error(traceback.format_exc())
        sys.exit(1)

    # --- Proceed with Generation --- 
    logging.info(f"Starting conversation data generation using base arguments: {args}") 
    
    # Store tuples of (turns_list, topic, scenario, style, include_points, actual_p1_name, actual_p2_name)
    temp_conversation_data = [] 
    num_successfully_generated = 0 
    total_turns = 0 
    jsonl_save_successful = False 
    hf_dataset_dict = None 
    validation_passed = True
    total_llm_conversation_time = 0.0 
    total_llm_arg_generation_time = 0.0 # Initialize timer
    total_llm_variation_time = 0.0 

    # These will store the definitive persona info AFTER it's set (manual or generated)
    base_persona1 = None
    base_persona1_desc = None
    base_persona2 = None
    base_persona2_desc = None
    initial_topic = None
    initial_scenario = None
    initial_style = None
    initial_include_points = None
    
    # Initialize variables to store the last successful context used
    last_successful_topic = None
    last_successful_scenario = None
    last_successful_style = None
    last_successful_include_points = None
    
    # NEW: Variables for image URLs
    base_persona1_image_url = None
    base_persona2_image_url = None
    
    # Flag to track if base arguments are determined
    base_args_determined = False
    
    if args.creative_brief:
        logging.info("Creative brief provided. Generating detailed arguments ONCE using the LLM...")
        if any([args.topic, args.persona1, args.persona1_desc, args.persona2, args.persona2_desc, args.scenario, args.style]):
             logging.warning("Both --creative-brief and detailed arguments were provided. Ignoring detailed arguments and generating from brief.")

        arg_gen_start_time = time.monotonic()
        # Pass the specific search term arguments
        generated_args = generate_args_from_brief_safe(args.creative_brief, text_generator, tokenizer, 
                                                args.persona1_search_term, args.persona2_search_term)
        arg_gen_end_time = time.monotonic()
        arg_gen_duration = arg_gen_end_time - arg_gen_start_time
        total_llm_arg_generation_time = arg_gen_duration # Assign duration here
        
        if generated_args is None:
            logging.error(f"Failed to generate or parse detailed arguments from the creative brief (took {arg_gen_duration:.2f}s). Exiting.")
            sys.exit(1)
        logging.info(f"Single argument generation from brief took {arg_gen_duration:.2f}s.")
            
        # Overwrite args namespace - These will be used for ALL examples
        base_persona1 = generated_args.get('persona1')
        base_persona1_desc = generated_args.get('persona1_desc')
        # Add a default description if persona1_desc was not generated
        if base_persona1_desc is None:
            base_persona1_desc = f"A character named {base_persona1}. Speaks naturally with a distinct voice."
            logging.warning(f"persona1_desc was missing. Using default description: '{base_persona1_desc}'")
        base_persona2 = generated_args.get('persona2')
        base_persona2_desc = generated_args.get('persona2_desc')
        # Add a default description if persona2_desc was not generated
        if base_persona2_desc is None:
            base_persona2_desc = f"A character named {base_persona2}. Speaks naturally with a distinct voice."
            logging.warning(f"persona2_desc was missing. Using default description: '{base_persona2_desc}'")
        initial_topic = generated_args.get('topic')
        initial_scenario = generated_args.get('scenario')
        initial_style = generated_args.get('style')
        initial_include_points = generated_args.get('include_points')
        
        logging.info("--- Generated Arguments (used for all examples) ---")
        logging.info(f"  Topic: {initial_topic}")
        logging.info(f"  Persona 1: {base_persona1} ({base_persona1_desc})")
        logging.info(f"  Persona 2: {base_persona2} ({base_persona2_desc})")
        logging.info(f"  Scenario: {initial_scenario}")
        logging.info(f"  Style: {initial_style}")
        logging.info(f"  Include Points: {initial_include_points}")
        logging.info("-------------------------------------------------")
        
        # NEW: Perform Image Search for Base Personas
        logging.info("Attempting to find representative images for personas...")
        img_search_start = time.monotonic()
        if args.persona1_search_term:
            logging.info(f"Using search term '{args.persona1_search_term}' for Persona 1 image.")
            base_persona1_image_url = get_persona_image_url(args.persona1_search_term)
        else:
            base_persona1_image_url = get_persona_image_url(base_persona1) # Use name if no term

        if args.persona2_search_term:
            logging.info(f"Using search term '{args.persona2_search_term}' for Persona 2 image.")
            base_persona2_image_url = get_persona_image_url(args.persona2_search_term)
        else:
            base_persona2_image_url = get_persona_image_url(base_persona2) # Use name if no term
        img_search_end = time.monotonic()
        logging.info(f"Image search completed (took {img_search_end - img_search_start:.2f}s).")
        logging.info(f"  Persona 1 Image URL: {base_persona1_image_url}")
        logging.info(f"  Persona 2 Image URL: {base_persona2_image_url}")
        
        base_args_determined = True
    elif not args.creative_brief:
        logging.info("Using manually provided detailed arguments.")
        # Validate required manual args are provided
        required_manual_args = ['topic', 'persona1', 'persona1_desc', 'persona2', 'persona2_desc', 'scenario', 'style']
        missing_manual = [f'--{arg}' for arg in required_manual_args if getattr(args, arg) is None]
        if missing_manual:
             # Check if any general generation args were provided, which implies intent to generate
            generation_intent_args = ['num_examples', 'output_file', 'max_new_tokens', 'upload_to_hub', 'force_upload', 'validate_local_save']
            if any(getattr(args, gen_arg) is not None for gen_arg in generation_intent_args if gen_arg in args and hasattr(args, gen_arg)) or args.num_examples != parser.get_default('num_examples'): # Check non-None or non-default
                 logging.error(f"Missing required detailed arguments for manual generation mode: {', '.join(missing_manual)}")
                 logging.error("Please provide all required detailed arguments or use --creative-brief.")
                 sys.exit(1)
            else:
                 # If no generation args were changed and detailed args are missing, maybe user just ran script with no args?
                 parser.print_help()
                 sys.exit(0)
        
        # Set base variables from args 
        base_persona1 = args.persona1
        base_persona1_desc = args.persona1_desc
        base_persona2 = args.persona2
        base_persona2_desc = args.persona2_desc
        initial_topic = args.topic
        initial_scenario = args.scenario
        initial_style = args.style
        initial_include_points = args.include_points
        
        # Also set last successful values to same as initial values
        last_successful_topic = initial_topic
        last_successful_scenario = initial_scenario
        last_successful_style = initial_style
        last_successful_include_points = initial_include_points
        
        # Try to get images
        logging.info("Attempting to find representative images for personas...")
        img_search_start = time.monotonic()
        base_persona1_image_url = get_persona_image_url(base_persona1)
        base_persona2_image_url = get_persona_image_url(base_persona2)
        img_search_end = time.monotonic()
        logging.info(f"Image search completed (took {img_search_end - img_search_start:.2f}s).")
        logging.info(f"  Persona 1 Image URL: {base_persona1_image_url}")
        logging.info(f"  Persona 2 Image URL: {base_persona2_image_url}")
        
        base_args_determined = True

    # Check if required base args are set before proceeding
    if not base_args_determined or base_persona1 is None or base_persona2 is None:
        logging.error("Persona 1 and Persona 2 must be defined either via --creative-brief or manual arguments.")
        sys.exit(1)

    # --- Proceed with Generation Loop --- 
    logging.info(f"Starting conversation data generation using base arguments: {args}") 

    # --- Generation Loop ---
    gen_loop_start_time = time.monotonic()
    try:
        for i in range(args.num_examples):
            example_start_time = time.monotonic()
            logging.info(f"Processing example {i+1}/{args.num_examples}...")
            
            # --- Topic/Scenario Variation Step (IF brief mode) ---
            current_topic = initial_topic
            current_scenario = initial_scenario
            current_style = initial_style
            current_include_points = initial_include_points # Re-set each loop
            
            if args.creative_brief: # Check if brief mode is active
                variation_gen_start_time = time.monotonic()
                variation_args = generate_topic_variation(
                    args.creative_brief,
                    base_persona1, base_persona1_desc,
                    base_persona2, base_persona2_desc,
                    initial_topic, initial_scenario, initial_style, # Pass initial values
                    text_generator, tokenizer
                )
                variation_gen_end_time = time.monotonic()
                variation_gen_duration = variation_gen_end_time - variation_gen_start_time
                total_llm_variation_time += variation_gen_duration
                
                if variation_args:
                    logging.info(f"  Using generated variation for example {i+1} (took {variation_gen_duration:.2f}s)." )
                    current_topic = variation_args.get('topic', initial_topic)
                    current_scenario = variation_args.get('scenario', initial_scenario)
                    current_style = variation_args.get('style', initial_style)
                    # Note: include_points are not varied currently
                    logging.debug(f"    Variation Topic: {current_topic}")
                    logging.debug(f"    Variation Scenario: {current_scenario}")
                    logging.debug(f"    Variation Style: {current_style}")
                else:
                    logging.warning(f"  Failed to generate variation for example {i+1} (took {variation_gen_duration:.2f}s). Using initial topic/scenario/style.")
                    # Fallback to initial args already set in current_ vars
            
            # --- Conversation Generation Step --- 
            logging.info(f"  Generating conversation for example {i+1}...")
            # Use BASE personas but CURRENT topic/scenario/style
            messages = create_generation_prompt(
                current_topic, 
                base_persona1, 
                base_persona2, 
                base_persona1_desc, 
                base_persona2_desc, 
                current_scenario, 
                current_style, 
                current_include_points # Using initial points for now
            )
            try:
                prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception as e:
                logging.error(f"  Failed to apply chat template for conversation generation (example {i+1}): {e}. Skipping.")
                continue

            raw_conversation = None
            llm_call_start_time = time.monotonic()
            try:
                outputs = text_generator(
                    prompt_text,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=0.75, # Increased temperature
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id,
                    num_return_sequences=1
                )
                llm_call_end_time = time.monotonic()
                llm_duration = llm_call_end_time - llm_call_start_time
                total_llm_conversation_time += llm_duration
                logging.info(f"    LLM generation for example {i+1} took {llm_duration:.2f}s.")
                if outputs and isinstance(outputs, list) and 'generated_text' in outputs[0]:
                    if prompt_text in outputs[0]['generated_text']:
                         raw_conversation = outputs[0]['generated_text'][len(prompt_text):].strip()
                    else:
                        logging.warning(f"Prompt text not found in conversation generation response for example {i+1}. Using full output.")
                        raw_conversation = outputs[0]['generated_text'].strip()
                    if not raw_conversation:
                        logging.warning(f"LLM returned empty response for conversation generation (example {i+1}). Skipping.")
                        continue
                    # Validate start prefix more carefully
                    if not (raw_conversation.startswith(f"{base_persona1}:") or raw_conversation.lstrip().startswith(f"{base_persona1}:") or \
                            raw_conversation.startswith(f"{base_persona2}:") or raw_conversation.lstrip().startswith(f"{base_persona2}:")):
                        logging.warning(f"Generated text for example {i+1} does not start with expected persona prefix. Skipping. Output: {raw_conversation[:100]}...")
                        continue
                else:
                    logging.warning(f"Unexpected output format from text_generator for conversation generation (example {i+1}): {outputs}. Skipping.")
                    continue
            except Exception as e:
                llm_call_end_time = time.monotonic() # Capture time even on failure
                logging.error(f"  Error during LLM conversation generation (example {i+1}): {e} (attempt took {llm_call_end_time - llm_call_start_time:.2f}s). Skipping.")
                continue 
            if raw_conversation:
                 parsing_start_time = time.monotonic()
                 # Pass base_persona1 and base_persona2 names for parsing
                 structured_turns, actual_p1_name, actual_p2_name = parse_conversation_to_sharegpt(raw_conversation, base_persona1, base_persona2) 
                 parsing_end_time = time.monotonic()
                 # Check if structured_turns and persona names are valid
                 if structured_turns and actual_p1_name and actual_p2_name:
                    # Store tuple with turns and the specific context used for this convo
                    # AND the actual names returned by the parser
                    temp_conversation_data.append((structured_turns, current_topic, current_scenario, current_style, current_include_points, actual_p1_name, actual_p2_name))
                    num_successfully_generated += 1
                    
                    # Update last successful textual vars with the context used for THIS example
                    last_successful_topic = current_topic
                    last_successful_scenario = current_scenario
                    last_successful_style = current_style
                    last_successful_include_points = current_include_points
                    # Image URLs remain the base ones determined earlier

                    logging.debug(f"Stored structured conversation {len(temp_conversation_data)} (P1: '{actual_p1_name}', P2: '{actual_p2_name}')..." )
                    logging.debug(f"  Parsing for example {i+1} took {parsing_end_time - parsing_start_time:.4f}s.")
                 else:
                     logging.warning(f"  Failed to parse generated text (example {i+1}). Skipping. (Parsing attempt took {parsing_end_time - parsing_start_time:.4f}s)")
            else:
                 logging.warning(f"Skipping parsing for example {i+1} because raw_conversation was empty or generation failed.")
            example_end_time = time.monotonic()
            logging.debug(f"  Total processing time for example {i+1}: {example_end_time - example_start_time:.2f}s.")

    except Exception as e:
         logging.error(f"An unexpected error occurred during the main generation loop: {e}")
    gen_loop_end_time = time.monotonic()
    logging.info(f"Generation loop finished (took {gen_loop_end_time - gen_loop_start_time:.2f}s).")
    # Update summary logging for arg generation
    if args.creative_brief:
        # Only one call was made
        logging.info(f"Initial argument generation LLM call took: {total_llm_arg_generation_time:.2f}s.")         
    if num_successfully_generated > 0:
         avg_llm_conv_time = total_llm_conversation_time / num_successfully_generated
         logging.info(f"Total time spent in conversation LLM calls: {total_llm_conversation_time:.2f}s (Avg: {avg_llm_conv_time:.2f}s / successful example).")
    else:
        logging.info("No examples successfully generated.")

    # --- Flatten Data and Write JSON Lines File ---
    flatten_start_time = time.monotonic()
    # Calculate total turns correctly from the stored tuples
    total_turns = sum(len(conv_tuple[0]) for conv_tuple in temp_conversation_data)
    
    if temp_conversation_data:
        try:
            with open(args.output_file, 'w', encoding='utf-8') as f_out:
                current_turn_count = 0 
                # No need for human_speaker/gpt_speaker variables here anymore
                    
                # Iterate through the stored tuples
                for idx, (conversation_turns, topic, scenario, style, include_pts, p1_name, p2_name) in enumerate(temp_conversation_data):
                    turn_counter = 0 # Reset turn counter for each conversation
                    for turn in conversation_turns:
                        role = turn.get('from', '')
                        # Use the names stored in the tuple for this conversation, falling back to base persona names
                        if role == 'human':
                            speaker_name = p1_name if p1_name else base_persona1
                        elif role == 'gpt': 
                            speaker_name = p2_name if p2_name else base_persona2
                        else:
                            speaker_name = 'Unknown'
                            
                        if not speaker_name: # Double-check for None or empty strings
                            logging.warning(f"Speaker name is empty for turn {turn_counter} in conversation {idx}. Using 'Unknown'.")
                            speaker_name = 'Unknown'
                             
                        turn_data = {
                            "conversation_id": idx,
                            "turn_number": turn_counter,
                            "role": role, 
                            "speaker_name": speaker_name, # Use correct name
                            "topic": topic,             
                            "scenario": scenario,       
                            "style": style,             
                            "include_points": include_pts if include_pts else "", 
                            "content": turn.get('value', '')
                        }
                        # Add debug log here
                        logging.debug(f"Writing turn data: {turn_data}") 
                        f_out.write(json.dumps(turn_data) + '\n')
                        current_turn_count += 1
                        turn_counter += 1 # Increment turn counter
            if current_turn_count == total_turns:
                logging.info(f"Successfully wrote {total_turns} turns to {args.output_file}")
                jsonl_save_successful = True
            else:
                logging.error(f"Mismatch writing turns! Expected {total_turns}, wrote {current_turn_count}. File might be corrupt.")
                jsonl_save_successful = False
        except IOError as e:
            logging.error(f"Failed to write to output file {args.output_file}: {e}")
        except Exception as e:
            logging.error(f"An unexpected error occurred during JSONL writing: {e}")
    else:
        logging.warning("No conversations successfully generated/parsed, skipping file writing.")
    flatten_end_time = time.monotonic()
    if temp_conversation_data: 
        logging.info(f"Flattening and writing {total_turns} turns took {flatten_end_time - flatten_start_time:.2f}s.")

    # --- Optional: Validate Locally Saved JSONL File ---
    if args.validate_local_save and jsonl_save_successful:
        validation_start_time = time.monotonic()
        # ... (Validation logic) ...
        validation_end_time = time.monotonic()
        logging.info(f"Local JSONL validation took {validation_end_time - validation_start_time:.2f}s.")
        # ... (Log success/failure) ...
    # ... (else if validate but no save) ...

    # --- Load JSONL into Dataset Object (if save was successful) ---
    if jsonl_save_successful:
        logging.warning("Using parameters from the LAST successfully generated example for the dataset card text, and base images.")
        load_dataset_start_time = time.monotonic()
        dataset_created = False
        hf_dataset_dict = None # Initialize
        try: # Main try for all dataset loading attempts
            # Attempt standard HF load_dataset first
            logging.info("Attempting standard load_dataset('json')...")
            # Define features needed for standard load - CONSISTENTLY use int64 based on inference
            features = Features({
                'conversation_id': Value('int64'),
                'turn_number': Value('int64'),
                'role': Value('string'),
                'speaker_name': Value('string'),
                'topic': Value('string'),
                'scenario': Value('string'),
                'style': Value('string'),
                'include_points': Value('string'),
                'content': Value('string')
            })
            hf_dataset_dict = load_dataset('json', data_files=args.output_file, split='train', features=features)
            if not isinstance(hf_dataset_dict, DatasetDict): hf_dataset_dict = DatasetDict({'train': hf_dataset_dict})
            dataset_created = True # Mark as created if standard load succeeds
            logging.info("Standard load_dataset('json') successful.")
 
            # Add DatasetInfo IF dataset was created successfully within the main try block
            if isinstance(hf_dataset_dict, DatasetDict) and 'train' in hf_dataset_dict:
                yaml_metadata = """---\nlicense: unknown\ntags:\n- conversational\n- synthetic\n---\n"""
                # Use the parameters from the *last* successful iteration
                # Use BASE persona info determined earlier for consistency in card
                final_persona1 = base_persona1
                final_persona1_desc = base_persona1_desc
                final_persona2 = base_persona2
                final_persona2_desc = base_persona2_desc
                
                # Use LAST generated topic/scenario/style for description
                final_topic = last_successful_topic
                final_scenario = last_successful_scenario
                final_style = last_successful_style
                final_include_points = last_successful_include_points

                # Construct image markdown, handle None case
                persona1_image_md = f"![{final_persona1}]({base_persona1_image_url})" if base_persona1_image_url else "(No image found)"
                persona2_image_md = f"![{final_persona2}]({base_persona2_image_url})" if base_persona2_image_url else "(No image found)"

                markdown_body = f"""# {final_persona1} & {final_persona2}: {final_topic} - Generated by Conversation Dataset Generator

This dataset was generated using the Conversation Dataset Generator script available at [https://cahlen.github.io/conversation-dataset-generator/](https://cahlen.github.io/conversation-dataset-generator/).

## Generation Parameters

*   **Number of Conversations Requested:** {args.num_examples}
*   **Number of Conversations Successfully Generated:** {num_successfully_generated}
*   **Total Turns:** {total_turns}
*   **Model ID:** `{args.model_id}`
*   **Creative Brief Used:** `{args.creative_brief if args.creative_brief else 'N/A (Manual Mode)'}`
*   **Persona 1 Search Term Used:** `{args.persona1_search_term if args.persona1_search_term else 'N/A'}`
*   **Persona 2 Search Term Used:** `{args.persona2_search_term if args.persona2_search_term else 'N/A'}`

**Note:** When using Creative Brief mode, Persona 1 ({base_persona1}) and Persona 2 ({base_persona2}) were determined once from the brief and kept consistent. The topic, scenario, and style may have been automatically varied for each example based on the original brief, aiming for diversity. The parameters shown below reflect the configuration used for the *last successfully generated example*.

*   **Topic:** `{final_topic}`
*   **Scenario:** `{final_scenario}`
*   **Style:** `{final_style}`
*   **Included Points:** `{final_include_points if final_include_points else 'None'}`

## Personas

**{final_persona1}**
{persona1_image_md}
*Description:* `{final_persona1_desc}` -> maps to `role: human`

**{final_persona2}**
{persona2_image_md}
*Description:* `{final_persona2_desc}` -> maps to `role: gpt`

## Usage

To use this dataset:

**1. Clone the repository:**
```bash
git lfs install
git clone https://huggingface.co/datasets/{args.upload_to_hub}
```

**2. Load in Python:**
```python
from datasets import load_dataset

dataset = load_dataset("{args.upload_to_hub}")

# Access the data (e.g., the training split)
print(dataset['train'][0])
```

## LoRA Training Example (Basic)

Below is a basic example of how you might use this dataset to fine-tune a small model like `google/gemma-2b-it` using LoRA with the PEFT and TRL libraries.

**Note:** This requires installing additional libraries: `pip install -U transformers datasets accelerate peft trl bitsandbytes torch`

```python
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from trl import SFTTrainer

# 1. Load the dataset
dataset_id = "{args.upload_to_hub}"
dataset = load_dataset(dataset_id)

# 2. Load Base Model & Tokenizer (using a small model like Gemma 2B)
model_id = "google/gemma-2b-it"

# Quantization Config (optional, for efficiency)
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16 # or torch.float16
)

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
# Set padding token if necessary (Gemma's is <pad>)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

# Model
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    device_map="auto", # Automatically place model shards
    trust_remote_code=True
)

# Prepare model for k-bit training if using quantization
model = prepare_model_for_kbit_training(model)

# 3. LoRA Configuration
lora_config = LoraConfig(
    r=8, # Rank
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], # Adjust based on model architecture
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 4. Training Arguments (minimal example)
training_args = TrainingArguments(
    output_dir="./lora-adapter-{final_persona1}-{final_persona2}", # Choose a directory
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=1, # Use 1 epoch for a quick demo
    logging_steps=10,
    save_steps=50, # Save adapter periodically
    fp16=False, # Use bf16 if available, otherwise fp16
    bf16=torch.cuda.is_bf16_supported(),
    optim="paged_adamw_8bit", # Use paged optimizer for efficiency
    report_to="none" # Disable wandb/tensorboard for simple example
)

# 5. Create SFTTrainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset['train'], # Assumes 'train' split exists
    peft_config=lora_config,
    tokenizer=tokenizer,
    args=training_args,
    max_seq_length=512, # Adjust as needed
    dataset_text_field="content", # Use content field directly 
    packing=True, # Pack sequences for efficiency
)

# 6. Train
print("Starting LoRA training...")
trainer.train()

### 7. Save the LoRA adapter
# Use a fixed string for the example output directory
trainer.save_model("./lora-adapter-output-directory") 
print(f"LoRA adapter saved to ./lora-adapter-output-directory")
```
"""

                format_desc = f"""## Dataset Format (JSON Lines source)

Each row in the dataset contains the following keys:
- conversation_id: Unique identifier for the conversation
- turn_number: The sequential number of the turn within a conversation
- role: Either 'human' or 'gpt' indicating who is speaking
- speaker_name: The actual name of the speaker (e.g., '{final_persona1}' or '{final_persona2}')
- topic: The conversation topic
- scenario: The scenario in which the conversation takes place
- style: The stylistic direction for the conversation
- include_points: Specific points to include in the conversation
- content: The actual text content of the turn
"""

                description_content = yaml_metadata.strip() + "\n\n" + markdown_body.strip() + "\n\n" + format_desc.strip()

                # Bump version for schema change
                # Ensure features are defined before creating DatasetInfo
                features = Features({
                    'conversation_id': Value('int64'), # Keep consistent with loading
                    'turn_number': Value('int64'),
                    'role': Value('string'),
                    'speaker_name': Value('string'),
                    'topic': Value('string'),
                    'scenario': Value('string'),
                    'style': Value('string'),
                    'include_points': Value('string'),
                    'content': Value('string')
                })
                ds_info = DatasetInfo(description=description_content, version="1.7.0", license="unknown", features=features)
                hf_dataset_dict['train'].info.update(ds_info)
                logging.info("Dataset loaded and detailed DatasetInfo updated successfully.")
            elif not isinstance(hf_dataset_dict, DatasetDict):
                logging.warning("Loaded dataset dict does not contain a 'train' split. Cannot add metadata.")
                dataset_created = False # Mark as failed if split is missing
        except Exception as e: # Aligned with the main try block around line 994
            hf_dataset_dict = None # Ensure dataset is None on any loading failure
            logging.error(f"An error occurred during the main generation or dataset processing: {e}")
            import traceback
            logging.error(f"Detailed error: {traceback.format_exc()}")
            dataset_created = False # Explicitly mark as not created on major error
            # Optionally re-raise or exit if the error is fatal
            # sys.exit(1)

    # --- Upload to Hugging Face Hub (if requested and DatasetDict created AND VALIDATED) ---
    # Dedented to be at the main script level, outside the main try/except
    if args.upload_to_hub and hf_dataset_dict is not None and dataset_created:
        # --- Determine if upload should proceed ---
        proceed_with_upload = False # Default to no
        if args.force_upload:
            logging.info("Force upload specified, proceeding without confirmation.")
            proceed_with_upload = True
        else:
            # Ask for confirmation
            try:
                confirm_upload = input(f"Dataset generated ({num_successfully_generated} examples, {total_turns} turns). Upload to Hugging Face Hub repository {args.upload_to_hub}? (yes/no): ")
                if confirm_upload.lower() == 'yes':
                    proceed_with_upload = True
                else:
                    logging.info("Upload cancelled by user.")
            except EOFError:
                logging.warning("Could not get user input (EOFError). Upload cancelled.")
            except Exception as e:
                logging.warning(f"An error occurred during upload confirmation: {e}. Upload cancelled.")

        # --- Conditional Upload Block ---
        if proceed_with_upload:
            # Check for token first
            token = HfFolder.get_token()
            if not token:
                logging.error("Hugging Face token not found. Please login using `huggingface-cli login`. Cannot upload.")
            else:
                api = HfApi(token=token)
                dataset_push_successful = False
                # --- Push Dataset ---
                push_ds_start_time = time.monotonic()
                try:
                    hf_dataset_dict.push_to_hub(args.upload_to_hub, private=False) # Assuming public for now
                    push_ds_end_time = time.monotonic()
                    logging.info(f"Dataset push to {args.upload_to_hub} took {push_ds_end_time - push_ds_start_time:.2f}s.")
                    dataset_push_successful = True
                except Exception as e:
                    push_ds_end_time = time.monotonic()
                    logging.error(f"Failed to push dataset: {e} (attempt took {push_ds_end_time - push_ds_start_time:.2f}s)")

                # --- Upload README ---
                if dataset_push_successful:
                    if 'train' in hf_dataset_dict and hf_dataset_dict['train'].info.description:
                        upload_readme_start_time = time.monotonic()
                        try:
                            card_content = hf_dataset_dict['train'].info.description
                            readme_content_bytes = card_content.encode('utf-8')
                            readme_file_in_memory = io.BytesIO(readme_content_bytes)

                            api.upload_file(
                                path_or_fileobj=readme_file_in_memory,
                                path_in_repo="README.md",
                                repo_id=args.upload_to_hub,
                                repo_type="dataset"
                            )
                            upload_readme_end_time = time.monotonic()
                            logging.info(f"README.md upload took {upload_readme_end_time - upload_readme_start_time:.2f}s.")
                        except Exception as readme_e:
                            upload_readme_end_time = time.monotonic()
                            logging.warning(f"Failed to upload README.md: {readme_e} (attempt took {upload_readme_end_time - upload_readme_start_time:.2f}s)")
                    elif not ('train' in hf_dataset_dict and hf_dataset_dict['train'].info.description):
                        logging.warning("Could not find description in DatasetInfo to upload as README.md.")
                    else:
                        logging.error("Dataset object missing 'train' split or description unexpectedly.")
    elif args.upload_to_hub and (hf_dataset_dict is None or not dataset_created):
        logging.warning("Upload requested, but dataset object was not created/validated successfully. Skipping upload.")

    script_end_time = time.monotonic()
    logging.info(f"Finished generation. Successfully generated {num_successfully_generated} conversations ({total_turns} turns).")
    logging.info(f"Total script execution time: {script_end_time - script_start_time:.2f}s.")
