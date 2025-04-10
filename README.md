# Conversation Dataset Generator ✨

<a href="https://github.com/cahlen/conversation-dataset-generator"><img src="https://upload.wikimedia.org/wikipedia/commons/1/1e/Speech_bubble_icon.svg" alt="Conversation Icon" width="40"></a> | [View Project Page](https://cahlen.github.io/conversation-dataset-generator/) | [View on GitHub](https://github.com/cahlen/conversation-dataset-generator)

*Craft High-Quality Dialogue Data for Your LLMs.*

## Overview

Ever wish you could generate *just* the right kind of conversational data? Whether you're fine-tuning a Large Language Model (LLM) for a specific **style** or **persona**, need dialogue for a creative project, or want to explore complex **topics** in a natural flow, the Conversation Dataset Generator is here to help!

This powerful and flexible Python script leverages Hugging Face's `transformers` library to put you in control. You can either:

1.  **Specify Everything (Manual Mode):** Define the exact `topic`, `personas` (with descriptions!), `scenario`, `style`, and even specific `keywords` to include.
2.  **Use a Creative Spark (Brief Mode):** Provide a high-level `creative-brief` (like *"Sherlock Holmes explains TikTok trends to a confused Dr. Watson"*) and let the script use an LLM to brainstorm the detailed parameters *for you*. This mode automatically generates **topic/scenario variations** for each example while keeping the core personas consistent, enhancing dataset diversity. Furthermore, you can optionally provide specific **web search terms** (`--persona1-search-term`, `--persona2-search-term`) to fetch real-time context about the personas, allowing the LLM to generate more accurate descriptions and dialogue even for individuals or characters not well-represented in its training data. **Note:** In this mode, the script also attempts to find representative images for the base personas, which are included in the generated dataset card if uploaded to the Hugging Face Hub.

Either way, the output is a clean **JSON Lines (`.jsonl`)** file, perfect for downstream tasks. Each line represents a single turn with a rich set of keys: `conversation_id` (int64), `turn_number` (int64), `role` (string), `speaker_name` (string), `topic` (string), `scenario` (string), `style` (string), `include_points` (string), and `content` (string) – readily compatible with popular LLM training frameworks and NLP pipelines.

## Why Use Conversation Dataset Generator? (Applications & Training Goals)

Unlock the potential of your LLMs or accelerate your creative process! This script empowers you to generate targeted datasets for various goals:

1.  **Style Specialization:** Need an LLM that sounds like a pirate? A valley girl? A formal news anchor? Generate large datasets with a consistent `--style` (or varied styles linked to topics in brief mode) to train models or LoRA adapters that master specific conversational nuances.
    *   *Goal:* Create a model that *sounds* authentic in a target context.
2.  **Persona Embodiment (Including Niche Characters):** Build believable characters. Define distinct personas using `--persona1/2` and `--persona1/2-desc`. **Crucially, in brief mode, you can provide `--personaX-search-term` arguments to fetch web context.** This allows the script to generate informed descriptions and dialogue even for specific individuals, lesser-known figures, or custom characters the base LLM might not know well, enabling fine-tuning for highly specific persona adoption.
    *   *Goal:* Create interactive characters with consistent personalities, even those outside the LLM's core knowledge.
3.  **Topic/Scenario Fluency & Conditioning:** Enhance a model's ability to discuss particular `--topic`s within specific `--scenario`s naturally. Generate dialogues that explore subjects from different angles (especially with brief mode's variation). The added columns allow for fine-tuning models conditioned on topic or scenario.
    *   *Goal:* Improve conversational performance in specific domains or settings; train models to adapt responses based on context.
4.  **Instruction Adherence:** Train models to better follow constraints. Use `--include-points` to generate data where specific keywords *should* appear, improving the model's ability to incorporate requirements into creative outputs. The `include_points` column allows analysis of how well the points were integrated.
    *   *Goal:* Make models more reliable at following specific instructions within a dialogue.
5.  **Creative Content Generation:** Break writer's block! Draft dialogue for scripts (TV, film, games), podcasts, or stories by defining compelling personas, scenarios, and styles. Explore possibilities quickly, especially using the brief mode's variation feature.
    *   *Goal:* Accelerate creative writing and dialogue prototyping.
6.  **Dialogue Flow Analysis:** The `turn_number` column facilitates studying conversation progression, identifying patterns in openings, mid-dialogue shifts, or closings.
    *   *Goal:* Understand and model natural conversation structure.

## Generating Data for Specific Goals (Examples)

Let's see it in action!

### Using Detailed Arguments (Manual Mode)

Specify exactly what you want.

---

**Example 1: Training a "Sitcom Banter" Style LoRA**

*Goal: Create a LoRA that makes an LLM generate witty, observational dialogue reminiscent of a classic sitcom.*

```bash
python generate.py \
  --num-examples 1000 \
  --topic "the absurdity of everyday errands" \
  --persona1 "Alex" \
  --persona1-desc "slightly neurotic, prone to overthinking, often uses rhetorical questions" \
  --persona2 "Sam" \
  --persona2-desc "more laid-back, often amused by Alex's antics, responds with dry wit" \
  --scenario "waiting in line at the post office" \
  --style "observational, witty, fast-paced banter, slightly absurd, like Seinfeld" \
  --include-points "long lines, confusing forms, questionable package handling, passive aggression" \
  --output-file sitcom_style_dataset.jsonl \
  --model-id meta-llama/Meta-Llama-3-8B-Instruct
```

*Explanation: High volume (`--num-examples 1000`), consistent detailed parameters (especially `--style` and descriptive personas) focus the data on the target style.*

---

**Example 2: Training a "Helpful Coding Mentor" Persona LoRA**

*Goal: Fine-tune a model to act as a patient, encouraging coding mentor.*

```bash
python generate.py \
  --num-examples 500 \
  --topic "debugging a common Python error (e.g., IndexError)" \
  --persona1 "MentorBot" \
  --persona1-desc "a patient, knowledgeable, and encouraging Python tutor AI. Uses analogies, asks guiding questions rather than giving direct answers, celebrates small successes." \
  --persona2 "Learner" \
  --persona2-desc "a beginner programmer feeling slightly stuck but eager to learn, expresses confusion clearly." \
  --scenario "working through a coding problem together online via chat" \
  --style "supportive, clear, step-by-step, educational, positive reinforcement" \
  --include-points "traceback, variable scope, print debugging, list index, off-by-one, debugging process" \
  --output-file mentor_persona_dataset.jsonl \
  --model-id meta-llama/Meta-Llama-3-8B-Instruct
```

*Explanation: Focus is on detailed `--persona1-desc` capturing the desired mentor traits (patience, guiding questions) and a supportive `--style` to shape the mentor's voice.*

---

**Example 3: Training a Topic-Focused LoRA ("Explaining Quantum Computing Simply")**

*Goal: Make the LLM more fluent and natural when explaining a complex topic conversationally.*

```bash
python generate.py \
  --num-examples 750 \
  --topic "basic concepts of quantum computing" \
  --persona1 "QuantumGuru" \
  --persona1-desc "an expert simplifying quantum concepts using everyday analogies (like coin flips for superposition). Patient and enjoys teaching." \
  --persona2 "CuriousChris" \
  --persona2-desc "intelligent but new to quantum, asks clarifying questions, tries to relate concepts to familiar things." \
  --scenario "a casual conversation over coffee trying to understand new tech trends" \
  --style "simplified, analogy-driven, patient, engaging, avoiding deep jargon where possible" \
  --include-points "qubit, superposition, entanglement, potential applications, uncertainty, classical vs quantum" \
  --output-file quantum_topic_dataset.jsonl \
  --model-id meta-llama/Meta-Llama-3-8B-Instruct
```

*Explanation: Teaches the *conversational flow* of explaining the specific `--topic`, reinforced by simplifying personas, analogy-driven descriptions, and style.*

---

**Example 4: Enhancing Instruction Adherence (Specific Constraints)**

*Goal: Train the model to better incorporate specific keywords or constraints during generation.*

```bash
python generate.py \
  --num-examples 800 \
  --topic "benefits of renewable energy sources" \
  --persona1 "EcoAdvocate" \
  --persona1-desc "passionate environmental scientist, presents facts and figures clearly, optimistic tone." \
  --persona2 "SkepticSam" \
  --persona2-desc "concerned about costs and grid reliability, asks challenging but fair questions, slightly pessimistic tone." \
  --scenario "a public town hall meeting discussion about local energy policy" \
  --style "informative but persuasive debate, addressing counterarguments respectfully" \
  --include-points "solar panel efficiency, wind turbine placement, grid stability, battery storage, long-term cost savings, carbon emissions, job creation" \
  --output-file instruction_adherence_dataset.jsonl \
  --model-id meta-llama/Meta-Llama-3-8B-Instruct
```

*Explanation: Training on data where specific `--include-points` were required reinforces the model's ability to follow constraints within a natural dialogue structure.*

---

**Example 5: Generating Data for Creative Writing (Sci-Fi Pilot Scene)**

*Goal: Draft dialogue for a specific scene in a science fiction TV pilot.*

```bash
python generate.py \
  --num-examples 20 \
  --topic "analyzing strange readings from an unknown alien artifact" \
  --persona1 "Captain Eva Rostova" \
  --persona1-desc "experienced, cautious starship captain, focused on procedure and crew safety. Speaks formally." \
  --persona2 "Dr. Aris Thorne" \
  --persona2-desc "brilliant but impulsive xeno-archaeologist, eager for discovery, sometimes disregards protocol. Speaks excitedly, uses technical jargon." \
  --scenario "on the bridge of the starship 'Odyssey' examining scan results displayed on a large viewscreen" \
  --style "tense, suspenseful, professional sci-fi dialogue, sense of wonder mixed with potential danger" \
  --include-points "unknown energy signature, unusual material composition, potential risks, isolation, first contact protocol" \
  --output-file scifi_scene_dialogue.jsonl \
  --model-id meta-llama/Meta-Llama-3-8B-Instruct
```

*Explanation: All parameters work together to create dialogue for a specific fictional moment. Lower `--num-examples` is suitable for drafting multiple variations of the scene.*

---

### Using a Creative Brief (Automatic Argument & Topic Variation)

Just provide an idea, and let the LLM handle the details and create variety! Optionally add web search terms for richer personas.

---

**Example 6: Generating Varied Historical Banter from a Brief**

*Goal: Quickly generate diverse dialogue between consistent historical figures without defining all details manually.*

```bash
python generate.py \
  --creative-brief "A philosophical debate between Leonardo da Vinci and Marie Curie about the nature of discovery." \
  --num-examples 25 \
  --output-file brief_historical_debate.jsonl \
  --model-id meta-llama/Meta-Llama-3-8B-Instruct \
  --upload-to-hub YourUser/VariedHistoricalDebate
```

*Explanation: The script uses the LLM to interpret the `--creative-brief`, generate initial detailed parameters (personas, topic, etc.). Then, for each of the 25 examples, it generates a *new, related* topic/scenario (e.g., discussing specific inventions, the ethics of science, the role of observation) while keeping the da Vinci/Curie personas consistent.*

---

**Example 7: Generating Dialogue for a Specific Person using Web Search**

*Goal: Create dialogue involving a specific, possibly less famous individual by providing web search terms for context.*

```bash
python generate.py \
  --creative-brief "Generate a conversation between tech reviewer Marques Brownlee (MKBHD) and legendary filmmaker Stanley Kubrick about the design philosophy of smartphones vs. cinema cameras." \
  --num-examples 5 \
  --persona1-search-term "Marques Brownlee MKBHD tech review style personality" \
  --persona2-search-term "Stanley Kubrick filmmaker personality directing style meticulous" \
  --output-file mkbhd_kubrick_web_terms_5.jsonl
```

*Explanation: The script uses the LLM for the overall brief interpretation and topic variation. However, it uses the provided `--personaX-search-term` arguments to fetch context from DuckDuckGo. This context helps the LLM generate more accurate `--personaX-desc` arguments, enabling conversations involving specific individuals the base model might not know well.*

---

**Example 8: Generating Fantasy Dialogue from a Brief**

*Goal: Create diverse dialogue for a fantasy setting from a simple concept.*

```bash
python generate.py \
  --creative-brief "An ancient, wise dragon trying to explain magic to a skeptical, pragmatic dwarf blacksmith." \
  --num-examples 50 \
  --output-file brief_fantasy_talk.jsonl \
  --validate-local-save
```

*Explanation: The script generates initial parameters from the brief, then varies the topic/scenario (e.g., explaining different types of magic, the cost of spells, magical artifacts vs. forged items) for each of the 50 examples, keeping the dragon and dwarf personas.*

---

**Example 9: Generating Absurdist Comedy Variations from a Brief**

*Goal: Generate surreal, varied dialogue based on an unusual pairing.*

```bash
python generate.py \
  --creative-brief "A sentient existentialist toaster discussing the meaning of crumbs with a flock of nihilistic pigeons in a park." \
  --num-examples 10 \
  --output-file brief_toaster_pigeons.jsonl
```

*Explanation: Perfect for highly imaginative scenarios! The script generates varied crumb-related topics/scenarios (e.g., the futility of sweeping, the beauty of decay, pigeons judging bread types) for the toaster and pigeons across 10 examples.*

---

**Example 10: Generating Specific Genre Dialogue (Noir) from a Brief**

*Goal: Quickly generate dialogue fitting a specific genre like Noir using only a brief.*

```bash
python generate.py \
  --creative-brief "A hardboiled detective interrogating a nervous informant about a stolen artifact in a smoky, rain-slicked alley." \
  --num-examples 10 \
  --output-file brief_noir_interrogation.jsonl \
  --model-id meta-llama/Meta-Llama-3-8B-Instruct
```

*Explanation: The `--creative-brief` provides strong genre cues (hardboiled detective, nervous informant, smoky alley). The LLM generates appropriate personas, topics, scenarios, and a noir style, varying the specifics (e.g., the nature of the artifact, the informant's specific fear) across the examples.*

---

**Example 11: Generating Dialogue for Specific Historical Figures using Web Search**

*Goal: Create dialogue between specific, potentially niche historical figures by providing web search terms for context.*

```bash
python generate.py \
  --creative-brief "Conversation between pioneering computer scientist Grace Hopper and minimalist artist Donald Judd about optimizing naval logistics vs. arranging metal boxes." \
  --num-examples 5 \
  --persona1-search-term "Grace Hopper admiral computer scientist personality nickname Amazing Grace COBOL" \
  --persona2-search-term "Donald Judd artist minimalism Marfa Texas personality meticulous" \
  --output-file hopper_judd_web_search.jsonl \
  --model-id meta-llama/Meta-Llama-3-8B-Instruct
```

*Explanation: The brief sets the stage. The `--personaX-search-term` arguments guide the LLM's argument generation step by providing specific web context for Grace Hopper and Donald Judd, helping capture their distinct personalities and fields, even if they aren't strongly represented in the base model's training.*

---

**Example 12: Generating Dialogue for Specific Fictional Characters using Web Search**

*Goal: Create dialogue between well-known but perhaps less common fictional characters using web search to solidify their personas.*

```bash
python generate.py \
  --creative-brief "A discussion between the AI assistant Clippy and the philosophical robot Marvin the Paranoid Android about the inherent suffering of existence vs. offering unsolicited help." \
  --num-examples 8 \
  --persona1-search-term "Microsoft Clippy paperclip assistant personality annoying helpful interruption" \
  --persona2-search-term "Marvin the Paranoid Android Hitchhiker's Guide personality depressed intelligent brain the size of a planet" \
  --output-file clippy_marvin_web_search.jsonl \
  --model-id meta-llama/Meta-Llama-3-8B-Instruct
```

*Explanation: Similar to the historical example, the brief provides the core idea, while the `--personaX-search-term` arguments provide specific context scraped from the web about Clippy and Marvin, ensuring their iconic (and contrasting) personalities are captured accurately during the initial argument generation, leading to more authentic dialogue.*

---

## Output Format (JSON Lines locally, Hugging Face Dataset on Hub)

Understanding where your data goes:

1.  **Local File (`.jsonl`):** The script always saves the generated data locally first to the path specified by `--output-file`. This is a **JSON Lines** file: each line is a complete JSON object representing a single turn.

    ```json
    {"conversation_id": 0, "turn_number": 0, "role": "human", "speaker_name": "Alex", "topic": "the absurdity of everyday errands", "scenario": "waiting in line at the post office", "style": "observational, witty, fast-paced banter, slightly absurd, like Seinfeld", "include_points": "long lines, confusing forms, questionable package handling, passive aggression", "content": "Seriously, Sam, look at this line. Is time moving slower in here? Are we in some kind of bureaucratic vortex?"}
    {"conversation_id": 0, "turn_number": 1, "role": "gpt", "speaker_name": "Sam", "topic": "the absurdity of everyday errands", "scenario": "waiting in line at the post office", "style": "observational, witty, fast-paced banter, slightly absurd, like Seinfeld", "include_points": "long lines, confusing forms, questionable package handling, passive aggression", "content": "Only if the vortex requires triplicate forms for entry. And possibly a blood sample. Did you fill out the 7B/Stroke-6 form for *existing* in the line?"}
    {"conversation_id": 1, "turn_number": 0, "role": "human", "speaker_name": "Alex", "topic": "the existential dread of choosing coffee beans", "scenario": "staring blankly at a shelf in a grocery store", "style": "observational, witty, fast-paced banter, slightly absurd, like Seinfeld", "include_points": "origin, roast level, ethical sourcing, paralysis by analysis", "content": "Single origin Ethiopian Yirgacheffe... or the house blend... medium roast... dark roast... Sam, how do people *choose*?"}
    ```

    Each row has the following keys:

    *   `conversation_id` (int64): Identifier grouping turns within the dataset (0-indexed).
    *   `turn_number` (int64): The sequence number of the turn within its conversation (0-indexed).
    *   `role` (string): Speaker role (`human` or `gpt`, mapping from Persona 1 and Persona 2 respectively).
    *   `speaker_name` (string): The actual name of the speaker for this turn (e.g., 'Alex', 'Sam').
    *   `topic` (string): The specific topic generated/used for this conversation.
    *   `scenario` (string): The specific scenario generated/used for this conversation.
    *   `style` (string): The specific style generated/used for this conversation.
    *   `include_points` (string): Comma-separated list of keywords requested for inclusion in this conversation (or empty string if none).
    *   `content` (string): The text content of the turn.

2.  **Hugging Face Hub Upload (Optional):** If you provide a repo ID via `--upload-to-hub`, the script performs a two-step upload after generation (and optional local validation):
    *   **Step 1: Load & Push Dataset:** It loads the local `.jsonl` file into a Hugging Face `DatasetDict` object (`datasets.load_dataset('json', ...)`), ensuring features like `conversation_id` and `turn_number` are correctly typed (as `int64`). It then generates a detailed dataset card (README) using the run parameters (based on the *last successfully generated example* when using topic variation), **including any found persona images**, and attaches it to the `DatasetInfo`. Crucially, the `DatasetInfo` includes the `Features` definition matching the full schema. Finally, it pushes the `DatasetDict` to your Hub repository using `push_to_hub()`.
    *   **Step 2: Upload Custom README:** It retrieves the generated dataset card content from the `DatasetInfo`, encodes it to bytes (`utf-8`), and uploads these bytes directly as the `README.md` file using `HfApi.upload_file`. This ensures your repository displays a rich, informative dataset card reflecting the generation parameters and the full data schema.

The final dataset on the Hub will have the full `conversation_id`, `turn_number`, `role`, `speaker_name`, `topic`, `scenario`, `style`, `include_points`, `content` structure and should display correctly in the dataset previewer.

## Model & Fine-Tuning Notes

Leveraging the generated data:

*   **Generation Model:** Uses `meta-llama/Meta-Llama-3-8B-Instruct` by default for both argument generation (in brief mode) and conversation generation. You can change this with `--model-id` to any compatible Hugging Face text-generation model (results may vary!). Using larger/more capable models might yield better results, especially for complex briefs or nuanced styles.
*   **Fine-Tuning Suitability:** This data is ideal for Parameter-Efficient Fine-Tuning (PEFT) methods like **LoRA**. You can create specialized LoRA adapters for style, persona, or topic without the cost of retraining the entire base model.
*   **LoRA Benefits:** Smaller footprint, faster training, modular (mix and match adapters!), easily shareable.
*   **Base Models:** For best results when fine-tuning, start with strong instruction-following base models like Llama 3 Instruct, Mistral Instruct, Mixtral Instruct, Qwen2 Instruct, Gemma Instruct, etc.

## Technical Details

The nitty-gritty:

### Requirements

*   Python 3.8+
*   Key Libraries: **`torch`**, **`transformers`**, **`accelerate`**, **`datasets`**, **`huggingface_hub`**, **`pandas`**. Add **`duckduckgo-search`** if using the web search term arguments (`--personaX-search-term`). Optional: **`bitsandbytes`** for quantization (used in README example).
*   **GPU:** A powerful GPU with sufficient VRAM and CUDA support is *highly recommended*. Creative Brief mode involves multiple LLM calls per example (initial args [using web context if provided], topic variation, conversation), increasing GPU usage. Manual mode uses one LLM call per example.
*   **CPU/Memory:** A capable CPU and adequate RAM are recommended.
*   **Web Search:** Using `--personaX-search-term` requires an internet connection for DuckDuckGo searches during argument generation.

### Installation

Get up and running:

1.  **Clone the repository (Optional):**
    ```bash
    git clone https://github.com/cahlen/conversation-dataset-generator.git
    cd conversation-dataset-generator
    ```

2.  **Create & Activate Virtual Environment (Recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate # On Windows use `venv\Scripts\activate`
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: If `torch` is in requirements.txt, pip might install a CPU or older CUDA version. For optimal GPU usage, consider installing PyTorch separately first, matching your CUDA version - see step 4).*

4.  **Install Specific PyTorch Version (Optional but Recommended for GPU):**
    For specific CUDA versions, install PyTorch *after* other dependencies:
    ```bash
    # Example for CUDA 12.1 (Check official PyTorch site for your version)
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

    # Find the command for your specific CUDA version:
    # https://pytorch.org/get-started/locally/
    ```
    *(Ensure your NVIDIA driver version supports your chosen CUDA version!)*

5.  **Install Optional `bitsandbytes` (for LoRA example):**
    ```bash
    pip install bitsandbytes
    ```

6.  **Login to Hugging Face Hub (Optional, for uploading):**
    To use `--upload-to-hub`, log in via the terminal:
    ```bash
    huggingface-cli login
    # Follow prompts to enter your HF token (read or write permissions needed)
    ```

### Basic Usage

Two ways to generate:

**Script Flow Overview:**

```text
1. Start generate.py
2. Check arguments:
   |--> If --delete-repo provided:
   |     Confirm deletion -> Delete repo(s) -> Exit.
   |--> If --creative-brief provided:
   |     Optional: Perform web search (if --personaX-search-term)
   |     Generate base args (personas, initial topic/scenario/style) using LLM (+ web context)
   |     Optional: Search for persona images
   |     Loop N times (--num-examples):
   |       Generate topic/scenario variation using LLM
   |       Generate conversation using LLM (base personas + varied context)
   |       Parse & Store Conversation
   |     -> Save to JSONL -> Optional Upload -> Exit.
   |--> Else (Manual Mode):
   |     Use provided --topic, --personas, etc.
   |     Optional: Search for persona images
   |     Loop N times (--num-examples):
   |       Generate conversation using LLM (fixed context)
   |       Parse & Store Conversation
   |     -> Save to JSONL -> Optional Upload -> Exit.
```

1.  **Manual Mode (Detailed Arguments):**
    ```bash
    python generate.py --persona1 "Wizard" --persona1-desc "Grumpy, old, prone to muttering spells" \
                       --persona2 "Knight" --persona2-desc "Overly cheerful, oblivious to Wizard's mood" \
                       --topic "The best way to polish armor without magic" \
                       --scenario "Stuck in a dungeon waiting room with bad Muzak" \
                       --style "Comedic, bickering, contrasting personalities" \
                       --num-examples 10 --output-file manual_test.jsonl
    ```

2.  **Creative Brief Mode (with Topic Variation & Optional Web Search):**
    ```bash
    # Without web search
    python generate.py --creative-brief "A pirate captain trying to order coffee at a modern minimalist cafe" \
                       --num-examples 15 --output-file brief_pirate.jsonl

    # With web search for specific personas
    python generate.py --creative-brief "Conversation between Tech Lead Tina and Junior Dev Joe about effective code reviews" \
                       --num-examples 10 \
                       --persona1-search-term "Typical Tech Lead responsibilities personality traits communication" \
                       --persona2-search-term "Junior Developer challenges learning curve receiving feedback" \
                       --output-file brief_tech_review.jsonl
    ```

### Argument Reference

Tailor your generation precisely. Provide EITHER `--creative-brief` OR the set of detailed manual arguments. Use `--delete-repo` only for deleting repositories.

**Mode Selection**

*   `--creative-brief STR`: Provide a high-level concept (e.g., *"Godzilla ordering takeout sushi"*). The script uses the LLM specified by `--model-id` to first generate the detailed arguments (topic, personas, etc.) automatically, potentially informed by web context if search terms are provided (see below). Then, for each requested example, it generates a *new, related* topic/scenario variation while keeping the initially generated personas consistent. If you provide this, any manual `--topic`, `--persona1`, etc., arguments are ignored.
*   `--delete-repo USERNAME/REPO_ID [USERNAME/REPO_ID ...]`: **DANGER ZONE.** Use this argument *instead of* generation arguments to permanently delete one or more Hugging Face Hub dataset repositories. **THIS ACTION IS IRREVERSIBLE.** You will be asked for confirmation.

**Creative Brief Web Context (Optional - Only used with `--creative-brief`)**

*   `--persona1-search-term STR`: If provided along with `--creative-brief`, the script will perform a web search (via DuckDuckGo) using this exact term. The fetched text snippets will be added as context to the prompt used for generating the main arguments (including `--persona1-desc`), helping the LLM create a more informed persona. Ideal for less common or specific characters/individuals.
*   `--persona2-search-term STR`: Same as above, but for Persona 2.

**Detailed Arguments (Manual Mode)**

*(Required if not using `--creative-brief` or `--delete-repo`)*

*   `--topic STR`: Central topic/subject of the conversation.
*   `--persona1 STR`: Name of the first speaker (this name will map to the `human` role in the output data).
*   `--persona1-desc STR`: Detailed description of the first speaker's personality, background, speech patterns, quirks, etc. (Crucial for generation quality!).
*   `--persona2 STR`: Name of the second speaker (maps to the `gpt` role).
*   `--persona2-desc STR`: Detailed description of the second speaker.
*   `--scenario STR`: The setting, situation, or context for the conversation.
*   `--style STR`: Desired tone, mood, and linguistic style (e.g., "formal debate", "casual chat", "Shakespearean insults", "valley girl slang", "hardboiled detective noir").
*   `--include-points STR`: Optional comma-separated list of keywords or talking points the conversation should try to naturally incorporate (e.g., `"time travel paradox,grandfather,temporal mechanics"`).

**General Arguments (Applicable to Generation Modes)**

*   `--num-examples INT`: How many distinct conversation examples to generate. (Default: 3)
*   `--output-file PATH`: Path to save the output JSON Lines (`.jsonl`) file. (Default: `generated_data.jsonl`)
*   `--model-id STR`: Hugging Face model ID (e.g., `meta-llama/Meta-Llama-3-8B-Instruct`). **Crucially, this model is used for BOTH the conversation generation AND the argument generation step if `--creative-brief` is used.** Choose a strong instruction-following model. (Default: 'meta-llama/Meta-Llama-3-8B-Instruct')
*   `--max-new-tokens INT`: Max tokens the LLM can generate in the main conversation step. Adjust based on desired conversation length and model limits. (Default: 768)
*   `--upload-to-hub STR`: Your Hugging Face Hub repository ID (e.g., `YourUsername/YourDatasetName`) to upload the results to. The script will create the repo if it doesn't exist. Requires prior login. (Default: None)
*   `--force-upload`: Skip the confirmation prompt when uploading to the Hub. Use with caution! (Default: False)
*   `--validate-local-save`: Perform basic checks on the locally saved `.jsonl` file after writing. (Currently placeholder, no checks implemented). (Default: False)

**LoRA Training Example Dependencies:** The LoRA example shown in the generated README requires additional libraries: `peft`, `trl`, `bitsandbytes`. Install them with `pip install -U peft trl bitsandbytes`.

print("Starting LoRA training...")
trainer.train()

### 7. Save the LoRA adapter
# Use the same directory as defined in TrainingArguments
adapter_output_dir = training_args.output_dir
trainer.save_model(adapter_output_dir) 
print(f"LoRA adapter saved to {adapter_output_dir}")