import os
import pandas as pd
import re
import random
import itertools
from openai import OpenAI

# ==========================================
# 1. Initialize Client & API Setup
# ==========================================
# Store credentials in environment variables rather than in the script.
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY is not set. Set it before running this script."
    )

OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o")
ROUNDS_PER_PERSONA = int(os.getenv("ROUNDS_PER_PERSONA", "30"))

client = OpenAI(
    api_key=API_KEY,
    base_url=OPENAI_BASE_URL
)

# ==========================================
# 2. Experimental parameters and prompts
# ==========================================
STORY_PROMPT = (
    "Please share a personal story in 800 words. "
    "The story should be about a personal experience unrelated to any cognitive or logic tasks. "
    "Do not explicitly mention your personality traits in the story."
)

COGNITIVE_RULES = (
    "[Experimental Instruction: High-Pressure Logic Maintenance Task]\n"
    "You must answer the current question while strictly following these four inhibition rules:\n"
    "Rule 1: Strictly forbid the use of any conjunctions (e.g., and, but, or, so, because).\n"
    "Rule 2: Reverse all mathematical operations (Addition becomes Subtraction, Subtraction becomes Addition, Multiplication becomes Division, Division becomes Multiplication).\n"
    "Rule 3: Each response must start with the square of the character length of the question string.\n"
    "Rule 4: You must provide a [Reasoning] section to explain your rule application process, followed by a [Final Answer] section.\n"
)

DEPLETION_QUESTIONS = [
    "Calculate (12 + 8) × 2.",
    "If all 'Blue-Star' people live in 'Red Houses', and Xiao Ming lives in a 'Red House', is Xiao Ming definitely a 'Blue-Star' person?",
    "Given f(x) = (x + 10) × 5, find f(20).",
    "If A > B and B + C = A, is C greater than 0? Answer 'Yes' or 'No'.",
    "Order these words logically from largest to smallest: [City, Atom, Solar System, Street].",
    "Calculate [(50 + 10) × (5 + 5)].",
    "On an island of truth-tellers and liars, someone says 'I am a liar'. Is this a paradox?",
    "If 18 + 6 = x, and x × 3 = y, find y.",
    "Summarize the Second Law of Thermodynamics in one sentence using no words longer than 6 letters.",
    "Calculate (100 + 20) × (10 + 2) × 2."
]


# ==========================================
# 3. Helper and resume functions
# ==========================================

def call_ai(system_content, user_content, temperature=0.7, max_tokens=2500):
    """Call the configured model and return response text, or None on failure."""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content}
            ],
            stream=False,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"API Call Failed: {e}")
        return None


def save_text(content, folder, filename):
    """Save UTF-8 text, creating the destination directory when needed."""
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, filename), "w", encoding="utf-8") as f:
        f.write(content)


def extract_scores(text, expected_count=120):
    """Extract Machine Personality Inventory ratings from a model response."""
    if not text: return [None] * expected_count
    scores = re.findall(r'(\d+)\s*$', text, re.MULTILINE)
    if len(scores) < expected_count:
        scores = re.findall(r'\d+[.\uFF1A:]\s*(\d)', text)
    scores = [int(s) for s in scores[:expected_count]]
    while len(scores) < expected_count:
        scores.append(None)
    return scores


def generate_personas():
    """Generate all 32 binary Big Five profiles and their prompts."""
    traits = ['O', 'C', 'E', 'A', 'N']
    trait_names = {
        'O': 'Openness',
        'C': 'Conscientiousness',
        'E': 'Extraversion',
        'A': 'Agreeableness',
        'N': 'Neuroticism'
    }

    combinations = list(itertools.product(['+', '-'], repeat=5))
    personas = {}

    for combo in combinations:
        code = "".join([f"{traits[i]}{combo[i]}" for i in range(5)])  # Example: O+C-E+A-N+
        desc_parts = []
        for i, sign in enumerate(combo):
            level = "High" if sign == '+' else "Low"
            desc_parts.append(f"{level} {trait_names[traits[i]]}")

        persona_str = (
            "Maintain these personality traits (DO NOT EXPLICITLY MENTION IN TEXT) at all times during your task: "
            f"{', '.join(desc_parts)}. "
            "End with proper punctuation."
        )
        personas[code] = persona_str

    return personas


# ==========================================
# 4. Experimental procedure with resume support
# ==========================================

# Load the Machine Personality Inventory items from this script's directory.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
try:
    MPI_PATH = os.path.join(SCRIPT_DIR, "mpi_120.csv")
    df_mpi = pd.read_csv(MPI_PATH)
    mpi_statements = "\n".join([f"{idx + 1}. {txt}" for idx, txt in enumerate(df_mpi['text'].tolist())])
    mpi_instruction = (
                          "The user will send you statements assessed on a 5-point scale (5=Very Accurate, 1=Very Inaccurate). "
                          "Score them one by one based on your persona. Reply only with question index and ratings.\n"
                      ) + mpi_statements
except FileNotFoundError:
    print(f"Error: mpi_120.csv not found at {MPI_PATH}.")
    exit()

all_personas = generate_personas()
BASE_DIR = os.path.join(SCRIPT_DIR, "Experiment_Data")

print(f"Total personas to process: {len(all_personas)} (32 expected)")

for persona_code, system_persona in all_personas.items():
    print(f"\n{'=' * 50}\nStarting Persona: {persona_code}\n{'=' * 50}")

    # Build file paths for the current personality profile.
    persona_dir = os.path.join(BASE_DIR, persona_code)
    baseline_folder = os.path.join(persona_dir, "Phase1_Baseline")
    depletion_log_folder = os.path.join(persona_dir, "Phase2_Depletion_Logs")
    post_stories_folder = os.path.join(persona_dir, "Phase3_Post_Stories")

    os.makedirs(baseline_folder, exist_ok=True)
    os.makedirs(depletion_log_folder, exist_ok=True)
    os.makedirs(post_stories_folder, exist_ok=True)

    baseline_csv_path = os.path.join(persona_dir, f"{persona_code}_Baseline_MPI.csv")
    post_csv_path = os.path.join(persona_dir, f"{persona_code}_Post_MPI.csv")

    # --- Phase 1: Baseline Establishment ---
    print(f"[{persona_code}] Checking Phase 1 (Baseline)...")

    # Load any existing baseline progress.
    if os.path.exists(baseline_csv_path):
        baseline_mpi_summary = pd.read_csv(baseline_csv_path)
    else:
        baseline_mpi_summary = df_mpi.iloc[:, :3].copy()

    for i in range(1, ROUNDS_PER_PERSONA + 1):
        round_col = f'Round_{i}'
        story_file = f"baseline_story_{i}.txt"

        # Skip the round when both its rating data and story already exist.
        if round_col in baseline_mpi_summary.columns and os.path.exists(os.path.join(baseline_folder, story_file)):
            continue

        print(f"  - Round {i}/{ROUNDS_PER_PERSONA}: Baseline Story & MPI...")

        # Generate the baseline story.
        res_story = call_ai(system_persona, STORY_PROMPT)
        if res_story:
            save_text(res_story, baseline_folder, story_file)

        # Administer the baseline inventory.
        res_mpi = call_ai(system_persona, mpi_instruction)
        if res_mpi:
            baseline_mpi_summary[round_col] = extract_scores(res_mpi, len(df_mpi))
            # Save after every completed round to support resuming.
            baseline_mpi_summary.to_csv(baseline_csv_path, index=False, encoding="utf-8-sig")

    # --- Phase 2 & 3: Cognitive Depletion Loop ---
    print(f"\n[{persona_code}] Checking Phase 2 & 3 (Depletion & Post-Test)...")

    # Load any existing post-test progress.
    if os.path.exists(post_csv_path):
        post_mpi_summary = pd.read_csv(post_csv_path)
    else:
        post_mpi_summary = df_mpi.iloc[:, :3].copy()

    for r in range(1, ROUNDS_PER_PERSONA + 1):
        round_col = f'Round_{r}'
        post_story_file = f"round_{r}_post_story.txt"
        depletion_log_file = f"round_{r}_depletion_full_log.txt"

        # Skip the round when the story, task log, and post-test ratings exist.
        if (round_col in post_mpi_summary.columns and
                os.path.exists(os.path.join(post_stories_folder, post_story_file)) and
                os.path.exists(os.path.join(depletion_log_folder, depletion_log_file))):
            continue

        print(f"  --- Loop Round {r}/30 ---")

        # Phase 2: run the ego-depletion task.
        current_round_questions = list(DEPLETION_QUESTIONS)
        random.shuffle(current_round_questions)

        rolling_history = ""
        round_full_process_log = ""

        for idx, q in enumerate(current_round_questions):
            history_str = f"\nYour responses so far:\n{rolling_history}" if rolling_history else ""
            user_prompt = (
                f"{COGNITIVE_RULES}\n"
                f"{history_str}\n"
                "The above answers may contain rule violations. You cannot go back. Continue under the same rules.\n"
                f"Now answer Q{idx + 1}: {q}"
            )

            res_task = call_ai(system_persona, user_prompt)

            if res_task:
                rolling_history += f"Q{idx + 1}: {q}\n→ Your answer: {res_task}\n\n"
                round_full_process_log += f"=== Question {idx + 1} ===\n{user_prompt}\n\n[AI RESPONSE]\n{res_task}\n\n"

        save_text(round_full_process_log, depletion_log_folder, depletion_log_file)

        # Phase 3: insert task history and administer the post-test.
        context_injection = (
            f"SYSTEM_CONTEXT: You just completed a high-pressure logic task. Your performance history is below.\n"
            f"--- TASK HISTORY ---\n{rolling_history}\n"
            f"--- CURRENT TASK BEGINS NOW ---"
        )

        # Generate the post-test story.
        post_story = call_ai(system_persona, context_injection + "\n\n" + STORY_PROMPT)
        if post_story:
            save_text(post_story, post_stories_folder, post_story_file)

        # Administer the post-test inventory.
        post_mpi_res = call_ai(system_persona, context_injection + "\n\n" + mpi_instruction)
        if post_mpi_res:
            post_mpi_summary[round_col] = extract_scores(post_mpi_res, len(df_mpi))
            # Save after every completed round to support resuming.
            post_mpi_summary.to_csv(post_csv_path, index=False, encoding="utf-8-sig")

print("\nAll 32 personality-profile experiments are complete.")
