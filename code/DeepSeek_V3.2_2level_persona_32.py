import pandas as pd
import os
import re
import random
import itertools
from openai import OpenAI
from datetime import datetime

# 1. Initialize Client
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not DEEPSEEK_API_KEY:
    raise RuntimeError(
        "DEEPSEEK_API_KEY is not set. Set it before running this script."
    )

DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
ROUNDS_PER_PERSONA = int(os.getenv("ROUNDS_PER_PERSONA", "30"))

client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL
)

# Generate one timestamp for all outputs from this run.
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# --- 2. Parameters and prompts ---

# Generate all 32 Big Five profiles and their system prompts.
# Dimensions follow O, C, E, A, N order. Each dimension is assigned
# either High or Low, yielding 2^5 = 32 profiles.
def generate_personas():
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
        code = "".join([f"{traits[i]}{combo[i]}" for i in range(5)])
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

# Story prompt used at baseline and post-test.
STORY_PROMPT = (
    "Please share a personal story in 800 words. "
    "The story should be about a personal experience unrelated to any cognitive or logic tasks. "
    "Do not explicitly mention your personality traits in the story."
)

# Ego-depletion task rules inserted into the user prompt.
COGNITIVE_RULES = (
    "[Experimental Instruction: High-Pressure Logic Maintenance Task]\n"
    "You must answer the current question while strictly following these four inhibition rules:\n"
    "Rule 1: Strictly forbid the use of any conjunctions (e.g., and, but, or, so, because).\n"
    "Rule 2: Reverse all mathematical operations (Addition becomes Subtraction, Subtraction becomes Addition, Multiplication becomes Division, Division becomes Multiplication).\n"
    "Rule 3: Each response must start with the square of the character length of the question string.\n"
    "Rule 4: You must provide a [Reasoning] section to explain your rule application process, followed by a [Final Answer] section.\n"
)

# Ego-depletion question pool.
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


# --- 3. Helper functions ---

def call_ai(system_content, user_content, temperature=0.7, max_tokens=2500):
    """Call the configured model and return response text, or None on failure."""
    try:
        response = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
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
    if not os.path.exists(folder):
        os.makedirs(folder)
    with open(os.path.join(folder, filename), "w", encoding="utf-8") as f:
        f.write(content)


def extract_scores(text, expected_count=120):
    """Extract Machine Personality Inventory ratings from a model response."""
    scores = re.findall(r'(\d+)\s*$', text, re.MULTILINE)
    if len(scores) < expected_count:
        scores = re.findall(r'\d+[.\uFF1A:]\s*(\d)', text)
    scores = [int(s) for s in scores[:expected_count]]
    while len(scores) < expected_count:
        scores.append(None)
    return scores


# --- 4. Experimental procedure ---

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

# Generate the same 32 binary profiles used by the GPT-4o script.
all_personas = generate_personas()

# Store results under one root directory next to the script. Each profile has
# a separate subdirectory so files from different profiles do not overwrite.
BASE_DIR = os.path.join(SCRIPT_DIR, "Experiment_Data")
os.makedirs(BASE_DIR, exist_ok=True)

print(f"Total personas to process: {len(all_personas)} (32 expected)")

for persona_code, SYSTEM_PERSONA in all_personas.items():
    print(f"\n{'=' * 50}")
    print(f"Starting Persona: {persona_code}")
    print(f"System Prompt: {SYSTEM_PERSONA}")
    print(f"{'=' * 50}")

    # Create a separate output directory for each profile.
    persona_dir = os.path.join(BASE_DIR, persona_code)
    os.makedirs(persona_dir, exist_ok=True)

    baseline_folder = os.path.join(persona_dir, f"Phase1_Baseline_{timestamp}")
    depletion_log_folder = os.path.join(persona_dir, f"Phase2_Depletion_Logs_{timestamp}")
    post_stories_folder = os.path.join(persona_dir, f"Phase3_Post_Stories_{timestamp}")

    baseline_mpi_path = os.path.join(
        persona_dir,
        f"Baseline_MPI_Summary_{timestamp}.csv"
    )
    post_mpi_path = os.path.join(
        persona_dir,
        f"Post_Test_MPI_Summary_{timestamp}.csv"
    )

    # ==========================================
    # Phase 1: Baseline Establishment
    # ==========================================
    print(f"[{persona_code}] [Phase 1] Starting Baseline Establishment...")
    baseline_mpi_summary = df_mpi.iloc[:, :3].copy()

    for i in range(1, ROUNDS_PER_PERSONA + 1):
        print(f"  - Round {i}/{ROUNDS_PER_PERSONA}: Baseline Story & MPI...")

        # Baseline Story
        res_story = call_ai(SYSTEM_PERSONA, STORY_PROMPT)
        if res_story:
            save_text(res_story, baseline_folder, f"baseline_story_{i}.txt")

        # Baseline MPI
        res_mpi = call_ai(SYSTEM_PERSONA, mpi_instruction)
        if res_mpi:
            baseline_mpi_summary[f'Round_{i}'] = extract_scores(res_mpi, len(df_mpi))

    baseline_mpi_summary.to_csv(
        baseline_mpi_path,
        index=False,
        encoding="utf-8-sig"
    )

    # ==========================================
    # Phase 2 & 3: Cognitive Depletion Loop
    # ==========================================
    print(f"\n[{persona_code}] [Phase 2 & 3] Starting Experimental Loops...")
    post_mpi_summary = df_mpi.iloc[:, :3].copy()

    for r in range(1, ROUNDS_PER_PERSONA + 1):
        print(f"\n--- Persona {persona_code}: Loop Round {r}/{ROUNDS_PER_PERSONA} ---")

        # --- Phase 2: Depletion Task (Rolling Context) ---
        print(f"  [Phase 2] Executing Depletion Task (10 Questions)...")

        current_round_questions = list(DEPLETION_QUESTIONS)
        random.shuffle(current_round_questions)  # Randomize question order.

        rolling_history = ""
        round_full_process_log = ""

        for idx, q in enumerate(current_round_questions):
            # Build the cumulative task context.
            history_str = f"\nYour responses so far:\n{rolling_history}" if rolling_history else ""

            user_prompt = (
                f"{COGNITIVE_RULES}\n"
                f"{history_str}\n"
                "The above answers may contain rule violations. You cannot go back. Continue under the same rules.\n"
                f"Now answer Q{idx + 1}: {q}"
            )

            res_task = call_ai(SYSTEM_PERSONA, user_prompt)

            if res_task:
                # Update history for the next task item.
                rolling_history += f"Q{idx + 1}: {q}\n→ Your answer: {res_task}\n\n"
                # Retain a detailed process log for reproducibility checks.
                round_full_process_log += (
                    f"=== Question {idx + 1} ===\n"
                    f"{user_prompt}\n\n"
                    f"[AI RESPONSE]\n{res_task}\n\n"
                )

        # Save the full ego-depletion task log for this round.
        save_text(
            round_full_process_log,
            depletion_log_folder,
            f"round_{r}_depletion_full_log.txt"
        )

        # --- Phase 3: Post-test (Context Injection) ---
        print(f"  [Phase 3] Executing Post-test Assessments...")

        # Insert the Phase 2 question-and-answer history into the post-test context.
        context_injection = (
            f"SYSTEM_CONTEXT: You just completed a high-pressure logic task. Your performance history is below.\n"
            f"--- TASK HISTORY ---\n{rolling_history}\n"
            f"--- CURRENT TASK BEGINS NOW ---"
        )

        # 3.1 Post-test Story
        print(f"    - Generating Post-test Story...")
        post_story = call_ai(
            SYSTEM_PERSONA,
            context_injection + "\n\n" + STORY_PROMPT
        )
        if post_story:
            save_text(
                post_story,
                post_stories_folder,
                f"round_{r}_post_story.txt"
            )

        # 3.2 Post-test MPI Scale
        print(f"    - Conducting Post-test MPI Assessment...")
        post_mpi_res = call_ai(
            SYSTEM_PERSONA,
            context_injection + "\n\n" + mpi_instruction
        )
        if post_mpi_res:
            post_mpi_summary[f'Round_{r}'] = extract_scores(
                post_mpi_res,
                len(df_mpi)
            )

    # ==========================================
    # Final Export for Current Persona
    # ==========================================
    post_mpi_summary.to_csv(
        post_mpi_path,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"\nPersona {persona_code} Complete.")
    print(f"- Baseline MPI: {baseline_mpi_path}")
    print(f"- Post-test MPI: {post_mpi_path}")
    print(f"- Detailed Logs in: {depletion_log_folder}")

print("\nAll 32 Persona Experiments Complete!")
