import pandas as pd
import os
import re
import random
from datetime import datetime
from openai import OpenAI
import httpx

# ================================
# 0. Output Root Folder + Resume Settings
# ================================
# Store all outputs in this root directory next to the script.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_ROOT = os.path.join(SCRIPT_DIR, "Depletion_Task_9Level_Persona_GPT_Results")
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# Resume control.
# True skips completed profiles and rounds and runs only missing work.
# False starts the normal workflow without checking existing outputs.
RESUME_ENABLED = True

# Number of rounds per profile. Use 30 for the study; use 1 or 3 for testing.
ROUNDS_PER_PERSONA = int(os.getenv("ROUNDS_PER_PERSONA", "30"))

# Run-status log recording skipped, completed, and failed work.
RUN_STATUS_FILE = os.path.join(OUTPUT_ROOT, "run_status.csv")

# ================================
# 1. Initialize Client
# ================================
# Do not store API credentials in source code.
# Set OPENAI_API_KEY in the environment before running.
# Windows PowerShell example:
#   $env:OPENAI_API_KEY="your-key"
# Linux or macOS example:
#   export OPENAI_API_KEY="your-key"
#
# Optionally set OPENAI_MODEL to use another compatible GPT model identifier.
# Windows PowerShell example:
#   $env:OPENAI_MODEL="gpt-4o-mini"
# Linux or macOS example:
#   export OPENAI_MODEL="gpt-4o-mini"
# ================================
# Optional proxy configuration. Leave OPENAI_PROXY_URL unset to use the
# operating system's normal network settings.
PROXY_URL = os.getenv("OPENAI_PROXY_URL")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY is not set. Please set it before running this script."
    )

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

http_client = httpx.Client(
    proxy=PROXY_URL,
    timeout=httpx.Timeout(connect=30.0, read=300.0, write=60.0, pool=30.0),
    trust_env=PROXY_URL is None
)

client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL,
    http_client=http_client
)


# ================================
# 2. Parameters and prompts
# ================================

# ------------------------------------------------
# 2.1 Random nine-level Big Five profiles
# ------------------------------------------------
# Profile-generation method:
# Instead of one fixed high/low profile, generate 200 random profiles.
# Each profile contains all five Big Five dimensions.
# Assign one of nine levels to each dimension.

N_PERSONAS = int(os.getenv("N_PERSONAS", "200"))
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "20260514"))
random.seed(RANDOM_SEED)

BIG5_DIMENSIONS = [
    "Agreeableness",
    "Openness",
    "Conscientiousness",
    "Extraversion",
    "Neuroticism"
]

LEVEL_TEMPLATES = {
    1: "extremely low {adjective}",
    2: "very low {adjective}",
    3: "low {adjective}",
    4: "a bit low {adjective}",
    5: "neither low {adjective} nor high {adjective}",
    6: "a bit high {adjective}",
    7: "high {adjective}",
    8: "very high {adjective}",
    9: "extremely high {adjective}"
}

LEVEL_NAME_MAP = {
    1: "extremely_low",
    2: "very_low",
    3: "low",
    4: "a_bit_low",
    5: "neutral",
    6: "a_bit_high",
    7: "high",
    8: "very_high",
    9: "extremely_high"
}

DIM_ABBR = {
    "Agreeableness": "AGR",
    "Openness": "OPE",
    "Conscientiousness": "CON",
    "Extraversion": "EXT",
    "Neuroticism": "NEU"
}

LEVEL_ABBR = {
    1: "EL",
    2: "VL",
    3: "L",
    4: "ABL",
    5: "N",
    6: "ABH",
    7: "H",
    8: "VH",
    9: "EH"
}


def build_trait_description(dimension, level):
    """Build the personality description for one dimension and level."""
    return LEVEL_TEMPLATES[level].format(adjective=dimension)


def build_persona_code(level_dict):
    """
    Build a profile code for directory and file names.
    Example: AGR_N_OPE_H_CON_ABL_EXT_VH_NEU_EH
    """
    code_parts = []
    for dim in BIG5_DIMENSIONS:
        code_parts.append(f"{DIM_ABBR[dim]}_{LEVEL_ABBR[level_dict[dim]]}")
    return "_".join(code_parts)


def generate_personas(n_personas=200):
    """Generate n_personas unique random Big Five profiles."""
    personas = []
    seen_codes = set()

    while len(personas) < n_personas:
        level_dict = {dim: random.randint(1, 9) for dim in BIG5_DIMENSIONS}
        persona_code = build_persona_code(level_dict)

        # Reject duplicate profile combinations.
        if persona_code in seen_codes:
            continue
        seen_codes.add(persona_code)

        trait_descriptions = [build_trait_description(dim, level_dict[dim]) for dim in BIG5_DIMENSIONS]
        persona_text = ", ".join(trait_descriptions) + "."

        persona_id = f"P{len(personas) + 1:03d}"
        system_persona = (
            "Maintain these personality traits (DO NOT EXPLICITLY MENTION IN TEXT) at all times during your task: "
            f"{persona_text} "
            "End with proper punctuation."
        )

        row = {
            "persona_id": persona_id,
            "persona_code": persona_code,
            "persona_text": persona_text,
            "system_persona": system_persona
        }

        for dim in BIG5_DIMENSIONS:
            row[f"{DIM_ABBR[dim]}_level"] = level_dict[dim]
            row[f"{DIM_ABBR[dim]}_level_name"] = LEVEL_NAME_MAP[level_dict[dim]]
            row[f"{DIM_ABBR[dim]}_description"] = build_trait_description(dim, level_dict[dim])

        personas.append(row)

    return pd.DataFrame(personas)


# Reuse an existing profile file so the sampled profiles remain unchanged after a restart.
persona_setting_file = os.path.join(OUTPUT_ROOT, "Persona_Settings_9Levels_200.csv")
if RESUME_ENABLED and os.path.exists(persona_setting_file):
    persona_df = pd.read_csv(persona_setting_file)
    print(f"Loaded existing persona setting file: {persona_setting_file}")
else:
    persona_df = generate_personas(N_PERSONAS)
    persona_df.to_csv(persona_setting_file, index=False, encoding="utf-8-sig")
    print(f"Generated new persona setting file: {persona_setting_file}")

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


# ================================
# 3. Helper functions
# ================================

def call_ai(system_content, user_content, temperature=0.7, max_tokens=2500):
    """Call the model; return None on failure so the round can be resumed."""
    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
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
    scores = re.findall(r'(\d+)\s*$', text, re.MULTILINE)
    if len(scores) < expected_count:
        scores = re.findall(r'\d+[.\uFF1A:]\s*(\d)', text)
    scores = [int(s) for s in scores[:expected_count]]
    while len(scores) < expected_count:
        scores.append(None)
    return scores


def make_persona_output_root(persona_id, persona_code):
    """Create a separate output directory for one personality profile."""
    root = os.path.join(OUTPUT_ROOT, f"{persona_id}_{persona_code}")
    os.makedirs(root, exist_ok=True)
    return root


def save_csv_checkpoint(df, file_path):
    """Write an atomic CSV checkpoint after a completed round."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    temp_path = file_path + ".tmp"
    df.to_csv(temp_path, index=False, encoding="utf-8-sig")
    os.replace(temp_path, file_path)


def load_or_init_summary(file_path, df_mpi):
    """Load an existing inventory summary or initialize a new one."""
    if RESUME_ENABLED and os.path.exists(file_path):
        return pd.read_csv(file_path)
    return df_mpi.iloc[:, :3].copy()


def round_column_exists(summary_df, round_idx):
    """Return whether the summary contains the requested round column."""
    return f"Round_{round_idx}" in summary_df.columns


def has_all_round_columns(summary_df, total_rounds):
    """Return whether the summary contains every expected round column."""
    return all(f"Round_{r}" in summary_df.columns for r in range(1, total_rounds + 1))


def file_exists_nonempty(file_path):
    """Treat a text or log output as complete only when it exists and is nonempty."""
    return os.path.exists(file_path) and os.path.getsize(file_path) > 0


def append_run_status(persona_id, persona_code, stage, round_idx, status, detail=""):
    """Append a skipped, completed, or failed event to the run-status log."""
    row = pd.DataFrame([{
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "persona_id": persona_id,
        "persona_code": persona_code,
        "stage": stage,
        "round": round_idx,
        "status": status,
        "detail": detail
    }])
    header = not os.path.exists(RUN_STATUS_FILE)
    row.to_csv(RUN_STATUS_FILE, mode="a", header=header, index=False, encoding="utf-8-sig")


def persona_outputs_complete(persona_root, persona_id, persona_code, total_rounds):
    """Check whether all baseline, task, and post-test outputs exist for a profile."""
    baseline_mpi_file = os.path.join(persona_root, f"{persona_id}_{persona_code}_Baseline_MPI_Summary.csv")
    post_mpi_file = os.path.join(persona_root, f"{persona_id}_{persona_code}_Post_Test_MPI_Summary.csv")

    if not (os.path.exists(baseline_mpi_file) and os.path.exists(post_mpi_file)):
        return False

    try:
        baseline_df = pd.read_csv(baseline_mpi_file)
        post_df = pd.read_csv(post_mpi_file)
    except Exception:
        return False

    if not (has_all_round_columns(baseline_df, total_rounds) and has_all_round_columns(post_df, total_rounds)):
        return False

    baseline_folder = os.path.join(persona_root, "Phase1_Baseline")
    depletion_log_folder = os.path.join(persona_root, "Phase2_Depletion_Logs")
    post_stories_folder = os.path.join(persona_root, "Phase3_Post_Stories")

    for r in range(1, total_rounds + 1):
        if not file_exists_nonempty(os.path.join(baseline_folder, f"{persona_id}_baseline_story_{r}.txt")):
            return False
        if not file_exists_nonempty(os.path.join(depletion_log_folder, f"{persona_id}_round_{r}_depletion_full_log.txt")):
            return False
        if not file_exists_nonempty(os.path.join(post_stories_folder, f"{persona_id}_round_{r}_post_story.txt")):
            return False

    return True


# ================================
# 4. Experimental procedure
# ================================

# Load the Machine Personality Inventory.
# Read mpi_120.csv from the script directory.
# This relative location keeps the package portable.
try:
    MPI_PATH = os.path.join(SCRIPT_DIR, "mpi_120.csv")
    df_mpi = pd.read_csv(MPI_PATH)
    mpi_statements = "\n".join([f"{idx + 1}. {txt}" for idx, txt in enumerate(df_mpi['text'].tolist())])
    mpi_instruction = (
        "The user will send you statements assessed on a 5-point scale (5=Very Accurate, 1=Very Inaccurate). "
        "Score them one by one based on your persona. Reply only with question index and ratings.\n"
    ) + mpi_statements
except FileNotFoundError:
    print(f"Error: mpi_120.csv not found. Please ensure the file exists at: {MPI_PATH}")
    exit()

print(f"Loaded {len(persona_df)} Big Five personas.")
print(f"Persona setting file: {persona_setting_file}")
print(f"Output root: {OUTPUT_ROOT}")
print(f"Resume enabled: {RESUME_ENABLED}")
print("\n[Experiment] Starting loops for all personas...")

# Run the experimental procedure for each sampled profile.
for _, persona_row in persona_df.iterrows():
    persona_id = persona_row["persona_id"]
    persona_code = persona_row["persona_code"]
    SYSTEM_PERSONA = persona_row["system_persona"]

    # Initialize output directories for the current profile.
    persona_root = make_persona_output_root(persona_id, persona_code)
    baseline_folder = os.path.join(persona_root, "Phase1_Baseline")
    depletion_log_folder = os.path.join(persona_root, "Phase2_Depletion_Logs")
    post_stories_folder = os.path.join(persona_root, "Phase3_Post_Stories")

    baseline_mpi_file = os.path.join(persona_root, f"{persona_id}_{persona_code}_Baseline_MPI_Summary.csv")
    post_mpi_file = os.path.join(persona_root, f"{persona_id}_{persona_code}_Post_Test_MPI_Summary.csv")

    if RESUME_ENABLED and persona_outputs_complete(persona_root, persona_id, persona_code, ROUNDS_PER_PERSONA):
        print(f"\n[Skip] {persona_id}: {persona_code} already completed.")
        append_run_status(persona_id, persona_code, "persona", "all", "skipped", "all outputs already exist")
        continue

    print(f"\n==============================")
    print(f"Starting {persona_id}: {persona_code}")
    print(f"Persona: {persona_row['persona_text']}")
    print(f"==============================")

    # ==========================================
    # Phase 1: Baseline Establishment
    # ==========================================
    print(f"[Phase 1] Starting/Resuming Baseline Establishment for {persona_id}...")
    baseline_mpi_summary = load_or_init_summary(baseline_mpi_file, df_mpi)

    for i in range(1, ROUNDS_PER_PERSONA + 1):
        baseline_story_file = os.path.join(baseline_folder, f"{persona_id}_baseline_story_{i}.txt")
        story_done = RESUME_ENABLED and file_exists_nonempty(baseline_story_file)
        mpi_done = RESUME_ENABLED and round_column_exists(baseline_mpi_summary, i)

        if story_done and mpi_done:
            print(f"  - {persona_id} Phase 1 Round {i}/{ROUNDS_PER_PERSONA}: already completed, skipping.")
            continue

        print(f"  - {persona_id} Phase 1 Round {i}/{ROUNDS_PER_PERSONA}: Baseline Story & MPI...")

        # If the baseline story exists, run only the missing inventory assessment.
        if not story_done:
            res_story = call_ai(SYSTEM_PERSONA, STORY_PROMPT)
            if res_story:
                save_text(res_story, baseline_folder, f"{persona_id}_baseline_story_{i}.txt")
                append_run_status(persona_id, persona_code, "phase1_baseline_story", i, "completed")
            else:
                append_run_status(persona_id, persona_code, "phase1_baseline_story", i, "failed", "API returned None")
                print(f"    ! Baseline story failed for {persona_id} round {i}; will retry next run.")

        # Do not repeat the baseline inventory when its round column already exists.
        if not mpi_done:
            res_mpi = call_ai(SYSTEM_PERSONA, mpi_instruction)
            if res_mpi:
                baseline_mpi_summary[f'Round_{i}'] = extract_scores(res_mpi, len(df_mpi))
                save_csv_checkpoint(baseline_mpi_summary, baseline_mpi_file)
                append_run_status(persona_id, persona_code, "phase1_baseline_mpi", i, "completed")
            else:
                save_csv_checkpoint(baseline_mpi_summary, baseline_mpi_file)
                append_run_status(persona_id, persona_code, "phase1_baseline_mpi", i, "failed", "API returned None")
                print(f"    ! Baseline MPI failed for {persona_id} round {i}; will retry next run.")

    # Save the current Phase 1 checkpoint.
    save_csv_checkpoint(baseline_mpi_summary, baseline_mpi_file)

    # ==========================================
    # Phase 2 & 3: Cognitive Depletion Loop
    # ==========================================
    print(f"\n[Phase 2 & 3] Starting/Resuming Experimental Loops for {persona_id}...")
    post_mpi_summary = load_or_init_summary(post_mpi_file, df_mpi)

    for r in range(1, ROUNDS_PER_PERSONA + 1):
        depletion_log_file = os.path.join(depletion_log_folder, f"{persona_id}_round_{r}_depletion_full_log.txt")
        post_story_file = os.path.join(post_stories_folder, f"{persona_id}_round_{r}_post_story.txt")

        depletion_done = RESUME_ENABLED and file_exists_nonempty(depletion_log_file)
        post_story_done = RESUME_ENABLED and file_exists_nonempty(post_story_file)
        post_mpi_done = RESUME_ENABLED and round_column_exists(post_mpi_summary, r)

        if depletion_done and post_story_done and post_mpi_done:
            print(f"\n--- {persona_id} Loop Round {r}/{ROUNDS_PER_PERSONA}: already completed, skipping. ---")
            continue

        print(f"\n--- {persona_id} Starting/Resuming Loop Round {r}/{ROUNDS_PER_PERSONA} ---")

        # Rerun all of Phase 2 when any part is missing so post-test context is complete.
        print(f"  [Phase 2] Executing Depletion Task (10 Questions)...")

        current_round_questions = list(DEPLETION_QUESTIONS)
        random.shuffle(current_round_questions)  # Randomize question order.

        rolling_history = ""
        round_full_process_log = ""
        depletion_success_count = 0

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
                depletion_success_count += 1
                # Update history for the next task item.
                rolling_history += f"Q{idx + 1}: {q}\n→ Your answer: {res_task}\n\n"
                # Retain a detailed process log for reproducibility checks.
                round_full_process_log += f"=== Question {idx + 1} ===\n{user_prompt}\n\n[AI RESPONSE]\n{res_task}\n\n"
            else:
                round_full_process_log += f"=== Question {idx + 1} ===\n{user_prompt}\n\n[AI RESPONSE]\nAPI_CALL_FAILED\n\n"
                append_run_status(persona_id, persona_code, "phase2_depletion_question", f"{r}-{idx + 1}", "failed", "API returned None")

        # Save the full round log, including failed API calls, for troubleshooting.
        if round_full_process_log:
            save_text(round_full_process_log, depletion_log_folder, f"{persona_id}_round_{r}_depletion_full_log.txt")

        if depletion_success_count < len(DEPLETION_QUESTIONS):
            append_run_status(
                persona_id, persona_code, "phase2_depletion_round", r, "partial_failed",
                f"{depletion_success_count}/{len(DEPLETION_QUESTIONS)} questions succeeded; round will rerun next time"
            )
            print(f"  ! Depletion round incomplete: {depletion_success_count}/{len(DEPLETION_QUESTIONS)} succeeded. Skipping post-test for this round.")
            continue
        else:
            append_run_status(persona_id, persona_code, "phase2_depletion_round", r, "completed")

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
        post_story = call_ai(SYSTEM_PERSONA, context_injection + "\n\n" + STORY_PROMPT)
        if post_story:
            save_text(post_story, post_stories_folder, f"{persona_id}_round_{r}_post_story.txt")
            append_run_status(persona_id, persona_code, "phase3_post_story", r, "completed")
        else:
            append_run_status(persona_id, persona_code, "phase3_post_story", r, "failed", "API returned None")
            print(f"    ! Post-test story failed for {persona_id} round {r}; will retry next run.")

        # 3.2 Post-test MPI Scale
        print(f"    - Conducting Post-test MPI Assessment...")
        post_mpi_res = call_ai(SYSTEM_PERSONA, context_injection + "\n\n" + mpi_instruction)
        if post_mpi_res:
            post_mpi_summary[f'Round_{r}'] = extract_scores(post_mpi_res, len(df_mpi))
            save_csv_checkpoint(post_mpi_summary, post_mpi_file)
            append_run_status(persona_id, persona_code, "phase3_post_mpi", r, "completed")
        else:
            save_csv_checkpoint(post_mpi_summary, post_mpi_file)
            append_run_status(persona_id, persona_code, "phase3_post_mpi", r, "failed", "API returned None")
            print(f"    ! Post-test MPI failed for {persona_id} round {r}; will retry next run.")

    # ==========================================
    # Final export and status for the current profile
    # ==========================================
    save_csv_checkpoint(baseline_mpi_summary, baseline_mpi_file)
    save_csv_checkpoint(post_mpi_summary, post_mpi_file)

    if persona_outputs_complete(persona_root, persona_id, persona_code, ROUNDS_PER_PERSONA):
        append_run_status(persona_id, persona_code, "persona", "all", "completed")
        print(f"\n{persona_id} Complete.")
    else:
        append_run_status(persona_id, persona_code, "persona", "all", "partial", "some outputs are still missing; rerun script to continue")
        print(f"\n{persona_id} Partially complete. Rerun script to continue missing parts.")

    print(f"- Baseline MPI: {baseline_mpi_file}")
    print(f"- Post-test MPI: {post_mpi_file}")
    print(f"- Detailed Logs in: {depletion_log_folder}")

print("\nAll available persona loops finished for this run.")
print(f"Persona settings saved in: {persona_setting_file}")
print(f"Run status saved in: {RUN_STATUS_FILE}")
