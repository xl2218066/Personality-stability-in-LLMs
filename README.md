# Ego-depletion tasks and personality stability in large language models

## Description

This repository contains the Python code and summary datasets for the study "Ego-depletion tasks affect personality stability in large language models." The study compares the Big Five personality scores of GPT-4o and DeepSeek-V3.2 before and after a rule-intensive ego-depletion task. It includes a binary personality-assignment design with 32 profiles and a nine-level design with 200 sampled profiles.

The code files have English comments and documentation. API credentials are read from environment variables and are not stored in the repository.

## Repository contents

```text
README.md
requirements.txt
code/
  DeepSeek_V3.2_2level_persona_32.py
  DeepSeek_V3.2_9level_persona_200.py
  GPT-4o_2level_persona_32.py
  GPT-4o_9level_persona_200.py
data/
  DeepSeek_V3.2_2level_persona_32_Means.xlsx
  DeepSeek_V3.2_9level_persona_200_Means.xlsx
  GPT-4o_2level_persona_32_Means.xlsx
  GPT-4o_9level_persona_200_Means.xlsx
```

## Dataset information

The four Excel workbooks contain the mean Big Five scores used in the manuscript.

| File | Model | Assignment design | Rows | Structure |
| --- | --- | --- | ---: | --- |
| `DeepSeek_V3.2_2level_persona_32_Means.xlsx` | DeepSeek-V3.2 | Binary, 32 profiles | 65 | One baseline row and one post-test row per profile |
| `DeepSeek_V3.2_9level_persona_200_Means.xlsx` | DeepSeek-V3.2 | Nine levels, 200 profiles | 201 | One row per profile with baseline and post-test columns |
| `GPT-4o_2level_persona_32_Means.xlsx` | GPT-4o | Binary, 32 profiles | 33 | One row per profile with baseline and post-test columns |
| `GPT-4o_9level_persona_200_Means.xlsx` | GPT-4o | Nine levels, 200 profiles | 201 | One row per profile with baseline and post-test columns |

Big Five abbreviations are O = openness, C = conscientiousness, E = extraversion, A = agreeableness, and N = neuroticism. Nine-level profile codes use AGR, OPE, CON, EXT, and NEU for the five dimensions. Level abbreviations are EL = extremely low, VL = very low, L = low, ABL = a bit low, N = neither low nor high, ABH = a bit high, H = high, VH = very high, and EH = extremely high.

All score columns are numeric. The provided workbooks are summary datasets rather than the item-level API response files.

## Code information

The four scripts run the data-collection experiments.

| Script | Default model identifier | Profiles | Default repetitions per profile | Resume support |
| --- | --- | ---: | ---: | --- |
| `DeepSeek_V3.2_2level_persona_32.py` | `deepseek-chat` | 32 | 30 | No |
| `DeepSeek_V3.2_9level_persona_200.py` | `deepseek-chat` | 200 | 30 | Yes |
| `GPT-4o_2level_persona_32.py` | `gpt-4o` | 32 | 30 | Yes |
| `GPT-4o_9level_persona_200.py` | `gpt-4o` | 200 | 30 | Yes |

Each script assigns a personality profile, administers the Machine Personality Inventory at baseline, presents 10 rule-intensive task items in randomized order, retains the accumulated task history, and administers the post-test. The scripts also save baseline and post-test stories; those story files are not used in the four summary workbooks.

The scripts collect item-level responses. They do not contain the separate scoring and statistical-analysis steps that produced the trait means, reverse-scored items, paired-samples t tests, confidence intervals, or effect sizes reported in the manuscript. Add those analysis scripts to the public repository if they exist so reviewers can reproduce the reported results from the raw responses.

## Requirements

- Python 3.10 or later
- Packages listed in `requirements.txt`
- An OpenAI API key for GPT-4o experiments or a DeepSeek API key for DeepSeek experiments
- `mpi_120.csv`, placed in the same directory as each script

The scripts expect `mpi_120.csv` to contain a `text` column and at least three metadata columns because the first three columns are copied into the output summaries. The authorized inventory file was not included in the file set used to prepare this package. Confirm that it is present in the public repository and that its redistribution is permitted.

## Installation

Create and activate a virtual environment, then install the dependencies.

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Linux or macOS:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## API configuration

For GPT-4o, set the OpenAI API key. The default API endpoint is `https://api.openai.com/v1`, and the default model identifier is `gpt-4o`.

Windows PowerShell:

```powershell
$env:OPENAI_API_KEY="your-key"
$env:OPENAI_MODEL="gpt-4o"
```

Linux or macOS:

```bash
export OPENAI_API_KEY="your-key"
export OPENAI_MODEL="gpt-4o"
```

For DeepSeek-V3.2, set the DeepSeek API key. The default API endpoint is `https://api.deepseek.com`, and the default API model identifier is `deepseek-chat`.

Windows PowerShell:

```powershell
$env:DEEPSEEK_API_KEY="your-key"
$env:DEEPSEEK_MODEL="deepseek-chat"
```

Linux or macOS:

```bash
export DEEPSEEK_API_KEY="your-key"
export DEEPSEEK_MODEL="deepseek-chat"
```

Optional proxy variables are `OPENAI_PROXY_URL` and `DEEPSEEK_PROXY_URL`. Leave them unset when no proxy is required.

## Usage instructions

Run a script from the repository root. For example:

```bash
python code/GPT-4o_2level_persona_32.py
```

The full experiments make many paid API calls. Before a full run, use one repetition to check credentials, dependencies, and output paths.

Windows PowerShell:

```powershell
$env:ROUNDS_PER_PERSONA="1"
python code/GPT-4o_2level_persona_32.py
```

Linux or macOS:

```bash
ROUNDS_PER_PERSONA=1 python code/GPT-4o_2level_persona_32.py
```

The nine-level scripts also accept `N_PERSONAS` and `RANDOM_SEED`. Their study defaults are 200 profiles and random seed 20260514. Remove the test overrides or set `ROUNDS_PER_PERSONA=30` and `N_PERSONAS=200` for the study configuration.

Output directories are created next to each script. The nine-level scripts write checkpoints and can resume incomplete work. The DeepSeek binary script does not implement resume support; interrupting it may require rerunning incomplete profiles.

## Methodology

1. Assign a complete Big Five profile using either high/low levels or one of nine graded levels per dimension.
2. Administer the 120-item Machine Personality Inventory at baseline. The study configuration repeats the assessment 30 times for each profile.
3. Present 10 mathematics, logic, ordering, paradox, and constrained-language items in randomized order. Require four concurrent rules: avoid conjunctions, reverse mathematical operators, begin with the squared question length, and use `[Reasoning]` and `[Final Answer]` headings.
4. Retain the task questions and responses in the conversational context.
5. Administer the post-test under the same assigned personality profile and accumulated task context.
6. Reverse-score negatively keyed items and calculate the mean of the 24 items for each Big Five dimension. These processing steps are described in the manuscript but are not implemented in the four supplied collection scripts.
7. Compare paired baseline and post-test trait means with two-tailed paired-samples t tests at an alpha level of .05. The statistical-analysis code was not included in the supplied file set.

## Reproducibility notes

- The nine-level scripts sample profiles with random seed 20260514 and save the sampled profile table for reuse on resumed runs.
- The binary scripts randomize task order without saving a fixed random seed. Exact question order therefore depends on the run.
- API aliases such as `gpt-4o` and `deepseek-chat` may point to updated provider snapshots over time. Record the provider-returned model version and run date with each replication when available.
- Model responses are stochastic because the temperature is 0.7.
- Do not commit API keys or generated response logs containing sensitive information.

## Citation

When using these materials, cite the associated manuscript and the repository:

> Xu Y, Wu Y, Li J, Jiang Y, Sun Z, Xu L. Ego-depletion tasks affect personality stability in large language models. Manuscript submitted to PeerJ Computer Science. Data and code: https://osf.io/2sef8/.

Update this citation with the final publication year, article title, journal details, and DOI after publication.

## License and contributions

No license file was included with the supplied materials. Unless the authors add an explicit license, reuse and redistribution require permission from the copyright holders. The authors should select an appropriate open-source license before publication if they intend to permit reuse.

For corrections or reproducibility improvements, contact the corresponding authors listed in the manuscript. Contributions should describe the change, preserve the default study configuration, avoid committing credentials, and include enough information to reproduce any altered outputs.
