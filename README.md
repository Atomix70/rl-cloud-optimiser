# Intelligent Cloud Resource Management — RL Training & Evaluation

A multi-objective deep reinforcement learning system for cloud autoscaling. A
Proximal Policy Optimization (PPO) agent learns to set the number of virtual
machines / pods to run, jointly minimising infrastructure cost and Service-Level
Agreement (SLA) breaches. This repository contains the simulator, the agent, the
baselines, the full experimental evaluation, and an explainability chatbot.

> This is the **research/training** project. The live Kubernetes deployment lives
> in the sibling project `cloud-rl-deployment` — see `../cloud-rl-deployment/README.md`
> for how to run the trained policy against a real Minikube cluster.

---

## What this project does

- Simulates a cloud cluster under a realistic workload derived from the
  **Google Cluster Trace 2011**.
- Trains a **PPO agent** (implemented from scratch) to balance cost vs SLA.
- Compares it against **threshold HPA**, **proportional HPA**, and a **DQN** baseline.
- Produces a **cost–SLA Pareto frontier**, **statistical significance** tests, and
  a **cross-provider generalisation** test on the Alibaba 2018 trace.
- Provides a **RAG chatbot** that explains the agent's decisions in natural language.

---

## Requirements

- **Python 3.11**
- macOS / Linux (developed on macOS, Apple M4 Pro; CPU-only is fine — no GPU needed)
- ~2 GB free disk for data + models
- For the chatbot only: **Ollama** with the `mistral` model pulled

---

## 1. Setup

```bash
# clone / enter the project folder
cd cloud-rl-project

# create and activate a virtual environment
python3.11 -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate

# install dependencies
pip install -r requirements.txt
```

If there is no `requirements.txt`, install the core packages directly:

```bash
pip install torch numpy pandas matplotlib gymnasium scipy \
            sentence-transformers faiss-cpu ollama gradio
```

---

## 2. Get the data

This project uses two public datasets. **If you only want to run the agent and
reproduce the results, you can skip the raw downloads** — the pre-computed
`trace_params.json` (Google workload statistics) is all the simulator needs, and
`alibaba_params.json` covers the generalisation test. Only download the raw
traces if you want to re-run the extraction notebooks (01 and 04) from scratch.

### 2a. Google Cluster Trace 2011 (training data)

The agent is trained on workload statistics extracted from the **Google
ClusterData-2011-2** trace (a ~29-day trace of a ~12,500-machine Google Borg
cell, released May 2011). The full compressed trace is ~41 GB, but this project
only uses the **`task_events`** table.

The data lives in a public Google Cloud Storage bucket named
`clusterdata-2011-2`. No Google account is required to download it. The
recommended tool is `gcloud storage` (or the older `gsutil`):

```bash
# install the Google Cloud SDK first: https://cloud.google.com/sdk/docs/install

# create the data folder
mkdir -p trace_data

# download only the task_events table (what this project uses)
gcloud storage cp -r gs://clusterdata-2011-2/task_events ./trace_data/
```

> The `task_events` table is split into ~500 gzipped part-files. This project's
> extraction (`01_explore.ipynb`) reads the part-files from `trace_data/`.
> **[VERIFY]** Confirm the folder path in `01_explore.ipynb` matches where you
> placed the files (the notebook expects them under `trace_data/`).

Dataset home and full schema: https://github.com/google/cluster-data
(see `ClusterData2011_2.md` for the bucket details and `SHA256SUM` verification.)

### 2b. Alibaba Cluster Trace 2018 (generalisation-test data)

The cross-provider generalisation test uses the **`machine_usage`** table from
the **Alibaba cluster-trace-v2018** (~4000 machines over 8 days). This project
uses its `cpu_util_percent` column to derive a workload rhythm.

The trace is hosted in the Alibaba `clusterdata` repository; the download link is
provided after a short (under a minute) survey:

- Repository: https://github.com/alibaba/clusterdata
- Go to the `cluster-trace-v2018` directory and follow the survey/download link
  to obtain `machine_usage.csv`.

```bash
# after downloading, place the CSV in the project root as:
#   alibaba_machine_usage.csv
```

> **[VERIFY]** `04_generalisation.ipynb` reads `alibaba_machine_usage.csv` from
> the project root. The full `machine_usage` table is large; this project only
> needs enough rows to cover ~7 days (the notebook trims to the first 7×24×12
> five-minute samples).

### 2c. Skipping the downloads (fastest path)

If the repository already contains `trace_params.json` and `alibaba_params.json`
(the small extracted-statistics files), you can run everything **except** the two
extraction notebooks without downloading any raw data. This is the recommended
path for anyone who just wants to reproduce the results.

---

## 3. Project structure

```
cloud-rl-project/
├── env.py                 # the CloudClusterEnv simulator (Gymnasium)
├── agent.py               # ActorCritic, DQN, PPO/DQN training
├── evaluate.py            # shared evaluation functions
├── trace_params.json      # extracted workload statistics (produced by 01)
├── 01_explore.ipynb            # extract workload stats from the raw trace
├── 02_simulator.ipynb          # validate the environment
├── 03_ppo_agent.ipynb          # train PPO (Pareto sweep) + HPA comparison
├── 04_generalisation.ipynb     # Alibaba cross-provider test
├── 05_surge_test.ipynb         # surge mechanism validation
├── 06_hint_experiment.ipynb    # operator-hint benefit (±5 VM/step)
├── 07_hint_experiment_2.ipynb  # operator-hint benefit (±1 VM/step)
├── 08_rag_chatbot.ipynb        # RAG explainability + function-calling control
├── 09_dqn_baseline.ipynb       # DQN baseline + three-way comparison
├── 10_statistical_tests.ipynb  # 30-seed significance (paired t-tests)
├── 11_report_figures.ipynb     # generate all figures
└── report_figures/             # generated figures (created by 11)
```

---

## 4. How to run (recommended order)

Run the notebooks in order. Each imports from the shared modules
(`env.py`, `agent.py`, `evaluate.py`) — no code is duplicated.

| Step | Notebook | What it does | Approx time |
|------|----------|--------------|-------------|
| 1 | `01_explore.ipynb` | Extract workload stats → `trace_params.json` | ~2 min |
| 2 | `02_simulator.ipynb` | Sanity-check the environment | <1 min |
| 3 | `03_ppo_agent.ipynb` | Train 4 PPO configs (Pareto sweep) | ~1 hour |
| 4 | `09_dqn_baseline.ipynb` | Train the DQN baseline | ~20 min |
| 5 | `10_statistical_tests.ipynb` | 30-seed significance tests | ~2 min |
| 6 | `04_generalisation.ipynb` | Alibaba cross-provider test | ~2 min |
| 7 | `05` / `06` / `07` | Surge + operator-hint experiments | ~40 min total |
| 8 | `08_rag_chatbot.ipynb` | RAG explainability (needs Ollama) | ~5 min |
| 9 | `11_report_figures.ipynb` | Generate all figures | ~1 min |

> **Note on training time:** training is CPU-only and stochastic. A full PPO run
> is 250,000 steps (~15-20 min per config). If you only want to *evaluate*
> pre-trained models, skip the training cells — evaluation is fast.

---

## 5. Key results you should see

After running steps 3–5, the 30-seed comparison should show (approximately):

| Controller | Cost | SLA breaches |
|-----------|------|--------------|
| Threshold HPA | ~291 | ~1785 |
| DQN | ~170 | ~3684 |
| **PPO (SLA-focused)** | **~178** | **~878** |

PPO reduces cost by ~39% and breaches by ~51% vs threshold HPA (p < 0.001).

> Exact numbers vary slightly by training seed — PPO training is stochastic. Cost
> is very stable; breach counts vary between runs but always beat the baselines.

---

## 6. Running the explainability chatbot (optional)

The chatbot (notebook 08) needs a local LLM via **Ollama**:

```bash
# install Ollama (see https://ollama.com), then:
ollama pull mistral
ollama serve          # keep this running in a separate terminal
```

Then run `08_rag_chatbot.ipynb`. It launches a Gradio chat interface where you
can ask why the agent made scaling decisions, or send it surge warnings.

---

## 7. Locked design decisions (for anyone modifying the code)

These are intentional and should not be changed without understanding the impact:

- **32-dimensional state**: 5-step CPU + memory history, 9 situational scalars,
  3 operator-hint slots.
- **Continuous action** in [-1, 1] → ±5 VMs/step, clipped to [2, 20].
- **Normalised breach rate** in the reward (breaches ÷ jobs-due) — this prevents
  the agent from cheating by always running the maximum VMs. **Do not** revert to
  raw breach counts.
- **15-minute** control interval; **1 week = 672 steps** per episode.

---

## Troubleshooting

- **`ModuleNotFoundError`** → activate the venv (`source venv/bin/activate`).
- **`scipy` not found** in notebook 10 → `pip install scipy`.
- **Chatbot errors** → make sure `ollama serve` is running and `mistral` is pulled.
- **Figure cell fails on `boxplot`** → newer matplotlib renamed `labels` to
  `tick_labels`; already handled in `11_report_figures.ipynb`.
