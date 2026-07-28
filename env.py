import numpy as np
import gymnasium as gym
from gymnasium import spaces

# ---- cluster configuration ----
MIN_PODS = 2
MAX_PODS = 20
VM_CPU_CAP = 1.0                          # CPU capacity per VM
VM_MEM_CAP = 1.0                          # memory capacity per VM
JOBS_PER_STEP_PER_VM = 25                 # nominal throughput (for utilisation/state)

# ---- timing ----
STEP_MINUTES = 15
STEPS_PER_HOUR = 60 // STEP_MINUTES       # 4
STEPS_PER_DAY  = STEPS_PER_HOUR * 24      # 96
STEPS_PER_WEEK = STEPS_PER_DAY * 7        # 672  (one episode = one week)

# ---- workload scaling (calibrated so 2-20 VM range is meaningful) ----
WORKLOAD_SCALE = 0.008

LAMBDA_COST = 0.4
LAMBDA_SLA  = 1.5
LAMBDA_UTIL = 0.1

# ---- surge configuration (synthetic spikes for hint training) ----
SURGE_LEAD_STEPS = 4          # how many steps before a surge the hint activates
SURGE_MIN, SURGE_MAX = 3.5, 5.0   # surge magnitude range
SURGE_DUR_MIN, SURGE_DUR_MAX = 8, 14   # surge duration in steps

print("Setup complete. Steps per week:", STEPS_PER_WEEK)


class CloudClusterEnv(gym.Env):
    """Simulated cloud cluster. The agent chooses how many VMs to run,
    balancing cost against SLA compliance, using CPU and memory signals."""

    def __init__(self, stats, lambda_cost=LAMBDA_COST, lambda_sla=LAMBDA_SLA,
                 lambda_util=LAMBDA_UTIL, seed=None, enable_surges=False,
                 enable_hints=False, min_surges=1, max_surges=3):
        super().__init__()
        self.stats = stats
        self.lambda_cost = lambda_cost
        self.lambda_sla = lambda_sla
        self.lambda_util = lambda_util
        self._rng = np.random.default_rng(seed)
        self.enable_surges = enable_surges     # turn synthetic surges on/off
        self.enable_hints = enable_hints       # turn hint signalling on/off
        self.min_surges = min_surges       
        self.max_surges = max_surges       
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(0.0, 1.0, shape=(32,), dtype=np.float32)

    # ---------------------------------------------------------------- reset
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.active_vms = 4
        self.queue = []
        self.cost_total = 0.0
        self.total_breaches = 0
        self.history = [[0.0, 0.0, 0.0, 0.0] for _ in range(5)]
        self.prev_queue_len = 0
        self.hint_active = 0.0
        self.hint_magnitude = 0.0
        self.hint_time_to_event = 0.0
        self.decision_log = []

        # ---- schedule surges for this episode ----
        self.surges = []
        if self.enable_surges:
            n_surges = self._rng.integers(self.min_surges, self.max_surges + 1)
            for _ in range(n_surges):
                start = int(self._rng.integers(20, STEPS_PER_WEEK - 20))
                duration = int(self._rng.integers(SURGE_DUR_MIN, SURGE_DUR_MAX))
                magnitude = float(self._rng.uniform(SURGE_MIN, SURGE_MAX))
                self.surges.append({'start': start,
                                    'end': start + duration,
                                    'magnitude': magnitude})

        return self._build_state(), {}

    # --------------------------------------------------------- time helper
    def _current_day_hour(self):
        total_hours = self.step_count // STEPS_PER_HOUR
        hour = int(total_hours % 24)
        day = int((total_hours // 24) % 7)
        return day, hour

    # ------------------------------------------------- surge helpers
    def _surge_multiplier(self):
        """Arrival multiplier at the current step (1.0 if no active surge)."""
        for surge in self.surges:
            if surge['start'] <= self.step_count < surge['end']:
                return surge['magnitude']
        return 1.0

    def _update_hints(self):
        """Set hint slots if a surge is approaching within the lead window."""
        # default: no hint
        self.hint_active = 0.0
        self.hint_magnitude = 0.0
        self.hint_time_to_event = 0.0
        if not self.enable_hints:
            return
        for surge in self.surges:
            steps_until = surge['start'] - self.step_count
            # activate hint during the lead window before the surge starts
            if 0 < steps_until <= SURGE_LEAD_STEPS:
                self.hint_active = 1.0
                # NOTE: with the current surge range (3.5-5.0) this always clips
                # to 1.0, so hint_magnitude carries no gradation — the trained
                # hint models learned from hint_active and time_to_event only.
                # Kept as-is: changing the divisor would shift the input
                # distribution of the already-trained hint models.
                self.hint_magnitude = float(np.clip(surge['magnitude'] / 3.0, 0.0, 1.0))
                # time-to-event: 1.0 = imminent, smaller = further away
                self.hint_time_to_event = float(1.0 - steps_until / SURGE_LEAD_STEPS)
                return

    # ------------------------------------------------------ job generation
    def _generate_jobs(self):
        day, hour = self._current_day_hour()
        s = self.stats[str(day)][str(hour)]
        base_mean = (s['arrival_rate'] * WORKLOAD_SCALE) / STEPS_PER_HOUR
        surge_mult = self._surge_multiplier()               # ← surge applied
        n_jobs = self._rng.poisson(base_mean * surge_mult)
        deadline_by_class = {0: 2, 1: 4, 2: 8, 3: 16}
        new_jobs = []
        for _ in range(n_jobs):
            cpu = float(np.clip(self._rng.normal(s['avg_cpu'], s['cpu_std']), 0.001, 1.0))
            mem = float(np.clip(self._rng.normal(s['avg_mem'], s['mem_std']), 0.001, 1.0))
            sched = int(self._rng.choice([0, 1, 2, 3], p=s['class_distribution']))
            new_jobs.append({'cpu': cpu, 'mem': mem,
                             'deadline': self.step_count + deadline_by_class[sched]})
        return new_jobs

    # ------------------------------------------------- assign jobs to VMs
    def _assign_jobs(self):
        vm_cpu_load = [0.0] * self.active_vms
        vm_mem_load = [0.0] * self.active_vms
        still_waiting = []
        jobs_processed = 0
        for job in self.queue:
            placed = False
            for vm_idx in sorted(range(self.active_vms), key=lambda i: vm_cpu_load[i]):
                cpu_ok = vm_cpu_load[vm_idx] + job['cpu'] <= VM_CPU_CAP
                mem_ok = vm_mem_load[vm_idx] + job['mem'] <= VM_MEM_CAP
                if cpu_ok and mem_ok:
                    vm_cpu_load[vm_idx] += job['cpu']
                    vm_mem_load[vm_idx] += job['mem']
                    jobs_processed += 1
                    placed = True
                    break
            if not placed:
                still_waiting.append(job)
        self.queue = still_waiting
        if self.active_vms > 0:
            avg_cpu = float(np.mean(vm_cpu_load)); max_cpu = float(np.max(vm_cpu_load))
            avg_mem = float(np.mean(vm_mem_load)); max_mem = float(np.max(vm_mem_load))
        else:
            avg_cpu = max_cpu = avg_mem = max_mem = 0.0
        return jobs_processed, avg_cpu, max_cpu, avg_mem, max_mem

    # ---------------------------------------------------- deadline checks
    def _check_deadlines(self):
        breaches = 0
        surviving = []
        for job in self.queue:
            if self.step_count > job['deadline']:
                breaches += 1
            else:
                surviving.append(job)
        self.queue = surviving
        return breaches

    # ---------------------------------------------------------- the state
    def _build_state(self):
        hist = np.array(self.history, dtype=np.float32)
        avg_cpu_hist = hist[:, 0]; max_cpu_hist = hist[:, 1]
        avg_mem_hist = hist[:, 2]; max_mem_hist = hist[:, 3]

        capacity = self.active_vms * JOBS_PER_STEP_PER_VM
        queue_depth_ratio = float(np.clip(
            len(self.queue) / capacity if capacity > 0 else 1.0, 0.0, 1.0))

        growth = len(self.queue) - self.prev_queue_len
        queue_growth = float(np.clip(
            0.5 + (growth / capacity if capacity > 0 else 0.0), 0.0, 1.0))

        if len(self.queue) > 0:
            near = sum(1 for j in self.queue if j['deadline'] - self.step_count <= 2)
            sla_pressure = near / len(self.queue)
            slacks = [j['deadline'] - self.step_count for j in self.queue]
            min_slack = float(np.clip(min(slacks) / 16.0, 0.0, 1.0))
        else:
            sla_pressure = 0.0
            min_slack = 1.0

        active_vms_norm = self.active_vms / MAX_PODS
        cost_norm = float(np.clip(self.cost_total / STEPS_PER_WEEK, 0.0, 1.0))

        day, hour = self._current_day_hour()
        hour_sin = (np.sin(2 * np.pi * hour / 24) + 1) / 2
        hour_cos = (np.cos(2 * np.pi * hour / 24) + 1) / 2
        day_norm = day / 6.0

        hint = [self.hint_active, self.hint_magnitude, self.hint_time_to_event]

        return np.concatenate([
            avg_cpu_hist, max_cpu_hist, avg_mem_hist, max_mem_hist,
            [queue_depth_ratio], [queue_growth], [sla_pressure], [min_slack],
            [active_vms_norm], [cost_norm], [hour_sin], [hour_cos], [day_norm],
            hint,
        ]).astype(np.float32)

    # ----------------------------------------------------------- the step
    def step(self, action):
        # update hint slots BEFORE the agent's next observation reflects them
        self._update_hints()                                # ← NEW

        # 1. apply action
        delta = int(round(float(action[0]) * 5))
        self.active_vms = int(np.clip(self.active_vms + delta, MIN_PODS, MAX_PODS))

        # 2. generate jobs (surge applied inside)
        self.queue.extend(self._generate_jobs())

        # 3. assign to VMs
        jobs_processed, avg_cpu, max_cpu, avg_mem, max_mem = self._assign_jobs()

        # 4. deadlines -> breaches
        breaches = self._check_deadlines()
        self.total_breaches += breaches

        # 5. cost & utilisation
        cost = self.active_vms / MAX_PODS
        self.cost_total += cost
        utilisation = max(avg_cpu, avg_mem)

        jobs_due_this_step = jobs_processed + breaches
        breach_rate = breaches / jobs_due_this_step if jobs_due_this_step > 0 else 0.0

        # 6. reward
        reward = (- self.lambda_cost * cost
                  - self.lambda_sla * breach_rate
                  + self.lambda_util * utilisation)

        # 7. history
        self.history.append([avg_cpu, max_cpu, avg_mem, max_mem])
        self.history.pop(0)

        day, hour = self._current_day_hour()
        self.decision_log.append({
            'step': self.step_count, 'day': day, 'hour': hour,
            'active_vms': self.active_vms,
            'avg_cpu': round(avg_cpu, 3), 'avg_mem': round(avg_mem, 3),
            'queue': len(self.queue), 'breaches': breaches,
            'reward': round(reward, 3),
            'hint_active': self.hint_active,          # ← logged for RAG/analysis
        })

        self.prev_queue_len = len(self.queue)
        self.step_count += 1

        obs = self._build_state()
        done = self.step_count >= STEPS_PER_WEEK
        info = {'cost': cost, 'breaches': breaches, 'utilisation': utilisation,
                'active_vms': self.active_vms, 'queue': len(self.queue),
                'avg_cpu': avg_cpu, 'avg_mem': avg_mem,
                'surge_mult': self._surge_multiplier(),   # ← for diagnostics
                'hint_active': self.hint_active}
        return obs, reward, done, False, info


print("CloudClusterEnv defined.")