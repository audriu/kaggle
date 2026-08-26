"""Lean episode runner for the kaggriculture environment.

`kaggle_environments.Environment.run()` costs ~5.2 s per 720-step episode, of which
~98% is framework overhead: per-step action schema validation, `structify` of the
observation, stdout/stderr redirection, a `deepcopy` of the agent-visible state, and
retaining every step in `env.steps`. The rules interpreter itself is only ~0.11 ms/step
(~0.08 s per episode).

FastEnv keeps one mutable state, calls `kaggriculture.interpreter` directly and throws
away history. It is the *same* interpreter, so the rules cannot drift -- only the
plumbing is removed. `sim/difftest.py` asserts trajectory equality against the reference.

Caveat: agents are handed the live observation rather than a per-step deepcopy, so an
agent that mutates its observation would corrupt the episode. Pass `guard=True` to
restore reference-identical copying (slow) when validating an untrusted agent.
"""

import copy

from kaggle_environments import make
from kaggle_environments.envs.kaggriculture import kaggriculture as K
from kaggle_environments.utils import structify

TURNS_PER_DAY = 24


class FastEnv:
    """One kaggriculture episode, driven step by step with no framework overhead."""

    __slots__ = ("state", "cfg", "info", "done", "steps_total", "step_i", "guard")

    def __init__(self, seed, episode_steps=720, num_agents=2, guard=False, **cfg_overrides):
        base = dict(make("kaggriculture").configuration)
        base.update(cfg_overrides)
        base["episodeSteps"] = episode_steps
        base["seed"] = seed
        self.cfg = structify(base)
        self.info = {}
        self.done = False
        self.steps_total = episode_steps
        self.step_i = 0
        self.guard = guard

        self.state = structify([
            {
                "action": None,
                "reward": 0.0,
                "info": {},
                "observation": {"player": i, "remainingOverageTime": 60},
                "status": "ACTIVE",
            }
            for i in range(num_agents)
        ])
        # First interpreter call populates farms/market/town via _initialize().
        K.interpreter(self.state, self)
        self.state[0].observation.step = 0

    # `interpreter` reads `env.configuration`, `env.info` and `env.done`.
    @property
    def configuration(self):
        return self.cfg

    def observation(self, i):
        """Agent-visible observation for player `i` (shared fields + own private)."""
        obs = self.state[i].observation
        return copy.deepcopy(obs) if self.guard else obs

    def step(self, actions):
        """Apply one joint action. `actions[i]` is the dict returned by agent i."""
        for i, a in enumerate(actions):
            self.state[i].action = a
        self.state[0].observation.step = self.step_i
        K.interpreter(self.state, self)
        self.step_i += 1
        if self.state[0].status == "DONE":
            self.done = True
        return self.state

    def rewards(self):
        farms = self.state[0].observation.farms
        return [float(farms[s.observation.player]["money"]) for s in self.state]

    def run(self, agents):
        """Play a whole episode. `agents` are callables obs -> action dict."""
        # The framework records `episodeSteps` states; steps[0] is the initial state,
        # so the interpreter runs episodeSteps-1 times.
        for _ in range(self.steps_total - 1):
            actions = [fn(self.observation(i)) for i, fn in enumerate(agents)]
            self.step(actions)
            if self.done:
                break
        return self.rewards()


def load_agent(path):
    """Load an agent file the way Kaggle does: exec'd with no __file__, last callable wins.

    Built-in names ("pass", "random", "starter") resolve to the env's own agents.
    """
    if path in K.agents:
        return K.agents[path]
    src = open(path).read()
    ns = {}
    exec(compile(src, path, "exec"), ns)
    for name in ("agent", "act", "main"):
        if callable(ns.get(name)):
            return ns[name]
    last = [v for v in ns.values() if callable(v) and getattr(v, "__module__", None) is None]
    if not last:
        raise ValueError(f"no callable agent found in {path}")
    return last[-1]


def play(agents, seed, **kw):
    """Convenience: run one episode and return final rewards."""
    fns = [load_agent(a) if isinstance(a, str) else a for a in agents]
    return FastEnv(seed, **kw).run(fns)
