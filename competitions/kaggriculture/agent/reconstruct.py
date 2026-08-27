"""Rebuild a rollout-capable simulator from an agent-visible observation.

This is the week-3 bridge (research plan section 8): at act time on Kaggle the
agent holds ONLY its observation dict -- the exact object FastEnv.observation(i)
returns -- and needs a forward simulator to score candidate plans by rollout.
`make_sim(obs)` builds one that quacks like sim.fastenv.FastEnv (`step`,
`observation`, `rewards`, `done`, plus `me`) but skips `make()` + the
interpreter's `_initialize` call, so constructing it costs 0.39 ms warm
(measured on the loaded 16-core box, so pessimistic 2-4x; cold first call
56 ms, dominated by the one-time `make("kaggriculture")` config load that is
cached at module level -- pure config, no episode state).

What the observation does and does not carry (audited against the env source,
kaggle_environments 1.32.7, envs/kaggriculture/kaggriculture.py):

EXACT -- copied verbatim from the obs:
  * BOTH farms, wholesale: the spec marks `farms` shared, and the interpreter
    mirrors `obs0.farms` to every player each step. That includes every
    per-tile field the daily bookkeeping reads (`watered_today`,
    `consecutive_unwatered`, `yield_units`, `planted_day`, `max_lifespan_step`,
    `fertilized_until_day`, `fed_today`, `cared_today`, `consecutive_unfed`,
    `pending_care_bonus`, `fertilizer_available`), money, farmer/hand
    positions, `unlocked_quadrants` and `hires_today` (which prices the next
    Fibonacci hire). There is NO hidden per-tile state.
  * Own private: shed, seeds, per-unit carried inventories.
  * Market (inventory + prices; `params` too if the config ever overrides
    them), town.unlocked_shops, day, hour. `step` is derived as
    day * turnsPerDay + hour, which matters because the step counter drives
    ALL scheduled events: town-shop consume at step % 4 == 0, town-center at
    step % 24 == 0, plant decay against `max_lifespan_step`, and end-of-day at
    (step + 1) % 24 == 0. Seeding `step_i` this way is what makes an
    hour-13 reconstruction resume mid-day instead of re-firing daily events
    (train/test_reconstruct.py locksteps a (day 12, hour 13) reconstruction
    against the real env and asserts bit-identical state).

ESTIMATED -- the only invented state:
  * Opponent private (shed / seeds / carried). Default: the env's own
    `_new_private()` (all-zero shed and seeds, one empty inventory) -- farms
    are independent, so this only matters through the opponent's future market
    orders. `opponent_private` lets a caller inject a better estimate (e.g.
    inferred from observed sells) without touching this module.

UNKNOWABLE -- future randomness:
  * The episode seed. `resolve_episode_seed` (kaggle_environments/utils.py)
    scrubs `configuration.seed` to None and hides the value in
    `env.info["seed"]`, so no agent can ever recover it. It feeds exactly one
    thing: the per-day `random.Random((seed * 1_000_003) ^ day)` stream,
    created fresh inside `_end_of_day` and consumed farm-0 weeds -> farm-1
    weeds -> shop-unlock draw. No RNG state survives between days, so there is
    nothing to reconstruct -- future weed/shop draws are genuinely unknowable.
    `rng_seed` stands in; holding it fixed across candidate rollouts is the
    common-random-numbers trick (all candidates face the same imagined weeds).
  * episodeSteps is not in the obs either (the agent is 1-arg); the cached
    default (720, the competition setting) is used.

Ground truth (train/test_reconstruct.py, policy-vs-starter, loaded box):
given the TRUE opponent private + TRUE seed the reconstruction is bit-identical
to the real env for 72 locksteps from every probe point (full-state JSON
fingerprint per step), i.e. all default-mode divergence comes from the two
documented estimated/unknowable inputs. With defaults (empty opponent private,
wrong seed), player-0 money diverges by mean $1 / max $6 at K=2 days, $4/$20
at K=3, $9/$41 at K=5 (n=20 probe points, day 5-20 incl. an hour-13 one).
Ablations split that: injecting the true seed (RNG error removed) leaves
$3/$13 at K=5, injecting the true opponent private leaves $9/$38 -- so vs
starter the weed/shop RNG dominates and the empty-opponent estimate is nearly
free. CAVEAT: starter barely moves the market; a heavy-selling opponent will
grow the opponent-private share -- that is what `opponent_private` is for.
Spearman rank fidelity over 10 theta-variant candidates (real end-money spread
up to $857/context) -- the thing the search actually needs -- is mean rho
1.000 at K=2 and 0.998 at K=3 (n=6 contexts, worst single context 0.988).
A 30-point day/hour robustness sweep of 5-day rollouts raised zero exceptions,
all money finite. make_sim + 2-day rollout: 24 ms end-to-end.

Kaggle-deployable on purpose: stdlib + kaggle_environments only (both exist on
the eval workers -- the env itself runs there), no __file__, no repo imports.
Payloads are converted to PLAIN dicts/lists because that is what the reference
state actually holds: `_initialize` builds plain dicts and only the outer
step/observation shells are Structs, so a faithful clone keeps the same types.
"""

import copy

from kaggle_environments import make
from kaggle_environments.envs.kaggriculture import kaggriculture as K
from kaggle_environments.utils import structify

_BASE_CFG = None  # cached pure config; make() is ~350 ms once, dict() reuse ~free


def _base_cfg():
    global _BASE_CFG
    if _BASE_CFG is None:
        cfg = dict(make("kaggriculture").configuration)
        cfg["seed"] = None  # scrubbed, exactly as resolve_episode_seed leaves it
        _BASE_CFG = cfg
    return _BASE_CFG


def _plain(o):
    """Deep-copy to plain dicts/lists (accepts Kaggle Structs, which are dict
    subclasses). Game state is dicts/lists/scalars all the way down."""
    if isinstance(o, dict):
        return {k: _plain(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_plain(v) for v in o]
    return o


class ReconSim:
    """A kaggriculture episode resumed from an agent-visible observation.

    Same stepping surface as sim.fastenv.FastEnv (`step`, `observation`,
    `rewards`, `done`, `run`), driven by the same `K.interpreter`, plus:
      me        -- the reconstructing player's id (obs["player"])
      run_steps -- roll N steps with an agent pair, return rewards()
    """

    __slots__ = ("state", "cfg", "info", "done", "steps_total", "step_i", "guard", "me")

    def __init__(self, obs, opponent_private=None, rng_seed=0, episode_steps=720,
                 guard=False):
        farms = _plain(K.get(obs, "farms", None) or [])
        if not farms:
            raise ValueError("observation has no farms; cannot reconstruct")
        num_agents = len(farms)
        me = int(K.get(obs, "player", 0))
        day = int(K.get(obs, "day", 0))
        hour = int(K.get(obs, "hour", 0))

        self.cfg = structify(dict(_base_cfg(), episodeSteps=int(episode_steps)))
        turns_per_day = max(1, int(K.get(self.cfg, "turnsPerDay", 24)))
        # The step counter is the clock for every scheduled event (town consume,
        # decay, end-of-day); day/hour pin it exactly, including mid-day.
        self.step_i = day * turns_per_day + hour
        self.info = {"seed": int(rng_seed)}  # feeds ONLY the daily weed/shop RNG
        self.done = False
        self.steps_total = int(episode_steps)
        self.guard = guard
        self.me = me

        privates = []
        for i in range(num_agents):
            if i == me:
                privates.append(_plain(K.get(obs, "private", None)) or K._new_private())
            elif opponent_private is not None:
                privates.append(copy.deepcopy(_plain(opponent_private)))
            else:
                privates.append(K._new_private())

        market = _plain(K.get(obs, "market", None)) or K._new_market()
        town = _plain(K.get(obs, "town", None)) or K._new_town()
        overage = float(K.get(obs, "remainingOverageTime", 60) or 60)

        # Same shell shape FastEnv builds; the interpreter shares farms/market/
        # town across players by reference, so mirror that from the start.
        self.state = structify([
            {
                "action": None,
                "reward": 0.0,
                "info": {},
                "observation": {"player": i,
                                "remainingOverageTime": overage if i == me else 60},
                "status": "ACTIVE",
            }
            for i in range(num_agents)
        ])
        for i in range(num_agents):
            o = self.state[i].observation
            o.farms = farms
            o.market = market
            o.town = town
            o.day = day
            o.hour = hour
            o.private = privates[i]
        self.state[0].observation.step = self.step_i
        # obs0 has farms -> the interpreter's _initialize branch never fires.

    # `interpreter` reads `env.configuration`, `env.info` and `env.done`.
    @property
    def configuration(self):
        return self.cfg

    def observation(self, i):
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

    def run_steps(self, agents, n_steps):
        """Roll forward up to `n_steps` (stops at episode end); returns rewards()."""
        for _ in range(n_steps):
            if self.done or self.step_i >= self.steps_total - 1:
                break
            self.step([fn(self.observation(i)) for i, fn in enumerate(agents)])
        return self.rewards()

    def run(self, agents):
        """Play out the rest of the episode."""
        return self.run_steps(agents, self.steps_total)


def make_sim(obs, opponent_private=None, rng_seed=0):
    """Observation -> forward simulator. See the module docstring for exactly
    what is reconstructed vs estimated vs unknowable.

    obs              -- the dict FastEnv.observation(i) / Kaggle hand the agent
    opponent_private -- optional {"shed":..,"seeds":..,"inventories":..}
                        estimate for the opponent (default: empty)
    rng_seed         -- stand-in for the hidden episode seed; drives future
                        weed spawns and shop unlocks. Keep it constant across
                        candidate rollouts for common random numbers.
    """
    return ReconSim(obs, opponent_private=opponent_private, rng_seed=rng_seed)
