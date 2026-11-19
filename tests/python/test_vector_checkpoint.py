"""Tests for saving and loading the vector environment's state."""

import numpy as np
import pytest
from ale_py import AtariVectorEnv
from gymnasium.utils.env_checker import data_equivalence

NUM_ENVS = 4
# Short episodes with a small pool, so the steps after a save cross episode
# boundaries and draw start states from the pool.
CHECKPOINT_KWARGS = {
    "num_envs": NUM_ENVS,
    "max_num_frames_per_episode": 400,
    "reset_pool_size": 8,
}


def step_all(env: AtariVectorEnv, actions: np.ndarray):
    return [env.step(step_actions) for step_actions in actions]


def warmed_up_env(warmup_steps: int = 150, **kwargs):
    env = AtariVectorEnv("breakout", **(CHECKPOINT_KWARGS | kwargs))
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    for _ in range(warmup_steps):
        env.step(rng.integers(0, env.single_action_space.n, size=NUM_ENVS))
    return env


def continuation_actions(env: AtariVectorEnv, num_steps: int = 300):
    rng = np.random.default_rng(1)
    return rng.integers(0, env.single_action_space.n, size=(num_steps, NUM_ENVS))


def count_episode_ends(results):
    return sum(int(np.sum(term | trunc)) for _, _, term, trunc, _ in results)


@pytest.mark.parametrize("repeat_action_probability", [0.0, 0.25])
@pytest.mark.parametrize("autoreset_mode", ["NextStep", "SameStep"])
@pytest.mark.parametrize("grayscale", [True, False])
def test_load_resumes_the_saved_rollout(
    repeat_action_probability: float, autoreset_mode: str, grayscale: bool
):
    """Loading a state replays exactly the steps that followed its save.

    Sticky actions, both autoreset modes and RGB frames each keep state that
    the emulator snapshot alone would not restore.
    """
    env = warmed_up_env(
        repeat_action_probability=repeat_action_probability,
        autoreset_mode=autoreset_mode,
        grayscale=grayscale,
    )
    actions = continuation_actions(env)

    state = env.save_state()
    original = step_all(env, actions)
    env.load_state(state)
    resumed = step_all(env, actions)

    assert count_episode_ends(original) >= NUM_ENVS
    assert data_equivalence(original, resumed)
    env.close()


def test_load_into_new_env_resumes_the_saved_rollout():
    """A newly constructed env needs no reset to resume from a loaded state."""
    env = warmed_up_env(repeat_action_probability=0.25)
    actions = continuation_actions(env)
    state = env.save_state()
    original = step_all(env, actions)
    env.close()

    new_env = AtariVectorEnv(
        "breakout", **CHECKPOINT_KWARGS, repeat_action_probability=0.25
    )
    new_env.load_state(state)
    resumed = step_all(new_env, actions)

    assert data_equivalence(original, resumed)
    new_env.close()


def test_state_is_independent_of_later_steps():
    """Saving the same point twice gives the same bytes, after any detour."""
    env = warmed_up_env()
    state = env.save_state()
    step_all(env, continuation_actions(env, num_steps=50))
    env.load_state(state)

    assert env.save_state() == state
    env.close()
