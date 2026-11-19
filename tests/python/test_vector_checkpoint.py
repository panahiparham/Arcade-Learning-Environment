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


class TestRejectedCheckpoints:
    """Checkpoints are refused whenever they could not resume exactly."""

    def test_save_before_reset(self):
        """An env that was never reset has no state worth saving."""
        env = AtariVectorEnv("breakout", **CHECKPOINT_KWARGS)
        with pytest.raises(RuntimeError, match="preceding reset"):
            env.save_state()
        env.close()

    def test_async_mode(self):
        """With batch_size < num_envs, some envs are always mid-step."""
        env = AtariVectorEnv("breakout", **CHECKPOINT_KWARGS, batch_size=2)
        env.reset(seed=0)
        with pytest.raises(RuntimeError, match="batch_size == num_envs"):
            env.save_state()
        with pytest.raises(RuntimeError, match="batch_size == num_envs"):
            env.load_state(warmed_up_env(warmup_steps=0).save_state())
        env.close()

    def test_between_send_and_recv(self):
        """Workers may be using the envs until the pending recv returns."""
        env = warmed_up_env(warmup_steps=0)
        state = env.save_state()
        env.send(np.zeros(NUM_ENVS, dtype=np.int32))
        with pytest.raises(RuntimeError, match="between send"):
            env.save_state()
        with pytest.raises(RuntimeError, match="between send"):
            env.load_state(state)

        env.recv()
        env.load_state(state)
        env.close()

    def test_different_num_envs(self):
        """A state only fits a vectorizer with as many envs."""
        state = warmed_up_env(warmup_steps=0).save_state()
        env = AtariVectorEnv("breakout", **(CHECKPOINT_KWARGS | {"num_envs": 2}))
        with pytest.raises(RuntimeError, match="num_envs=4"):
            env.load_state(state)
        env.close()

    def test_different_observation_settings(self):
        """Frame buffers saved at one image size do not fit another."""
        state = warmed_up_env(warmup_steps=0).save_state()
        env = AtariVectorEnv("breakout", **CHECKPOINT_KWARGS, img_height=42)
        with pytest.raises(RuntimeError, match="observation settings"):
            env.load_state(state)
        env.close()

    def test_different_rom(self):
        """The emulator refuses a state saved for another game."""
        state = warmed_up_env(warmup_steps=0).save_state()
        env = AtariVectorEnv("pong", **CHECKPOINT_KWARGS)
        with pytest.raises(RuntimeError, match="different ROM"):
            env.load_state(state)
        env.close()

    @pytest.mark.parametrize(
        "corrupt",
        [
            pytest.param(lambda state: b"", id="empty"),
            pytest.param(lambda state: state[: len(state) // 2], id="truncated"),
        ],
    )
    def test_corrupt_state(self, corrupt):
        """Damaged bytes raise an error instead of loading garbage."""
        env = warmed_up_env(warmup_steps=0)
        with pytest.raises(RuntimeError):
            env.load_state(corrupt(env.save_state()))
        env.close()

    def test_unknown_format_version(self):
        """A state from a newer layout is refused before any env is touched."""
        env = warmed_up_env(warmup_steps=0)
        state = env.save_state()
        with pytest.raises(RuntimeError, match="Unsupported"):
            env.load_state(b"\x02" + state[1:])
        env.close()
