from training.grpo.reward_adapter import aggregate_episode_reward


def test_aggregate_episode_reward_sums_step_rewards():
    assert aggregate_episode_reward([0.1, 0.2, 0.3], method="sum") == 0.6
