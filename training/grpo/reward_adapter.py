def aggregate_episode_reward(rewards: list[float], method: str = "sum") -> float:
    if not rewards:
        return 0.0
    if method == "last":
        return float(rewards[-1])
    return float(sum(rewards))
