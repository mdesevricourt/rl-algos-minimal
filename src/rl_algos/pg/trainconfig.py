from dataclasses import dataclass


@dataclass
class TrainConfig:
    """Configuration for training parameters."""

    epoch: int = 50  # number of epochs to train
    N: int = 5  # number of trajectories per epoch
    T: int = 999  # number of steps per trajectory
    env: str = "AntBulletEnv-v0"  # environment name
    algo: str = "ppo"  # algorithm name
    lr: float = 0.01
    gamma: float = 0.99  # discount factor
    target_update: int = 1  # number of epochs between two print statements

    def __str__(self) -> str:
        return f"TrainConfig(epoch={self.epoch}, N={self.N}, T={self.T}, env='{self.env}', algo='{self.algo}', lr={self.lr}, gamma={self.gamma}, target_update={self.target_update})"
