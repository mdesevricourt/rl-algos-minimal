# rl-algos-minimal

Minimal, reproducible implementations of core deep reinforcement learning algorithms (PPO, MAML, etc.) with clean code, configs, and experiment results.


The script takes several command-line arguments to configure the environment, algorithm, and training parameters.

---

## Usage

Run the script from the command line:

```bash
python train.py --algo <algorithm> [--env <environment>] [--epoch <epochs>]
```

## Arguments

| Argument  | Required | Type | Default           | Description                                                           |
| --------- | -------- | ---- | ----------------- | --------------------------------------------------------------------- |
| `--algo`  | Yes      | str  | *None*            | Algorithm to use. Must be one of: `pg`, `pgb`, `ppo`.                 |
| `--env`   | No       | str  | `AntBulletEnv-v0` | Environment ID (any gym-compatible environment, e.g., `CartPole-v1`). |
| `--epoch` | No       | int  | `50`              | Number of training epochs.                                            |

