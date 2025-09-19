import surrol.gym
import torch.multiprocessing as mp
import hydra
from trainers.rl_trainer import RLTrainer

@hydra.main(version_base=None, config_path="./configs", config_name="train_ddpg")
def main(cfg):
    exp = RLTrainer(cfg)
    exp.train()
    # exp.resume_train()

if __name__ == "__main__":
    # For multiprocessing with CUDA, 'spawn' is often safer than 'fork'
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass
    main()