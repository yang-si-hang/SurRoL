import surrol.gym
import os
import hydra
from trainers.rl_trainer import RLTrainer

@hydra.main(version_base=None, config_path="./configs", config_name="eval")
def main(cfg):
    # 只构建，不训练，直接评估
    exp = RLTrainer(cfg)
    # 调用内置的评估 checkpoint 函数
    exp.eval_ckpt()

if __name__ == "__main__":
    main()