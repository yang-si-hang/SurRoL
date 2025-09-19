import surrol.gym  # 确保注册
import gym, numpy as np, os, psutil, time
DEMO = "/home/ubuntu/ysh/Code/SurRoL/Benchmark/state_based/surrol/data/demo/data_NeedlePickRL-v0_random_200.npz"

def print_mem(tag):
    p = psutil.Process()
    rss = p.memory_info().rss / 1024 / 1024
    print(f"[MEM]{tag}: {rss:.1f} MB")

print_mem("start")

env = gym.make("NeedlePickRL-v0")
obs = env.reset()
print("obs shape/type:", type(obs))
print_mem("after env.reset")


# 做一次 dummy rollout
for i in range(50):
    a = env.action_space.sample()
    o,r,dn,info = env.step(a)
    if dn:
        env.reset()
print_mem("after 50 random steps")

time.sleep(2)

env.close()