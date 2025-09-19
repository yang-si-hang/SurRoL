""" 用来可视化仓库锁构建的环境 (human 模式)
created by ysh @ 2025-09-17
"""
import time
import gym
import pybullet as p

# 触发注册（surrol.gym.__init__ 里会执行多个 register）
import surrol.gym  # noqa: F401

def main():
    # human 渲染模式 → SurRoLEnv 会使用 p.GUI 弹出 PyBullet 窗口
    env = gym.make('NeedlePickRL-v0', render_mode='human')
    obs = env.reset()
    ret = 0.0
    for t in range(500):
        # 随机动作，仅用于目视场景；实际训练时由策略产生命令
        action = env.action_space.sample()
        obs, r, done, info = env.step(action)
        ret += float(r)
        # 如果你希望显式渲染帧（多数情况下 p.GUI 已经在绘制）
        try:
            env.render(mode='human')
        except Exception:
            pass
        time.sleep(1.0 / 60.0)
        if done:
            break
    print(f"return={ret:.3f}, steps={t+1}")
    env.close()

def show_env(env_name, render_mode='human'):
    env = gym.make(env_name, render_mode=render_mode)
    obs = env.reset()

    try:
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
    except Exception:
        pass
    try:
        p.configureDebugVisualizer(p.COV_ENABLE_KEYBOARD_SHORTCUTS, 1)
    except Exception:
        pass
    try:
        p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 1)
    except Exception:
        pass

    print("已打开 PyBullet GUI。使用鼠标旋转/平移/缩放相机进行观察。按 Ctrl+C 退出。")
    t_end = time.time() + 300  # 最多运行 5 分钟
    try:
        while time.time() < t_end:
            # 对于 GUI 渲染，通常不需要显式调用 render；但不报错的话可以保留
            try:
                env.render(mode="human")
            except Exception:
                pass
            time.sleep(1.0 / 60.0)
    except KeyboardInterrupt:
        pass
    finally:
        env.close()
        print("已关闭环境。")
if __name__ == "__main__":
    # main()
    show_env("PsmEnvSoft-v0", render_mode="human")