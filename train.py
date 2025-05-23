import argparse
import numpy as np
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from autonomous_parking_env import ParkingEnv

def make_env(render=False):
    def _init():
        env = ParkingEnv(render_mode="human" if render else None)
        return Monitor(env)
    return _init

class RewardLoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.ep_rewards = []
        self.current_ep_reward = 0
        self.ep_distance_reward = 0
        self.ep_angle_reward = 0
        self.ep_penalty_reward = 0
        self.ep_lane_penalty = 0
        self.ep_slot_bonus = 0
        self.ep_heading_penalty = 0  # Add heading penalty logging

        self.distance = 0

    def _on_step(self) -> bool:
        reward = self.locals.get('rewards')[0]  # for vectorized env
        done = self.locals.get('dones')[0]
        info = self.locals.get('infos')[0]

        self.current_ep_reward += reward
        self.ep_distance_reward += info.get('ep_distance_reward', 0)
        self.ep_angle_reward += info.get('ep_angle_reward', 0)
        self.ep_penalty_reward += info.get('ep_penalty_reward', 0)
        self.ep_lane_penalty += info.get('ep_lane_penalty', 0)
        self.ep_slot_bonus += info.get('ep_slot_bonus', 0)
        self.ep_heading_penalty += info.get('ep_heading_penalty', 0)  # Add heading penalty
        self.ep_success += info.get('ep_success', 0)
        self.distance = info.get('distance_in_meters', 0)

        if done:
            self.ep_rewards.append(self.current_ep_reward)
            print(f"Episode finished: total_reward={self.current_ep_reward:.2f}")
            print(f"  Distance reward sum: {self.ep_distance_reward:.2f}")
            print(f"  Angle reward sum: {self.ep_angle_reward:.2f}")
            print(f"  Penalty reward sum: {self.ep_penalty_reward:.2f}")
            print(f"  Lane penalty sum: {self.ep_lane_penalty:.2f}")
            print(f"  Slot bonus sum: {self.ep_slot_bonus:.2f}")
            print(f"  Heading penalty sum: {self.ep_heading_penalty:.2f}")  # Log heading penalty
            print(f"  Success: {self.ep_success:.2f}")

            # Reset counters for next episode
            self.current_ep_reward = 0
            self.ep_distance_reward = 0
            self.ep_angle_reward = 0
            self.ep_penalty_reward = 0
            self.ep_lane_penalty = 0
            self.ep_slot_bonus = 0
            self.ep_heading_penalty = 0
            self.ep_success = 0
            self.distance = 0

        return True


def train(timesteps: int, render=False, continue_training=False):
    env = DummyVecEnv([make_env(render)])
    n_act = env.action_space.shape[-1]

    sigma_start = np.array([0.95, 0.99], dtype=np.float32)
    sigma_end = np.array([0.05, 0.05], dtype=np.float32)
    noise = NormalActionNoise(mean=np.zeros(n_act), sigma=sigma_start.copy())

    policy_kwargs = dict(net_arch=[30, 45, 20])

    if continue_training:
        print("Loading existing model to continue training...")
        model = DDPG.load("ddpg_parking_li_long.zip", env=env, device="auto")
        # Optional: You may want to restore the noise sigma value to some intermediate value
        # For example, keep sigma at sigma_end if near end of training
        noise.sigma = sigma_end
        model.action_noise = noise
    else:
        model = DDPG("MlpPolicy", env,
                     learning_rate=0.002,
                     buffer_size=100_000,
                     batch_size=140,
                     tau=0.01,
                     gamma=0.92,
                     action_noise=noise,
                     verbose=1,
                     policy_kwargs=policy_kwargs,
                     device="auto")

    callback = RewardLoggingCallback()

    print("Starting training...")
    total_steps = 0
    while total_steps < timesteps:
        model.learn(total_timesteps=1000, reset_num_timesteps=False, progress_bar=True, callback=callback)
        total_steps += 1000

        # Decay noise sigma linearly over time
        frac = max(0, 1 - total_steps / timesteps)
        noise.sigma = sigma_end + frac * (sigma_start - sigma_end)
        print(f"Step: {total_steps}, Noise Sigma: {noise.sigma}")

    model.save("ddpg_parking_li_long.zip")
    env.close()


def play():
    env = DummyVecEnv([make_env(True)])
    model = DDPG.load("ddpg_parking_li_long.zip", env=env, device="auto")
    obs = env.reset()
    done = False
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, info = env.step(action)
        env.render()
        done = terminated[0] 
        if done:
            obs = env.reset()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--timesteps", type=int)
    group.add_argument("--play", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--continue", dest="continue_training", action="store_true", help="Continue training from saved model")
    args = parser.parse_args()

    if args.play:
        play()
    else:
        train(args.timesteps, args.render, args.continue_training)