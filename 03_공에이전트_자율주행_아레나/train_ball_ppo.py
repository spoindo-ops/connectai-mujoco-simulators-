import os
import gymnasium as gym
import ball_nav_env
from stable_baselines3 import PPO

def main():
    print("=" * 60)
    print("BallNav-v0 공 에이전트 PPO 학습 시작 (8강 공식 환경)")
    print("=" * 60)

    env = gym.make("BallNav-v0")

    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128])),
        learning_rate=3e-4,
        verbose=1
    )

    print("훈련 중... 잠시만 기다려주세요...")
    model.learn(total_timesteps=10000)
    
    save_path = os.path.join(os.path.dirname(__file__), "ball_ppo_model.zip")
    model.save(save_path)
    print(f"학습 완료 및 모델 저장: {save_path}")

if __name__ == "__main__":
    main()
