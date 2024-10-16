from transformers import AutoModelForCausalLM
from transformers import PPOModel
from transformers import RewardModel
from trl import PPOv2Trainer
from trl import PPOv2Config
from datasets import load_dataset
from baseline.model import T5Model
import sentencepiece
import gym

env = gym.make('CartPole-v1')

# Load a pre-trained transformer model
model = T5Model()

config = PPOv2Config(
    batch_size=32,            # Number of experiences in each mini-batch
    learning_rate=5e-5,       # Initial learning rate
    gamma=0.99,               # Discount factor for rewards
    lam=0.95,                 # GAE (Generalized Advantage Estimation) parameter
    vf_coef=0.5,              # Coefficient for the value function loss
    max_grad_norm=0.5,        # Gradient clipping to avoid exploding gradients
    output_dir = './output'
)


# Define the PPO trainer (using PPO as an example of policy optimization)
trainer = PPOv2Trainer(
    policy=model,
    config=config,
    tokenizer = model.get_tokenizer()
    ref_policy = PPOModel.from_pretrained("t5-based")
    reward_model = RewardModel()
    train_dataset = [
    {
        "prompt": "a star is a kind of celestial object / celestial body.",
        "chosen": ["all x.(Star(x) -> (CelestialObject(x) & CelestialBody(x)))", "all x.(Star(x) -> CelestialObject(x) & CelestialBody(x))"]
        "reject": ["all x.(Star(x) -> (CelestialObject(x) | CelestialBody(x)))", "all x y.((Star(x) & CelestialObject(y)) -> CelestialBody(x))", "all x y.((Star(x) & CelestialObject(y)) -> CelestialBody(x,y))", "all x.(Star(x) -> CelestialObject(x))", "all x y z.((Star(x) & CelestialObject(y) & CelestialBody(z)) -> (CelestialObject(x,y) & CelestialBody(x,z)))", "all x y z.((Star(x) & CelestialObject(y) & CelestialBody(z)) -> (CelestialObject(y) & CelestialBody(z)))", "all x y z.((Star(x) & CelestialObject(y) & CelestialBody(z)) -> (CelestialObject(x) & CelestialBody(x)))", "all x y z.((Star(x) & CelestialObject(y) & CelestialBody(z)) -> (CelestialObject(x,y) | CelestialBody(x,z)))",
        "all x y.((Star(x) & CelestialObject(y) & CelestialBody(x)) -> CelestialObject(x,y))", "all x y.((Star(x) & CelestialObject(y) & CelestialBody(x)) -> (CelestialObject(x) & CelestialBody(y)))", "all x y z.((Star(x) & CelestialObject(y) & CelestialBody(z)) -> (CelestialObject(x) | CelestialBody(x)))", "all x y z.((Star(x) & CelestialObject(y) & CelestialBody(z)) -> CelestialObject(x,y,z))"]
    }]

)


def train(total_episodes):
    for episode in range(total_episodes):
        observations = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = trainer.get_action(observations)
            observations, reward, done, _ = env.step(action)
            total_reward += reward

        trainer.update_policy()
        print(f"Episode {episode} reward: {total_reward}")

print("test")