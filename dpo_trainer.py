from transformers import AutoModelForCausalLM
from trl import PPOv2Trainer
from trl import PPOv2Config
from trl import DPOTrainer
from trl import DPOConfig
from transformers import T5ForConditionalGeneration, T5Config, T5Tokenizer
from datasets import load_dataset
from baseline.model import T5Model
import sentencepiece
import gym
from datasets import Dataset


env = gym.make('CartPole-v1')

# Load a pre-trained transformer model
config = T5Config.from_pretrained("t5-base")

model = T5ForConditionalGeneration(config = config)

training_args = DPOConfig(output_dir="./Qwen2-0.5B-DPO", logging_steps=10)

train_dataset = {
    "prompt": ["a star is a kind of celestial object / celestial body."],
    "chosen": [
        "all x.(Star(x) -> (CelestialObject(x) & CelestialBody(x)))"],
    "rejected": [
        "all x.(Star(x) -> (CelestialObject(x) | CelestialBody(x)))"
    ]
}

train_data = Dataset.from_dict(train_dataset)

trainer = DPOTrainer(model=model, ref_model=None, args = training_args, tokenizer=T5Tokenizer.from_pretrained('t5-base'), train_dataset = train_data)

trainer.train()

'''


{
        "prompt": "a star is a kind of celestial object / celestial body.",
        "chosen": ["all x.(Star(x) -> (CelestialObject(x) & CelestialBody(x)))", "all x.(Star(x) -> CelestialObject(x) & CelestialBody(x))"],
        "reject": ["all x.(Star(x) -> (CelestialObject(x) | CelestialBody(x)))", "all x y.((Star(x) & CelestialObject(y)) -> CelestialBody(x))", "all x y.((Star(x) & CelestialObject(y)) -> CelestialBody(x,y))", "all x.(Star(x) -> CelestialObject(x))", "all x y z.((Star(x) & CelestialObject(y) & CelestialBody(z)) -> (CelestialObject(x,y) & CelestialBody(x,z)))", "all x y z.((Star(x) & CelestialObject(y) & CelestialBody(z)) -> (CelestialObject(y) & CelestialBody(z)))", "all x y z.((Star(x) & CelestialObject(y) & CelestialBody(z)) -> (CelestialObject(x) & CelestialBody(x)))", "all x y z.((Star(x) & CelestialObject(y) & CelestialBody(z)) -> (CelestialObject(x,y) | CelestialBody(x,z)))",
        "all x y.((Star(x) & CelestialObject(y) & CelestialBody(x)) -> CelestialObject(x,y))", "all x y.((Star(x) & CelestialObject(y) & CelestialBody(x)) -> (CelestialObject(x) & CelestialBody(y)))", "all x y z.((Star(x) & CelestialObject(y) & CelestialBody(z)) -> (CelestialObject(x) | CelestialBody(x)))", "all x y z.((Star(x) & CelestialObject(y) & CelestialBody(z)) -> CelestialObject(x,y,z))"]
    },
    {
        "prompt": "apparent magnitude is a measure of the brightness of a celestial object / celestial body as observed on earth.",
        "chosen": ["all x y.((ApparentMagnitude(x) & CelestialObject(y) & CelestialBody(y) & ObservedOnEarth(x,y)) -> MeasuresBrightness(x,y))", "all x y.((AppearingMagnitude(x) & CelestialObject(y) & CelestialBody(y)) -> MeasuresBrightnessAsObservedOnEarth(x,y))"],
        "reject": ["all x y.((ApparentMagnitude(x) & CelestialObject(y) & CelestialBody(y)) -> MeasuresBrightnessAsObservedOnEarth(x,y))",
                   "all x.(ApparentMagnitude(x) -> (MeasuresBrightness(x) & CelestialObject(x) & CelestialBody(x) & ObservedOnEarth(x)))",
                   "all x y.((ApparentMagnitude(x) & CelestialObject(y) & CelestialBody(y)) -> MeasuresBrightness(x,y))",
                   "all x.(ApparentMagnitude(x) -> (MeasuresBrightnessOfCelestialObject(x) & MeasuresBrightnessOfCelestialBody(x) & ObservedOnEarth(x)))",
                   "all x y z.((ApparentMagnitude(x) & CelestialObject(y) & CelestialBody(z)) -> MeasuresBrightness(x,y,z))",
                   "all x y z.((ApparentMagnitude(x) & CelestialObject(y) & CelestialBody(z)) -> MeasuresBrightnessAsObservedOnEarth(x,y,z))",
                   "all x y z.((ApparentMagnitude(x) & CelestialObject(y) & CelestialBody(z) & ObservedOnEarth(x)) -> MeasuresBrightness(x,y,z))",
                   "all x y z.((ApparentMagnitude(x) & CelestialObject(y) & CelestialBody(z)) -> (MeasuresBrightness(x,y,z) & ObservedOnEarth(x)))",
                   "all x.(ApparentMagnitude(x) -> (MeasuresBrightnessOfCelestialObject(x) & ObservedOnEarth(x)))",
                   "all x.(ApparentMagnitude(x) -> (MeasuresBrightness(x) & ObservedOnEarth(x)))",
                   "all x y.((ApparentMagnitude(x) & CelestialObject(y) & CelestialBody(z)) -> MeasuresBrightness(x,y,z))"]
    },
    {
        "prompt": "apparent magnitude is a measure of the brightness of a star as observed on earth.",
        "chosen": ["all x y.((ApparentMagnitude(x) & Star(y) & ObservedOnEarth(x,y)) -> MeasuresBrightness(x,y))",
                   "all x y.((AppearingMagnitude(x) & Star(y)) -> MeasuresBrightnessAsObservedOnEarth(x,y))",
                   "all x y.((AppearingMagnitude(x) & Star(y)) -> MeasuresBrightness(x,y))"],
        "reject": ["all x.(ApparentMagnitude(x) -> (MeasuresBrightnessOfStar(x) & ObservedOnEarth(x)))",
                   "all x y.((AppearingMagnitude(x) & Star(y) & ObservedOnEarth(x,y)) -> MeasuresBrightness(x,y))",
                   "all x y.((ApparentMagnitude(x) & Star(y) & ObservedOnEarth(y)) -> MeasuresBrightness(x,y))",
                   "all x y.((AppearingMagnitude(x) & Star(y) & ObservedOnEarth(y)) -> MeasuresBrightness(x,y))",
                   "all x y z.((ApparentMagnitude(x) & Star(y) & Earth(z)) -> MeasuresBrightness(x,y,z))",
                   "all x y z.((AppearingMagnitude(x) & Star(y) & Earth(z)) -> MeasuresBrightness(x,y,z))",
                   "all x.(AppearingMagnitude(x) -> (MeasuresBrightnessOfStar(x) & ObservedOnEarth(x)))",
                   "all x.(ApparentMagnitude(x) -> MeasuresBrightnessOfStar(x))",
                   "all x y.((AppearingMagnitude(x) & Star(y) & ObservedOnEarth(y,x)) -> MeasuresBrightness(x,y))",
                   "all x.(ApparentMagnitude(x) -> (MeasuresBrightness(x) & ObservedOnEarth(x)))",
                   "all x y.((AppealingMagnitude(x) & Star(y) & ObservedOnEarth(y)) -> MeasuresBrightness(x,y))",
                   "all x y z.((ApparentMagnitude(x) & Star(y) & Earth(z)) -> MeasuresBrightnessAsObserved(x,y,z))"]
    }
    }

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
    },
    {
        "prompt": "apparent magnitude is a measure of the brightness of a celestial object / celestial body as observed on earth.",
        "chosen": ["all x y.((ApparentMagnitude(x) & CelestialObject(y) & CelestialBody(y) & ObservedOnEarth(x,y)) -> MeasuresBrightness(x,y))", "all x y.((AppearingMagnitude(x) & CelestialObject(y) & CelestialBody(y)) -> MeasuresBrightnessAsObservedOnEarth(x,y))"]
        "reject": ["all x y.((ApparentMagnitude(x) & CelestialObject(y) & CelestialBody(y)) -> MeasuresBrightnessAsObservedOnEarth(x,y))",
                   "all x.(ApparentMagnitude(x) -> (MeasuresBrightness(x) & CelestialObject(x) & CelestialBody(x) & ObservedOnEarth(x)))",
                   "all x y.((ApparentMagnitude(x) & CelestialObject(y) & CelestialBody(y)) -> MeasuresBrightness(x,y))",
                   "all x.(ApparentMagnitude(x) -> (MeasuresBrightnessOfCelestialObject(x) & MeasuresBrightnessOfCelestialBody(x) & ObservedOnEarth(x)))",
                   "all x y z.((ApparentMagnitude(x) & CelestialObject(y) & CelestialBody(z)) -> MeasuresBrightness(x,y,z))",
                   "all x y z.((ApparentMagnitude(x) & CelestialObject(y) & CelestialBody(z)) -> MeasuresBrightnessAsObservedOnEarth(x,y,z))",
                   "all x y z.((ApparentMagnitude(x) & CelestialObject(y) & CelestialBody(z) & ObservedOnEarth(x)) -> MeasuresBrightness(x,y,z))",
                   "all x y z.((ApparentMagnitude(x) & CelestialObject(y) & CelestialBody(z)) -> (MeasuresBrightness(x,y,z) & ObservedOnEarth(x)))",
                   "all x.(ApparentMagnitude(x) -> (MeasuresBrightnessOfCelestialObject(x) & ObservedOnEarth(x)))",
                   "all x.(ApparentMagnitude(x) -> (MeasuresBrightness(x) & ObservedOnEarth(x)))",
                   "all x y.((ApparentMagnitude(x) & CelestialObject(y) & CelestialBody(z)) -> MeasuresBrightness(x,y,z))"]
    },
    {
        "prompt": "apparent magnitude is a measure of the brightness of a star as observed on earth."
        "chosen": ["all x y.((ApparentMagnitude(x) & Star(y) & ObservedOnEarth(x,y)) -> MeasuresBrightness(x,y))",
                   "all x y.((AppearingMagnitude(x) & Star(y)) -> MeasuresBrightnessAsObservedOnEarth(x,y))",
                   "all x y.((AppearingMagnitude(x) & Star(y)) -> MeasuresBrightness(x,y))"]
        "reject": ["all x.(ApparentMagnitude(x) -> (MeasuresBrightnessOfStar(x) & ObservedOnEarth(x)))",
                   "all x y.((AppearingMagnitude(x) & Star(y) & ObservedOnEarth(x,y)) -> MeasuresBrightness(x,y))",
                   "all x y.((ApparentMagnitude(x) & Star(y) & ObservedOnEarth(y)) -> MeasuresBrightness(x,y))",
                   "all x y.((AppearingMagnitude(x) & Star(y) & ObservedOnEarth(y)) -> MeasuresBrightness(x,y))",
                   "all x y z.((ApparentMagnitude(x) & Star(y) & Earth(z)) -> MeasuresBrightness(x,y,z))",
                   "all x y z.((AppearingMagnitude(x) & Star(y) & Earth(z)) -> MeasuresBrightness(x,y,z))",
                   "all x.(AppearingMagnitude(x) -> (MeasuresBrightnessOfStar(x) & ObservedOnEarth(x)))",
                   "all x.(ApparentMagnitude(x) -> MeasuresBrightnessOfStar(x))",
                   "all x y.((AppearingMagnitude(x) & Star(y) & ObservedOnEarth(y,x)) -> MeasuresBrightness(x,y))",
                   "all x.(ApparentMagnitude(x) -> (MeasuresBrightness(x) & ObservedOnEarth(x)))",
                   "all x y.((AppealingMagnitude(x) & Star(y) & ObservedOnEarth(y)) -> MeasuresBrightness(x,y))",
                   "all x y z.((ApparentMagnitude(x) & Star(y) & Earth(z)) -> MeasuresBrightnessAsObserved(x,y,z))"]
    }
    ]

)
##

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

'''