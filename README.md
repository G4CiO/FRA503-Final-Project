# FRA503-Final-Project

## **1. Project Overview**
### 1.1 Objective
To design a car parking system using Deep Reinforcement Learning (DRL), specifically the DDPG algorithm, to allow a vehicle to autonomously park into a vertical parking slot with minimal reliance on physical sensors.
### 1.2 Goal
Train an intelligent agent to learn optimal parking behavior through interaction in a simulated environment using only kinematic data.
## **2. Scopes**
- Simulate a vertical parking environment using a 2D coordinate system.
- Implement a low-speed kinematic car model (rear-wheel driven) and Set the car to have no slip and dynamics
- Apply Deep Deterministic Policy Gradient (DDPG) for continuous control.
- Design a reward function based on distance, heading angle, and safety penalties.
- Evaluate the model’s ability to generalize from different initial positions.
- Focus on software simulation only (no real-world sensors or physical robot used).
## **3. Problem Formulation**
We aim to develop an agent that can learn to park a vehicle by selecting continuous steering and displacement actions based on its position and orientation, while respecting physical and safety constraints, and maximizing proximity and alignment with the parking spot.

**Inputs (State):**
- The current position of the vehicle: 
(
𝑥
,
𝑦
)
(x,y)

- The heading angle of the vehicle: 
𝛼
α

**Outputs (Action):**
- The steering angle: 
𝛽
β

- The displacement of the vehicle: 
𝑠
s

**Objective:**

To find a sequence of actions 
(
𝛽
,
𝑠
)
(β,s) that guides the vehicle:

- Closely and accurately to the parking point

- With a final heading angle of approximately 90°

- Without colliding or pressing any boundary lines

**Constraints:**
- The steering angle 
𝛽
β is limited to 
[
−
30
∘
,
30
∘
]
[−30 
∘
 ,30 
∘
 ]

- The displacement 
𝑠
s per step is limited to 
[
−
1.0
,
0.2
]
 meters

- The vehicle must not cross or press parking lines (line pressure penalty)

- The motion follows a kinematic model without slippage

## **4. Techniques**

### 4.1 Create Environment

#### 4.1.1 Car Kinematic Model
The car kinematic model is implemented in the `_Car` class within the `autonomous_parking_env.py` file. The model represents the car as a rectangle and tracks its position using rear axle coordinates `(x, y)` and a heading angle `alpha`. The motion of the car is based on physics and geometry principles. The `integrate_once` method updates the car's position and heading based on the displacement (`disp_m`) and steering angle (`beta`). The class initializes the car with its position and orientation, updating the graphical representation accordingly.

```python
class _Car(pygame.sprite.Sprite):
    def __init__(self, rear_px):
        super().__init__()
        surf = pygame.Surface((CAR_LENGTH, CAR_WIDTH), pygame.SRCALPHA)
        surf.fill(BLUE)
        self.base = surf
        self.image = surf
        self.rect = self.image.get_rect()
        self.x, self.y = map(float, rear_px)  # rear-axle position
        self.alpha = 0.0  # heading angle (rad)
        self.disp_m = self.beta = 0.0
        self._update_center()
        self.rect = self.image.get_rect(center=(int(self.cx), int(self.cy)))
```

#### 4.1.2 State
The environment's state at any time is represented by the car's relative position to the parking slot center and its heading angle. This is captured as a NumPy array:
```python
obs = np.array([
    (self.car.cx - SLOT_CENTER[0]) / PX_PER_M,
    (self.car.cy - SLOT_CENTER[1]) / PX_PER_M,
    self.car.alpha
], dtype=np.float32)
```
The variables reflect the car's `x` and `y` relative to the parking slot center, normalized by `PX_PER_M`, and its heading `alpha`.

#### 4.1.3 Action
The car's actions consist of two continuous variables:
- `disp_m`: The displacement (forward or backward movement).
- `steer`: The steering angle, limited by `MAX_STEER_RAD`.

The action space is defined as:
```python
self.action_space = spaces.Box(
    low=np.array([MIN_DISP_M, -MAX_STEER_RAD], np.float32),
    high=np.array([MAX_DISP_M, MAX_STEER_RAD], np.float32),
    shape=(2,), dtype=np.float32
)
```
This allows the agent to control both the movement and steering in a continuous manner.

#### 4.1.4 Reward and Penalty
The reward system encourages the car to park correctly while avoiding collisions and lane violations. The total reward is composed of:
- `distance_reward`: Rewards for approaching the parking slot.
- `angle_reward`: Rewards for aligning the heading.
- `penalty_reward`: Penalties for collisions or leaving the screen.
- `lane_penalty`: Penalizes moving outside the parking lane.
- `slot_bonus`: Large bonus when parked inside the slot.
- `heading_penalty`: Penalizes misalignment.

The total reward formula:
```python
reward = distance_reward + angle_reward + penalty_reward + lane_penalty + slot_bonus + heading_penalty
```

#### 4.1.5 Parking Conditions
Parking success is defined by:
- Distance to the slot center `d` less than 0.1 * PX_PER_M (10 cm).
- Heading angle difference within 10 degrees.

When these conditions are met:
```python
success = (d <= 0.1 * PX_PER_M) and (abs(math.degrees(angle_diff)) <= 10)
if success:
    reward += SUCCESS_REWARD
```
where `SUCCESS_REWARD = 100000.0`.

---

### 4.2 Create Train Model
The training utilizes the DDPG algorithm from Stable Baselines3, configured with continuous action spaces and normal action noise to promote exploration.

```python
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
```

**Parameters:**
- `learning_rate`: Controls how quickly the model learns.
- `buffer_size`: Size of the experience replay buffer.
- `batch_size`: Number of samples per training batch.
- `tau`: For soft updates.
- `gamma`: Discount factor for future rewards.
- `action_noise`: Adds randomness for exploration.

**Training Loop:**
The model is trained over a set number of timesteps, with exploration noise decaying over time to favor exploitation:
```python
model.learn(total_timesteps=1000, reset_num_timesteps=False, progress_bar=True, callback=callback)
```

---

## **5. Simulation**
The simulation visualizes the car's behavior within the environment using Pygame. It loads a trained model and continuously predicts actions to control the car.

```python
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
```

**Visualization:**
- The `render()` function updates the graphical window.
- The agent's actions are predicted based on current state.
- The environment resets when parking is successful or fails, allowing continuous testing and visualization.

## **6. Result**

## **7. Analysis**