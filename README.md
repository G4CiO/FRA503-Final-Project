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
#### 4.1.1 Car kinematic model
#### 4.1.2 State
#### 4.1.3 Action
#### 4.1.4 Reward and Penalty
#### 4.1.5 Parking conditions
### 4.2 Create train model
## **5. Simulation**

## **6. Result**

## **7. Analysis**