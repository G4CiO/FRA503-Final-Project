# autonomous_parking_env.py (Environment)

from __future__ import annotations
import math, numpy as np, pygame
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding

# Constants
PX_PER_M = 60
CAR_LEN_M, CAR_W_M = 3.0, 2.0
CAR_LENGTH, CAR_WIDTH = int(CAR_LEN_M * PX_PER_M), int(CAR_W_M * PX_PER_M)
WHEEL_BASE = int(2.5 * PX_PER_M)

MIN_DISP_M, MAX_DISP_M = -0.8, 0.2  # backward max -0.5 m/step, forward max 0.2 m/step
MAX_STEER_RAD = math.radians(30)

SCREEN_W, SCREEN_H = 1440, 960
LINE_Y, LINE_THICK = 700, 4
ROAD_Y = 100
FPS = 60

WHITE, BLACK = (255, 255, 255), (0, 0, 0)
BLUE, RED, GREEN = (50, 50, 200), (200, 0, 0), (0, 200, 0)

TOLERANCE = int(0.3 * PX_PER_M)
SLOT_W = CAR_WIDTH + 2 * TOLERANCE
SLOT_D = CAR_LENGTH + TOLERANCE
SLOT_CENTER = (SCREEN_W // 2, LINE_Y + SLOT_D // 2)

START_REAR = (100 + (SLOT_CENTER[0] + 4.5 * PX_PER_M - WHEEL_BASE // 2),
              (ROAD_Y + LINE_Y - 50) // 2)

HEADING_VEC = PX_PER_M


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

    def _update_center(self):
        self.cx = self.x + (WHEEL_BASE / 2) * math.cos(self.alpha)
        self.cy = self.y + (WHEEL_BASE / 2) * math.sin(self.alpha)

    def set_action(self, disp_m, steer):
        self.disp_m = float(np.clip(disp_m, MIN_DISP_M, MAX_DISP_M))
        self.beta = float(np.clip(steer, -MAX_STEER_RAD, MAX_STEER_RAD))

    def integrate_once(self):
        disp_px = self.disp_m * PX_PER_M
        self.x += disp_px * math.cos(self.alpha) * math.cos(self.beta)
        self.y += disp_px * math.sin(self.alpha) * math.cos(self.beta)
        self.alpha += (disp_px / WHEEL_BASE) * math.sin(self.beta)
        self._update_center()
        self.image = pygame.transform.rotate(self.base, -math.degrees(self.alpha))
        self.rect = self.image.get_rect(center=(int(self.cx), int(self.cy)))


class ParkingEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": FPS}

    def __init__(self, *, render_mode=None, max_steps=200, seed=None):
        super().__init__()
        self._render_mode = render_mode
        self.max_steps = max_steps

        # Define action space: 2 continuous actions - displacement (forward/backward) and steering angle
        self.action_space = spaces.Box(
            low=np.array([MIN_DISP_M, -MAX_STEER_RAD], np.float32),
            high=np.array([MAX_DISP_M, MAX_STEER_RAD], np.float32),
            shape=(2,), dtype=np.float32
        )

        # Define observation space: x, y relative to parking slot center and heading angle (radians)
        obs_high = np.array([SCREEN_W / PX_PER_M, SCREEN_H / PX_PER_M, math.pi], np.float32)
        self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)

        # Initialize pygame window or surface
        pygame.init()
        self._screen = pygame.display.set_mode((SCREEN_W, SCREEN_H)) \
            if render_mode == "human" else pygame.Surface((SCREEN_W, SCREEN_H))
        self._clock = pygame.time.Clock()

        # Create a mask for the parking slot boundaries for collision checking
        self._slot_mask = self._make_wall_mask()
        self.seed(seed)
        self.reset(seed=seed)

        self.OFFSCREEN_PENALTY = -10  # Base offscreen penalty (not currently used in step)

        # Cumulative reward trackers for each reward component (useful for logging)
        self.ep_distance_reward = 0.0
        self.ep_angle_reward = 0.0
        self.ep_penalty_reward = 0.0  # Penalty for collisions/offscreen

        self.ep_lane_penalty = 0.0    # Penalty for leaving driving corridor (lane)
        self.ep_slot_bonus = 0.0      # Bonus for entering parking slot
        self.ep_heading_penalty = 0.0 # Penalty for bad heading alignment

        self.d = 0.0
        self.distance_in_meters = 0.0  # Distance to parking slot center in meters

    def _make_wall_mask(self):
        # Draw the parking boundaries on a transparent surface for collision detection
        surf = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
        pygame.draw.line(surf, BLACK, (0, ROAD_Y), (SCREEN_W, ROAD_Y), LINE_THICK)  # Road boundary line
        x1 = (SCREEN_W - SLOT_W) // 2
        x2 = x1 + SLOT_W
        y1 = LINE_Y
        y2 = y1 + SLOT_D
        # Draw slot walls: left, right, top, bottom boundaries of the parking slot
        for p, q in [((0, y1), (x1, y1)), ((x2, y1), (SCREEN_W, y1)),
                     ((x1, y1), (x1, y2)), ((x2, y1), (x2, y2)),
                     ((x1, y2), (x2, y2))]:
            pygame.draw.line(surf, BLACK, p, q, LINE_THICK)
        return pygame.mask.from_surface(surf)

    def seed(self, seed=None):
        # Set numpy random seed for reproducibility
        self.np_random, _ = seeding.np_random(seed)

    def reset(self, *, seed=None, options=None):
        # Reset environment and episode data
        if seed is not None:
            self.seed(seed)
        self.car = _Car(START_REAR)  # Initialize car at starting rear-axle position
        self.steps = 0
        self.d_max = 10 * PX_PER_M  # Max distance used for normalization in rewards

        # Reset cumulative rewards for each episode
        self.ep_distance_reward = 0.0
        self.ep_angle_reward = 0.0
        self.ep_penalty_reward = 0.0
        self.ep_lane_penalty = 0.0
        self.ep_slot_bonus = 0.0
        self.ep_heading_penalty = 0.0

        # Observation: relative position (meters) of car center to parking slot center, plus heading angle (radians)
        obs = np.array([
            (self.car.cx - SLOT_CENTER[0]) / PX_PER_M,
            (self.car.cy - SLOT_CENTER[1]) / PX_PER_M,
            self.car.alpha
        ], dtype=np.float32)

        if self._render_mode == "human":
            self.render()

        return obs, {}

    def step(self, action):
        # Apply action: displacement and steering to the car
        self.car.set_action(*action)
        prev = self._snapshot()  # Save current state to restore if collision occurs
        self.car.integrate_once()
        self.steps += 1

        # Check for collision or offscreen (off the visible window)
        collision = self._collision()
        off_left = self.car.rect.right < 0
        off_right = self.car.rect.left > SCREEN_W
        off_top = self.car.rect.bottom < 0
        off_bot = self.car.rect.top > SCREEN_H
        offscreen = off_left or off_right or off_top or off_bot
        collision = collision or offscreen

        # ----------- Key distance calculation -----------
        # Calculate Euclidean distance (in pixels) between car center and parking slot center
        # This distance is used in distance-based reward (normalized by d_max)
        self.d = math.hypot(self.car.cx - SLOT_CENTER[0], self.car.cy - SLOT_CENTER[1])
        R_d = -self.d / self.d_max  # Negative normalized distance reward (closer is better)
        self.distance_in_meters = self.d / PX_PER_M  # Convert to meters for logging

        # ----------- Angle / Heading calculation -----------
        desired_heading = -math.pi / 2  # Desired heading aligned with parking slot (-90 degrees)
        angle_diff = (self.car.alpha - desired_heading + math.pi) % (2 * math.pi) - math.pi
        R_a = -abs(angle_diff) / math.pi  # Normalized negative penalty for heading error

        # ----------- Weight factor for blending distance and angle reward --------
        y_diff = self.car.cy - LINE_Y
        w = 0 if y_diff < 0 else min(1.0, y_diff / SLOT_D)

        # Distance reward weighted more at the start (far from slot), angle reward weighted more near slot
        distance_reward = 10 * (1 - w) * R_d
        angle_reward = 10 * w * R_a

        # ----------- Lane penalty -----------
        lane_left = (SCREEN_W // 2) - SLOT_W * 1.5
        lane_right = (SCREEN_W // 2) + SLOT_W * 1.5
        lane_penalty = -15 if not (lane_left <= self.car.cx <= lane_right) else 0

        # ----------- Slot bonus -----------
        slot_x1 = (SCREEN_W - SLOT_W) // 2
        slot_x2 = slot_x1 + SLOT_W
        slot_y1 = LINE_Y
        slot_y2 = slot_y1 + SLOT_D
        in_slot_area = (slot_x1 <= self.car.cx <= slot_x2) and (slot_y1 <= self.car.cy <= slot_y2)
        slot_bonus = 10000.0 if in_slot_area else 0

        # ----------- Offscreen penalty -----------
        OFFSCREEN_PENALTY = -500.0
        penalty_reward = OFFSCREEN_PENALTY if collision else 0

        # ----------- Heading penalty -----------
        heading_penalty = -3 * (abs(angle_diff) / math.pi)  # Penalize heading error

        # ----------- Aggregate total reward -----------
        reward = distance_reward + angle_reward + penalty_reward + lane_penalty + slot_bonus + heading_penalty


        # ----------- Accumulate each reward component for logging --------
        self.ep_distance_reward += distance_reward
        self.ep_angle_reward += angle_reward
        self.ep_penalty_reward += penalty_reward
        self.ep_lane_penalty += lane_penalty
        self.ep_slot_bonus += slot_bonus
        self.ep_heading_penalty += heading_penalty

        # ----------- Success condition -----------
        success = (self.d <= 0.1 * PX_PER_M) and (abs(math.degrees(angle_diff)) <= 10)
        SUCCESS_REWARD = 100000.0
        if success:
            reward += SUCCESS_REWARD

        # ----------- Episode termination flags -----------
        terminated = collision or success
        truncated = self.steps >= self.max_steps

        if collision:
            self._restore(prev)  # Rollback state on collision to previous step

        # ----------- Prepare observation -----------
        obs = np.array([
            (self.car.cx - SLOT_CENTER[0]) / PX_PER_M,
            (self.car.cy - SLOT_CENTER[1]) / PX_PER_M,
            self.car.alpha
        ], np.float32)

        if self._render_mode == "human":
            self.render()

        # ----------- Info dictionary with cumulative rewards and status flags -----------
        info = {
            "success": success,
            "collision": collision,
            "offscreen": offscreen,
            "ep_distance_reward": self.ep_distance_reward,
            "ep_angle_reward": self.ep_angle_reward,
            "ep_penalty_reward": self.ep_penalty_reward,
            "ep_lane_penalty": self.ep_lane_penalty,
            "ep_slot_bonus": self.ep_slot_bonus,
            "ep_heading_penalty": self.ep_heading_penalty,
            "distance_in_meters": self.distance_in_meters,
        }

        return obs, reward, terminated, truncated, info


    def render(self):
        s = self._screen
        s.fill(WHITE)

        # Draw lane and parking slot walls
        pygame.draw.line(s, BLACK, (0, ROAD_Y), (SCREEN_W, ROAD_Y), LINE_THICK)
        x1 = (SCREEN_W - SLOT_W) // 2
        x2 = x1 + SLOT_W
        y1 = LINE_Y
        y2 = y1 + SLOT_D
        for p, q in [((0, y1), (x1, y1)), ((x2, y1), (SCREEN_W, y1)),
                    ((x1, y1), (x1, y2)), ((x2, y1), (x2, y2)),
                    ((x1, y2), (x2, y2))]:
            pygame.draw.line(s, BLACK, p, q, LINE_THICK)

        # Parking point (target) – green circle
        pygame.draw.circle(s, GREEN, SLOT_CENTER, 10)  # increased size for visibility

        # Car sprite
        s.blit(self.car.image, self.car.rect)

        # Car reference point – blue circle
        pygame.draw.circle(s, (255, 255, 0 ), (int(self.car.cx), int(self.car.cy)), 8)

        # Heading arrows: forward (nose) → RED
        fx = self.car.cx + math.cos(self.car.alpha) * HEADING_VEC
        fy = self.car.cy + math.sin(self.car.alpha) * HEADING_VEC
        pygame.draw.line(s, RED, (self.car.cx, self.car.cy), (fx, fy), 4)

        # Backward (tail) → BLACK
        bx = self.car.cx - math.cos(self.car.alpha) * HEADING_VEC
        by = self.car.cy - math.sin(self.car.alpha) * HEADING_VEC
        pygame.draw.line(s, BLACK, (self.car.cx, self.car.cy), (bx, by), 4)

        if self._render_mode == "human":
            pygame.display.flip()


            if self._render_mode == "human":
                pygame.display.flip()

    def close(self):
        pygame.quit()

    def _collision(self):
        mask = pygame.mask.from_surface(self.car.image)
        return bool(self._slot_mask.overlap(mask, (self.car.rect.left, self.car.rect.top)))

    def _snapshot(self):
        return (self.car.x, self.car.y, self.car.alpha,
                self.car.cx, self.car.cy, self.car.image, self.car.rect.copy())

    def _restore(self, snap):
        (self.car.x, self.car.y, self.car.alpha,
         self.car.cx, self.car.cy, self.car.image, self.car.rect) = snap


