"""
autonomous_parking_env_with_collision.py

– Car agent (3 m×2 m) using rear‐axle kinematics (state at rear‐axle)
– Geometric center (x₀,y₀) used for rendering and parking‐success
– Pixel‐perfect collision (masks + rollback) against U‐shaped slot
– Keyboard control for continuous speed and steering
– Parking‐success check per Li & Long (2021) Eq. (8), with reverse‐orientation target (270°)
– Prints "Ouch!" whenever the car crashes into a wall
Scale: 100 px = 1 m
"""

import pygame, sys, math

# === Constants & Scaling ===
PX_PER_M         = 100                # 100 px = 1 m
TOLERANCE        = int(0.15 * PX_PER_M)
CAR_LENGTH       = int(3.0 * PX_PER_M)
CAR_WIDTH        = int(2.0 * PX_PER_M)
WHEEL_BASE       = int(2.5 * PX_PER_M)

SLOT_WIDTH       = CAR_WIDTH + 2 * TOLERANCE
SLOT_DEPTH       = CAR_LENGTH + TOLERANCE

SCREEN_WIDTH, SCREEN_HEIGHT = 1440, 960
FPS = 60

LINE_Y     = 500
LINE_THICK = 8

MAX_STEER_DEG    = 30
MAX_STEER        = math.radians(MAX_STEER_DEG)
MAX_DISPLACEMENT = 0.2 * PX_PER_M    # 0.2 m/s → 20 px/s
MIN_DISPLACEMENT = -1.0 * PX_PER_M   # –1 m/s → –100 px/s

HEADING_LEN = PX_PER_M              # 1 m → 100 px

WHITE = (255,255,255)
BLACK = (  0,  0,  0)
BLUE  = ( 50, 50,200)
RED   = (200,  0,  0)
GREEN = (  0,200,  0)

# Slot‐center (xₚ,yₚ) for parking‐success
SLOT_CENTER = (SCREEN_WIDTH // 2, LINE_Y + SLOT_DEPTH // 2)
# Initial rear‐axle start position
START_REAR  = (SCREEN_WIDTH // 2, LINE_Y - SLOT_DEPTH // 2 - 50)


class Car(pygame.sprite.Sprite):
    """Rear‐axle kinematics; geometric center computed for rendering & parking."""
    def __init__(self, rear_pos):
        super().__init__()
        self.original_image = pygame.Surface((CAR_LENGTH, CAR_WIDTH), pygame.SRCALPHA)
        pygame.draw.rect(self.original_image, BLUE, (0, 0, CAR_LENGTH, CAR_WIDTH))
        self.image = self.original_image
        self.rect  = self.image.get_rect()

        # Kinematic state at rear‐axle midpoint
        self.x, self.y = float(rear_pos[0]), float(rear_pos[1])
        self.alpha    = 0.0      # heading (rad)
        self.s        = 0.0      # speed (px/s)
        self.beta     = 0.0      # steering angle (rad)

        # Geometric center (x₀,y₀), updated in `update`
        self.x0 = self.x + (WHEEL_BASE / 2)
        self.y0 = self.y

    def update(self, dt):
        # — Rear‐wheel kinematics (Eq. 1)
        dx     = self.s * dt * math.cos(self.alpha) * math.cos(self.beta)
        dy     = self.s * dt * math.sin(self.alpha) * math.cos(self.beta)
        dalpha = (self.s * dt / WHEEL_BASE) * math.sin(self.beta)

        self.x     += dx
        self.y     += dy
        self.alpha += dalpha

        # — Geometric center (x₀,y₀) for render & parking condition
        self.x0 = self.x + (WHEEL_BASE / 2) * math.cos(self.alpha)
        self.y0 = self.y + (WHEEL_BASE / 2) * math.sin(self.alpha)

        # — Rotate sprite about its center
        angle_deg    = -math.degrees(self.alpha)
        self.image   = pygame.transform.rotate(self.original_image, angle_deg)
        self.rect    = self.image.get_rect(center=(int(self.x0), int(self.y0)))

    def set_action(self, displacement, steer_angle):
        self.s    = max(min(displacement, MAX_DISPLACEMENT), MIN_DISPLACEMENT)
        self.beta = max(min(steer_angle,   MAX_STEER),        -MAX_STEER)


def create_slot_walls():
    x1    = (SCREEN_WIDTH - SLOT_WIDTH) // 2
    x2    = x1 + SLOT_WIDTH
    y_top = LINE_Y - LINE_THICK // 2
    y_bot = LINE_Y + SLOT_DEPTH - LINE_THICK // 2
    return [
        pygame.Rect(0,    y_top, x1,              LINE_THICK),
        pygame.Rect(x2,   y_top, SCREEN_WIDTH-x2, LINE_THICK),
        pygame.Rect(x1 - LINE_THICK//2, LINE_Y, LINE_THICK, SLOT_DEPTH),
        pygame.Rect(x2 - LINE_THICK//2, LINE_Y, LINE_THICK, SLOT_DEPTH),
        pygame.Rect(x1,   y_bot, SLOT_WIDTH,     LINE_THICK),
    ]


def draw_parking_slot(surface):
    x1 = (SCREEN_WIDTH - SLOT_WIDTH) // 2
    x2 = x1 + SLOT_WIDTH
    y_top, y_bot = LINE_Y, LINE_Y + SLOT_DEPTH
    pygame.draw.line(surface, BLACK, (0,    y_top), (x1,   y_top), LINE_THICK)
    pygame.draw.line(surface, BLACK, (x2,   y_top), (SCREEN_WIDTH, y_top), LINE_THICK)
    pygame.draw.line(surface, BLACK, (x1, y_top), (x1, y_bot), LINE_THICK)
    pygame.draw.line(surface, BLACK, (x2, y_top), (x2, y_bot), LINE_THICK)
    pygame.draw.line(surface, BLACK, (x1, y_bot), (x2, y_bot), LINE_THICK)


def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Parking Env – Reverse Orientation Parking")
    clock = pygame.time.Clock()

    car   = Car(START_REAR)
    walls = create_slot_walls()

    # Precompute wall mask for pixel‐perfect collision
    wall_surf = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
    draw_parking_slot(wall_surf)
    wall_mask = pygame.mask.from_surface(wall_surf)

    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0  # seconds elapsed

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False

        # — Controls to (s, β)
        keys  = pygame.key.get_pressed()
        disp  = (MAX_DISPLACEMENT if keys[pygame.K_UP]
                 else MIN_DISPLACEMENT if keys[pygame.K_DOWN]
                 else 0.0)
        steer = (-MAX_STEER if keys[pygame.K_LEFT]
                 else MAX_STEER if keys[pygame.K_RIGHT]
                 else 0.0)
        car.set_action(disp, steer)

        # — Snapshot for rollback
        prev = (car.x, car.y, car.alpha, car.x0, car.y0, car.image, car.rect.copy())

        # — Update kinematics & sprite
        car.update(dt)

        # — Pixel‐perfect collision + rollback
        car_mask = pygame.mask.from_surface(car.image)
        offset   = (car.rect.left, car.rect.top)
        if wall_mask.overlap(car_mask, offset):
            # Collision detected: rollback and print "Ouch!"
            car.x, car.y, car.alpha, car.x0, car.y0, car.image, car.rect = prev
            print("Ouch!")

        # — Render
        screen.fill(WHITE)
        screen.blit(wall_surf, (0, 0))
        screen.blit(car.image, car.rect)

        # — Heading marker from geometric center
        cx, cy = int(car.x0), int(car.y0)
        pygame.draw.circle(screen, RED,   (cx, cy), 5)
        hx = car.x0 + math.cos(car.alpha) * HEADING_LEN
        hy = car.y0 + math.sin(car.alpha) * HEADING_LEN
        pygame.draw.line(screen, GREEN, (cx, cy), (int(hx), int(hy)), 4)

        # — Parking‐success check with reverse‐orientation target (270°) —
        xp, yp       = SLOT_CENTER
        d            = math.hypot(car.x0 - xp, car.y0 - yp)
        alpha_deg    = math.degrees(car.alpha) % 360
        target_deg   = 270
        heading_err  = min(abs(alpha_deg - target_deg),
                           360 - abs(alpha_deg - target_deg))

        if d <= 0.1 * PX_PER_M and heading_err <= 10:
            pygame.draw.circle(screen, GREEN, (cx, cy), 12, width=3)
            print(f"Parked (reverse)! d={d/PX_PER_M:.3f} m, Δα={heading_err:.1f}°")

        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
