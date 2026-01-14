#!/usr/bin/env python3
"""
Small Game - A maze game controlled by hand gestures
Written by Claude, designed by Daniel Goldenholz 2026
"""

import pygame
import cv2
import numpy as np
import random
import math
import time
from collections import deque
from enum import Enum
import threading
import urllib.request
import os

# MediaPipe imports - handle different versions
try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    USE_TASKS_API = True
except ImportError:
    import mediapipe as mp
    USE_TASKS_API = False

# Initialize Pygame and its mixer for sound
pygame.init()
pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)

# Game constants
MAZE_SIZE = 30
CELL_SIZE = 20
MAZE_OFFSET_X = 100
MAZE_OFFSET_Y = 100
FPS = 60
COUNTDOWN_TIME = 90  # seconds (larger maze needs more time)

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (100, 100, 100)
DARK_GRAY = (50, 50, 50)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
RED = (255, 0, 0)
ORANGE = (255, 165, 0)
PURPLE = (148, 0, 211)
CYAN = (0, 255, 255)
CAT_ORANGE = (255, 140, 0)
CAT_DARK = (200, 100, 0)

class GameState(Enum):
    TITLE = 1
    PLAYING = 2
    LEVEL_COMPLETE = 3
    GAME_OVER = 4
    RECONFIGURING = 5

class Direction(Enum):
    NONE = 0
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4

class SoundGenerator:
    """Generate simple synthesized sounds"""

    @staticmethod
    def generate_tone(frequency, duration, volume=0.3, wave_type='sine'):
        """Generate a tone with given frequency and duration"""
        sample_rate = 44100
        n_samples = int(sample_rate * duration)
        t = np.linspace(0, duration, n_samples, False)

        if wave_type == 'sine':
            wave = np.sin(2 * np.pi * frequency * t)
        elif wave_type == 'square':
            wave = np.sign(np.sin(2 * np.pi * frequency * t))
        elif wave_type == 'sawtooth':
            wave = 2 * (t * frequency - np.floor(0.5 + t * frequency))
        else:
            wave = np.sin(2 * np.pi * frequency * t)

        # Apply envelope to avoid clicks
        envelope = np.ones(n_samples)
        attack = int(0.01 * sample_rate)
        release = int(0.05 * sample_rate)
        envelope[:attack] = np.linspace(0, 1, attack)
        envelope[-release:] = np.linspace(1, 0, release)

        wave = wave * envelope * volume

        # Convert to 16-bit stereo
        wave = (wave * 32767).astype(np.int16)
        stereo_wave = np.column_stack((wave, wave))

        return pygame.sndarray.make_sound(stereo_wave)

    @staticmethod
    def generate_move_sound():
        """Short blip for movement"""
        return SoundGenerator.generate_tone(440, 0.05, 0.2, 'sine')

    @staticmethod
    def generate_wall_sound():
        """Thud sound for hitting wall"""
        return SoundGenerator.generate_tone(100, 0.1, 0.3, 'square')

    @staticmethod
    def generate_easy_exit_sound():
        """Sound for easy exit"""
        sounds = []
        sounds.append(SoundGenerator.generate_tone(523, 0.15, 0.3, 'sine'))  # C
        sounds.append(SoundGenerator.generate_tone(659, 0.15, 0.3, 'sine'))  # E
        return sounds

    @staticmethod
    def generate_medium_exit_sound():
        """Sound for medium exit"""
        sounds = []
        sounds.append(SoundGenerator.generate_tone(523, 0.1, 0.3, 'sine'))   # C
        sounds.append(SoundGenerator.generate_tone(659, 0.1, 0.3, 'sine'))   # E
        sounds.append(SoundGenerator.generate_tone(784, 0.15, 0.3, 'sine'))  # G
        return sounds

    @staticmethod
    def generate_hard_exit_sound():
        """Triumphant sound for hard exit"""
        sounds = []
        sounds.append(SoundGenerator.generate_tone(523, 0.1, 0.4, 'sine'))   # C
        sounds.append(SoundGenerator.generate_tone(659, 0.1, 0.4, 'sine'))   # E
        sounds.append(SoundGenerator.generate_tone(784, 0.1, 0.4, 'sine'))   # G
        sounds.append(SoundGenerator.generate_tone(1047, 0.2, 0.4, 'sine'))  # High C
        return sounds

    @staticmethod
    def generate_countdown_warning():
        """Warning beep for low time"""
        return SoundGenerator.generate_tone(880, 0.1, 0.4, 'square')

    @staticmethod
    def generate_game_over_sound():
        """Sad sound for game over"""
        sounds = []
        sounds.append(SoundGenerator.generate_tone(400, 0.2, 0.3, 'sawtooth'))
        sounds.append(SoundGenerator.generate_tone(300, 0.2, 0.3, 'sawtooth'))
        sounds.append(SoundGenerator.generate_tone(200, 0.4, 0.3, 'sawtooth'))
        return sounds

    @staticmethod
    def generate_bonus_sound():
        """Sound for time bonus"""
        sounds = []
        for i in range(5):
            freq = 600 + i * 100
            sounds.append(SoundGenerator.generate_tone(freq, 0.08, 0.3, 'sine'))
        return sounds

    @staticmethod
    def generate_reconfigure_sound():
        """Whooshing sound for maze reconfiguration"""
        sounds = []
        # Descending whoosh
        for i in range(8):
            freq = 800 - i * 80
            sounds.append(SoundGenerator.generate_tone(freq, 0.1, 0.2, 'sawtooth'))
        return sounds


class Particle:
    """Particle for swirling animation during maze reconfiguration"""

    def __init__(self, x, y, target_x, target_y, color):
        self.x = float(x)
        self.y = float(y)
        self.target_x = float(target_x)
        self.target_y = float(target_y)
        self.color = color
        self.size = random.randint(3, 8)
        self.angle = random.uniform(0, 2 * math.pi)
        self.angular_speed = random.uniform(0.1, 0.3)
        self.radius = random.uniform(20, 100)
        self.progress = 0.0
        self.speed = random.uniform(0.01, 0.03)

    def update(self):
        """Update particle position with swirling motion"""
        self.progress += self.speed
        self.angle += self.angular_speed

        if self.progress >= 1.0:
            self.x = self.target_x
            self.y = self.target_y
            return True  # Particle reached destination

        # Interpolate position with swirling
        t = self.progress
        # Ease in-out
        t = t * t * (3 - 2 * t)

        base_x = self.x + (self.target_x - self.x) * t
        base_y = self.y + (self.target_y - self.y) * t

        # Add swirling motion that decreases as we approach target
        swirl_factor = (1 - t) * self.radius
        self.draw_x = base_x + math.cos(self.angle) * swirl_factor
        self.draw_y = base_y + math.sin(self.angle) * swirl_factor

        return False

    def draw(self, surface):
        """Draw the particle"""
        if hasattr(self, 'draw_x'):
            # Fade color based on progress
            fade = 1 - self.progress * 0.5
            color = tuple(int(c * fade) for c in self.color)
            pygame.draw.circle(surface, color, (int(self.draw_x), int(self.draw_y)), self.size)


class MazeGenerator:
    """Generate a challenging maze with multiple independently reachable exits"""

    @staticmethod
    def generate(size=30, start_pos=None):
        """Generate a maze using recursive backtracking with added complexity"""
        # Initialize maze with all walls
        maze = [[1 for _ in range(size)] for _ in range(size)]

        # Default start position
        if start_pos is None:
            start_pos = (1, 1)

        # Ensure start position is on odd coordinates for maze algorithm
        sx = start_pos[0] if start_pos[0] % 2 == 1 else start_pos[0] + 1
        sy = start_pos[1] if start_pos[1] % 2 == 1 else start_pos[1] + 1
        sx = max(1, min(sx, size - 2))
        sy = max(1, min(sy, size - 2))

        # Use iterative approach with stack to avoid recursion limit
        def carve_iterative(start_x, start_y):
            stack = [(start_x, start_y)]
            maze[start_y][start_x] = 0

            while stack:
                x, y = stack[-1]
                directions = [(0, -2), (0, 2), (-2, 0), (2, 0)]
                random.shuffle(directions)

                found = False
                for dx, dy in directions:
                    nx, ny = x + dx, y + dy
                    if 0 < nx < size-1 and 0 < ny < size-1 and maze[ny][nx] == 1:
                        # Carve through wall
                        maze[y + dy // 2][x + dx // 2] = 0
                        maze[ny][nx] = 0
                        stack.append((nx, ny))
                        found = True
                        break

                if not found:
                    stack.pop()

        # Start from the specified position
        carve_iterative(sx, sy)

        # Ensure the actual start position is also clear
        maze[start_pos[1]][start_pos[0]] = 0

        # Add some extra paths to create loops and make maze more interesting
        # This removes some walls to create multiple routes
        for _ in range(size // 3):
            x = random.randint(2, size - 3)
            y = random.randint(2, size - 3)
            if maze[y][x] == 1:
                # Check if removing this wall connects two passages
                neighbors = 0
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    if maze[y + dy][x + dx] == 0:
                        neighbors += 1
                if neighbors >= 2:
                    maze[y][x] = 0

        # Define three separate regions for exits (different corners/edges)
        # Easy: top-right area, Medium: bottom-left area, Hard: bottom-right corner
        regions = {
            'easy': (size // 2, size - 4, 4, size // 2 - 4),      # top-right
            'medium': (4, size // 2 - 4, size // 2, size - 4),    # bottom-left
            'hard': (size - 8, size - 4, size - 8, size - 4)      # bottom-right corner
        }

        # Calculate distances from start
        distances = MazeGenerator.calculate_distances(maze, start_pos)

        def find_exit_in_region(min_x, max_x, min_y, max_y, min_dist=0):
            """Find a suitable exit position within a region"""
            candidates = []
            for y in range(min_y, min(max_y + 1, size - 1)):
                for x in range(min_x, min(max_x + 1, size - 1)):
                    if maze[y][x] == 0 and (x, y) != start_pos:
                        dist = distances.get((x, y), 0)
                        if dist >= min_dist:
                            candidates.append((x, y, dist))

            if candidates:
                # Sort by distance and pick one from the further half
                candidates.sort(key=lambda c: c[2])
                idx = len(candidates) * 2 // 3  # Pick from further 1/3
                return candidates[min(idx, len(candidates) - 1)][:2]
            return None

        # Find exits in each region with minimum distance requirements
        easy_exit = find_exit_in_region(*regions['easy'], min_dist=size // 4)
        medium_exit = find_exit_in_region(*regions['medium'], min_dist=size // 2)
        hard_exit = find_exit_in_region(*regions['hard'], min_dist=size)

        # Fallback positions if regions don't have suitable exits
        if not easy_exit:
            easy_exit = (size - 4, 4)
        if not medium_exit:
            medium_exit = (4, size - 4)
        if not hard_exit:
            hard_exit = (size - 2, size - 2)

        # Ensure all exits are passable
        for exit_pos in [easy_exit, medium_exit, hard_exit]:
            maze[exit_pos[1]][exit_pos[0]] = 0
            # Also ensure there's a path to each exit by clearing adjacent if needed
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = exit_pos[0] + dx, exit_pos[1] + dy
                if 0 < nx < size - 1 and 0 < ny < size - 1:
                    if maze[ny][nx] == 0:
                        break
            else:
                # No adjacent passage, create one
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nx, ny = exit_pos[0] + dx, exit_pos[1] + dy
                    if 0 < nx < size - 1 and 0 < ny < size - 1:
                        maze[ny][nx] = 0
                        break

        # Verify all exits are reachable from start position
        distances = MazeGenerator.calculate_distances(maze, start_pos)
        for exit_name, exit_pos in [('easy', easy_exit), ('medium', medium_exit), ('hard', hard_exit)]:
            if exit_pos not in distances:
                # Exit not reachable, carve a path to it
                MazeGenerator._carve_path_to(maze, start_pos, exit_pos)

        exits = {
            'easy': easy_exit,
            'medium': medium_exit,
            'hard': hard_exit
        }

        return maze, exits

    @staticmethod
    def _carve_path_to(maze, start, end):
        """Carve a path from start towards end"""
        x, y = start
        ex, ey = end
        size = len(maze)

        while (x, y) != (ex, ey):
            maze[y][x] = 0
            # Move towards end
            if x < ex and x + 1 < size - 1:
                x += 1
            elif x > ex and x - 1 > 0:
                x -= 1
            elif y < ey and y + 1 < size - 1:
                y += 1
            elif y > ey and y - 1 > 0:
                y -= 1
            else:
                break
        maze[ey][ex] = 0

    @staticmethod
    def calculate_distances(maze, start):
        """BFS to calculate distances from start"""
        distances = {start: 0}
        queue = deque([start])
        size = len(maze)

        while queue:
            x, y = queue.popleft()
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if (0 <= nx < size and 0 <= ny < size and
                    maze[ny][nx] == 0 and (nx, ny) not in distances):
                    distances[(nx, ny)] = distances[(x, y)] + 1
                    queue.append((nx, ny))

        return distances


class HandGestureDetector:
    """Detect hand gestures using MediaPipe Tasks API"""

    # Hand landmark indices
    WRIST = 0
    INDEX_FINGER_MCP = 5
    INDEX_FINGER_TIP = 8
    MIDDLE_FINGER_TIP = 12
    RING_FINGER_TIP = 16
    PINKY_TIP = 20

    def __init__(self):
        self.cap = None
        self.current_direction = Direction.NONE
        self.running = False
        self.frame = None
        self.lock = threading.Lock()
        self.hand_landmarker = None
        self.latest_result = None

        # Download model if needed
        self.model_path = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
        if not os.path.exists(self.model_path):
            print("Downloading hand detection model...")
            model_url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            urllib.request.urlretrieve(model_url, self.model_path)
            print("Model downloaded.")

    def _result_callback(self, result, output_image, timestamp_ms):
        """Callback for async hand detection results"""
        self.latest_result = result

    def start(self):
        """Start the camera capture thread"""
        # Initialize hand landmarker
        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.LIVE_STREAM,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            result_callback=self._result_callback
        )
        self.hand_landmarker = vision.HandLandmarker.create_from_options(options)

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop the camera capture"""
        self.running = False
        if self.cap:
            self.cap.release()
        if self.hand_landmarker:
            self.hand_landmarker.close()

    def _capture_loop(self):
        """Background thread for camera capture and gesture detection"""
        frame_timestamp = 0
        while self.running:
            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    # Flip horizontally for mirror effect
                    frame = cv2.flip(frame, 1)

                    # Convert to RGB for MediaPipe
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # Create MediaPipe Image
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

                    # Process frame asynchronously
                    frame_timestamp += 33  # ~30fps timestamps
                    self.hand_landmarker.detect_async(mp_image, frame_timestamp)

                    direction = Direction.NONE

                    # Check latest results
                    if self.latest_result and self.latest_result.hand_landmarks:
                        landmarks = self.latest_result.hand_landmarks[0]
                        direction = self._detect_pointing_direction(landmarks)

                    with self.lock:
                        self.current_direction = direction
                        self.frame = frame

            time.sleep(0.033)  # ~30 fps

    def _detect_pointing_direction(self, landmarks):
        """Detect pointing direction from hand landmarks"""
        # Get key points (landmarks is a list of NormalizedLandmark)
        wrist = landmarks[self.WRIST]
        index_tip = landmarks[self.INDEX_FINGER_TIP]
        index_mcp = landmarks[self.INDEX_FINGER_MCP]

        # Calculate pointing direction based on index finger relative to wrist
        dx = index_tip.x - wrist.x
        dy = index_tip.y - wrist.y

        # Determine primary direction
        if abs(dx) > abs(dy):
            # Horizontal pointing
            if dx > 0.1:
                return Direction.RIGHT
            elif dx < -0.1:
                return Direction.LEFT
        else:
            # Vertical pointing
            if dy < -0.1:
                return Direction.UP
            elif dy > 0.1:
                return Direction.DOWN

        return Direction.NONE

    def get_direction(self):
        """Get current detected direction"""
        with self.lock:
            return self.current_direction

    def get_frame(self):
        """Get current camera frame"""
        with self.lock:
            return self.frame.copy() if self.frame is not None else None


class Cat:
    """Vector graphic cat character"""

    def __init__(self, x, y, cell_size):
        self.grid_x = x
        self.grid_y = y
        self.cell_size = cell_size
        self.facing = Direction.RIGHT
        self.move_cooldown = 0
        self.animation_frame = 0

    def draw(self, surface, offset_x, offset_y):
        """Draw the cat as a vector graphic"""
        # Calculate pixel position
        px = offset_x + self.grid_x * self.cell_size + self.cell_size // 2
        py = offset_y + self.grid_y * self.cell_size + self.cell_size // 2

        size = self.cell_size * 0.8
        half = size // 2

        # Body (oval)
        body_rect = pygame.Rect(px - half, py - half * 0.6, size, size * 0.7)
        pygame.draw.ellipse(surface, CAT_ORANGE, body_rect)

        # Head (circle)
        head_radius = int(size * 0.35)
        head_x = px
        head_y = py - int(size * 0.2)
        pygame.draw.circle(surface, CAT_ORANGE, (head_x, head_y), head_radius)

        # Ears (triangles)
        ear_size = head_radius * 0.6
        # Left ear
        left_ear = [
            (head_x - head_radius * 0.6, head_y - head_radius * 0.5),
            (head_x - head_radius * 0.9, head_y - head_radius * 1.2),
            (head_x - head_radius * 0.2, head_y - head_radius * 0.9)
        ]
        pygame.draw.polygon(surface, CAT_ORANGE, left_ear)
        pygame.draw.polygon(surface, (255, 180, 180), [
            (left_ear[0][0] + 2, left_ear[0][1] + 2),
            (left_ear[1][0] + 2, left_ear[1][1] + 4),
            (left_ear[2][0] - 2, left_ear[2][1] + 2)
        ])

        # Right ear
        right_ear = [
            (head_x + head_radius * 0.6, head_y - head_radius * 0.5),
            (head_x + head_radius * 0.9, head_y - head_radius * 1.2),
            (head_x + head_radius * 0.2, head_y - head_radius * 0.9)
        ]
        pygame.draw.polygon(surface, CAT_ORANGE, right_ear)
        pygame.draw.polygon(surface, (255, 180, 180), [
            (right_ear[0][0] - 2, right_ear[0][1] + 2),
            (right_ear[1][0] - 2, right_ear[1][1] + 4),
            (right_ear[2][0] + 2, right_ear[2][1] + 2)
        ])

        # Eyes
        eye_radius = int(head_radius * 0.25)
        eye_y = head_y - int(head_radius * 0.1)
        # Left eye
        pygame.draw.circle(surface, WHITE, (head_x - int(head_radius * 0.4), eye_y), eye_radius)
        pygame.draw.circle(surface, BLACK, (head_x - int(head_radius * 0.4), eye_y), eye_radius // 2)
        # Right eye
        pygame.draw.circle(surface, WHITE, (head_x + int(head_radius * 0.4), eye_y), eye_radius)
        pygame.draw.circle(surface, BLACK, (head_x + int(head_radius * 0.4), eye_y), eye_radius // 2)

        # Nose (small triangle)
        nose_y = head_y + int(head_radius * 0.2)
        nose_points = [
            (head_x, nose_y + 3),
            (head_x - 3, nose_y - 2),
            (head_x + 3, nose_y - 2)
        ]
        pygame.draw.polygon(surface, (255, 100, 100), nose_points)

        # Whiskers
        whisker_y = nose_y + 2
        for i in range(3):
            angle = -20 + i * 20
            # Left whiskers
            start = (head_x - head_radius * 0.3, whisker_y + i * 2)
            end = (head_x - head_radius * 1.2, whisker_y - 5 + i * 5)
            pygame.draw.line(surface, DARK_GRAY, start, end, 1)
            # Right whiskers
            start = (head_x + head_radius * 0.3, whisker_y + i * 2)
            end = (head_x + head_radius * 1.2, whisker_y - 5 + i * 5)
            pygame.draw.line(surface, DARK_GRAY, start, end, 1)

        # Tail (curved line)
        tail_points = []
        for i in range(10):
            t = i / 9
            tx = px + half * 0.8 + t * half * 0.8
            ty = py + int(math.sin(t * math.pi + self.animation_frame * 0.2) * 8)
            tail_points.append((tx, ty))

        if len(tail_points) > 1:
            pygame.draw.lines(surface, CAT_DARK, False, tail_points, 3)

        self.animation_frame += 1

    def move(self, direction, maze):
        """Move the cat in the given direction if possible"""
        if self.move_cooldown > 0:
            return False

        dx, dy = 0, 0
        if direction == Direction.UP:
            dy = -1
        elif direction == Direction.DOWN:
            dy = 1
        elif direction == Direction.LEFT:
            dx = -1
        elif direction == Direction.RIGHT:
            dx = 1
        else:
            return False

        new_x = self.grid_x + dx
        new_y = self.grid_y + dy

        # Check bounds and walls
        if (0 <= new_x < len(maze[0]) and 0 <= new_y < len(maze) and
            maze[new_y][new_x] == 0):
            self.grid_x = new_x
            self.grid_y = new_y
            self.facing = direction
            self.move_cooldown = 10  # Frames to wait before next move
            return True

        return False

    def update(self):
        """Update cat state"""
        if self.move_cooldown > 0:
            self.move_cooldown -= 1


class Game:
    """Main game class"""

    def __init__(self):
        # Set up full screen display
        self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        self.screen_width, self.screen_height = self.screen.get_size()
        pygame.display.set_caption("Small Game")

        self.clock = pygame.time.Clock()
        self.state = GameState.TITLE
        self.running = True

        # Calculate maze positioning for center of screen
        self.cell_size = min(
            (self.screen_width - 300) // MAZE_SIZE,
            (self.screen_height - 200) // MAZE_SIZE
        )
        self.maze_offset_x = (self.screen_width - MAZE_SIZE * self.cell_size) // 2
        self.maze_offset_y = (self.screen_height - MAZE_SIZE * self.cell_size) // 2

        # Game state
        self.level = 1
        self.score = 0
        self.maze = None
        self.exits = None
        self.cat = None
        self.countdown = COUNTDOWN_TIME
        self.last_tick = 0
        self.level_score = 0
        self.bonus_awarded = False
        self.keyboard_direction = Direction.NONE

        # Reconfiguration state
        self.reconfigure_timer = 30  # 30 second timer
        self.reconfigure_last_tick = 0
        self.particles = []
        self.old_maze = None
        self.new_maze = None
        self.new_exits = None
        self.reconfigure_progress = 0

        # 3D view settings
        self.view_3d_width = 250
        self.view_3d_height = 180
        self.view_3d_x = 0  # Set after screen init
        self.view_3d_y = 0
        self.cat_facing = Direction.RIGHT

        # Pawprint tracking - stores positions the cat has visited
        self.pawprints = set()

        # Initialize sounds
        self._init_sounds()

        # Initialize gesture detector
        self.gesture_detector = HandGestureDetector()

        # Fonts
        self.title_font = pygame.font.Font(None, 72)
        self.large_font = pygame.font.Font(None, 48)
        self.medium_font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)

        # Set 3D view position (lower right corner)
        self.view_3d_x = self.screen_width - self.view_3d_width - 20
        self.view_3d_y = self.screen_height - self.view_3d_height - 20

    def _init_sounds(self):
        """Initialize all game sounds"""
        self.sounds = {
            'move': SoundGenerator.generate_move_sound(),
            'wall': SoundGenerator.generate_wall_sound(),
            'easy_exit': SoundGenerator.generate_easy_exit_sound(),
            'medium_exit': SoundGenerator.generate_medium_exit_sound(),
            'hard_exit': SoundGenerator.generate_hard_exit_sound(),
            'warning': SoundGenerator.generate_countdown_warning(),
            'game_over': SoundGenerator.generate_game_over_sound(),
            'bonus': SoundGenerator.generate_bonus_sound(),
            'reconfigure': SoundGenerator.generate_reconfigure_sound()
        }

    def _play_sound_sequence(self, sounds, delay=150):
        """Play a sequence of sounds with delay"""
        def play():
            for i, sound in enumerate(sounds):
                pygame.time.wait(delay * i)
                sound.play()

        thread = threading.Thread(target=play, daemon=True)
        thread.start()

    def start_new_level(self):
        """Start a new level"""
        self.maze, self.exits = MazeGenerator.generate(MAZE_SIZE)
        self.cat = Cat(1, 1, self.cell_size)
        self.countdown = COUNTDOWN_TIME
        self.last_tick = time.time()
        self.level_score = 0
        self.bonus_awarded = False
        self.reconfigure_timer = 30
        self.reconfigure_last_tick = time.time()
        self.particles = []
        self.cat_facing = Direction.RIGHT
        self.pawprints = {(1, 1)}  # Start position
        self.state = GameState.PLAYING

    def handle_events(self):
        """Handle pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    if self.state == GameState.TITLE:
                        self.gesture_detector.start()
                        self.start_new_level()
                    elif self.state == GameState.LEVEL_COMPLETE:
                        self.level += 1
                        self.start_new_level()
                    elif self.state == GameState.GAME_OVER:
                        self.level = 1
                        self.score = 0
                        self.start_new_level()
                # Arrow key controls as fallback
                elif self.state == GameState.PLAYING:
                    if event.key == pygame.K_UP:
                        self.keyboard_direction = Direction.UP
                    elif event.key == pygame.K_DOWN:
                        self.keyboard_direction = Direction.DOWN
                    elif event.key == pygame.K_LEFT:
                        self.keyboard_direction = Direction.LEFT
                    elif event.key == pygame.K_RIGHT:
                        self.keyboard_direction = Direction.RIGHT

    def update(self):
        """Update game state"""
        if self.state == GameState.RECONFIGURING:
            self._update_reconfiguration()
            return

        if self.state != GameState.PLAYING:
            return

        # Update countdown
        current_time = time.time()
        if current_time - self.last_tick >= 1:
            self.countdown -= 1
            self.last_tick = current_time

            # Warning sound at low time
            if self.countdown <= 10 and self.countdown > 0:
                self.sounds['warning'].play()

            # Game over if time runs out
            if self.countdown <= 0:
                self._play_sound_sequence(self.sounds['game_over'])
                self.state = GameState.GAME_OVER
                return

        # Update reconfigure timer
        if current_time - self.reconfigure_last_tick >= 1:
            self.reconfigure_timer -= 1
            self.reconfigure_last_tick = current_time

            # Trigger reconfiguration when timer hits 0
            if self.reconfigure_timer <= 0:
                self._start_reconfiguration()
                return

        # Get gesture direction or use keyboard fallback
        direction = self.gesture_detector.get_direction()
        if direction == Direction.NONE and self.keyboard_direction != Direction.NONE:
            direction = self.keyboard_direction
            self.keyboard_direction = Direction.NONE  # Reset after use

        # Move cat and track facing direction
        if direction != Direction.NONE:
            if self.cat.move(direction, self.maze):
                self.sounds['move'].play()
                self.cat_facing = direction  # Update facing direction
                # Track pawprints
                self.pawprints.add((self.cat.grid_x, self.cat.grid_y))
            else:
                # Hit a wall
                if self.cat.move_cooldown == 0:
                    self.sounds['wall'].play()
                    self.cat.move_cooldown = 10

        # Update cat
        self.cat.update()

        # Check for exit reached
        cat_pos = (self.cat.grid_x, self.cat.grid_y)

        if cat_pos == self.exits['easy']:
            self.level_score = 10
            self._play_sound_sequence(self.sounds['easy_exit'])
            self._complete_level()
        elif cat_pos == self.exits['medium']:
            self.level_score = 50
            self._play_sound_sequence(self.sounds['medium_exit'])
            self._complete_level()
        elif cat_pos == self.exits['hard']:
            self.level_score = 100
            self._play_sound_sequence(self.sounds['hard_exit'])
            self._complete_level()

    def _start_reconfiguration(self):
        """Start maze reconfiguration animation"""
        self._play_sound_sequence(self.sounds['reconfigure'], 100)

        # Store current maze
        self.old_maze = [row[:] for row in self.maze]

        # Generate new maze from cat's current position
        cat_pos = (self.cat.grid_x, self.cat.grid_y)
        self.new_maze, self.new_exits = MazeGenerator.generate(MAZE_SIZE, start_pos=cat_pos)

        # Create particles for cells that change
        self.particles = []
        for y in range(MAZE_SIZE):
            for x in range(MAZE_SIZE):
                old_val = self.old_maze[y][x]
                new_val = self.new_maze[y][x]

                # Calculate screen positions
                px = self.maze_offset_x + x * self.cell_size + self.cell_size // 2
                py = self.maze_offset_y + y * self.cell_size + self.cell_size // 2

                if old_val != new_val:
                    # Create particles that swirl from old position to new
                    # Start from random positions around the cell
                    for _ in range(3):
                        start_x = px + random.uniform(-50, 50)
                        start_y = py + random.uniform(-50, 50)
                        color = DARK_GRAY if new_val == 1 else CYAN
                        self.particles.append(Particle(start_x, start_y, px, py, color))

        self.reconfigure_progress = 0
        self.state = GameState.RECONFIGURING

    def _update_reconfiguration(self):
        """Update reconfiguration animation"""
        self.reconfigure_progress += 0.02

        # Update all particles
        all_done = True
        for particle in self.particles:
            if not particle.update():
                all_done = False

        # Check if animation is complete
        if self.reconfigure_progress >= 1.0 or all_done:
            # Apply new maze
            self.maze = self.new_maze
            self.exits = self.new_exits
            self.particles = []
            self.reconfigure_timer = 30
            self.reconfigure_last_tick = time.time()
            self.state = GameState.PLAYING

    def _complete_level(self):
        """Handle level completion"""
        # Add bonus if time remaining
        if self.countdown > 0:
            self.bonus_awarded = True
            self.level_score += 50
            pygame.time.wait(200)
            self._play_sound_sequence(self.sounds['bonus'], 80)

        self.score += self.level_score
        self.state = GameState.LEVEL_COMPLETE

    def draw(self):
        """Draw the game"""
        self.screen.fill(BLACK)

        if self.state == GameState.TITLE:
            self._draw_title_screen()
        elif self.state == GameState.PLAYING:
            self._draw_game()
        elif self.state == GameState.RECONFIGURING:
            self._draw_reconfiguring()
        elif self.state == GameState.LEVEL_COMPLETE:
            self._draw_level_complete()
        elif self.state == GameState.GAME_OVER:
            self._draw_game_over()

        pygame.display.flip()

    def _draw_title_screen(self):
        """Draw the title screen"""
        # Title
        title1 = self.title_font.render("Small Game", True, CYAN)
        title1_rect = title1.get_rect(center=(self.screen_width // 2, self.screen_height // 3))
        self.screen.blit(title1, title1_rect)

        # Subtitle
        subtitle = self.medium_font.render("written by Claude, designed by Daniel Goldenholz 2026", True, WHITE)
        subtitle_rect = subtitle.get_rect(center=(self.screen_width // 2, self.screen_height // 3 + 60))
        self.screen.blit(subtitle, subtitle_rect)

        # Instructions
        instructions = [
            "Use hand gestures or ARROW KEYS to control the cat",
            "Point UP, DOWN, LEFT, or RIGHT to move",
            "Find the exits: EASY (10 pts), MEDIUM (50 pts), HARD (100 pts)",
            "Complete before time runs out for a 50 point bonus!",
            "",
            "Press SPACE to start",
            "Press Q to quit anytime"
        ]

        y = self.screen_height // 2 + 50
        for line in instructions:
            text = self.small_font.render(line, True, GRAY)
            text_rect = text.get_rect(center=(self.screen_width // 2, y))
            self.screen.blit(text, text_rect)
            y += 30

        # Draw a sample cat
        sample_cat = Cat(0, 0, 60)
        sample_cat.draw(self.screen, self.screen_width // 2 - 30, self.screen_height // 2 - 80)

    def _draw_game(self):
        """Draw the main game"""
        # Draw maze
        for y in range(MAZE_SIZE):
            for x in range(MAZE_SIZE):
                rect = pygame.Rect(
                    self.maze_offset_x + x * self.cell_size,
                    self.maze_offset_y + y * self.cell_size,
                    self.cell_size,
                    self.cell_size
                )

                if self.maze[y][x] == 1:
                    pygame.draw.rect(self.screen, DARK_GRAY, rect)
                else:
                    pygame.draw.rect(self.screen, (20, 20, 30), rect)

                pygame.draw.rect(self.screen, GRAY, rect, 1)

        # Draw exits
        for exit_type, pos in self.exits.items():
            rect = pygame.Rect(
                self.maze_offset_x + pos[0] * self.cell_size,
                self.maze_offset_y + pos[1] * self.cell_size,
                self.cell_size,
                self.cell_size
            )

            if exit_type == 'easy':
                color = GREEN
                label = "E"
            elif exit_type == 'medium':
                color = YELLOW
                label = "M"
            else:
                color = RED
                label = "H"

            pygame.draw.rect(self.screen, color, rect)
            text = self.small_font.render(label, True, BLACK)
            text_rect = text.get_rect(center=rect.center)
            self.screen.blit(text, text_rect)

        # Draw cat
        self.cat.draw(self.screen, self.maze_offset_x, self.maze_offset_y)

        # Draw HUD
        self._draw_hud()

        # Draw gesture indicator
        self._draw_gesture_indicator()

        # Draw 3D view
        self._draw_3d_view()

    def _draw_hud(self):
        """Draw the heads-up display"""
        # Level
        level_text = self.medium_font.render(f"Level: {self.level}", True, WHITE)
        self.screen.blit(level_text, (20, 20))

        # Score
        score_text = self.medium_font.render(f"Score: {self.score}", True, WHITE)
        self.screen.blit(score_text, (20, 60))

        # Countdown timer
        timer_color = WHITE if self.countdown > 10 else RED
        timer_text = self.large_font.render(f"Time: {self.countdown}", True, timer_color)
        timer_rect = timer_text.get_rect(midtop=(self.screen_width // 2, 20))
        self.screen.blit(timer_text, timer_rect)

        # Reconfigure timer (below main timer)
        reconfig_color = CYAN if self.reconfigure_timer > 10 else ORANGE
        reconfig_text = self.medium_font.render(f"Reconfigure: {self.reconfigure_timer}s", True, reconfig_color)
        reconfig_rect = reconfig_text.get_rect(midtop=(self.screen_width // 2, 70))
        self.screen.blit(reconfig_text, reconfig_rect)

        # Exit legend
        legend_y = self.screen_height - 100
        legend_items = [
            ("EASY: 10 pts", GREEN),
            ("MEDIUM: 50 pts", YELLOW),
            ("HARD: 100 pts", RED)
        ]

        for i, (text, color) in enumerate(legend_items):
            label = self.small_font.render(text, True, color)
            self.screen.blit(label, (20, legend_y + i * 25))

    def _draw_gesture_indicator(self):
        """Draw the gesture indicator in upper right corner"""
        indicator_x = self.screen_width - 200
        indicator_y = 20

        # Draw camera preview if available
        frame = self.gesture_detector.get_frame()
        if frame is not None:
            # Resize frame for preview
            preview_width = 160
            preview_height = 120
            frame = cv2.resize(frame, (preview_width, preview_height))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = np.rot90(frame)
            frame = pygame.surfarray.make_surface(frame)
            self.screen.blit(frame, (indicator_x, indicator_y))
            indicator_y += preview_height + 10

        # Draw current gesture
        direction = self.gesture_detector.get_direction()
        direction_names = {
            Direction.NONE: "No gesture",
            Direction.UP: "UP",
            Direction.DOWN: "DOWN",
            Direction.LEFT: "LEFT",
            Direction.RIGHT: "RIGHT"
        }

        gesture_text = self.medium_font.render(direction_names[direction], True, CYAN)
        self.screen.blit(gesture_text, (indicator_x, indicator_y))

        # Draw direction arrow
        arrow_size = 40
        arrow_x = indicator_x + 80
        arrow_y = indicator_y + 50

        if direction == Direction.UP:
            points = [(arrow_x, arrow_y - arrow_size),
                      (arrow_x - arrow_size//2, arrow_y),
                      (arrow_x + arrow_size//2, arrow_y)]
            pygame.draw.polygon(self.screen, CYAN, points)
        elif direction == Direction.DOWN:
            points = [(arrow_x, arrow_y + arrow_size),
                      (arrow_x - arrow_size//2, arrow_y),
                      (arrow_x + arrow_size//2, arrow_y)]
            pygame.draw.polygon(self.screen, CYAN, points)
        elif direction == Direction.LEFT:
            points = [(arrow_x - arrow_size, arrow_y),
                      (arrow_x, arrow_y - arrow_size//2),
                      (arrow_x, arrow_y + arrow_size//2)]
            pygame.draw.polygon(self.screen, CYAN, points)
        elif direction == Direction.RIGHT:
            points = [(arrow_x + arrow_size, arrow_y),
                      (arrow_x, arrow_y - arrow_size//2),
                      (arrow_x, arrow_y + arrow_size//2)]
            pygame.draw.polygon(self.screen, CYAN, points)

    def _get_wallpaper_color(self, grid_x, grid_y, wall_x_offset, wall_y, shade):
        """Get wallpaper color based on location with patterns"""
        # Different zones have different wallpaper styles
        zone = ((grid_x // 5) + (grid_y // 5)) % 4

        if zone == 0:
            # Red damask pattern
            pattern = ((int(wall_x_offset * 10) + int(wall_y * 8)) % 3 == 0)
            base_r, base_g, base_b = 120, 40, 50
            if pattern:
                base_r, base_g, base_b = 150, 50, 60
        elif zone == 1:
            # Blue stripes
            stripe = (int(wall_x_offset * 15) % 4 < 2)
            if stripe:
                base_r, base_g, base_b = 40, 60, 120
            else:
                base_r, base_g, base_b = 30, 45, 90
        elif zone == 2:
            # Green floral
            flower = ((int(wall_x_offset * 8) % 3 == 0) and (int(wall_y * 6) % 3 == 0))
            base_r, base_g, base_b = 50, 90, 50
            if flower:
                base_r, base_g, base_b = 80, 130, 70
        else:
            # Purple geometric
            geo = ((int(wall_x_offset * 12) + int(wall_y * 12)) % 5 < 2)
            if geo:
                base_r, base_g, base_b = 80, 50, 100
            else:
                base_r, base_g, base_b = 60, 40, 80

        # Apply distance shading
        factor = shade / 255.0
        return (int(base_r * factor), int(base_g * factor), int(base_b * factor))

    def _get_carpet_color(self, floor_x, floor_y, distance):
        """Get carpet color with texture based on position"""
        # Different carpet zones
        zone = ((int(floor_x) // 4) + (int(floor_y) // 4)) % 5

        # Base colors for different carpet types
        if zone == 0:
            # Rich burgundy carpet
            base = (100, 30, 40)
            pattern_color = (120, 40, 50)
        elif zone == 1:
            # Navy blue carpet
            base = (25, 35, 80)
            pattern_color = (35, 50, 100)
        elif zone == 2:
            # Forest green carpet
            base = (30, 70, 40)
            pattern_color = (40, 90, 50)
        elif zone == 3:
            # Golden carpet
            base = (120, 90, 40)
            pattern_color = (150, 110, 50)
        else:
            # Purple carpet
            base = (70, 40, 90)
            pattern_color = (90, 55, 110)

        # Add texture pattern
        tex_x = int(floor_x * 4) % 3
        tex_y = int(floor_y * 4) % 3
        has_pattern = (tex_x + tex_y) % 2 == 0

        color = pattern_color if has_pattern else base

        # Apply distance fog
        fog_factor = max(0.2, 1.0 - distance * 0.06)
        return tuple(int(c * fog_factor) for c in color)

    def _draw_3d_view(self):
        """Draw an enhanced 3D first-person view of what the cat sees"""
        view_surface = pygame.Surface((self.view_3d_width, self.view_3d_height))

        cat_x = self.cat.grid_x
        cat_y = self.cat.grid_y
        half_height = self.view_3d_height // 2

        # Get player angle based on facing direction
        if self.cat_facing == Direction.UP:
            player_angle = -math.pi / 2
        elif self.cat_facing == Direction.DOWN:
            player_angle = math.pi / 2
        elif self.cat_facing == Direction.LEFT:
            player_angle = math.pi
        else:  # RIGHT
            player_angle = 0

        # Draw ceiling gradient first
        for y in range(half_height):
            # Darker at top, lighter near horizon
            shade = int(20 + (y / half_height) * 40)
            # Add some color variation for a more interesting ceiling
            ceiling_color = (shade // 2, shade // 2, shade)
            pygame.draw.line(view_surface, ceiling_color, (0, y), (self.view_3d_width, y))

        # Draw floor with perspective and carpet texture
        for y in range(half_height, self.view_3d_height):
            # Calculate the distance for this floor row
            row_distance = half_height / (y - half_height + 0.1)

            for x in range(0, self.view_3d_width, 2):
                # Calculate the actual floor position
                ray_angle = player_angle + (x / self.view_3d_width - 0.5) * (math.pi / 3)

                floor_x = cat_x + 0.5 + math.cos(ray_angle) * row_distance
                floor_y = cat_y + 0.5 + math.sin(ray_angle) * row_distance

                grid_fx, grid_fy = int(floor_x), int(floor_y)

                # Check if this floor position has pawprints
                has_pawprint = (grid_fx, grid_fy) in self.pawprints

                # Get carpet color
                color = self._get_carpet_color(floor_x, floor_y, row_distance)

                # Draw pawprint overlay if cat has been here
                if has_pawprint:
                    # Add orange-ish tint for pawprints
                    paw_pattern = ((int(floor_x * 6) % 4 < 2) and (int(floor_y * 6) % 4 < 2))
                    if paw_pattern:
                        color = (min(255, color[0] + 60), min(255, color[1] + 30), color[2])

                pygame.draw.rect(view_surface, color, (x, y, 2, 1))

        # Raycasting for walls
        num_rays = self.view_3d_width
        fov = math.pi / 3

        # Store ray hit data for drawing
        ray_data = []

        for i in range(num_rays):
            ray_angle = player_angle + (i / num_rays - 0.5) * fov

            ray_dx = math.cos(ray_angle)
            ray_dy = math.sin(ray_angle)

            # DDA raycasting for more accurate results
            rx, ry = float(cat_x) + 0.5, float(cat_y) + 0.5
            distance = 0
            max_dist = 20
            hit = False
            hit_exit = None
            hit_x, hit_y = 0, 0
            wall_x_offset = 0

            step = 0.05
            while distance < max_dist and not hit:
                rx += ray_dx * step
                ry += ray_dy * step
                distance += step

                grid_x, grid_y = int(rx), int(ry)

                if 0 <= grid_x < MAZE_SIZE and 0 <= grid_y < MAZE_SIZE:
                    if self.maze[grid_y][grid_x] == 1:
                        hit = True
                        hit_x, hit_y = grid_x, grid_y
                        # Calculate where on the wall we hit (for texture mapping)
                        wall_x_offset = (rx - grid_x) if abs(ray_dx) > abs(ray_dy) else (ry - grid_y)

                    # Check exits
                    for exit_type, pos in self.exits.items():
                        if (grid_x, grid_y) == pos:
                            hit_exit = exit_type
                else:
                    hit = True
                    hit_x, hit_y = grid_x, grid_y

            # Correct fisheye effect
            corrected_distance = distance * math.cos(ray_angle - player_angle)

            ray_data.append({
                'distance': corrected_distance,
                'hit_exit': hit_exit,
                'hit_x': hit_x,
                'hit_y': hit_y,
                'wall_x_offset': wall_x_offset,
                'raw_distance': distance
            })

        # Draw walls with textures
        for i, data in enumerate(ray_data):
            distance = data['distance']
            if distance <= 0:
                distance = 0.1

            # Calculate wall height with perspective
            wall_height = min(self.view_3d_height * 2, int(self.view_3d_height / (distance * 0.4)))

            y_start = half_height - wall_height // 2
            y_end = half_height + wall_height // 2

            # Distance-based shading
            shade = max(30, min(255, int(255 - distance * 12)))

            # Draw wall column with texture
            for y in range(max(0, y_start), min(self.view_3d_height, y_end)):
                # Calculate wall texture Y coordinate
                wall_y = (y - y_start) / max(1, wall_height)

                if data['hit_exit'] == 'easy':
                    # Glowing green exit
                    glow = int(50 * math.sin(time.time() * 5 + i * 0.1))
                    color = (0, min(255, shade + glow), 0)
                elif data['hit_exit'] == 'medium':
                    # Glowing yellow exit
                    glow = int(40 * math.sin(time.time() * 4 + i * 0.1))
                    color = (min(255, shade + glow), min(255, shade + glow), 0)
                elif data['hit_exit'] == 'hard':
                    # Glowing red exit
                    glow = int(60 * math.sin(time.time() * 6 + i * 0.1))
                    color = (min(255, shade + glow), 0, 0)
                else:
                    # Regular wall with wallpaper
                    color = self._get_wallpaper_color(
                        data['hit_x'], data['hit_y'],
                        data['wall_x_offset'], wall_y, shade
                    )

                view_surface.set_at((i, y), color)

            # Draw wall trim/molding at top and bottom
            if y_start >= 0 and y_start < self.view_3d_height:
                trim_shade = min(255, shade + 30)
                pygame.draw.line(view_surface, (trim_shade, trim_shade // 2, 0),
                               (i, max(0, y_start)), (i, max(0, y_start + 2)))
            if y_end < self.view_3d_height and y_end > 0:
                trim_shade = max(0, shade - 20)
                pygame.draw.line(view_surface, (trim_shade // 2, trim_shade // 3, 0),
                               (i, min(self.view_3d_height - 1, y_end - 2)),
                               (i, min(self.view_3d_height - 1, y_end)))

        # Draw frame/border
        pygame.draw.rect(view_surface, (139, 90, 43), (0, 0, self.view_3d_width, self.view_3d_height), 3)
        pygame.draw.rect(view_surface, (101, 67, 33), (2, 2, self.view_3d_width - 4, self.view_3d_height - 4), 1)

        # Blit to screen
        self.screen.blit(view_surface, (self.view_3d_x, self.view_3d_y))

        # Label with decorative style
        label = self.small_font.render("Cat's Eye View", True, (139, 90, 43))
        self.screen.blit(label, (self.view_3d_x + 5, self.view_3d_y - 22))

        # Show pawprint indicator
        pawprint_text = self.small_font.render(f"Explored: {len(self.pawprints)} tiles", True, CAT_ORANGE)
        self.screen.blit(pawprint_text, (self.view_3d_x, self.view_3d_y + self.view_3d_height + 5))

    def _draw_reconfiguring(self):
        """Draw the reconfiguration animation"""
        # Draw the transitioning maze (blend between old and new)
        for y in range(MAZE_SIZE):
            for x in range(MAZE_SIZE):
                rect = pygame.Rect(
                    self.maze_offset_x + x * self.cell_size,
                    self.maze_offset_y + y * self.cell_size,
                    self.cell_size,
                    self.cell_size
                )

                # Blend colors based on progress
                old_val = self.old_maze[y][x] if self.old_maze else 0
                new_val = self.new_maze[y][x] if self.new_maze else 0

                if old_val == new_val:
                    # No change
                    if old_val == 1:
                        pygame.draw.rect(self.screen, DARK_GRAY, rect)
                    else:
                        pygame.draw.rect(self.screen, (20, 20, 30), rect)
                else:
                    # Transitioning cell - make it glow
                    t = self.reconfigure_progress
                    if new_val == 1:
                        # Becoming wall
                        color = (int(50 * t), int(50 * t), int(50 * t))
                    else:
                        # Becoming passage
                        glow = int(100 * (1 - t))
                        color = (20 + glow, 20 + glow, 30 + glow * 2)
                    pygame.draw.rect(self.screen, color, rect)

                pygame.draw.rect(self.screen, GRAY, rect, 1)

        # Draw particles
        for particle in self.particles:
            particle.draw(self.screen)

        # Draw cat
        self.cat.draw(self.screen, self.maze_offset_x, self.maze_offset_y)

        # Draw "Reconfiguring" text
        text = self.large_font.render("MAZE RECONFIGURING...", True, CYAN)
        text_rect = text.get_rect(center=(self.screen_width // 2, 50))
        self.screen.blit(text, text_rect)

        # Draw progress bar
        bar_width = 400
        bar_height = 20
        bar_x = (self.screen_width - bar_width) // 2
        bar_y = 90
        pygame.draw.rect(self.screen, GRAY, (bar_x, bar_y, bar_width, bar_height), 2)
        fill_width = int(bar_width * self.reconfigure_progress)
        pygame.draw.rect(self.screen, CYAN, (bar_x, bar_y, fill_width, bar_height))

    def _draw_level_complete(self):
        """Draw level complete screen"""
        # Title
        title = self.title_font.render("Level Complete!", True, GREEN)
        title_rect = title.get_rect(center=(self.screen_width // 2, self.screen_height // 3))
        self.screen.blit(title, title_rect)

        # Score breakdown
        y = self.screen_height // 2 - 50

        exit_score = self.level_score - (50 if self.bonus_awarded else 0)
        score_text = self.medium_font.render(f"Exit Points: {exit_score}", True, WHITE)
        score_rect = score_text.get_rect(center=(self.screen_width // 2, y))
        self.screen.blit(score_text, score_rect)

        if self.bonus_awarded:
            y += 40
            bonus_text = self.medium_font.render("Time Bonus: +50", True, YELLOW)
            bonus_rect = bonus_text.get_rect(center=(self.screen_width // 2, y))
            self.screen.blit(bonus_text, bonus_rect)

        y += 60
        total_text = self.large_font.render(f"Level Score: {self.level_score}", True, CYAN)
        total_rect = total_text.get_rect(center=(self.screen_width // 2, y))
        self.screen.blit(total_text, total_rect)

        y += 50
        final_text = self.large_font.render(f"Total Score: {self.score}", True, WHITE)
        final_rect = final_text.get_rect(center=(self.screen_width // 2, y))
        self.screen.blit(final_text, final_rect)

        y += 80
        continue_text = self.medium_font.render("Press SPACE for next level", True, GRAY)
        continue_rect = continue_text.get_rect(center=(self.screen_width // 2, y))
        self.screen.blit(continue_text, continue_rect)

    def _draw_game_over(self):
        """Draw game over screen"""
        title = self.title_font.render("Time's Up!", True, RED)
        title_rect = title.get_rect(center=(self.screen_width // 2, self.screen_height // 3))
        self.screen.blit(title, title_rect)

        score_text = self.large_font.render(f"Final Score: {self.score}", True, WHITE)
        score_rect = score_text.get_rect(center=(self.screen_width // 2, self.screen_height // 2))
        self.screen.blit(score_text, score_rect)

        level_text = self.medium_font.render(f"Reached Level: {self.level}", True, GRAY)
        level_rect = level_text.get_rect(center=(self.screen_width // 2, self.screen_height // 2 + 50))
        self.screen.blit(level_text, level_rect)

        restart_text = self.medium_font.render("Press SPACE to restart", True, GRAY)
        restart_rect = restart_text.get_rect(center=(self.screen_width // 2, self.screen_height // 2 + 120))
        self.screen.blit(restart_text, restart_rect)

    def run(self):
        """Main game loop"""
        try:
            while self.running:
                self.handle_events()
                self.update()
                self.draw()
                self.clock.tick(FPS)
        finally:
            self.gesture_detector.stop()
            pygame.quit()


def main():
    """Entry point"""
    game = Game()
    game.run()


if __name__ == "__main__":
    main()
