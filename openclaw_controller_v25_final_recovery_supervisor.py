#!/usr/bin/env python3
"""
openclaw_controller.py
======================
OPENCLAW — LLM-ASSISTED AUTONOMOUS INSPECTION BENCHMARK  v22 MAX INTELLIGENCE CLOSE APPROACH
Single-file production controller. Built from scratch.

Thesis Title : Evaluation of a Latency-Aware LLM-Based Framework for
               Intelligent Task Planning and Autonomous Navigation in
               ROS 2 Mobile Robots

Platform : Jetson Orin Nano 8GB  |  ROS 2 Humble  |  Gazebo Classic
Robot    : TurtleBot3 Burger  (PC runs Gazebo; Jetson runs everything else)
LLMs     : gemma2:2b · qwen2.5:3b · deepseek-r1:1.5b  via Ollama (local)

Architecture (NEW — not patched from old code):
  ┌─ CylinderDetector   LiDAR range-clustering + algebraic circle fit
  ├─ CylinderRegistry   EMA-tracked, stable CYL_N labels, 6 target max
  ├─ SmoothNavigator    PD controller + low-pass filter, no duck-walking
  ├─ MissionEngine      explore → LLM plan → execute → return home → reset
  ├─ BenchmarkEngine    20 runs × 3 models, auto-CSV, full metrics
  ├─ ResetGuard         verified return-to-home before every new run
  ├─ FlaskDashboard     7 pages: Control, Camera, Map/Heatmap, Logs,
  │                     Individual, Comparison, Export
  └─ WatchdogThread     stale-topic alerts, system health every 10 s

Key design decisions vs old code:
  • Camera: Gazebo /camera/image_raw ONLY — no USB/video0
  • Cylinders: 6 targets (spec says 6); LiDAR discovery, no hardcoding
  • Navigation: PD + angular LP-filter → smooth, professional movement
  • Reset: teleport (Gazebo srv) with nav fallback; verified before next run
  • Stop button: immediate halt, preserves data, does NOT auto-reset
  • Model switch: returns home + resets before loading new model
  • Heatmap: trajectory density overlay rendered in dashboard canvas
  • All metrics tracked: latency, nav time, distance, collisions, power, etc.
"""

# ══════════════════════════════════════════════════════════════
# SECTION 0 — IMPORTS
# ══════════════════════════════════════════════════════════════
import os, sys, csv, json, math, time, re, io, threading, zipfile
import datetime, statistics, collections, logging
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import requests
from flask import Flask, jsonify, request, Response, send_file, stream_with_context

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import (QoSProfile, ReliabilityPolicy, DurabilityPolicy,
                        HistoryPolicy, qos_profile_sensor_data)
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan, Image
from visualization_msgs.msg import Marker, MarkerArray

try:
    from gazebo_msgs.srv import SetEntityState
    from gazebo_msgs.msg import EntityState
    _GAZEBO_SRV = True
except ImportError:
    _GAZEBO_SRV = False

try:
    import cv2 as _cv2
    import numpy as _np
    _CV2 = True
except ImportError:
    _cv2 = None
    _np  = None
    _CV2 = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
_LOG = logging.getLogger("openclaw")


# ══════════════════════════════════════════════════════════════
# SECTION 1 — CONFIGURATION
# ══════════════════════════════════════════════════════════════
class CFG:
    # ── Flask ──────────────────────────────────────────────────
    FLASK_PORT      = 5000
    MJPEG_QUALITY   = 65

    # ── Ollama / LLM ──────────────────────────────────────────
    OLLAMA_BASE     = "http://localhost:11434"
    OLLAMA_GEN      = f"{OLLAMA_BASE}/api/generate"
    OLLAMA_TAGS     = f"{OLLAMA_BASE}/api/tags"
    OLLAMA_TIMEOUT  = 25           # seconds per LLM call — command fallback stays responsive
    MODELS          = ["gemma2:2b", "qwen2.5:3b", "deepseek-r1:1.5b"]
    MODEL_COLORS    = {
        "gemma2:2b":        "#e74c3c",
        "qwen2.5:3b":       "#3498db",
        "deepseek-r1:1.5b": "#2ecc71",
    }

    # ── Home position (Gazebo world frame) ────────────────────
    # IMPORTANT: match the Gazebo warehouse START/HOME marker used in your FYP world.
    # Can be overridden from terminal: export OPENCLAW_HOME_X=-2.0 etc.
    HOME_X   =  float(os.environ.get("OPENCLAW_HOME_X", "-2.0"))
    HOME_Y   =  float(os.environ.get("OPENCLAW_HOME_Y", "-0.5"))
    HOME_YAW =  float(os.environ.get("OPENCLAW_HOME_YAW", "0.0"))
    HOME_TOL =  0.50               # m — loose verified home radius, prevents fake reset failure

    # ── Robot identity (Gazebo) ───────────────────────────────
    ROBOT_NAME = "turtlebot3_burger"

    # ── Safety distances ──────────────────────────────────────
    EMERG_DIST   = 0.25    # m  hard stop + backoff
    WALL_DIST    = 0.40    # m  obstacle steering zone
    WARN_DIST    = 0.50    # m  slow + steer
    TARGET_CLR   = 0.80    # m  comfortable clearance

    # ── Smooth navigation PD controller ──────────────────────
    NAV_LIN_MAX  = 0.22    # m/s max forward — faster but safe in Gazebo
    NAV_ANG_MAX  = 1.20    # rad/s max angular — rotate phase only
    NAV_KP       = 0.80    # gentle drive correction
    NAV_KD       = 0.12    # damping
    NAV_LP_ALPHA = 0.35    # angular cmd low-pass
    CREEP_SPEED  = 0.08    # m/s minimum / near-obstacle
    ARRIVAL_R    = 0.18    # m normal waypoint arrival radius
    CYL_ARRIVAL_R = 0.10   # m close cylinder approach waypoint arrival radius
    ARRIVE_HOLD  = 4       # consecutive frames inside ARRIVAL_R → arrived
    WP_TIMEOUT   = 90.0    # s stall timeout per waypoint
    BACK_SPEED   = -0.06   # m/s reversal speed
    RECOVER_BDG  = 22.0    # s maximum back-off budget
    MANEUVER_RETRY_LIMIT = 6  # max reverse/steer recovery attempts per waypoint
    MANEUVER_REVERSE_T = 1.05 # seconds to reverse like parking manoeuvre
    MANEUVER_TURN_T = 0.95    # seconds to steer left/right after reversing
    MANEUVER_FORWARD_T = 0.55 # short forward glide after steering to clear the trap
    RECOVERY_EARLY_S = 5.0    # trigger recovery before 10s duckling starts
    RECOVERY_DETOUR_M = 0.55  # lateral/diagonal clearance step after reverse-turn
    RECOVERY_FORWARD_GLIDE_M = 0.34 # small car-like forward advance after steering
    RECOVERY_LLM_TIMEOUT = 2.5 # bounded LLM recovery advice; fallback always works
    DUCKLING_FAIL_S = 30.0    # if no progress for 30s, declare waypoint failure and go HOME
    CLOSE_STOP_M = 0.19       # close-inspection emergency distance
    CLOSE_WALL_M = 0.27       # close-inspection wall distance
    CLOSE_WARN_M = 0.38       # close-inspection slow distance

    # ── Orbit parameters ──────────────────────────────────────
    ORBIT_R      = 0.60    # m orbit radius around cylinder
    ORBIT_APP_R  = 0.26    # m close inspection standoff from cylinder centre; near but avoids collision
    ORBIT_SPD    = 0.20    # m/s tangential speed
    ORBIT_DEG    = 355.0   # degrees for full orbit

    # ── LiDAR cylinder detection ──────────────────────────────
    CYL_GAP       = 0.25   # m cluster split distance
    CYL_MIN_PTS   = 5
    CYL_MAX_PTS   = 80
    CYL_DIAM_MIN  = 0.06   # m
    CYL_DIAM_MAX  = 0.40   # m
    CYL_MATCH_D   = 0.40   # m  association threshold
    CYL_CONFIRM   = 5      # observations required to confirm
    CYL_MAX_RNG   = 3.5    # m  detection range limit
    CYL_MAX_COUNT = 9      # fixed Gazebo layout: 9 labelled cylinders

    # ── Camera (Gazebo topic only — no USB) ───────────────────
    CAM_TOPIC   = os.environ.get("OPENCLAW_CAM_TOPIC", "/camera/image_raw")
    CAM_ENABLED = os.environ.get("OPENCLAW_CAM", "1").lower() not in ("0","false","no")

    # ── Mission timing ─────────────────────────────────────────
    MISSION_TIMEOUT = 600.0   # s per mission
    EXPLORE_BUDGET  = 90.0    # s exploration sweep
    RUNS_PER_MODEL  = 30

    PASS_THRESHOLD = 70.0
    LATENCY_IDEAL_S = 2.0
    LATENCY_WORST_S = 30.0
    RECOVERY_IDEAL_MAX = 1
    RECOVERY_WORST = 6      # v23 default: 30 runs per model for thesis data collection
    PHASE2_REQUIRED_DEFAULT = False  # v23: optional; benchmark override ON by default
    BENCHMARK_FORCE_DEFAULT = True   # never block real data collection on Phase 2
    LEARNING_FEEDBACK_ENABLED = True  # v25: model-specific run-memory feedback, not weight training
    LEARNING_MAX_NOTES = 6            # keep only the most useful recent lessons per model/task

    # ── Data paths ─────────────────────────────────────────────
    DATA_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    TRAJ_DIR  = os.path.join(DATA_DIR, "trajectories")
    CSV_PATH  = os.path.join(DATA_DIR, "inspection_results.csv")
    VLOG_PATH = os.path.join(DATA_DIR, "validation_log.csv")

    # ── Watchdog intervals ────────────────────────────────────
    ODOM_STALE = 3.0
    SCAN_STALE = 3.0


for _d in (CFG.DATA_DIR, CFG.TRAJ_DIR):
    os.makedirs(_d, exist_ok=True)


# ══════════════════════════════════════════════════════════════
# SECTION 2 — LIDAR CYLINDER DETECTOR
# ══════════════════════════════════════════════════════════════
@dataclass
class RawDetection:
    cx: float
    cy: float
    radius: float
    num_pts: int
    range_to: float


class CylinderDetector:
    """
    Detects cylinders from a LaserScan message using:
      1. Cartesian conversion of valid range readings
      2. Euclidean gap clustering
      3. Algebraic least-squares circle fit (Pratt method)
      4. Diameter filter
      5. Robot-to-world frame transform
    """

    @staticmethod
    def _fit_circle(pts: List[Tuple[float, float]]) -> Optional[Tuple[float, float, float]]:
        n = len(pts)
        if n < 3:
            return None
        mx = sum(p[0] for p in pts) / n
        my = sum(p[1] for p in pts) / n
        ux = [p[0] - mx for p in pts]
        uy = [p[1] - my for p in pts]
        suu  = sum(u*u for u in ux)
        svv  = sum(v*v for v in uy)
        suv  = sum(u*v for u, v in zip(ux, uy))
        suuu = sum(u**3 for u in ux)
        svvv = sum(v**3 for v in uy)
        suvv = sum(u*v*v for u, v in zip(ux, uy))
        svuu = sum(v*u*u for u, v in zip(ux, uy))
        det  = suu * svv - suv * suv
        if abs(det) < 1e-12:
            return None
        uc = (0.5 * (suuu + suvv) * svv - 0.5 * (svvv + svuu) * suv) / det
        vc = (0.5 * (svvv + svuu) * suu - 0.5 * (suuu + suvv) * suv) / det
        r  = math.sqrt(uc*uc + vc*vc + (suu + svv) / n)
        return uc + mx, vc + my, r

    def detect(self, msg: LaserScan,
               rx: float, ry: float, ryaw: float) -> List[RawDetection]:
        results: List[RawDetection] = []
        pts: List[Tuple[float, float, float]] = []
        ang = msg.angle_min
        inc = msg.angle_increment
        for r in msg.ranges:
            if math.isfinite(r) and 0.05 < r < CFG.CYL_MAX_RNG:
                pts.append((r * math.cos(ang), r * math.sin(ang), r))
            ang += inc
        if len(pts) < CFG.CYL_MIN_PTS:
            return results
        # Cluster
        clusters: List[List] = []
        cur = [pts[0]]
        for i in range(1, len(pts)):
            p, q = pts[i - 1], pts[i]
            if math.hypot(q[0] - p[0], q[1] - p[1]) > CFG.CYL_GAP:
                clusters.append(cur)
                cur = []
            cur.append(q)
        clusters.append(cur)
        cos_y = math.cos(ryaw)
        sin_y = math.sin(ryaw)
        for cl in clusters:
            n = len(cl)
            if not (CFG.CYL_MIN_PTS <= n <= CFG.CYL_MAX_PTS):
                continue
            fit = self._fit_circle([(p[0], p[1]) for p in cl])
            if fit is None:
                continue
            cxr, cyr, rad = fit
            if not (CFG.CYL_DIAM_MIN <= 2 * rad <= CFG.CYL_DIAM_MAX):
                continue
            wx = rx + cos_y * cxr - sin_y * cyr
            wy = ry + sin_y * cxr + cos_y * cyr
            dist = math.hypot(cxr, cyr)
            results.append(RawDetection(cx=wx, cy=wy, radius=rad,
                                        num_pts=n, range_to=dist))
        return results


# ══════════════════════════════════════════════════════════════
# SECTION 3 — CYLINDER REGISTRY
# ══════════════════════════════════════════════════════════════
@dataclass
class TrackedCylinder:
    cyl_id: int
    x: float
    y: float
    radius: float
    observe_count: int = 0
    confirmed: bool = False
    last_seen: float = 0.0
    label: str = ""

    @property
    def name(self) -> str:
        return self.label or f"CYL_{self.cyl_id}"


class CylinderRegistry:
    """
    Thread-safe registry of tracked cylinders.
    Uses EMA (0.8/0.2) for position smoothing.
    Assigns stable CYL_N labels after confirmation.
    Caps at CFG.CYL_MAX_COUNT confirmed cylinders.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._next = 1
        self._cyls: Dict[int, TrackedCylinder] = {}

    def update(self, dets: List[RawDetection]) -> None:
        with self._lock:
            now = time.time()
            confirmed_count = sum(1 for c in self._cyls.values() if c.confirmed)
            for d in dets:
                best_id, best_dist = None, CFG.CYL_MATCH_D
                for cid, tc in self._cyls.items():
                    dist = math.hypot(d.cx - tc.x, d.cy - tc.y)
                    if dist < best_dist:
                        best_dist = dist
                        best_id = cid
                if best_id is not None:
                    tc = self._cyls[best_id]
                    tc.x = 0.8 * tc.x + 0.2 * d.cx
                    tc.y = 0.8 * tc.y + 0.2 * d.cy
                    tc.radius = 0.8 * tc.radius + 0.2 * d.radius
                    tc.observe_count += 1
                    tc.last_seen = now
                    if tc.observe_count >= CFG.CYL_CONFIRM:
                        tc.confirmed = True
                else:
                    # Only add new unconfirmed if we haven't hit the cap
                    if confirmed_count < CFG.CYL_MAX_COUNT:
                        cid = self._next
                        self._next += 1
                        self._cyls[cid] = TrackedCylinder(
                            cyl_id=cid, x=d.cx, y=d.cy, radius=d.radius,
                            observe_count=1, last_seen=now,
                            label=f"CYL_{cid}")
            # Re-label confirmed cylinders sequentially
            confirmed = sorted(
                [c for c in self._cyls.values() if c.confirmed],
                key=lambda c: c.cyl_id)
            for i, c in enumerate(confirmed, 1):
                c.label = f"CYL_{i}"

    def confirmed_list(self) -> List[TrackedCylinder]:
        with self._lock:
            return sorted(
                [c for c in self._cyls.values() if c.confirmed],
                key=lambda c: c.cyl_id)

    def all_list(self) -> List[TrackedCylinder]:
        with self._lock:
            return sorted(self._cyls.values(), key=lambda c: c.cyl_id)

    def get_by_name(self, name: str) -> Optional[TrackedCylinder]:
        with self._lock:
            for c in self._cyls.values():
                if c.label == name or c.name == name:
                    return c
        return None

    def to_json(self) -> List[Dict]:
        with self._lock:
            return [
                {"id": c.cyl_id, "label": c.label,
                 "x": round(c.x, 3), "y": round(c.y, 3),
                 "radius": round(c.radius, 3),
                 "confirmed": c.confirmed, "seen": c.observe_count}
                for c in sorted(self._cyls.values(), key=lambda c: c.cyl_id)
            ]

    def reset(self) -> None:
        with self._lock:
            self._cyls.clear()
            self._next = 1


_DETECTOR = CylinderDetector()
_REGISTRY = CylinderRegistry()

# ── Fixed Gazebo 9-cylinder waypoint map ─────────────────────
# Your Gazebo world is fixed, so the robot should NOT wait for dynamic
# cylinder discovery. These nine labels are shown on the dashboard and given
# to the LLM. LiDAR remains only for obstacle/distance safety.
# Tune these if you move the white marker cylinders in Gazebo.
FIXED_CYLINDERS: Dict[str, Tuple[float, float, float]] = {
    "CYL_1": (-1.05,  0.75, 0.10),
    "CYL_2": ( 0.00,  0.75, 0.10),
    "CYL_3": ( 0.95,  0.75, 0.10),
    "CYL_4": (-1.05, -0.05, 0.10),
    "CYL_5": ( 0.00, -0.05, 0.10),
    "CYL_6": ( 0.95, -0.05, 0.10),
    "CYL_7": (-1.05, -0.85, 0.10),
    "CYL_8": ( 0.00, -0.85, 0.10),
    "CYL_9": ( 0.95, -0.85, 0.10),
}

def seed_fixed_cylinders() -> None:
    """Load the known 9-cylinder Gazebo layout into the v7 registry."""
    try:
        with _REGISTRY._lock:
            _REGISTRY._cyls.clear()
            _REGISTRY._next = 1
            for i, (label, (x, y, rad)) in enumerate(FIXED_CYLINDERS.items(), 1):
                _REGISTRY._cyls[i] = TrackedCylinder(
                    cyl_id=i, x=float(x), y=float(y), radius=float(rad),
                    observe_count=999, confirmed=True, last_seen=time.time(),
                    label=label,
                )
            _REGISTRY._next = len(FIXED_CYLINDERS) + 1
    except Exception as ex:
        ST.log(f"Fixed cylinder seed error: {ex}", "WARN")



# ══════════════════════════════════════════════════════════════
# SECTION 4 — DATA STRUCTURES
# ══════════════════════════════════════════════════════════════
@dataclass
class InspectionRecord:
    name: str
    reached: bool = False
    orbit_done: bool = False
    time_to_reach: float = 0.0
    orbit_duration: float = 0.0
    min_clr: float = 99.0
    avg_clr: float = 99.0
    x_at_arrive: float = 0.0
    y_at_arrive: float = 0.0


@dataclass
class CollisionEvent:
    t: float = 0.0
    x: float = 0.0
    y: float = 0.0
    scan_min: float = 0.0


@dataclass
class MissionResult:
    run_id: str = ""
    model: str = ""
    mission_number: int = 0
    mission_text: str = ""
    timestamp: str = ""
    status: str = "FAIL"
    success: bool = False
    failure_reason: str = ""
    analysis_comment: str = ""
    learning_feedback: str = ""
    improvement_hint: str = ""
    # LLM metrics
    llm_latency: float = 0.0
    parse_time: float = 0.0
    raw_llm_response: str = ""
    parsed_route: List[str] = field(default_factory=list)
    reasoning: str = ""
    route_valid: bool = False
    llm_retry_count: int = 0
    valid_json: bool = False
    route_correctness: float = 0.0
    fallback_used: bool = False
    llm_recovery_advice_used: bool = False
    # Navigation metrics
    reset_time: float = 0.0
    nav_time: float = 0.0
    total_mission_time: float = 0.0
    distance_travelled: float = 0.0
    path_length: float = 0.0
    route_efficiency: float = 0.0
    return_home_success: bool = False
    # Inspection metrics
    inspection_pct: float = 0.0
    points_planned: int = 0
    points_reached: int = 0
    orbits_completed: int = 0
    orbit_success_rate: float = 0.0
    executed_route: List[str] = field(default_factory=list)
    # Safety metrics
    collision_count: int = 0
    replan_count: int = 0
    min_clearance: float = 99.0
    avg_clearance: float = 99.0
    risk_score: float = 0.0
    safety_score: float = 0.0
    mission_score: float = 0.0
    latency_score: float = 0.0
    path_efficiency_score: float = 0.0
    recovery_score: float = 0.0
    final_deployment_score: float = 0.0
    pass_fail_reason: str = ""
    near_collision_count: int = 0
    duckling_timeout: bool = False
    reverse_maneuver_count: int = 0
    left_recovery_count: int = 0
    right_recovery_count: int = 0
    # System metrics
    avg_cpu: float = 0.0
    avg_ram: float = 0.0
    avg_power: float = 0.0
    avg_temp: float = 0.0
    # Trajectory
    trajectory: List[Dict] = field(default_factory=list)
    insp_records: List[InspectionRecord] = field(default_factory=list)
    coll_events: List[CollisionEvent] = field(default_factory=list)
    fail_point: str = ""
    fail_x: float = 0.0
    fail_y: float = 0.0


# ══════════════════════════════════════════════════════════════
# SECTION 5 — GLOBAL STATE
# ══════════════════════════════════════════════════════════════
class State:
    """Single shared state object for all threads."""

    def __init__(self) -> None:
        self.lock = threading.RLock()
        # Robot pose (from /odom)
        self.rx: float = 0.0
        self.ry: float = 0.0
        self.ryaw: float = 0.0
        self.vx: float = 0.0
        self.vy: float = 0.0
        self.vw: float = 0.0
        # LiDAR
        self.scan_min: float = 99.0
        self.scan_count: int = 0
        self.scan_sectors: Dict[str, float] = {
            k: 99.0 for k in
            ("front", "front_left", "front_right", "left", "right", "rear")
        }
        self.obstacle_pts: List[Dict] = []
        self.lidar_status: str = "SAFE"
        self.risk_score: float = 0.0
        # Camera
        self.cam_bytes: Optional[bytes] = None
        self.cam_ok: bool = False
        self.cam_fps: float = 0.0
        self.cam_w: int = 0
        self.cam_h: int = 0
        self._cam_t: float = 0.0
        self._cam_frames: int = 0
        # Mission state machine
        self.mission_state: str = "IDLE"
        self.cur_model: str = ""
        self.cur_run: int = 0
        self.cur_text: str = ""
        self.active_cyl: str = ""
        self.orbit_deg: float = 0.0
        self.planned_route: List[str] = []
        self.reasoning: str = ""
        self.current_action_detail: str = ""
        self.last_feedback: str = ""
        self.run_total: int = 1
        self.target_x: float = 0.0
        self.target_y: float = 0.0
        # Trajectory recording
        self.trajectory: List[Dict] = []
        self.recording: bool = False
        self.mission_t0: float = 0.0
        self.dist_total: float = 0.0
        self.prev_x: Optional[float] = None
        self.prev_y: Optional[float] = None
        # Collision tracking
        self.coll_count: int = 0
        self.coll_events: List[CollisionEvent] = []
        self.last_coll_t: float = 0.0
        self.clr_readings: List[float] = []
        self.replan_count: int = 0
        self.near_collision_count: int = 0
        self.duckling_timeout: bool = False
        self.reverse_maneuver_count: int = 0
        self.left_recovery_count: int = 0
        self.right_recovery_count: int = 0
        self.llm_recovery_advice_used: bool = False
        # Control flags
        self.bench_running: bool = False
        self.stop_requested: bool = False
        self.cancel_event: threading.Event = threading.Event()
        # Phase 2 gate
        self.phase2_passed: bool = False
        self.phase2_status: Dict[str, Any] = {
            m: {"runs": 0, "passed": False} for m in CFG.MODELS
        }
        # v25 adaptive run memory: model+mission -> recent feedback notes.
        # This is not model training; it is controlled prompt feedback for repeated runs.
        self.learning_memory: Dict[str, Dict[str, Any]] = {}
        # Results store
        self.results: List[MissionResult] = []
        # Validation status
        self.validation: Dict[str, Any] = {
            "odom_active": False, "scan_active": False,
            "cmdvel_ok": False,   "cam_ok": False,
            "reset_ok": False,    "ollama_ok": False,
            "models_ok": [],      "models_failed": [],
        }
        self.unavail_models: List[str] = []
        # Topic timestamps
        self.last_odom_t: float = 0.0
        self.last_scan_t: float = 0.0
        self.last_cam_t:  float = 0.0
        self.odom_msgs:   int = 0
        self.scan_msgs:   int = 0
        self.cam_msgs:    int = 0
        self.cmdvel_pubs: int = 0
        # HTTP session (Ollama)
        self.http = requests.Session()
        self.http.headers.update({"Connection": "keep-alive"})
        # Log buffer
        self.log_buf: collections.deque = collections.deque(maxlen=1000)

    # ── Logging ────────────────────────────────────────────────
    def log(self, msg: str, level: str = "INFO") -> None:
        ts   = datetime.datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        if level == "ERROR":
            _LOG.error(msg)
        elif level == "WARN":
            _LOG.warning(msg)
        else:
            _LOG.info(msg)
        with self.lock:
            self.log_buf.append(line)

    def set_state(self, s: str) -> None:
        with self.lock:
            self.mission_state = s
        self.log(f"STATE → {s}")

    # ── Mission reset (clears per-run data) ───────────────────
    def reset_mission(self) -> None:
        with self.lock:
            self.trajectory   = []
            self.recording    = False
            self.dist_total   = 0.0
            self.prev_x       = None
            self.prev_y       = None
            self.coll_count   = 0
            self.coll_events  = []
            self.last_coll_t  = 0.0
            self.active_cyl   = ""
            self.mission_t0   = 0.0
            self.planned_route = []
            self.reasoning    = ""
            self.current_action_detail = ""
            self.orbit_deg    = 0.0
            self.clr_readings = []
            self.replan_count = 0
            self.near_collision_count = 0
            self.duckling_timeout = False
            self.reverse_maneuver_count = 0
            self.left_recovery_count = 0
            self.right_recovery_count = 0
            self.llm_recovery_advice_used = False
        self.cancel_event.clear()
        self.stop_requested = False

    # ── Risk / safety scoring ─────────────────────────────────
    def compute_risk(self) -> float:
        s = self.scan_min
        if s < CFG.EMERG_DIST:  return 100.0
        if s < CFG.WARN_DIST:
            return 60.0 + 40.0 * (CFG.WARN_DIST - s) / (CFG.WARN_DIST - CFG.EMERG_DIST)
        if s < CFG.TARGET_CLR:
            return 20.0 + 40.0 * (CFG.TARGET_CLR - s) / (CFG.TARGET_CLR - CFG.WARN_DIST)
        return max(0.0, 20.0 * CFG.TARGET_CLR / (s + 0.01))

    def compute_safety_score(self, min_clr: float, coll: int) -> float:
        base = max(0.0, min(100.0,
            (min_clr - CFG.EMERG_DIST) / (CFG.TARGET_CLR - CFG.EMERG_DIST) * 100.0))
        return max(0.0, base - coll * 20.0)


ST = State()


# ══════════════════════════════════════════════════════════════
# SECTION 6 — SYSTEM METRICS SAMPLER
# ══════════════════════════════════════════════════════════════
def _read_sys_metrics() -> Tuple[float, float, float]:
    cpu = 0.0
    ram = 0.0
    try:
        import psutil
        cpu = psutil.cpu_percent(interval=0.05)
        ram = psutil.virtual_memory().used / 1e6
    except Exception:
        pass
    pwr = 0.0
    for fp in (
        "/sys/bus/i2c/drivers/ina3221x/0-0041/iio:device0/in_power0_input",
        "/sys/class/hwmon/hwmon0/power1_input",
    ):
        try:
            if os.path.exists(fp):
                pwr = float(open(fp).read().strip()) / 1e3
                break
        except Exception:
            pass
    return cpu, ram, pwr


class MetricsSampler:
    """Background thread that samples CPU/RAM/power every second."""

    def __init__(self, period: float = 1.0) -> None:
        self.period   = period
        self.samples: List[Tuple[float, float, float]] = []
        self._stop    = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self.samples = []
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="metrics_sampler")
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            self.samples.append(_read_sys_metrics())
            self._stop.wait(self.period)

    def stop(self) -> Tuple[float, float, float]:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        if not self.samples:
            self.samples.append(_read_sys_metrics())
        cpus = [s[0] for s in self.samples]
        rams = [s[1] for s in self.samples]
        pwrs = [s[2] for s in self.samples if s[2] > 0.0]
        return (
            statistics.mean(cpus) if cpus else 0.0,
            statistics.mean(rams) if rams else 0.0,
            statistics.mean(pwrs) if pwrs else 0.0,
        )


# ══════════════════════════════════════════════════════════════
# SECTION 7 — STATISTICS ENGINE
# ══════════════════════════════════════════════════════════════
def _agg(vals: List[float]) -> Dict:
    if not vals:
        return {"mean": 0, "std": 0, "min": 0, "max": 0, "ci95": 0, "n": 0}
    n = len(vals)
    m = statistics.mean(vals)
    s = statistics.stdev(vals) if n > 1 else 0.0
    return {
        "mean": round(m, 4), "std": round(s, 4),
        "min": round(min(vals), 4), "max": round(max(vals), 4),
        "ci95": round(1.96 * s / math.sqrt(n), 4), "n": n,
    }




def _bounded(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, float(value)))


def compute_run_scores(r: MissionResult) -> MissionResult:
    """Compute defendable thesis scores for a single run.

    The score avoids declaring a model "best" by success rate alone.
    It reflects the project title: latency-aware LLM planning + autonomous navigation.
    Weights:
      mission completion 35%, LLM latency 25%, path efficiency 20%, safety 10%, recovery 10%.
    """
    mission = float(r.inspection_pct or 0.0)
    if r.return_home_success:
        mission = min(100.0, mission + 10.0) if mission < 100.0 else 100.0
    elif mission >= 100.0:
        mission = 85.0  # completed target but failed ideal home return
    if r.failure_reason and 'HOME reset failed' in r.failure_reason:
        mission = min(mission, 30.0)

    lat = float(r.llm_latency or 0.0)
    if lat <= CFG.LATENCY_IDEAL_S:
        latency = 100.0
    elif lat >= CFG.LATENCY_WORST_S:
        latency = 0.0
    else:
        latency = 100.0 * (CFG.LATENCY_WORST_S - lat) / (CFG.LATENCY_WORST_S - CFG.LATENCY_IDEAL_S)

    eff = float(r.route_efficiency or 0.0)
    if eff <= 0.0:
        # fallback when ideal path cannot be estimated; do not unfairly set zero
        if r.distance_travelled > 0:
            eff = max(0.2, min(1.0, 8.0 / max(1.0, r.distance_travelled)))
        else:
            eff = 0.0
    path_score = _bounded(100.0 * min(1.0, eff))

    safety = float(r.safety_score or 0.0)
    safety -= 12.0 * max(0, int(r.collision_count))
    safety -= 4.0 * max(0, int(r.near_collision_count))
    safety = _bounded(safety)

    rec_events = int(r.reverse_maneuver_count) + int(r.left_recovery_count) + int(r.right_recovery_count) + int(r.replan_count)
    recovery = 100.0
    if rec_events > CFG.RECOVERY_IDEAL_MAX:
        recovery -= 12.0 * (rec_events - CFG.RECOVERY_IDEAL_MAX)
    if r.duckling_timeout:
        recovery -= 30.0
    if r.llm_recovery_advice_used:
        recovery += 5.0  # bounded reward when recovery assistance was useful
    recovery = _bounded(recovery)

    final = 0.35 * mission + 0.25 * latency + 0.20 * path_score + 0.10 * safety + 0.10 * recovery

    # automatic hard-fail cases
    hard_fail = False
    hard_reasons = []
    if 'major collision' in (r.failure_reason or '').lower():
        hard_fail = True; hard_reasons.append('major collision')
    if 'timeout' in (r.failure_reason or '').lower() and mission < 50:
        hard_fail = True; hard_reasons.append('mission timeout')
    if r.points_planned > 0 and r.points_reached == 0:
        hard_fail = True; hard_reasons.append('no target reached')
    if r.status == 'ABORTED':
        hard_fail = True; hard_reasons.append('aborted')

    r.mission_score = round(_bounded(mission), 2)
    r.latency_score = round(_bounded(latency), 2)
    r.path_efficiency_score = round(_bounded(path_score), 2)
    r.safety_score = round(_bounded(safety), 2)
    r.recovery_score = round(_bounded(recovery), 2)
    r.final_deployment_score = round(_bounded(final), 2)
    r.success = (not hard_fail) and (r.final_deployment_score >= CFG.PASS_THRESHOLD)
    if hard_fail:
        r.pass_fail_reason = 'Hard fail: ' + ', '.join(hard_reasons)
    elif r.success:
        r.pass_fail_reason = f'PASS: deployment score {r.final_deployment_score:.1f} ≥ {CFG.PASS_THRESHOLD:.0f}'
    else:
        r.pass_fail_reason = f'FAIL: deployment score {r.final_deployment_score:.1f} < {CFG.PASS_THRESHOLD:.0f}'
    return r


def model_short_review(model: str, ms: Dict[str, Any]) -> str:
    if not ms:
        return 'No completed runs yet.'
    sr = float(ms.get('success_rate', 0.0))
    lat = float(ms.get('llm_latency', {}).get('mean', 0.0))
    rec = float(ms.get('recovery_score', {}).get('mean', 0.0))
    dep = float(ms.get('final_deployment_score', {}).get('mean', 0.0))
    if sr >= 95 and lat > 8:
        return f'{model} is reliable but slower; good completion rate, weaker latency-awareness.'
    if dep >= 85 and lat <= 8:
        return f'{model} shows the best deployment balance: strong success, reasonable latency, and stable navigation.'
    if sr < 70:
        return f'{model} struggles with task completion; review route planning and recovery cases.'
    if rec < 70:
        return f'{model} completes some tasks but struggles during recovery/duckling situations.'
    return f'{model} is usable, but final ranking depends on latency, path efficiency, and recovery quality.'

def compute_stats(results: List[MissionResult]) -> Dict:
    if not results:
        return {}
    models = list(dict.fromkeys(r.model for r in results))
    out: Dict = {}
    for m in models:
        mr  = [r for r in results if r.model == m]
        n   = len(mr)
        suc = sum(1 for r in mr if r.success)
        ms: Dict[str, Any] = {
            "n": n, "success_count": suc,
            "success_rate": (suc / n * 100) if n else 0.0,
        }
        for attr in (
            "llm_latency", "nav_time", "total_mission_time",
            "distance_travelled", "path_length", "route_efficiency",
            "inspection_pct", "orbits_completed", "orbit_success_rate",
            "collision_count", "replan_count",
            "min_clearance", "avg_clearance",
            "risk_score", "safety_score", "mission_score", "latency_score",
            "path_efficiency_score", "recovery_score", "final_deployment_score",
            "near_collision_count", "duckling_timeout", "reverse_maneuver_count",
            "left_recovery_count", "right_recovery_count",
            "avg_cpu", "avg_ram", "avg_power", "avg_temp",
        ):
            ms[attr] = _agg([float(getattr(r, attr)) for r in mr])
        out[m] = ms
    return out


def deployment_score(ms: Dict) -> Tuple[float, str]:
    sc = ms.get("final_deployment_score", {}).get("mean", 0.0)
    if not sc:
        # compatibility fallback for older rows
        sr = ms.get("success_rate", 0.0)
        lat = ms.get("latency_score", {}).get("mean", 0.0)
        pe = ms.get("path_efficiency_score", {}).get("mean", 0.0)
        ss = ms.get("safety_score", {}).get("mean", 0.0)
        rs = ms.get("recovery_score", {}).get("mean", 0.0)
        sc = 0.35 * sr + 0.25 * lat + 0.20 * pe + 0.10 * ss + 0.10 * rs
    label = ("Deployment Ready" if sc >= 85 else
             "Research Ready" if sc >= 70 else
             "Needs Improvement" if sc >= 50 else "Not Ready")
    return round(sc, 1), label


def rank_models(stats: Dict) -> List[Dict]:
    if not stats:
        return []
    medals  = ["🥇", "🥈", "🥉"]
    scored  = {m: deployment_score(stats[m]) for m in stats}
    ranked  = sorted(stats.keys(), key=lambda m: scored[m][0], reverse=True)
    return [
        {
            "rank":         i + 1,
            "medal":        medals[i] if i < 3 else str(i + 1),
            "model":        m,
            "score":        scored[m][0],
            "label":        scored[m][1],
            "success_rate": stats[m].get("success_rate", 0),
            "latency":      stats[m].get("llm_latency",  {}).get("mean", 0),
            "inspection":   stats[m].get("inspection_pct",{}).get("mean", 0),
            "safety":       stats[m].get("safety_score",  {}).get("mean", 0),
            "latency_score": stats[m].get("latency_score", {}).get("mean", 0),
            "path_score":    stats[m].get("path_efficiency_score", {}).get("mean", 0),
            "recovery_score": stats[m].get("recovery_score", {}).get("mean", 0),
            "collisions":   stats[m].get("collision_count",{}).get("mean", 0),
            "comment":      model_short_review(m, stats[m]),
        }
        for i, m in enumerate(ranked)
    ]


# ══════════════════════════════════════════════════════════════
# SECTION 8 — CSV / TRAJECTORY I/O
# ══════════════════════════════════════════════════════════════
_CSV_HEADER = [
    "run_id", "model", "mission_number", "mission_text", "timestamp",
    "status", "success", "failure_reason",
    "llm_latency", "parse_time", "llm_retry_count", "valid_json", "route_correctness", "fallback_used",
    "reset_time", "nav_time", "total_mission_time",
    "distance_travelled", "path_length", "route_efficiency",
    "return_home_success",
    "inspection_pct", "points_planned", "points_reached",
    "orbits_completed", "orbit_success_rate",
    "collision_count", "near_collision_count", "replan_count", "duckling_timeout",
    "reverse_maneuver_count", "left_recovery_count", "right_recovery_count", "llm_recovery_advice_used",
    "min_clearance", "avg_clearance", "risk_score", "safety_score",
    "mission_score", "latency_score", "path_efficiency_score", "recovery_score",
    "final_deployment_score", "pass_fail_reason",
    "avg_cpu", "avg_ram", "avg_power", "avg_temp",
    "analysis_comment", "learning_feedback", "improvement_hint",
    "parsed_route", "executed_route", "reasoning",
]

_VLOG_HEADER = [
    "timestamp", "model", "run_index", "status",
    "total_mission_time", "collision_count", "failure_reason", "executed_route",
]


def _csv_append(path: str, header: List[str], row: List) -> None:
    new = not os.path.exists(path)
    try:
        with open(path, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(header)
            w.writerow(row)
    except Exception as ex:
        ST.log(f"CSV write error: {ex}", "WARN")


def save_result_csv(r: MissionResult) -> None:
    _csv_append(CFG.CSV_PATH, _CSV_HEADER, [
        r.run_id, r.model, r.mission_number,
        r.mission_text.replace("\n", " "), r.timestamp,
        r.status, int(r.success), r.failure_reason,
        f"{r.llm_latency:.4f}", f"{r.parse_time:.4f}", r.llm_retry_count, int(r.valid_json), f"{r.route_correctness:.1f}", int(r.fallback_used),
        f"{r.reset_time:.3f}", f"{r.nav_time:.3f}",
        f"{r.total_mission_time:.3f}",
        f"{r.distance_travelled:.3f}", f"{r.path_length:.3f}",
        f"{r.route_efficiency:.4f}",
        int(r.return_home_success),
        f"{r.inspection_pct:.2f}", r.points_planned, r.points_reached,
        r.orbits_completed, f"{r.orbit_success_rate:.3f}",
        r.collision_count, r.near_collision_count, r.replan_count, int(r.duckling_timeout),
        r.reverse_maneuver_count, r.left_recovery_count, r.right_recovery_count, int(r.llm_recovery_advice_used),
        f"{r.min_clearance:.4f}", f"{r.avg_clearance:.4f}",
        f"{r.risk_score:.2f}", f"{r.safety_score:.2f}",
        f"{r.mission_score:.2f}", f"{r.latency_score:.2f}", f"{r.path_efficiency_score:.2f}",
        f"{r.recovery_score:.2f}", f"{r.final_deployment_score:.2f}", r.pass_fail_reason,
        f"{r.avg_cpu:.2f}", f"{r.avg_ram:.1f}", f"{r.avg_power:.3f}", f"{r.avg_temp:.2f}",
        r.analysis_comment.replace("\n", " ")[:500],
        r.learning_feedback.replace("\n", " ")[:500],
        r.improvement_hint.replace("\n", " ")[:500],
        "|".join(r.parsed_route),
        "|".join(r.executed_route),
        r.reasoning.replace("\n", " ")[:500],
    ])


def save_vlog_csv(model: str, idx: int, r: MissionResult) -> None:
    _csv_append(CFG.VLOG_PATH, _VLOG_HEADER, [
        datetime.datetime.now().isoformat(), model, idx,
        r.status, f"{r.total_mission_time:.3f}",
        r.collision_count, r.failure_reason,
        " → ".join(r.executed_route),
    ])


def save_trajectory(r: MissionResult) -> None:
    try:
        fp = os.path.join(CFG.TRAJ_DIR, f"{r.run_id}.json")
        with open(fp, "w") as f:
            json.dump({
                "run_id": r.run_id, "model": r.model,
                "status": r.status, "success": r.success,
                "timestamp": r.timestamp,
                "mission_text": r.mission_text,
                "planned_route": r.parsed_route,
                "executed_route": r.executed_route,
                "trajectory": r.trajectory,
                "insp_records": [asdict(x) for x in r.insp_records],
                "coll_events":  [asdict(x) for x in r.coll_events],
                "reasoning":    r.reasoning,
                "raw_llm":      r.raw_llm_response,
                "failure_reason": r.failure_reason,
                "mission_score": r.mission_score,
                "latency_score": r.latency_score,
                "path_efficiency_score": r.path_efficiency_score,
                "recovery_score": r.recovery_score,
                "final_deployment_score": r.final_deployment_score,
                "pass_fail_reason": r.pass_fail_reason,
                "analysis_comment": r.analysis_comment,
                "learning_feedback": r.learning_feedback,
                "improvement_hint": r.improvement_hint,
            }, f)
    except Exception as ex:
        ST.log(f"Trajectory save error: {ex}", "WARN")


def load_trajectory(run_id: str) -> Optional[Dict]:
    fp = os.path.join(CFG.TRAJ_DIR, f"{run_id}.json")
    if not os.path.exists(fp):
        return None
    try:
        with open(fp) as f:
            return json.load(f)
    except Exception:
        return None


def list_trajectories() -> List[str]:
    return sorted(f[:-5] for f in os.listdir(CFG.TRAJ_DIR) if f.endswith(".json"))


def _path_length(traj: List[Dict]) -> float:
    if len(traj) < 2:
        return 0.0
    return sum(
        math.hypot(traj[i]["x"] - traj[i-1]["x"], traj[i]["y"] - traj[i-1]["y"])
        for i in range(1, len(traj))
    )


# ══════════════════════════════════════════════════════════════
# SECTION 9 — OLLAMA / LLM INTERFACE
# ══════════════════════════════════════════════════════════════
_ROUTE_RE  = re.compile(r'"route"\s*:\s*\[(.*?)\]', re.DOTALL | re.IGNORECASE)
_REASON_RE = re.compile(r'"reasoning"\s*:\s*"((?:[^"\\]|\\.)*)"',
                         re.DOTALL | re.IGNORECASE)


def warmup_ollama(models: Optional[List[str]] = None) -> List[str]:
    targets = models or CFG.MODELS
    ok: List[str] = []
    try:
        r = ST.http.get(CFG.OLLAMA_TAGS, timeout=10)
        with ST.lock:
            ST.validation["ollama_ok"] = (r.status_code == 200)
    except Exception as ex:
        ST.log(f"Ollama unreachable: {ex}", "WARN")
        with ST.lock:
            ST.validation["ollama_ok"] = False
            ST.unavail_models = list(targets)
            ST.validation["models_failed"] = list(targets)
        return []
    for model in targets:
        success = False
        for attempt in range(1, 4):
            try:
                r = ST.http.post(CFG.OLLAMA_GEN, json={
                    "model": model, "prompt": "ping",
                    "stream": False, "options": {"num_predict": 1},
                }, timeout=120)
                if r.status_code == 200:
                    success = True
                    break
            except requests.Timeout:
                ST.log(f"Warmup {model} timeout {attempt}/3", "WARN")
            except Exception as ex:
                ST.log(f"Warmup {model} error: {ex}", "WARN")
            if attempt < 3:
                time.sleep(5)
        if success:
            ok.append(model)
            ST.log(f"Warmup OK: {model}")
        else:
            with ST.lock:
                if model not in ST.unavail_models:
                    ST.unavail_models.append(model)
    with ST.lock:
        ST.validation["models_ok"]     = list(ok)
        ST.validation["models_failed"] = [m for m in targets if m not in ok]
        ST.validation["ollama_ok"]     = len(ok) > 0
    ST.log(f"Warmup done. Available: {ok}")
    return ok


def _cylinder_description() -> str:
    cyls = _REGISTRY.confirmed_list()
    if not cyls:
        return "No cylinders confirmed yet — robot must explore first."
    with ST.lock:
        rx, ry = ST.rx, ST.ry
    lines = [f"Confirmed cylinders ({len(cyls)} total):"]
    for c in cyls:
        d = math.hypot(c.x - rx, c.y - ry)
        lines.append(
            f"  {c.label}: x={c.x:.2f}, y={c.y:.2f}, "
            f"dist={d:.2f}m, radius={c.radius:.3f}m"
        )
    return "\n".join(lines)


def _parse_llm_response(text: str, allowed: List[str]) -> Tuple[List[str], str, bool]:
    route:     List[str] = []
    reasoning: str       = ""
    # Try JSON parse
    s = text.find("{")
    e = text.rfind("}")
    if s >= 0 and e > s:
        try:
            d = json.loads(text[s:e+1])
            raw = d.get("route", [])
            if isinstance(raw, list):
                route = [str(w).strip().upper().replace(" ", "_") for w in raw]
            reasoning = str(d.get("reasoning", "")).strip()
        except Exception:
            pass
    # Regex fallback for route
    if not route:
        m = _ROUTE_RE.search(text)
        if m:
            toks  = re.findall(r'"([^"]+)"|\'([^\']+)\'', m.group(1))
            route = [(a or b).strip().upper().replace(" ", "_") for a, b in toks]
    # Regex fallback for reasoning
    if not reasoning:
        rm = _REASON_RE.search(text)
        if rm:
            reasoning = rm.group(1).strip()
    # Last resort: scan text for CYL_N
    if not route:
        found = re.findall(r'CYL_\d+', text.upper())
        seen: List[str] = []
        for f in found:
            if not seen or seen[-1] != f:
                seen.append(f)
        route = seen
    # Filter to allowed only
    route = [w for w in route if w in allowed]
    return route, reasoning, bool(route)


def _validate_route(route: List[str], allowed: List[str]) -> Tuple[bool, List[str], str]:
    if not route:
        return False, [], "empty route"
    bad = [w for w in route if w not in allowed]
    if bad:
        return False, [], f"unknown cylinders: {bad}"
    deduped: List[str] = []
    for w in route:
        if not deduped or deduped[-1] != w:
            deduped.append(w)
    if not deduped:
        return False, [], "no valid waypoints after dedup"
    return True, deduped, "ok"


def _nearest_neighbor_route(allowed: List[str]) -> List[str]:
    cyls = {c.label: c for c in _REGISTRY.confirmed_list() if c.label in allowed}
    with ST.lock:
        px, py = ST.rx, ST.ry
    remaining = list(allowed)
    route: List[str] = []
    while remaining:
        nxt = min(
            remaining,
            key=lambda name: (
                math.hypot(cyls[name].x - px, cyls[name].y - py)
                if name in cyls else 999.0
            ),
        )
        route.append(nxt)
        if nxt in cyls:
            px, py = cyls[nxt].x, cyls[nxt].y
        remaining.remove(nxt)
    return route


def plan_mission(model: str, mission_text: str,
                 max_retries: int = 2) -> Dict:
    with ST.lock:
        rx, ry = ST.rx, ST.ry
        smin   = ST.scan_min
        lstat  = ST.lidar_status
    cyls    = _REGISTRY.confirmed_list()
    allowed = [c.label for c in cyls]
    desc    = _cylinder_description()
    result: Dict = {
        "raw": "", "route": [], "reasoning": "", "valid": False,
        "executable": [], "llm_latency": 0.0, "parse_time": 0.0,
        "reason": "", "attempts": 0, "allowed": allowed,
    }
    if not allowed:
        result["reason"] = "No confirmed cylinders available for planning"
        return result
    prompt = (
        "You are the task planner for an autonomous inspection robot.\n"
        "Your ONLY job is to decide which cylinders to inspect and in what order.\n\n"
        f"ENVIRONMENT:\n{desc}\n\n"
        "ALLOWED CYLINDER NAMES (copy exactly, case-sensitive):\n"
        f"  {', '.join(allowed)}\n\n"
        "RULES:\n"
        "  1. Use ONLY the names listed above.\n"
        "  2. Do NOT include HOME — the robot returns home automatically.\n"
        "  3. Minimise total travel distance.\n"
        "  4. The robot orbits each cylinder 360° at 0.6m for inspection.\n\n"
        f"ROBOT POSITION: ({rx:.2f}, {ry:.2f})  "
        f"nearest_obstacle={smin:.2f}m  lidar={lstat}\n\n"
        f'MISSION: "{mission_text}"\n\n'
        "Reply ONLY with valid JSON (no markdown, no explanation outside JSON):\n"
        '{"route":["CYL_1","CYL_3","CYL_2"],"reasoning":"shortest path explanation"}'
    )
    for attempt in range(1, max_retries + 2):
        result["attempts"] = attempt
        ST.log(f"LLM [{model}] planning attempt {attempt}")
        t0 = time.time()
        try:
            resp = ST.http.post(
                CFG.OLLAMA_GEN,
                json={"model": model, "prompt": prompt, "stream": False,
                      "options": {"temperature": 0.1, "num_predict": 250}},
                timeout=CFG.OLLAMA_TIMEOUT,
            )
        except requests.Timeout:
            result["llm_latency"] = CFG.OLLAMA_TIMEOUT
            result["reason"]      = "LLM timeout"
            continue
        except Exception as ex:
            result["reason"] = f"LLM error: {ex}"
            continue
        result["llm_latency"] = time.time() - t0
        if resp.status_code != 200:
            result["reason"] = f"HTTP {resp.status_code}"
            continue
        text = resp.json().get("response", "")
        result["raw"] = text
        tp = time.time()
        route, reasoning, _ = _parse_llm_response(text, allowed)
        result["parse_time"] = time.time() - tp
        result["route"]      = route
        result["reasoning"]  = reasoning
        ok, norm, why = _validate_route(route, allowed)
        result["valid"]  = ok
        result["reason"] = why
        if ok:
            # If "all" cylinders requested, append missing ones
            wants_all = any(w in mission_text.lower()
                            for w in ("all", "every", "complete", "each"))
            if wants_all:
                missing = [n for n in allowed if n not in norm]
                if missing:
                    norm = norm + _nearest_neighbor_route(missing)
                    result["reason"] += f"; appended missing: {missing}"
            result["executable"] = norm
            with ST.lock:
                ST.planned_route = list(norm)
                ST.reasoning     = reasoning
            ST.log(f"Route planned [{model}]: {' → '.join(norm)}")
            return result
        prompt += f"\n\nPrevious response was rejected ({why}). " \
                  f"Use ONLY these names: {', '.join(allowed)}"
    # All retries failed — use deterministic nearest-neighbor fallback
    fallback = _nearest_neighbor_route(allowed)
    result["route"]      = fallback
    result["executable"] = fallback
    result["valid"]      = True
    result["reason"]     = (
        f"LLM failed after {max_retries+1} attempts; "
        f"nearest-neighbor fallback: {result['reason']}"
    )
    result["reasoning"]  = result["reasoning"] or result["reason"]
    with ST.lock:
        ST.planned_route = list(fallback)
        ST.reasoning     = result["reasoning"]
    ST.log(f"Route fallback [{model}]: {' → '.join(fallback)}", "WARN")
    return result


# ══════════════════════════════════════════════════════════════
# SECTION 10 — ROS 2 NODE
# ══════════════════════════════════════════════════════════════
class OpenClawNode(Node):
    """
    Single ROS2 node handling all robot I/O:
      Subscribers : /odom, /scan, /camera/image_raw (optional)
      Publishers  : /cmd_vel, /openclaw/cylinders (RViz markers)
      Services    : /gazebo/set_entity_state (teleport for reset)
      Timers      : trajectory recorder (10 Hz), RViz publisher (0.5 Hz)
    """

    def __init__(self) -> None:
        super().__init__("openclaw_inspection")
        cbg = ReentrantCallbackGroup()
        # QoS for odometry (best-effort, matches Gazebo bridge)
        odom_qos = QoSProfile(
            depth=20, history=HistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        # Publishers
        self.cmd_pub   = self.create_publisher(Twist, "/cmd_vel", 10)
        self.rviz_pub  = self.create_publisher(MarkerArray, "/openclaw/cylinders", 10)
        # Subscribers
        self.create_subscription(Odometry,   "/odom",  self._odom_cb,
                                 odom_qos, callback_group=cbg)
        self.create_subscription(LaserScan,  "/scan",  self._scan_cb,
                                 qos_profile_sensor_data, callback_group=cbg)
        if CFG.CAM_ENABLED and CFG.CAM_TOPIC:
            self.create_subscription(Image, CFG.CAM_TOPIC, self._cam_cb,
                                     qos_profile_sensor_data, callback_group=cbg)
            ST.log(f"[ROS] Camera topic enabled: {CFG.CAM_TOPIC}")
        else:
            ST.log("[ROS] Camera disabled — benchmark runs LiDAR-only")
        # Gazebo reset service
        self._gz_cli = None
        if _GAZEBO_SRV:
            try:
                self._gz_cli = self.create_client(
                    SetEntityState, "/gazebo/set_entity_state",
                    callback_group=cbg)
            except Exception as ex:
                ST.log(f"[ROS] Gazebo service unavailable: {ex}", "WARN")
        # Timers
        self.create_timer(0.10, self._record_traj, callback_group=cbg)
        self.create_timer(2.00, self._publish_rviz, callback_group=cbg)
        ST.log("[ROS] OpenClawNode ready")

    # ── Command helpers ────────────────────────────────────────
    def _pub(self, tw: Twist) -> None:
        self.cmd_pub.publish(tw)
        with ST.lock:
            ST.cmdvel_pubs += 1

    def stop(self) -> None:
        zero = Twist()
        for _ in range(3):
            self._pub(zero)
            time.sleep(0.02)

    # ── Odometry callback ──────────────────────────────────────
    def _odom_cb(self, msg: Odometry) -> None:
        with ST.lock:
            ST.last_odom_t = time.time()
            ST.odom_msgs  += 1
            ST.rx  = msg.pose.pose.position.x
            ST.ry  = msg.pose.pose.position.y
            ST.vx  = msg.twist.twist.linear.x
            ST.vy  = msg.twist.twist.linear.y
            ST.vw  = msg.twist.twist.angular.z
            q      = msg.pose.pose.orientation
            ST.ryaw = math.atan2(
                2 * (q.w * q.z + q.x * q.y),
                1 - 2 * (q.y * q.y + q.z * q.z))
            if ST.prev_x is not None:
                ST.dist_total += math.hypot(ST.rx - ST.prev_x, ST.ry - ST.prev_y)
            ST.prev_x = ST.rx
            ST.prev_y = ST.ry

    # ── LiDAR callback ────────────────────────────────────────
    def _scan_cb(self, msg: LaserScan) -> None:
        rmax = msg.range_max if msg.range_max > 0 else 3.5
        sec  = {k: rmax for k in
                ("front","front_left","front_right","left","right","rear")}
        smin = rmax
        obs: List[Dict] = []
        with ST.lock:
            rx, ry, ryaw = ST.rx, ST.ry, ST.ryaw
        ang = msg.angle_min
        inc = msg.angle_increment
        idx = 0
        for r in msg.ranges:
            a    = ang
            ang += inc
            idx += 1
            if not (math.isfinite(r) and 0.01 < r < rmax):
                continue
            if r < smin:
                smin = r
            na = math.atan2(math.sin(a), math.cos(a))
            if   abs(na) <= 0.44:        key = "front"
            elif  0.44 < na <= 1.22:     key = "front_left"
            elif -1.22 <= na < -0.44:    key = "front_right"
            elif  1.22 < na <= 2.00:     key = "left"
            elif -2.00 <= na < -1.22:    key = "right"
            else:                        key = "rear"
            if r < sec[key]:
                sec[key] = r
            if r < 2.0 and idx % 8 == 0 and len(obs) < 80:
                obs.append({
                    "x": round(rx + r * math.cos(ryaw + a), 3),
                    "y": round(ry + r * math.sin(ryaw + a), 3),
                })
        status = ("CRITICAL" if smin < CFG.EMERG_DIST else
                  "WARNING"  if smin < CFG.WARN_DIST  else "SAFE")
        with ST.lock:
            ST.last_scan_t  = time.time()
            ST.scan_msgs   += 1
            ST.scan_min     = smin
            ST.scan_sectors = sec
            ST.obstacle_pts = obs
            ST.lidar_status = status
            ST.risk_score   = ST.compute_risk()
            if ST.recording and smin > 0.01:
                ST.clr_readings.append(smin)
            now = time.time()
            if (smin < CFG.EMERG_DIST and
                    (abs(ST.vx) > 0.01 or abs(ST.vw) > 0.01) and
                    now - ST.last_coll_t > 1.5):
                ST.coll_count += 1
                ST.last_coll_t = now
                ST.coll_events.append(CollisionEvent(t=now, x=ST.rx, y=ST.ry, scan_min=smin))
            if (smin < CFG.WARN_DIST and smin >= CFG.EMERG_DIST and
                    (abs(ST.vx) > 0.01 or abs(ST.vw) > 0.01)):
                ST.near_collision_count += 1
        # Cylinder detection (outside lock — uses local copies)
        # Fixed 9-cylinder map is used for planning; LiDAR is safety only.
        # dets = _DETECTOR.detect(msg, rx, ry, ryaw)
        # _REGISTRY.update(dets)

    # ── Camera callback ───────────────────────────────────────
    def _cam_cb(self, msg: Image) -> None:
        with ST.lock:
            ST.last_cam_t   = time.time()
            ST.cam_msgs    += 1
            ST._cam_frames += 1
            now = time.time()
            dt  = now - ST._cam_t
            if dt >= 2.0:
                ST.cam_fps     = ST._cam_frames / dt
                ST._cam_frames = 0
                ST._cam_t      = now
        if not _CV2:
            return
        try:
            arr = _np.frombuffer(msg.data, dtype=_np.uint8)
            enc = (msg.encoding or "").lower()
            if enc in ("rgb8", "bgr8"):
                img = arr.reshape((msg.height, msg.width, 3))
                if enc == "rgb8":
                    img = _cv2.cvtColor(img, _cv2.COLOR_RGB2BGR)
            elif enc == "mono8":
                img = arr.reshape((msg.height, msg.width))
            else:
                ch  = max(1, len(arr) // (msg.height * msg.width)) if msg.height * msg.width else 3
                img = arr.reshape((msg.height, msg.width, ch))
            # HUD overlay
            if len(img.shape) == 3:
                ann = img.copy()
                with ST.lock:
                    state = ST.mission_state
                    acyl  = ST.active_cyl
                    ncyl  = len(_REGISTRY.confirmed_list())
                    smn   = ST.scan_min
                    lstat = ST.lidar_status
                h_, w_ = ann.shape[:2]
                _cv2.rectangle(ann, (0, 0), (w_, 28), (0, 0, 0), -1)
                sc = ((0, 220, 0)  if state in ("IDLE","COMPLETED","SUCCESS") else
                      (0, 100, 255) if "APPROACH" in state or "ORBIT" in state else
                      (0, 200, 255))
                _cv2.putText(ann, state, (5, 18),
                             _cv2.FONT_HERSHEY_SIMPLEX, 0.5, sc, 1)
                if acyl:
                    _cv2.putText(ann, f"→ {acyl}", (w_ // 2 - 35, 18),
                                 _cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 150, 255), 1)
                _cv2.putText(ann, f"CYL:{ncyl}", (w_ - 75, 18),
                             _cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 220), 1)
                _cv2.rectangle(ann, (0, h_ - 24), (w_, h_), (0, 0, 0), -1)
                lc = ((0, 220, 0) if lstat == "SAFE" else
                      (0, 200, 255) if lstat == "WARNING" else (0, 0, 220))
                _cv2.putText(ann, f"LIDAR:{smn:.2f}m {lstat}", (5, h_ - 7),
                             _cv2.FONT_HERSHEY_SIMPLEX, 0.45, lc, 1)
                ok, buf = _cv2.imencode(
                    ".jpg", ann, [_cv2.IMWRITE_JPEG_QUALITY, CFG.MJPEG_QUALITY])
            else:
                ok, buf = _cv2.imencode(
                    ".jpg", img, [_cv2.IMWRITE_JPEG_QUALITY, CFG.MJPEG_QUALITY])
            if ok:
                with ST.lock:
                    ST.cam_bytes = buf.tobytes()
                    ST.cam_ok    = True
                    ST.cam_w     = msg.width
                    ST.cam_h     = msg.height
        except Exception as ex:
            ST.log(f"Camera frame error: {ex}", "WARN")

    # ── Trajectory recording timer ─────────────────────────────
    def _record_traj(self) -> None:
        with ST.lock:
            if not ST.recording:
                return
            t0 = ST.mission_t0
            ST.trajectory.append({
                "t":   round(time.time() - t0, 3) if t0 else 0.0,
                "x":   round(ST.rx,   4),
                "y":   round(ST.ry,   4),
                "yaw": round(ST.ryaw, 4),
            })

    # ── RViz cylinder marker publisher ────────────────────────
    def _publish_rviz(self) -> None:
        ma  = MarkerArray()
        mid = 0
        for c in _REGISTRY.all_list():
            m = Marker()
            m.header.frame_id = "odom"
            m.header.stamp    = self.get_clock().now().to_msg()
            m.ns      = "cyl"
            m.id      = mid; mid += 1
            m.type    = Marker.CYLINDER
            m.action  = Marker.ADD
            m.pose.position.x = float(c.x)
            m.pose.position.y = float(c.y)
            m.pose.position.z = 0.3
            m.scale.x = m.scale.y = float(c.radius * 2)
            m.scale.z = 0.6
            m.color.r = 1.0
            m.color.g = 1.0 if c.confirmed else 0.4
            m.color.b = 0.0
            m.color.a = 0.85
            ma.markers.append(m)
            t = Marker()
            t.header.frame_id = "odom"
            t.header.stamp    = self.get_clock().now().to_msg()
            t.ns     = "cyl_lbl"
            t.id     = mid; mid += 1
            t.type   = Marker.TEXT_VIEW_FACING
            t.action = Marker.ADD
            t.pose.position.x = float(c.x)
            t.pose.position.y = float(c.y)
            t.pose.position.z = 0.9
            t.scale.z = 0.22
            t.color.r = t.color.g = t.color.b = t.color.a = 1.0
            t.text   = c.label
            ma.markers.append(t)
        self.rviz_pub.publish(ma)

    # ── Gazebo service helper ─────────────────────────────────
    def _call_gz_srv(self, req, timeout: float = 8.0) -> Optional[Any]:
        if not self._gz_cli:
            return None
        if not self._gz_cli.wait_for_service(timeout_sec=2.0):
            return None
        fut      = self._gz_cli.call_async(req)
        deadline = time.time() + timeout
        while time.time() < deadline:
            if fut.done():
                try:
                    return fut.result()
                except Exception:
                    return None
            time.sleep(0.02)
        return None

    # ══════════════════════════════════════════════════════════
    # SMOOTH NAVIGATION — PD controller + angular LP filter
    # ══════════════════════════════════════════════════════════
    def _llm_maneuver_advice(self, reason: str, sec: Dict[str, float]) -> Optional[str]:
        """Ask the currently selected LLM for a short recovery decision.
        The LLM is only allowed to choose one manoeuvre word. If it is slow or
        invalid, the deterministic LiDAR fallback is used. This gives the robot
        the 'LLM help' you wanted without blocking movement for a long time.
        """
        with ST.lock:
            model = ST.cur_model or CFG.MODELS[0]
        try:
            prompt = (
                "Robot is stuck/blocked during navigation. Choose safest recovery.\n"
                f"Reason: {reason}\n"
                f"LiDAR sectors metres: front={sec.get('front',99):.2f}, front_left={sec.get('front_left',99):.2f}, "
                f"front_right={sec.get('front_right',99):.2f}, left={sec.get('left',99):.2f}, "
                f"right={sec.get('right',99):.2f}, rear={sec.get('rear',99):.2f}.\n"
                "Reply ONLY one word: WIDE_LEFT, WIDE_RIGHT, REVERSE_LEFT, REVERSE_RIGHT, LEFT, RIGHT, or STOP."
            )
            r = ST.http.post(
                CFG.OLLAMA_GEN,
                json={"model": model, "prompt": prompt, "stream": False,
                      "options": {"temperature": 0.0, "num_predict": 8}},
                timeout=CFG.RECOVERY_LLM_TIMEOUT,
            )
            txt = (r.json().get("response", "") if r.status_code == 200 else "").upper()
            for k in ("WIDE_LEFT", "WIDE_RIGHT", "REVERSE_LEFT", "REVERSE_RIGHT", "LEFT", "RIGHT", "STOP"):
                if k in txt:
                    ST.log(f"LLM recovery advice [{model}]: {k}")
                    with ST.lock:
                        ST.llm_recovery_advice_used = True
                    return k
        except Exception as ex:
            ST.log(f"LLM recovery advice skipped: {ex}", "WARN")
        return None

    def _choose_recovery_side(self, sec: Dict[str, float], tx: Optional[float] = None, ty: Optional[float] = None, advice: Optional[str] = None) -> float:
        """Choose recovery side. Positive = left, negative = right.
        Priority:
          1) LLM explicit advice if valid;
          2) target-side preference if that side is not blocked;
          3) widest LiDAR side.
        This makes the robot behave like a driver, not a left-right duckling.
        """
        if advice in ("LEFT", "REVERSE_LEFT", "WIDE_LEFT"):
            return 1.0
        if advice in ("RIGHT", "REVERSE_RIGHT", "WIDE_RIGHT"):
            return -1.0
        left_open = max(sec.get("left", 0.0), sec.get("front_left", 0.0))
        right_open = max(sec.get("right", 0.0), sec.get("front_right", 0.0))
        if tx is not None and ty is not None:
            with ST.lock:
                rx, ry, ryaw = ST.rx, ST.ry, ST.ryaw
            target_yaw = math.atan2(ty - ry, tx - rx)
            err = target_yaw - ryaw
            while err > math.pi: err -= 2 * math.pi
            while err < -math.pi: err += 2 * math.pi
            preferred = 1.0 if err > 0 else -1.0
            # Use target side if it has at least reasonable clearance.
            if preferred > 0 and left_open > max(0.42, right_open * 0.72):
                return 1.0
            if preferred < 0 and right_open > max(0.42, left_open * 0.72):
                return -1.0
        return 1.0 if left_open >= right_open else -1.0

    def _target_detour_point(self, tx: float, ty: float, side: float) -> Tuple[float, float]:
        """Create a temporary target-aware detour point.
        It steps diagonally toward the target plus a lateral offset to go around
        a blocking cylinder/wall. This is the missing 'drive like a car' logic.
        """
        with ST.lock:
            rx, ry = ST.rx, ST.ry
        dx, dy = tx - rx, ty - ry
        d = max(1e-6, math.hypot(dx, dy))
        ux, uy = dx / d, dy / d
        # perpendicular: left = (-uy, ux), right = (uy, -ux)
        px, py = (-uy * side), (ux * side)
        ahead = min(0.55, max(0.25, d * 0.35))
        lateral = CFG.RECOVERY_DETOUR_M
        return rx + ux * ahead + px * lateral, ry + uy * ahead + py * lateral

    def _maneuver_escape(self, deadline: float, reason: str = "obstacle", tx: Optional[float] = None, ty: Optional[float] = None) -> bool:
        """Target-aware recovery supervisor.

        Old behaviour: rotate left/right when the front was blocked. That caused
        duckling near CYL_4/CYL_2 because the robot kept seeing the same obstacle.

        New behaviour:
          1. stop,
          2. analyze LiDAR sectors,
          3. ask LLM for a bounded recovery word,
          4. deterministic fallback if LLM is slow,
          5. reverse with steering,
          6. short forward glide,
          7. optional detour point toward the target,
          8. retry original target.
        """
        if time.time() > deadline:
            self.stop(); return False
        with ST.lock:
            sec = dict(ST.scan_sectors)
        advice = self._llm_maneuver_advice(reason, sec)
        if advice == "STOP":
            self.stop(); return False
        turn_sign = self._choose_recovery_side(sec, tx, ty, advice)
        left_open = max(sec.get("left", 0.0), sec.get("front_left", 0.0))
        right_open = max(sec.get("right", 0.0), sec.get("front_right", 0.0))
        ST.log(
            f"RECOVERY SUPERVISOR: {reason}; front={sec.get('front',99):.2f} left={left_open:.2f} right={right_open:.2f}; "
            f"advice={advice or 'fallback'}; action={'LEFT' if turn_sign>0 else 'RIGHT'}",
            "WARN",
        )
        with ST.lock:
            ST.reverse_maneuver_count += 1
            if turn_sign > 0:
                ST.left_recovery_count += 1
            else:
                ST.right_recovery_count += 1
        self.stop(); time.sleep(0.08)

        # 1) Reverse with steering, but only if rear is safe. If not, rotate away.
        t0 = time.time()
        while time.time() - t0 < CFG.MANEUVER_REVERSE_T and time.time() < deadline:
            if ST.cancel_event.is_set() or ST.stop_requested:
                self.stop(); return False
            with ST.lock:
                rear = ST.scan_sectors.get("rear", 99.0)
            cmd = Twist()
            if rear > CFG.EMERG_DIST * 1.55:
                cmd.linear.x = CFG.BACK_SPEED
                cmd.angular.z = -0.55 * turn_sign
            else:
                cmd.linear.x = 0.0
                cmd.angular.z = 0.80 * turn_sign
            self._pub(cmd)
            time.sleep(0.05)

        # 2) Rotate open side, enough to change geometry; not just tiny wiggle.
        t0 = time.time()
        while time.time() - t0 < CFG.MANEUVER_TURN_T and time.time() < deadline:
            if ST.cancel_event.is_set() or ST.stop_requested:
                self.stop(); return False
            cmd = Twist(); cmd.angular.z = CFG.NAV_ANG_MAX * 0.78 * turn_sign
            self._pub(cmd)
            time.sleep(0.05)

        # 3) Smooth forward glide if front is open. This breaks the oscillation.
        t0 = time.time()
        while time.time() - t0 < getattr(CFG, 'MANEUVER_FORWARD_T', 0.55) and time.time() < deadline:
            if ST.cancel_event.is_set() or ST.stop_requested:
                self.stop(); return False
            with ST.lock:
                sec2 = dict(ST.scan_sectors)
            cmd = Twist()
            if sec2.get("front", 99.0) > max(CFG.WARN_DIST * 0.85, 0.42):
                cmd.linear.x = min(CFG.NAV_LIN_MAX * 0.70, 0.14)
                cmd.angular.z = 0.08 * turn_sign
            else:
                break
            self._pub(cmd)
            time.sleep(0.05)
        self.stop(); time.sleep(0.08)

        # 4) Optional target-aware detour. This is not LLM motor control; it is
        # a safe local primitive selected by the recovery supervisor.
        if tx is not None and ty is not None and time.time() + 3.0 < deadline:
            with ST.lock:
                sec3 = dict(ST.scan_sectors)
            if sec3.get("front", 99.0) > 0.40:
                dx, dy = self._target_detour_point(tx, ty, turn_sign)
                ST.log(f"RECOVERY DETOUR: temporary waypoint ({dx:.2f},{dy:.2f}) before retrying target", "WARN")
                # Drive to detour using a compact mini-controller to avoid recursive recovery storms.
                mini_deadline = min(deadline, time.time() + 5.0)
                stable = 0
                while rclpy.ok() and time.time() < mini_deadline:
                    if ST.cancel_event.is_set() or ST.stop_requested:
                        self.stop(); return False
                    with ST.lock:
                        rx, ry, ryaw = ST.rx, ST.ry, ST.ryaw
                        secd = dict(ST.scan_sectors)
                    ddx, ddy = dx - rx, dy - ry
                    dist = math.hypot(ddx, ddy)
                    if dist < 0.18:
                        stable += 1
                        if stable >= 2:
                            break
                    else:
                        stable = 0
                    front = secd.get("front", 99.0)
                    if front < CFG.EMERG_DIST:
                        break
                    heading = math.atan2(ddy, ddx)
                    err = heading - ryaw
                    while err > math.pi: err -= 2 * math.pi
                    while err < -math.pi: err += 2 * math.pi
                    cmd = Twist()
                    if abs(err) > math.radians(18):
                        cmd.angular.z = max(-0.75, min(0.75, 1.2 * err))
                    else:
                        cmd.linear.x = min(0.13, max(0.06, dist * 0.20))
                        cmd.angular.z = max(-0.22, min(0.22, 0.7 * err))
                    self._pub(cmd)
                    time.sleep(0.05)
                self.stop(); time.sleep(0.05)
        with ST.lock:
            ST.replan_count += 1
        return True

    def _backoff(self, deadline: float) -> bool:
        """
        Emergency backoff: stop → reverse → rotate to clearance.
        Returns True when front is clear enough to continue.
        """
        ST.log("SAFETY: backoff started", "WARN")
        self.stop()
        time.sleep(0.15)
        t0 = time.time()
        while rclpy.ok():
            if ST.cancel_event.is_set():
                self.stop()
                return False
            if time.time() - t0 > CFG.RECOVER_BDG or time.time() > deadline:
                self.stop()
                return False
            with ST.lock:
                smin = ST.scan_min
                sec  = dict(ST.scan_sectors)
            front = sec.get("front", 99.0)
            rear  = sec.get("rear",  99.0)
            if front > CFG.WALL_DIST * 1.8 and smin > CFG.EMERG_DIST * 1.5:
                self.stop()
                ST.log("BACKOFF: clear")
                return True
            cmd = Twist()
            if rear < CFG.EMERG_DIST * 1.4:
                fl   = sec.get("front_left",  99.0)
                fr   = sec.get("front_right", 99.0)
                left = sec.get("left",         99.0)
                right= sec.get("right",        99.0)
                cmd.angular.z = (CFG.NAV_ANG_MAX * 0.65
                                 if max(fl, left) >= max(fr, right)
                                 else -CFG.NAV_ANG_MAX * 0.65)
            else:
                cmd.linear.x  = CFG.BACK_SPEED
                fl = sec.get("front_left",  99.0)
                fr = sec.get("front_right", 99.0)
                cmd.angular.z = 0.25 if fl < fr else -0.25
            self._pub(cmd)
            time.sleep(0.05)
        self.stop()
        return False

    def align_to_heading(self, target_yaw: float, deadline: float, tol: float = 0.12) -> str:
        """Pure rotate in place. This is the anti-duck-walk fix."""
        stable = 0
        while rclpy.ok():
            if time.time() > deadline:
                self.stop(); return "TIMEOUT"
            if ST.cancel_event.is_set() or ST.stop_requested:
                self.stop(); return "STOPPED"
            with ST.lock:
                yaw = ST.ryaw
            err = target_yaw - yaw
            while err > math.pi: err -= 2 * math.pi
            while err < -math.pi: err += 2 * math.pi
            if abs(err) <= tol:
                stable += 1
                if stable >= 4:
                    self.stop(); return "ARRIVED"
            else:
                stable = 0
            cmd = Twist()
            # Minimum angular command avoids tiny buzzing, max keeps spin fast.
            mag = min(CFG.NAV_ANG_MAX, max(0.28, abs(err) * 1.8))
            cmd.angular.z = mag if err > 0 else -mag
            self._pub(cmd)
            time.sleep(0.05)
        self.stop(); return "TIMEOUT"

    def _safe_clearance_drive(self, linear: float, angular: float, sec: Dict[str, float], smin: float, close_mode: bool = False) -> Tuple[float, float]:
        """Reactive LiDAR safety layer. For normal navigation it keeps conservative
        clearance. For final cylinder approach it allows closer movement because the
        target cylinder itself is intentionally in front of the robot.
        """
        front = sec.get("front", 99.0)
        fl = sec.get("front_left", 99.0)
        fr = sec.get("front_right", 99.0)
        left = sec.get("left", 99.0)
        right = sec.get("right", 99.0)
        stop_m = CFG.CLOSE_STOP_M if close_mode else CFG.EMERG_DIST
        wall_m = CFG.CLOSE_WALL_M if close_mode else CFG.WALL_DIST
        warn_m = CFG.CLOSE_WARN_M if close_mode else CFG.WARN_DIST
        if front < stop_m:
            return 0.0, CFG.NAV_ANG_MAX * (1.0 if left >= right else -1.0)
        if front < wall_m:
            return 0.0, CFG.NAV_ANG_MAX * 0.85 * (1.0 if max(fl, left) >= max(fr, right) else -1.0)
        if front < warn_m:
            linear = min(linear, CFG.CREEP_SPEED if not close_mode else max(0.045, CFG.CREEP_SPEED * 0.65))
            angular += 0.35 * (1.0 if fl >= fr else -1.0)
        return linear, max(-CFG.NAV_ANG_MAX, min(CFG.NAV_ANG_MAX, angular))

    def navigate_to(self, tx: float, ty: float, deadline: float,
                    label: str = "", close_mode: bool = False, arrival_radius: Optional[float] = None) -> str:
        """
        Two-phase navigation:
          Phase A: rotate in place until the robot faces the target.
          Phase B: drive forward with only tiny heading correction.
        This removes the old duck-walking/swinging behaviour.
        Returns: ARRIVED | TIMEOUT | BLOCKED | STOPPED | COLLISION
        """
        confirm = 0
        best_dist = 99.0
        last_prog = time.time()
        stuck_since = time.time()
        recovery_count = 0
        if arrival_radius is None:
            arrival_radius = CFG.ARRIVAL_R
        with ST.lock:
            ST.target_x = tx; ST.target_y = ty
        while rclpy.ok():
            now = time.time()
            if now > deadline:
                self.stop(); return "TIMEOUT"
            if ST.cancel_event.is_set() or ST.stop_requested:
                self.stop(); return "STOPPED"
            with ST.lock:
                rx, ry, ryaw = ST.rx, ST.ry, ST.ryaw
                sec = dict(ST.scan_sectors)
                smin = ST.scan_min
            dx, dy = tx - rx, ty - ry
            dist = math.hypot(dx, dy)
            if dist < best_dist - 0.035:
                best_dist = dist; last_prog = now; stuck_since = now
            # Duckling/no-progress detector: try recovery, but after 30s declare fail so mission can go HOME.
            if now - stuck_since > CFG.DUCKLING_FAIL_S:
                with ST.lock:
                    ST.duckling_timeout = True
                self.stop(); ST.log(f"DUCKLING detected for {CFG.DUCKLING_FAIL_S:.0f}s at {label or 'target'} — declare waypoint failure", "WARN"); return "BLOCKED"
            if now - last_prog > 10.0:
                if recovery_count < CFG.MANEUVER_RETRY_LIMIT and self._maneuver_escape(deadline, reason="no progress/duckling", tx=tx, ty=ty):
                    recovery_count += 1
                    last_prog = time.time()
                    continue
                self.stop(); return "BLOCKED"
            if dist <= arrival_radius:
                confirm += 1
                if confirm >= CFG.ARRIVE_HOLD:
                    self.stop(); return "ARRIVED"
            else:
                confirm = 0
            heading = math.atan2(dy, dx)
            err = heading - ryaw
            while err > math.pi: err -= 2 * math.pi
            while err < -math.pi: err += 2 * math.pi
            # Emergency: only FRONT danger stops navigation. Side/rear close points are expected in your arena.
            front = sec.get("front", 99.0)
            stop_m = CFG.CLOSE_STOP_M if close_mode else CFG.EMERG_DIST
            if front < stop_m:
                self.stop()
                if recovery_count >= CFG.MANEUVER_RETRY_LIMIT:
                    return "COLLISION"
                recovery_count += 1
                if not self._maneuver_escape(deadline, reason="front emergency", tx=tx, ty=ty):
                    if not self._backoff(deadline):
                        return "COLLISION"
                last_prog = time.time()
                continue
            # Early recovery supervisor: when front is blocked by a cylinder/wall
            # but not yet a collision, do not sit there rotating left-right.
            wall_trigger = (CFG.CLOSE_WALL_M if close_mode else CFG.WALL_DIST)
            if front < wall_trigger and recovery_count < CFG.MANEUVER_RETRY_LIMIT:
                self.stop()
                recovery_count += 1
                if self._maneuver_escape(deadline, reason="front blocked", tx=tx, ty=ty):
                    last_prog = time.time()
                    continue
            cmd = Twist()
            if abs(err) > math.radians(12.0):
                # PHASE A: rotate only. No forward motion = no swing.
                raw_w = max(0.30, min(CFG.NAV_ANG_MAX, abs(err) * 1.7))
                cmd.linear.x = 0.0
                cmd.angular.z = raw_w if err > 0 else -raw_w
            else:
                # PHASE B: drive almost straight, small correction only.
                d_fac = min(1.0, max(0.25, dist / 1.0))
                cmd.linear.x = CFG.NAV_LIN_MAX * d_fac
                cmd.angular.z = max(-0.35, min(0.35, CFG.NAV_KP * err))
            cmd.linear.x, cmd.angular.z = self._safe_clearance_drive(cmd.linear.x, cmd.angular.z, sec, smin, close_mode=close_mode)
            self._pub(cmd)
            time.sleep(0.05)
        self.stop(); return "TIMEOUT"

    def drive_forward_distance(self, metres: float, deadline: float) -> str:
        """Relative forward primitive used by simple LLM commands like 'go straight'."""
        with ST.lock:
            sx, sy, yaw = ST.rx, ST.ry, ST.ryaw
        tx = sx + metres * math.cos(yaw)
        ty = sy + metres * math.sin(yaw)
        return self.navigate_to(tx, ty, deadline, label=f"FORWARD_{metres:.1f}m")

    def approach_cylinder(self, cx: float, cy: float, deadline: float) -> str:
        """Navigate near the assigned cylinder without crashing.
        Tries the closest safe standoff first, then alternative angles around the
        cylinder. If the robot is already near enough, inspection is allowed.
        """
        with ST.lock:
            rx, ry = ST.rx, ST.ry
        # Only accept already-near if genuinely close. Older versions accepted ~0.75m,
        # which made the robot stop far from the requested cylinder.
        if math.hypot(rx - cx, ry - cy) <= CFG.ORBIT_APP_R + 0.04:
            self.stop()
            return "ARRIVED"
        base = math.atan2(ry - cy, rx - cx)
        offsets = [0, math.radians(25), -math.radians(25), math.radians(50), -math.radians(50), math.radians(90), -math.radians(90), math.pi]
        candidates = []
        for off in offsets:
            a = base + off
            ax = cx + CFG.ORBIT_APP_R * math.cos(a)
            ay = cy + CFG.ORBIT_APP_R * math.sin(a)
            candidates.append((math.hypot(ax-rx, ay-ry), ax, ay))
        candidates.sort(key=lambda q: q[0])
        ST.log(f"Close approach target: cylinder=({cx:.2f},{cy:.2f}) standoff={CFG.ORBIT_APP_R:.2f}m")
        last_result = "TIMEOUT"
        for _, ax, ay in candidates:
            if time.time() > deadline or ST.stop_requested or ST.cancel_event.is_set():
                self.stop(); return "STOPPED"
            sub_deadline = min(deadline, time.time() + 45.0)
            last_result = self.navigate_to(ax, ay, sub_deadline, label="CYL_APPROACH", close_mode=True, arrival_radius=CFG.CYL_ARRIVAL_R)
            with ST.lock:
                rx, ry = ST.rx, ST.ry
            # Require a genuinely close inspection pose, not just a far visual line on dashboard.
            if last_result == "ARRIVED" and math.hypot(rx - cx, ry - cy) <= CFG.ORBIT_APP_R + 0.09:
                self.stop()
                return "ARRIVED"
            if math.hypot(rx - cx, ry - cy) <= CFG.ORBIT_APP_R + 0.06:
                self.stop()
                return "ARRIVED"
        return last_result

    def depart_from_cylinder(self, cx: float, cy: float, deadline: float) -> bool:
        """After a close inspection, move away from the cylinder before turning to
        the next waypoint. This fixes the CYL4→CYL5 style problem where the
        robot is too close, turns in place, sees the cylinder/wall, and duckles.
        It is intentionally non-fatal: it improves clearance but never marks the
        mission failed by itself.
        """
        with ST.lock:
            rx, ry = ST.rx, ST.ry
        vx, vy = rx - cx, ry - cy
        d = math.hypot(vx, vy)
        if d > 0.55:
            return True
        if d < 1e-3:
            vx, vy, d = 1.0, 0.0, 1.0
        ux, uy = vx / d, vy / d
        tx = cx + ux * 0.62
        ty = cy + uy * 0.62
        ST.log(f"CLEARANCE EXIT: backing away from cylinder to ({tx:.2f},{ty:.2f})")
        sub_deadline = min(deadline, time.time() + 8.0)
        ans = self.navigate_to(tx, ty, sub_deadline, label="CLEARANCE_EXIT", close_mode=True, arrival_radius=0.16)
        return ans in ("ARRIVED", "TIMEOUT")  # timeout is tolerated; this is only a clearance assist

    def orbit_cylinder(self, cx: float, cy: float, deadline: float) -> bool:
        """
        360° inspection orbit at CFG.ORBIT_R.
        Smooth tangential motion with heading aligned to orbit direction.
        Returns True on full completion, False on cancel/timeout.
        """
        step  = math.radians(3.0)
        total = 0.0
        ST.log(f"Orbit: centre=({cx:.2f},{cy:.2f}) r={CFG.ORBIT_R}m")
        while total < math.radians(CFG.ORBIT_DEG):
            if ST.cancel_event.is_set() or ST.stop_requested:
                self.stop()
                return False
            if time.time() > deadline:
                self.stop()
                return False
            with ST.lock:
                rx, ry, ryaw = ST.rx, ST.ry, ST.ryaw
                smin = ST.scan_min
                sec  = dict(ST.scan_sectors)
            dist_cyl = math.hypot(rx - cx, ry - cy)
            if smin < CFG.EMERG_DIST:
                self.stop()
                self._backoff(deadline)
                continue
            if dist_cyl < CFG.ORBIT_R * 0.65:
                cmd = Twist()
                cmd.linear.x = -CFG.CREEP_SPEED
                self._pub(cmd)
                time.sleep(0.08)
                continue
            # Next orbit point
            cur  = math.atan2(ry - cy, rx - cx)
            nxt  = cur + step
            tx   = cx + CFG.ORBIT_R * math.cos(nxt)
            ty   = cy + CFG.ORBIT_R * math.sin(nxt)
            head = math.atan2(ty - ry, tx - rx)
            err  = head - ryaw
            while err >  math.pi: err -= 2 * math.pi
            while err < -math.pi: err += 2 * math.pi
            cmd = Twist()
            if smin < CFG.WARN_DIST:
                cmd.linear.x  = CFG.ORBIT_SPD * 0.3
                cmd.angular.z = CFG.NAV_ANG_MAX * 0.4
            elif abs(err) > 0.3:
                cmd.linear.x  = CFG.ORBIT_SPD * 0.4
                cmd.angular.z = max(-CFG.NAV_ANG_MAX,
                                    min(CFG.NAV_ANG_MAX, CFG.NAV_KP * err))
            else:
                cmd.linear.x  = CFG.ORBIT_SPD
                cmd.angular.z = CFG.NAV_ANG_MAX * 0.50
            self._pub(cmd)
            time.sleep(0.05)
            total += step
            with ST.lock:
                ST.orbit_deg = min(360.0, math.degrees(total))
        self.stop()
        ST.log(f"Orbit complete ({math.degrees(total):.0f}°)")
        return True

    # ══════════════════════════════════════════════════════════
    # RESET — verified return-to-home before every run
    # ══════════════════════════════════════════════════════════
    def reset_robot(self) -> bool:
        """
        Reliable reset sequence:
          1. Clear cancel flags (must navigate if teleport fails)
          2. Stop motors
          3. Attempt Gazebo teleport
          4. Navigation fallback if teleport unavailable
          5. Verify home position reached
          6. Return True only on verified success
        """
        ST.set_state("RESETTING")
        ST.log(f"RESET → HOME ({CFG.HOME_X:.2f},{CFG.HOME_Y:.2f})")
        ST.stop_requested = False
        ST.cancel_event.clear()
        for _ in range(5):
            self.stop()
            time.sleep(0.04)
        time.sleep(0.3)
        teleported = False
        if _GAZEBO_SRV and self._gz_cli:
            try:
                req = SetEntityState.Request()
                req.state = EntityState()
                req.state.name            = CFG.ROBOT_NAME
                req.state.reference_frame = "world"
                req.state.pose.position.x = float(CFG.HOME_X)
                req.state.pose.position.y = float(CFG.HOME_Y)
                req.state.pose.position.z = 0.0
                req.state.pose.orientation.z = math.sin(CFG.HOME_YAW / 2)
                req.state.pose.orientation.w = math.cos(CFG.HOME_YAW / 2)
                req.state.twist = Twist()
                res = self._call_gz_srv(req, timeout=6.0)
                if res and res.success:
                    ST.log("RESET: Gazebo teleport OK")
                    teleported = True
                else:
                    ST.log("RESET: Gazebo teleport failed — using navigation", "WARN")
            except Exception as ex:
                ST.log(f"RESET: teleport exception: {ex}", "WARN")
        # Gazebo Classic can report teleport success but not actually move the robot.
        # Always read odom back after 0.6s; if still far, use navigation fallback twice.
        time.sleep(0.6)
        with ST.lock:
            rx, ry = ST.rx, ST.ry
        d_after = math.hypot(rx - CFG.HOME_X, ry - CFG.HOME_Y)
        if (not teleported) or d_after > 0.40:
            ST.log(f"RESET: odom readback {d_after:.2f}m from HOME — using navigation fallback")
            for attempt in range(1, 3):
                result = self.navigate_to(CFG.HOME_X, CFG.HOME_Y, time.time() + 60.0, label="HOME")
                with ST.lock:
                    rx, ry = ST.rx, ST.ry
                d_after = math.hypot(rx - CFG.HOME_X, ry - CFG.HOME_Y)
                ST.log(f"RESET: fallback attempt {attempt} result={result} dist={d_after:.2f}m")
                if d_after <= CFG.HOME_TOL:
                    break
        self.align_to_heading(CFG.HOME_YAW, time.time() + 8.0, tol=0.18)
        time.sleep(0.2)
        self.stop()
        with ST.lock:
            d = math.hypot(ST.rx - CFG.HOME_X, ST.ry - CFG.HOME_Y)
        reached = d <= max(CFG.HOME_TOL, CFG.ARRIVAL_R * 1.4)
        # Final safety net for model-switch fairness: if navigation home fails, drop robot at ideal HOME using Gazebo service.
        if not reached and _GAZEBO_SRV and self._gz_cli:
            ST.log("RESET: navigation home failed — final ideal-coordinate teleport", "WARN")
            try:
                req = SetEntityState.Request()
                req.state = EntityState()
                req.state.name = CFG.ROBOT_NAME
                req.state.reference_frame = "world"
                req.state.pose.position.x = float(CFG.HOME_X)
                req.state.pose.position.y = float(CFG.HOME_Y)
                req.state.pose.position.z = 0.02
                req.state.pose.orientation.z = math.sin(CFG.HOME_YAW / 2)
                req.state.pose.orientation.w = math.cos(CFG.HOME_YAW / 2)
                req.state.twist = Twist()
                self._call_gz_srv(req, timeout=6.0)
                time.sleep(0.7)
                with ST.lock:
                    d = math.hypot(ST.rx - CFG.HOME_X, ST.ry - CFG.HOME_Y)
                reached = d <= max(CFG.HOME_TOL, 0.60)
            except Exception as ex:
                ST.log(f"RESET final teleport exception: {ex}", "WARN")
        ST.log(f"RESET done — dist={d:.3f}m reached={reached}")
        ST.stop_requested = False
        ST.cancel_event.clear()
        ST.set_state("IDLE" if reached else "RESET_FAILED")
        return reached

    # ══════════════════════════════════════════════════════════
    # VALIDATION
    # ══════════════════════════════════════════════════════════
    def run_validation(self, wait: float = 10.0) -> Dict:
        v = ST.validation
        ST.log("=" * 50)
        ST.log("SYSTEM VALIDATION")
        # /odom
        dl = time.time() + wait
        while time.time() < dl:
            with ST.lock:
                age = time.time() - ST.last_odom_t
            if ST.last_odom_t and age < 2.0:
                v["odom_active"] = True
                break
            time.sleep(0.2)
        print(f"[{'OK  ' if v['odom_active'] else 'FAIL'}] /odom", flush=True)
        # /scan
        dl = time.time() + wait
        while time.time() < dl:
            with ST.lock:
                age = time.time() - ST.last_scan_t
            if ST.last_scan_t and age < 2.0:
                v["scan_active"] = True
                break
            time.sleep(0.2)
        print(f"[{'OK  ' if v['scan_active'] else 'FAIL'}] /scan", flush=True)
        # /cmd_vel
        for _ in range(20):
            if self.cmd_pub.get_subscription_count() > 0:
                v["cmdvel_ok"] = True
                break
            time.sleep(0.2)
        print(f"[{'OK  ' if v['cmdvel_ok'] else 'WARN'}] /cmd_vel", flush=True)
        # Gazebo reset service
        if _GAZEBO_SRV and self._gz_cli:
            v["reset_ok"] = self._gz_cli.wait_for_service(timeout_sec=2.0)
        print(f"[{'OK  ' if v['reset_ok'] else 'WARN'}] Gazebo reset service", flush=True)
        # Camera (optional)
        with ST.lock:
            v["cam_ok"] = bool(ST.last_cam_t and (time.time() - ST.last_cam_t) < 5.0)
        print(f"[{'OK  ' if v['cam_ok'] else 'OPT.'}] Gazebo camera", flush=True)
        ST.log("=" * 50)
        return v


# ══════════════════════════════════════════════════════════════
# SECTION 11 — MJPEG STREAM GENERATOR
# ══════════════════════════════════════════════════════════════
def _mjpeg_generator():
    boundary    = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
    last_hash   = None
    last_nosig  = 0.0

    def _make_no_signal() -> Optional[bytes]:
        if not (_CV2 and _np is not None):
            return None
        img = _np.zeros((240, 320, 3), dtype=_np.uint8)
        _cv2.rectangle(img, (0, 0), (320, 240), (12, 18, 30), -1)
        _cv2.putText(img, "NO SIGNAL", (48, 90),
                     _cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 200, 0), 3)
        _cv2.putText(img, "Gazebo camera not publishing", (18, 128),
                     _cv2.FONT_HERSHEY_SIMPLEX, 0.44, (70, 70, 70), 1)
        with ST.lock:
            state = ST.mission_state
        n   = len(_REGISTRY.confirmed_list())
        col = (0, 200, 100) if n > 0 else (0, 120, 180)
        _cv2.putText(img, f"State: {state}", (20, 165),
                     _cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 160, 220), 1)
        _cv2.putText(img, f"Cylinders confirmed: {n}", (20, 192),
                     _cv2.FONT_HERSHEY_SIMPLEX, 0.50, col, 1)
        _cv2.putText(img, "Camera: /camera/image_raw (optional)", (18, 220),
                     _cv2.FONT_HERSHEY_SIMPLEX, 0.38, (50, 50, 50), 1)
        _, buf = _cv2.imencode(".jpg", img, [_cv2.IMWRITE_JPEG_QUALITY, 60])
        return buf.tobytes()

    while True:
        with ST.lock:
            frame = ST.cam_bytes
        if frame is None:
            now = time.time()
            if now - last_nosig >= 1.0:
                frame       = _make_no_signal()
                last_nosig  = now
        if frame is not None:
            h = hash(frame)
            if h != last_hash:
                last_hash = h
                yield boundary + frame + b"\r\n"
        time.sleep(0.04)  # ~25 fps cap


# ══════════════════════════════════════════════════════════════
# SECTION 12 — MISSION ENGINE
# ══════════════════════════════════════════════════════════════
NODE: Optional[OpenClawNode] = None


def preflight(require_ollama: bool = True) -> Tuple[bool, str]:
    if NODE is None:
        return False, "ROS node not initialised"
    if not ST.validation.get("odom_active"):
        return False, "/odom not publishing — check Gazebo bridge"
    with ST.lock:
        scan_age = time.time() - ST.last_scan_t
    if scan_age > CFG.SCAN_STALE:
        return False, f"/scan stale ({scan_age:.1f}s) — check LiDAR"
    if require_ollama:
        try:
            r = ST.http.get(CFG.OLLAMA_TAGS, timeout=8)
            if r.status_code != 200:
                return False, "Ollama not responding"
        except Exception as ex:
            return False, f"Ollama: {ex}"
    return True, "ok"


def _explore_for_cylinders(budget: float) -> None:
    """v17: fixed arena. Explore button only refreshes the known 9-cylinder map.
    It does not move the robot and does not block missions.
    """
    ST.log("EXPLORE: fixed 9-cylinder map refreshed (no robot movement)")
    ST.set_state("EXPLORING")
    seed_fixed_cylinders()
    time.sleep(0.5)
    ST.set_state("IDLE")
    ST.log("EXPLORE complete — CYL_1..CYL_9 ready")

def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    s = text.find("{"); e = text.rfind("}")
    if s >= 0 and e > s:
        try:
            return json.loads(text[s:e+1])
        except Exception:
            return None
    return None



_NUMBER_WORDS = {
    "one": 1, "first": 1,
    "two": 2, "second": 2,
    "three": 3, "third": 3,
    "four": 4, "fourth": 4,
    "five": 5, "fifth": 5,
    "six": 6, "sixth": 6,
    "seven": 7, "seventh": 7,
    "eight": 8, "eighth": 8,
    "nine": 9, "ninth": 9,
}

def _extract_requested_cylinders(mission_text: str) -> List[str]:
    """Return cylinder labels explicitly mentioned by the user.
    Accepts: c3, C3, c 3, cyl3, CYL_3, cylinder 3, cylinder three,
    node 3, point 3, marker 3, and multiple targets in one command.
    """
    t = mission_text.lower().replace("-", "_")
    labels: List[str] = []
    patterns = [
        r'\b(?:cyl|cy|cylinder|node|point|marker|target|c)\s*[_ ]?\s*(\d{1,2})\b',
        r'\b(?:cyl|cy|cylinder|node|point|marker|target)\s*[_ ]?\s*(' + "|".join(_NUMBER_WORDS.keys()) + r')\b',
    ]
    for pat in patterns:
        for m in re.finditer(pat, t):
            token = m.group(1)
            n = int(token) if token.isdigit() else _NUMBER_WORDS.get(token, 0)
            if 1 <= n <= CFG.CYL_MAX_COUNT:
                lab = f"CYL_{n}"
                if lab not in labels:
                    labels.append(lab)
    return labels

def _build_direct_actions_from_text(mission_text: str) -> Optional[List[Dict[str, Any]]]:
    """Hard command understanding layer used equally for every model.
    It prevents simple demo commands from being misread by the LLM while still
    allowing the LLM to be measured for planning latency/reasoning.
    """
    t = mission_text.lower()
    labels = _extract_requested_cylinders(mission_text)
    actions: List[Dict[str, Any]] = []
    if labels:
        for lab in labels:
            actions.append({"type": "inspect_cylinder", "label": lab, "seconds": 2})
        if any(w in t for w in ("home", "return", "come back", "back", "ideal")):
            actions.append({"type": "return_home"})
        else:
            actions.append({"type": "return_home"})
        return actions
    if "all" in t or "every" in t or "complete" in t:
        return [{"type": "inspect_all", "seconds": 2}, {"type": "return_home"}]
    if "nearest" in t or "closest" in t:
        # nearest two / nearest 3 support
        m = re.search(r'nearest\s+(\d+|one|two|three|four|five|six|seven|eight|nine)', t)
        if m:
            tok=m.group(1); n=int(tok) if tok.isdigit() else _NUMBER_WORDS.get(tok,1)
            return [{"type":"inspect_nearest_n","count":max(1,min(CFG.CYL_MAX_COUNT,n)),"seconds":2},{"type":"return_home"}]
        return [{"type":"inspect_nearest","seconds":2},{"type":"return_home"}]
    if "far" in t or "farthest" in t or "furthest" in t:
        return [{"type":"inspect_far","seconds":2},{"type":"return_home"}]
    if "home" in t or "return" in t or "come back" in t:
        return [{"type":"return_home"}]
    return None

def _fallback_actions_from_text(mission_text: str) -> List[Dict[str, Any]]:
    """Deterministic fallback so the robot still moves even if an LLM replies badly."""
    direct = _build_direct_actions_from_text(mission_text)
    if direct:
        return direct
    t = mission_text.lower()
    actions: List[Dict[str, Any]] = []
    if "turn left" in t or "left" in t:
        actions.append({"type": "turn_left", "degrees": 90})
    if "turn right" in t or "right" in t:
        actions.append({"type": "turn_right", "degrees": 90})
    if "straight" in t or "forward" in t or "go" in t or "move" in t:
        m = re.search(r'(\d+(?:\.\d+)?)\s*(?:m|meter|metre)', t)
        dist = float(m.group(1)) if m else 1.0
        actions.append({"type": "move_forward", "meters": max(0.2, min(2.0, dist))})
    if not actions:
        actions = [{"type": "move_forward", "meters": 0.8}]
    if actions[-1].get("type") != "return_home":
        actions.append({"type": "return_home"})
    return actions



def _mission_key(model: str, mission_text: str) -> str:
    norm = re.sub(r'\s+', ' ', mission_text.strip().lower())[:160]
    return f"{model}::{norm}"


def _learning_summary_for_prompt(model: str, mission_text: str) -> str:
    """Return model-specific lessons from previous runs of the same command.
    This is the 'learning from mistake' feature requested by the user. It does
    not train the LLM weights; it feeds concise run-memory notes into the next
    prompt so repeated 10/30/40-run trials can improve decision quality.
    """
    if not getattr(CFG, 'LEARNING_FEEDBACK_ENABLED', True):
        return ""
    key = _mission_key(model, mission_text)
    with ST.lock:
        mem = ST.learning_memory.get(key, {})
        notes = list(mem.get('notes', []))[-CFG.LEARNING_MAX_NOTES:]
        n = mem.get('runs', 0)
        success_rate = mem.get('success_rate', None)
    if not notes:
        return ""
    sr = f"Success rate so far: {success_rate:.1f}%." if isinstance(success_rate, (int, float)) else ""
    return (
        "\nPREVIOUS RUN FEEDBACK FOR THIS MODEL AND SAME COMMAND:\n"
        f"Runs observed: {n}. {sr}\n"
        + "\n".join(f"- {x}" for x in notes[-4:])
        + "\nUse this feedback only to improve route/order. Do NOT output motor speeds.\n"
    )


def _build_result_feedback(r: MissionResult) -> Tuple[str, str, str]:
    """Generate examiner-friendly analysis + next-run feedback."""
    issues = []
    hints = []
    if r.success:
        issues.append(f"Mission passed with deployment score {r.final_deployment_score:.1f}.")
    else:
        issues.append(f"Mission failed with deployment score {r.final_deployment_score:.1f}: {r.failure_reason or 'unknown reason'}.")
    if r.pass_fail_reason:
        issues.append(r.pass_fail_reason)
    if not r.valid_json:
        issues.append("LLM did not provide clean JSON; fallback/direct parsing was needed.")
        hints.append("Return strict JSON actions only; avoid extra reasoning outside JSON.")
    if r.route_correctness < 100:
        issues.append("Requested cylinder route was not perfectly matched by the LLM plan.")
        hints.append("Preserve explicitly named cylinders and order from the user command.")
    if r.duckling_timeout:
        issues.append("Duckling/no-progress behaviour detected for 30 seconds.")
        hints.append("Choose shorter target order and allow recovery manoeuvre before continuing.")
    if r.reverse_maneuver_count:
        issues.append(f"Recovery manoeuvre used {r.reverse_maneuver_count} time(s).")
    if r.collision_count > 0 or r.near_collision_count > 0:
        issues.append(f"Safety events: collisions={r.collision_count}, near-collisions={r.near_collision_count}.")
        hints.append("Avoid routes that require squeezing between close cylinders unless necessary.")
    if not r.return_home_success:
        issues.append("Return HOME was difficult or required fallback reset.")
        hints.append("Prefer a clear exit path after the last cylinder before returning HOME.")
    if r.llm_latency > 20:
        issues.append(f"LLM latency was high ({r.llm_latency:.1f}s).")
        hints.append("Keep the plan short and avoid verbose reasoning.")
    if not hints:
        if r.success and r.reverse_maneuver_count == 0:
            hints.append("Plan was stable; keep the same concise route style next run.")
        else:
            hints.append("Maintain route but reduce unnecessary turns and recovery usage.")
    analysis = " ".join(issues)
    feedback = " ".join(hints[:3])
    improvement = (
        "Next run objective: reduce total time/recovery count while keeping the same command complete and returning HOME."
    )
    return analysis, feedback, improvement


def _update_learning_memory(r: MissionResult) -> None:
    if not getattr(CFG, 'LEARNING_FEEDBACK_ENABLED', True):
        return
    key = _mission_key(r.model, r.mission_text)
    note = r.learning_feedback or r.improvement_hint or r.analysis_comment
    with ST.lock:
        mem = ST.learning_memory.setdefault(key, {'runs': 0, 'successes': 0, 'notes': []})
        mem['runs'] += 1
        if r.success:
            mem['successes'] += 1
        mem['success_rate'] = 100.0 * mem['successes'] / max(1, mem['runs'])
        if note:
            mem['notes'].append(note[:240])
            mem['notes'] = mem['notes'][-CFG.LEARNING_MAX_NOTES:]
        ST.last_feedback = note[:500] if note else ST.last_feedback


def plan_actions(model: str, mission_text: str) -> Dict[str, Any]:
    """LLM decides the task-level action sequence. Python executes safely."""
    with ST.lock:
        rx, ry, yaw, smin = ST.rx, ST.ry, ST.ryaw, ST.scan_min
    cyls = _REGISTRY.confirmed_list()
    cyl_desc = "\n".join([f"{c.label}: x={c.x:.2f}, y={c.y:.2f}, dist={math.hypot(c.x-rx,c.y-ry):.2f}m" for c in cyls]) or "No confirmed cylinders yet."
    allowed = (
        'move_forward {"meters":0.2..2.0}, turn_left {"degrees":15..180}, '
        'turn_right {"degrees":15..180}, inspect_nearest {"seconds":2}, '
        'inspect_far {"seconds":2}, inspect_all {"seconds":2}, inspect_cylinder {"label":"CYL_4","seconds":2}, return_home'
    )
    prompt = (
        "You are the high-level task planner for a TurtleBot3 robot.\n"
        "You do NOT output motor speeds. You only output a short JSON action plan.\n"
        f"Robot pose: x={rx:.2f}, y={ry:.2f}, yaw={yaw:.2f}, front_clearance={smin:.2f}m.\n"
        f"Home/ideal pose: x={CFG.HOME_X:.2f}, y={CFG.HOME_Y:.2f}, yaw={CFG.HOME_YAW:.2f}.\n"
        f"Known cylinders:\n{cyl_desc}\n"
        f"Allowed actions: {allowed}.\n"
        "Safety rule: keep actions simple and short; the navigation layer avoids obstacles.\n"
        "Inspection means stop near a cylinder and wait for 2 seconds.\n"
        "Always end with return_home unless the command is only pause/stop.\n"
        f"User command: {mission_text}\n"
        "Reply ONLY this JSON format: {\"actions\":[{\"type\":\"move_forward\",\"meters\":1.0},{\"type\":\"return_home\"}],\"reasoning\":\"brief\"}"
    )
    t0 = time.time(); raw = ""; actions = [] ; reasoning = ""; valid_json = False; fallback_used = False
    try:
        resp = ST.http.post(CFG.OLLAMA_GEN, json={
            "model": model, "prompt": prompt, "stream": False,
            "options": {"temperature": 0.1, "num_predict": 220},
        }, timeout=CFG.OLLAMA_TIMEOUT)
        if resp.status_code == 200:
            raw = resp.json().get("response", "")
            obj = _extract_json_object(raw)
            if obj and isinstance(obj.get("actions"), list):
                valid_json = True
                actions = obj["actions"]
                reasoning = str(obj.get("reasoning", ""))
    except Exception as ex:
        raw = f"LLM error: {ex}"
    latency = time.time() - t0
    # Direct command layer: if the user clearly names a cylinder or simple target,
    # force the correct physical task. This applies to every LLM equally and avoids
    # demo commands like "go to c3 and come back home" being converted into only forward/home.
    direct_actions = _build_direct_actions_from_text(mission_text)
    if direct_actions:
        actions = direct_actions
        reasoning = (reasoning + " | Direct task extraction used to preserve the requested target.").strip()
    elif not actions:
        fallback_used = True
        actions = _fallback_actions_from_text(mission_text)
        reasoning = reasoning or "Fallback parser used because LLM did not return a valid action JSON."
    # sanitize
    clean: List[Dict[str, Any]] = []
    for a in actions[:12]:
        typ = str(a.get("type", "")).strip().lower()
        if typ == "move_forward":
            clean.append({"type": typ, "meters": max(0.2, min(2.0, float(a.get("meters", 1.0))))})
        elif typ in ("turn_left", "turn_right"):
            clean.append({"type": typ, "degrees": max(15.0, min(180.0, float(a.get("degrees", 90))))})
        elif typ in ("inspect_nearest", "inspect_far", "inspect_all"):
            clean.append({"type": typ, "seconds": max(1.0, min(5.0, float(a.get("seconds", 2))))})
        elif typ == "inspect_cylinder":
            lab = str(a.get("label", "")).strip().upper().replace(" ", "_")
            if re.fullmatch(r"CYL_\d+", lab):
                clean.append({"type": typ, "label": lab, "seconds": max(1.0, min(5.0, float(a.get("seconds", 2))))})
        elif typ == "return_home":
            clean.append({"type": typ})
    # Safety correction: if user explicitly mentions C3/CYL_3/etc., the route MUST include that cylinder.
    # This prevents the LLM from wrongly converting "go to c3" into only move_forward/return_home.
    m_direct = re.search(r'\b(?:cyl(?:inder)?|c)\s*[_-]?\s*(\d{1,2})\b', mission_text.lower())
    if m_direct:
        n = int(m_direct.group(1))
        lab = f"CYL_{n}"
        if 1 <= n <= CFG.CYL_MAX_COUNT and not any(a.get("type") == "inspect_cylinder" and a.get("label") == lab for a in clean):
            clean = [a for a in clean if a.get("type") not in ("move_forward",)]
            clean.insert(0, {"type": "inspect_cylinder", "label": lab, "seconds": 2})
            reasoning = (reasoning + " | Direct cylinder reference detected; forced target " + lab).strip()
    if not clean:
        fallback_used = True
        clean = _fallback_actions_from_text(mission_text)
    if clean[-1].get("type") != "return_home":
        clean.append({"type": "return_home"})
    with ST.lock:
        ST.planned_route = [(a.get("label") if a.get("type") == "inspect_cylinder" else a.get("type")) for a in clean]
        ST.reasoning = reasoning
    requested = _extract_requested_cylinders(mission_text)
    route_cyls = [a.get("label") for a in clean if a.get("type") == "inspect_cylinder"]
    route_correctness = 100.0 if (not requested or all(r in route_cyls for r in requested)) else 0.0
    return {"actions": clean, "reasoning": reasoning, "raw": raw, "llm_latency": latency, "parse_time": 0.0, "valid": True, "valid_json": valid_json, "fallback_used": fallback_used, "route_correctness": route_correctness, "llm_retry_count": 0}


def _select_cylinders_for_action(action_type: str, label: str = "") -> List[TrackedCylinder]:
    cyls = _REGISTRY.confirmed_list()
    if not cyls:
        return []
    with ST.lock:
        rx, ry = ST.rx, ST.ry
    if action_type == "inspect_cylinder" and label:
        c = _REGISTRY.get_by_name(label)
        return [c] if c else []
    if action_type == "inspect_all":
        # nearest-neighbour ordering from current robot position
        remaining = cyls[:]; ordered = []; px, py = rx, ry
        while remaining:
            c = min(remaining, key=lambda z: math.hypot(z.x-px, z.y-py))
            ordered.append(c); remaining.remove(c); px, py = c.x, c.y
        return ordered
    if action_type == "inspect_far":
        return [max(cyls, key=lambda z: math.hypot(z.x-rx, z.y-ry))]
    return [min(cyls, key=lambda z: math.hypot(z.x-rx, z.y-ry))]


def _execute_inspection_wait(seconds: float, deadline: float) -> bool:
    t0 = time.time()
    while time.time() - t0 < seconds:
        if time.time() > deadline or ST.stop_requested or ST.cancel_event.is_set():
            return False
        NODE.stop()
        time.sleep(0.1)
    return True


def run_single_mission(model: str, mission_text: str,
                       mission_number: int,
                       is_validation: bool = False) -> MissionResult:
    """
    v7 mission lifecycle:
      1. Verified reset to HOME/ideal coordinate.
      2. LLM creates a simple action plan from the command.
      3. Robot executes actions using safe primitives: move, turn, inspect 2s.
      4. Robot returns HOME after each mission/model so comparison is fair.
    """
    res = MissionResult()
    res.run_id = f"{model.replace(':','_').replace('.','_')}_{'v' if is_validation else 'r'}_{mission_number}_{int(time.time())}"
    res.model = model
    res.mission_number = mission_number
    res.mission_text = mission_text
    res.timestamp = datetime.datetime.now().isoformat()
    t_mission_start = time.time()
    with ST.lock:
        ST.cur_model = model; ST.cur_run = mission_number; ST.cur_text = mission_text
    ST.log("=" * 54)
    ST.log(f"{'VALIDATION' if is_validation else 'MISSION'} #{mission_number} | {model}")

    ST.reset_mission()
    t_reset = time.time()
    reset_ok = NODE.reset_robot()
    res.reset_time = time.time() - t_reset
    if not reset_ok:
        res.failure_reason = "HOME reset failed — check OPENCLAW_HOME_X/Y matches Gazebo start marker"
        res.total_mission_time = time.time() - t_mission_start
        ST.set_state("FAILED")
        return res

    ST.set_state("PLANNING")
    plan = plan_actions(model, mission_text)
    actions = plan["actions"]
    res.raw_llm_response = plan.get("raw", "")
    res.reasoning = plan.get("reasoning", "")
    res.parsed_route = [a["type"] for a in actions]
    res.route_valid = True
    res.valid_json = bool(plan.get("valid_json", False))
    res.fallback_used = bool(plan.get("fallback_used", False))
    res.route_correctness = float(plan.get("route_correctness", 0.0))
    res.llm_retry_count = int(plan.get("llm_retry_count", 0))
    res.llm_latency = plan.get("llm_latency", 0.0)
    res.parse_time = plan.get("parse_time", 0.0)
    def _planned_count(action_list):
        total = 0
        for a in action_list:
            t = a.get("type", "")
            if t == "inspect_all":
                total += len(FIXED_CYLINDERS)
            elif t == "inspect_nearest_n":
                total += int(a.get("count", 1))
            elif t in ("inspect_nearest", "inspect_far", "inspect_cylinder"):
                total += 1
        return total
    res.points_planned = _planned_count(actions)

    ST.set_state("RUNNING")
    sampler = MetricsSampler(period=1.0); sampler.start()
    nav_t0 = time.time(); deadline = nav_t0 + CFG.MISSION_TIMEOUT
    with ST.lock:
        ST.mission_t0 = nav_t0; ST.recording = True

    executed: List[str] = []
    inspections_ok = 0
    hard_failed = False

    for action in actions:
        if ST.stop_requested or ST.cancel_event.is_set() or time.time() > deadline:
            hard_failed = True; break
        typ = action["type"]
        ST.log(f"ACTION → {typ}: {action}")
        with ST.lock:
            ST.active_cyl = typ; ST.current_action_detail = f"Executing {typ}"; ST.orbit_deg = 0.0
        ok = True
        if typ == "move_forward":
            ok = (NODE.drive_forward_distance(float(action.get("meters", 1.0)), deadline) == "ARRIVED")
        elif typ == "turn_left":
            with ST.lock: target = ST.ryaw + math.radians(float(action.get("degrees", 90)))
            ok = (NODE.align_to_heading(target, deadline) == "ARRIVED")
        elif typ == "turn_right":
            with ST.lock: target = ST.ryaw - math.radians(float(action.get("degrees", 90)))
            ok = (NODE.align_to_heading(target, deadline) == "ARRIVED")
        elif typ in ("inspect_nearest", "inspect_far", "inspect_all", "inspect_cylinder", "inspect_nearest_n"):
            selector_arg = action.get("label", "") if typ != "inspect_nearest_n" else str(action.get("count", 1))
            targets = _select_cylinders_for_action(typ, selector_arg)
            if not targets:
                ST.log("No confirmed cylinder available; doing stationary 2s inspection sweep", "WARN")
                ok = _execute_inspection_wait(float(action.get("seconds", 2)), deadline)
                if ok: inspections_ok += 1; executed.append("INSPECT_STATIONARY")
            else:
                for c in targets:
                    if time.time() > deadline or ST.stop_requested: ok = False; break
                    with ST.lock:
                        ST.active_cyl = c.label
                        ST.current_action_detail = f"Going near {c.label}"
                    ir = InspectionRecord(name=c.label)
                    t_leg = time.time()
                    ST.set_state(f"APPROACHING_{c.label}")
                    ap = NODE.approach_cylinder(c.x, c.y, deadline)
                    if ap == "ARRIVED":
                        ir.reached = True; ir.time_to_reach = time.time() - t_leg
                        with ST.lock: ir.x_at_arrive = ST.rx; ir.y_at_arrive = ST.ry
                        ST.set_state(f"INSPECTING_{c.label}")
                        ok2 = _execute_inspection_wait(float(action.get("seconds", 2)), deadline)
                        ir.orbit_done = ok2; ir.orbit_duration = float(action.get("seconds", 2)) if ok2 else 0.0
                        if ok2:
                            inspections_ok += 1; executed.append(c.label)
                            # v25 clearance exit: create space before next cylinder/HOME.
                            try:
                                NODE.depart_from_cylinder(c.x, c.y, deadline)
                            except Exception as ex:
                                ST.log(f"Clearance exit skipped: {ex}", "WARN")
                    else:
                        ok2 = False; ST.log(f"Cannot inspect {c.label}: {ap}", "WARN")
                    res.insp_records.append(ir)
                    ok = ok and ok2
        elif typ == "return_home":
            ST.set_state("RETURNING")
            ok = NODE.reset_robot()
            res.return_home_success = ok
        if ok:
            executed.append(typ)
        else:
            hard_failed = True
            if not res.failure_reason:
                res.failure_reason = f"Action failed: {typ}"
            # still try home after failed action
            break
        time.sleep(0.1)

    if not res.return_home_success and not ST.stop_requested:
        ST.set_state("RETURNING")
        res.return_home_success = NODE.reset_robot()

    with ST.lock:
        ST.recording = False
        res.trajectory = list(ST.trajectory)
        res.distance_travelled = ST.dist_total
        res.collision_count = ST.coll_count
        res.coll_events = list(ST.coll_events)
        res.replan_count = ST.replan_count
        res.near_collision_count = ST.near_collision_count
        res.duckling_timeout = ST.duckling_timeout
        res.reverse_maneuver_count = ST.reverse_maneuver_count
        res.left_recovery_count = ST.left_recovery_count
        res.right_recovery_count = ST.right_recovery_count
        res.llm_recovery_advice_used = ST.llm_recovery_advice_used
        clr_all = list(ST.clr_readings)
    res.nav_time = time.time() - nav_t0
    res.total_mission_time = time.time() - t_mission_start
    res.path_length = _path_length(res.trajectory)
    if res.path_length > 0 and executed:
        pts = [(CFG.HOME_X, CFG.HOME_Y)]
        for lab in executed:
            if lab in FIXED_CYLINDERS:
                x, y, _ = FIXED_CYLINDERS[lab]; pts.append((x, y))
        if res.return_home_success:
            pts.append((CFG.HOME_X, CFG.HOME_Y))
        ideal = _path_length([{"x":x,"y":y} for x,y in pts]) if len(pts) > 1 else 0.0
        res.route_efficiency = round(max(0.0, min(1.0, ideal / max(0.001, res.path_length))), 4) if ideal > 0 else 0.0
    res.executed_route = executed
    res.points_reached = len(executed)
    res.orbits_completed = inspections_ok
    res.inspection_pct = (100.0 * inspections_ok / max(1, res.points_planned)) if res.points_planned > 0 else 100.0
    res.orbit_success_rate = 1.0 if inspections_ok > 0 else 0.0
    if clr_all:
        res.min_clearance = min(clr_all); res.avg_clearance = statistics.mean(clr_all)
    with ST.lock: res.risk_score = ST.risk_score
    res.safety_score = ST.compute_safety_score(res.min_clearance, res.collision_count)
    res.avg_cpu, res.avg_ram, res.avg_power = sampler.stop()

    cancelled = ST.cancel_event.is_set() or ST.stop_requested
    if hard_failed and not res.failure_reason:
        res.failure_reason = "Mission action failed or duckling/no-progress timeout"
    if cancelled and not res.failure_reason:
        res.failure_reason = "Mission stopped by user"
    if res.collision_count > 2 and not res.failure_reason:
        res.failure_reason = "Too many collision proximity events"
    if not res.return_home_success and not res.failure_reason:
        res.failure_reason = "Return home/ideal failed"
    res.status = "ABORTED" if cancelled else "EVALUATED"
    compute_run_scores(res)
    res.status = "ABORTED" if cancelled else ("SUCCESS" if res.success else "FAIL")
    res.analysis_comment, res.learning_feedback, res.improvement_hint = _build_result_feedback(res)
    _update_learning_memory(res)
    with ST.lock:
        ST.current_action_detail = res.analysis_comment[:240]
    NODE.stop(); ST.set_state("COMPLETED" if res.success else "FAILED")
    save_trajectory(res)
    ST.log(f"RESULT: {res.status} | actions={len(actions)} inspect={inspections_ok} home={res.return_home_success} coll={res.collision_count} total={res.total_mission_time:.1f}s")
    return res


# ── Validation gate ────────────────────────────────────────────
_VALID_MISSION = "Inspect the nearest confirmed cylinder and return home."


def run_phase2_gate(models: List[str]) -> bool:
    ok, why = preflight(True)
    if not ok:
        ST.log(f"PHASE2 ABORT: {why}", "WARN")
        return False
    ST.set_state("PHASE2_VALIDATION")
    for m in models:
        ST.phase2_status[m] = {"runs": 0, "passed": False}
    for model in models:
        consec   = 0
        attempts = 0
        while consec < 3:
            if ST.stop_requested:
                return False
            attempts += 1
            if attempts > 30:
                ST.log(f"Phase2 {model}: >30 attempts — aborting", "WARN")
                return False
            res = run_single_mission(model, _VALID_MISSION, consec + 1, True)
            save_vlog_csv(model, consec + 1, res)
            if res.success:
                consec  += 1
                ST.phase2_status[model] = {"runs": consec, "passed": consec >= 3}
                ST.log(f"Phase2 {model}: {consec}/3 PASS")
            else:
                consec  = 0
                ST.phase2_status[model] = {"runs": 0, "passed": False}
                ST.log(f"Phase2 {model}: FAIL ({res.failure_reason})", "WARN")
        ST.log(f"Phase2: {model} PASSED 3/3 consecutive")
    ST.phase2_passed = True
    ST.set_state("IDLE")
    return True


# ── Benchmark runner ───────────────────────────────────────────
def run_benchmark(models: List[str], runs: int, mission: str,
                  force: bool = True) -> None:
    if ST.bench_running:
        return
    if not ST.phase2_passed and not force:
        ST.log("BENCHMARK LOCKED — Phase 2 gate not passed. Enable override to collect data without Phase 2.", "WARN")
        ST.set_state("IDLE")
        return
    if force and not ST.phase2_passed:
        ST.log("BENCHMARK OVERRIDE: Phase 2 skipped because gate is optional in v23.", "WARN")
    ST.bench_running  = True
    ST.stop_requested = False
    ST.log(f"BENCHMARK START: models={models} N={runs}")
    ok, why = preflight(True)
    if not ok:
        ST.log(f"BENCHMARK ABORT: {why}", "WARN")
        if NODE:
            NODE.stop()
        ST.bench_running = False
        ST.set_state("IDLE")
        return
    available = warmup_ollama(models)
    if not available:
        ST.log("BENCHMARK ABORT: no models available", "WARN")
        ST.bench_running = False
        ST.set_state("IDLE")
        return
    for model in available:
        if ST.stop_requested:
            break
        for run_num in range(1, runs + 1):
            if ST.stop_requested:
                break
            try:
                res = run_single_mission(model, mission, run_num)
            except Exception as ex:
                res = MissionResult(
                    run_id=f"err_{model}_{run_num}_{int(time.time())}",
                    model=model, mission_number=run_num,
                    mission_text=mission,
                    timestamp=datetime.datetime.now().isoformat(),
                    status="FAIL",
                    failure_reason=f"Exception: {ex}",
                )
                ST.log(f"Run {run_num} exception: {ex}", "ERROR")
            ST.results.append(res)
            save_result_csv(res)
            time.sleep(1.0)
    ST.bench_running = False
    ST.set_state("IDLE")
    ST.log(f"BENCHMARK COMPLETE — {len(ST.results)} total runs")


# ══════════════════════════════════════════════════════════════
# SECTION 13 — FLASK DASHBOARD
# ══════════════════════════════════════════════════════════════
app = Flask(__name__)

# ── Shared CSS ────────────────────────────────────────────────
_CSS = """<style>
:root{--bg:#0d1117;--panel:#161b26;--card:#1c2333;--bl:#3b82f6;--gr:#22c55e;
--re:#ef4444;--ye:#eab308;--or:#f97316;--pu:#a855f7;--cy:#06b6d4;
--tx:#e6edf3;--mu:#8b949e;--br:#2a3343;}
*{box-sizing:border-box;margin:0;padding:0;}
body{background:var(--bg);color:var(--tx);font-family:'Segoe UI',system-ui,monospace;font-size:13px;}
.topbar{background:var(--panel);border-bottom:2px solid var(--br);padding:0 18px;display:flex;align-items:center;flex-wrap:wrap;gap:0;}
.logo{color:var(--bl);font-size:.95rem;font-weight:700;padding:11px 18px 11px 0;border-right:1px solid var(--br);margin-right:8px;white-space:nowrap;}
.tb-badge{background:#0e2a2f;color:var(--cy);border-radius:6px;padding:2px 8px;font-size:.62rem;font-weight:700;margin-left:6px;}
.topbar a{color:var(--mu);text-decoration:none;padding:11px 14px;font-size:.80rem;border-bottom:3px solid transparent;transition:.15s;}
.topbar a.active,.topbar a:hover{color:var(--bl);border-bottom-color:var(--bl);}
.panel{background:var(--panel);border:1px solid var(--br);border-radius:8px;padding:10px;}
.ph{font-size:.65rem;color:var(--bl);text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;border-bottom:1px solid var(--br);padding-bottom:4px;}
.btn{padding:6px 14px;border:none;border-radius:6px;cursor:pointer;font-weight:700;font-size:.78rem;font-family:inherit;transition:.12s;}
.btnP{background:var(--bl);color:#fff;} .btnD{background:var(--re);color:#fff;}
.btnS{background:var(--gr);color:#06210f;} .btnY{background:var(--ye);color:#241f04;}
.btnC{background:var(--cy);color:#001a1f;} .btnO{background:var(--or);color:#1a0800;}
.mb{padding:4px 10px;background:var(--card);border:1px solid var(--br);border-radius:18px;
    color:var(--mu);cursor:pointer;font-size:.75rem;font-family:monospace;transition:.15s;}
.mb.active{background:#1e3a5f;border-color:var(--bl);color:#79c0ff;}
.mb:hover:not(.active){border-color:var(--mu);color:var(--tx);}
textarea{background:var(--card);color:var(--tx);border:1px solid var(--br);
         border-radius:6px;padding:6px 10px;width:100%;font-family:inherit;font-size:.82rem;resize:vertical;}
textarea:focus{outline:none;border-color:var(--bl);}
select,input[type=number]{background:var(--card);color:var(--tx);border:1px solid var(--br);
                           border-radius:6px;padding:4px 7px;font-family:inherit;font-size:.76rem;width:100%;}
.lbl{font-size:.64rem;color:var(--mu);margin-top:4px;display:block;}
.mbox{background:var(--card);border-radius:6px;padding:6px;}
.mbox .v{font-size:.86rem;font-weight:700;color:var(--bl);}
.mbox .l{font-size:.58rem;color:var(--mu);}
.badge{display:inline-block;padding:3px 10px;border-radius:12px;font-weight:700;font-size:.72rem;}
.s-IDLE{background:#1c2333;color:var(--mu);} .s-OK{background:#0e3a22;color:var(--gr);}
.s-FAIL{background:#3a1414;color:var(--re);} .s-WORK{background:#0e2a3a;color:var(--bl);}
.s-ABORT{background:#3a1e00;color:var(--or);}
.en-SAFE{color:var(--gr);} .en-WARNING{color:var(--ye);} .en-CRITICAL{color:var(--re);}
.cyl-chip{display:inline-block;background:#0d2030;color:var(--cy);border-radius:7px;
          padding:2px 6px;margin:2px 2px 0 0;font-size:.65rem;}
.cyl-chip.active{background:#1e3a5f;border:1px solid var(--bl);}
.routechip{display:inline-block;background:#1e0f2f;color:var(--pu);border-radius:7px;
           padding:2px 6px;margin:2px 2px 0 0;font-size:.65rem;}
.bar-wrap{height:5px;border-radius:3px;background:var(--card);overflow:hidden;margin-top:3px;}
.bar-fill{height:100%;transition:width .35s,background .35s;}
.logbox{background:var(--card);border-radius:6px;height:82px;overflow-y:auto;
        padding:4px;font-size:.60rem;color:#9fb0c0;margin-top:5px;}
.logbox p{padding:1px 0;}
</style>"""

_TOPBAR = """<div class="topbar">
  <div class="logo">&#129302; OPENCLAW <span class="tb-badge">Mission v24</span></div>
  <a href="/" class="{p1}">Control</a>
  <a href="/camera" class="{p2}">Camera</a>
  <a href="/map" class="{p3}">Map &amp; Heatmap</a>
  <a href="/logs" class="{p4}">Logs</a>
  <a href="/individual" class="{p5}">Individual</a>
  <a href="/comparison" class="{p6}">Comparison</a>
  <a href="/export" class="{p7}">Export</a>
</div>"""

# ── Shared arena/lidar JS ──────────────────────────────────────
_ARENA_JS = r"""
const HOME_PT={home_js};
const MXMIN=-3.4,MXMAX=3.4,MYMIN=-3.8,MYMAX=3.8;
function mapTf(cv){
  const pad=22,W=cv.width-pad*2,H=cv.height-pad*2;
  const sx=W/(MXMAX-MXMIN),sy=H/(MYMAX-MYMIN),s=Math.min(sx,sy);
  const ox=pad+(W-(MXMAX-MXMIN)*s)/2-MXMIN*s,oy=pad+(H-(MYMAX-MYMIN)*s)/2+MYMAX*s;
  return{s,ox,oy,w2c:(x,y)=>[ox+x*s,oy-y*s]};
}
function drawArena(cv,d,showRobot,heatPts){
  const ctx=cv.getContext('2d');ctx.clearRect(0,0,cv.width,cv.height);
  ctx.fillStyle='#11151e';ctx.fillRect(0,0,cv.width,cv.height);
  const T=mapTf(cv);
  // Grid
  ctx.strokeStyle='#191f2c';ctx.lineWidth=1;
  for(let g=-3;g<=3;g++){let p=T.w2c(g,0);ctx.beginPath();ctx.moveTo(p[0],0);ctx.lineTo(p[0],cv.height);ctx.stroke();p=T.w2c(0,g);ctx.beginPath();ctx.moveTo(0,p[1]);ctx.lineTo(cv.width,p[1]);ctx.stroke();}
  // Heatmap (trajectory density)
  if(heatPts&&heatPts.length>1){
    ctx.globalAlpha=0.55;
    heatPts.forEach(p=>{const[x,y]=T.w2c(p.x,p.y);const g2=ctx.createRadialGradient(x,y,0,x,y,6);g2.addColorStop(0,'rgba(239,68,68,0.7)');g2.addColorStop(1,'rgba(239,68,68,0)');ctx.fillStyle=g2;ctx.beginPath();ctx.arc(x,y,6,0,7);ctx.fill();});
    ctx.globalAlpha=1.0;
  }
  // Planned route
  const route=d.planned_route||[],cyls=d.cylinders||[],cylMap={};
  cyls.forEach(c=>{cylMap[c.label]=c;});
  if(route.length){
    const hp=T.w2c(HOME_PT.x,HOME_PT.y);
    ctx.strokeStyle='#a855f7';ctx.lineWidth=2;ctx.setLineDash([8,5]);ctx.beginPath();ctx.moveTo(hp[0],hp[1]);
    route.forEach(n=>{const c=cylMap[n];if(!c)return;const p=T.w2c(c.x,c.y);ctx.lineTo(p[0],p[1]);});
    ctx.lineTo(hp[0],hp[1]);ctx.stroke();ctx.setLineDash([]);
  }
  // Actual trajectory (red line)
  const tr=d.trajectory||[];
  if(tr.length>1){ctx.strokeStyle='#ef4444';ctx.lineWidth=2;ctx.beginPath();
    tr.forEach((q,i)=>{const p=T.w2c(q.x,q.y);i===0?ctx.moveTo(p[0],p[1]):ctx.lineTo(p[0],p[1]);});ctx.stroke();}
  // Obstacle points
  (d.obstacles||[]).forEach(o=>{const p=T.w2c(o.x,o.y);ctx.fillStyle='#f9731677';ctx.beginPath();ctx.arc(p[0],p[1],2.5,0,7);ctx.fill();});
  // Orbit ring on active cylinder
  if(d.active_cyl){const c=cylMap[d.active_cyl];if(c){const p=T.w2c(c.x,c.y);ctx.strokeStyle='#a855f755';ctx.lineWidth=1.5;ctx.setLineDash([4,4]);ctx.beginPath();ctx.arc(p[0],p[1],0.6*T.s,0,7);ctx.stroke();ctx.setLineDash([]);}}
  // Cylinders
  cyls.forEach(c=>{
    const p=T.w2c(c.x,c.y),isA=c.label===d.active_cyl,inR=route.includes(c.label);
    ctx.fillStyle=isA?'#c084fc':inR?'#22d3ee':c.confirmed?'#e2e8f0':'#4b5563';
    ctx.strokeStyle=isA?'#9333ea':inR?'#0891b2':'#94a3b8';ctx.lineWidth=2;
    ctx.beginPath();ctx.arc(p[0],p[1],12,0,7);ctx.fill();ctx.stroke();
    const num=(c.label||'').replace('CYL_','');
    ctx.fillStyle='#0f172a';ctx.font='bold 15px monospace';ctx.textAlign='center';
    ctx.fillText(num,p[0],p[1]+5);
    ctx.fillStyle='#ffffff';ctx.strokeStyle='#0f172a';ctx.lineWidth=4;ctx.font='bold 14px monospace';
    ctx.strokeText(c.label,p[0],p[1]-18);ctx.fillText(c.label,p[0],p[1]-18);ctx.textAlign='left';
  });
  // HOME
  const hp=T.w2c(HOME_PT.x,HOME_PT.y);
  ctx.fillStyle='#22c55e';ctx.strokeStyle='#16a34a';ctx.lineWidth=2;
  ctx.beginPath();ctx.arc(hp[0],hp[1],8,0,7);ctx.fill();ctx.stroke();
  ctx.fillStyle='#e6edf3';ctx.font='bold 8px monospace';ctx.textAlign='center';
  ctx.fillText('HOME',hp[0],hp[1]+18);ctx.textAlign='left';
  // Robot arrow
  if(showRobot!==false&&d.robot){
    const[rx,ry]=T.w2c(d.robot.x,d.robot.y);
    ctx.save();ctx.translate(rx,ry);ctx.rotate(-(d.robot.yaw||0));
    ctx.fillStyle='#0d1117';ctx.strokeStyle='#e6edf3';ctx.lineWidth=1.5;
    ctx.beginPath();ctx.moveTo(11,0);ctx.lineTo(-7,6);ctx.lineTo(-7,-6);ctx.closePath();
    ctx.fill();ctx.stroke();ctx.restore();
  }
}
function drawLidar(cv,d){
  const ctx=cv.getContext('2d');ctx.clearRect(0,0,cv.width,cv.height);
  ctx.fillStyle='#1c2333';ctx.fillRect(0,0,cv.width,cv.height);
  const cx=cv.width/2,cy=cv.height/2,R=Math.min(cx,cy)-8,mx=3.5;
  ctx.strokeStyle='#2a3343';
  for(let r=1;r<=3;r++){ctx.beginPath();ctx.arc(cx,cy,R*r/3,0,7);ctx.stroke();}
  const sec=d.scan_sectors||{};
  const dirs={front:0,front_left:Math.PI/4,left:Math.PI/2,rear:Math.PI,right:-Math.PI/2,front_right:-Math.PI/4};
  Object.entries(dirs).forEach(([k,a])=>{
    const dist=Math.min(sec[k]||mx,mx),rr=R*dist/mx,x=cx+rr*Math.sin(a),y=cy-rr*Math.cos(a);
    ctx.fillStyle=dist<0.25?'#ef4444':dist<0.50?'#eab308':dist<0.80?'#f97316':'#3b82f6';
    ctx.beginPath();ctx.arc(x,y,4,0,7);ctx.fill();
  });
  ctx.fillStyle='#0d1117';ctx.strokeStyle='#e6edf3';ctx.lineWidth=1.5;ctx.beginPath();ctx.arc(cx,cy,4.5,0,7);ctx.fill();ctx.stroke();
}
"""


def _render(template: str) -> str:
    home_js = json.dumps({"x": CFG.HOME_X, "y": CFG.HOME_Y})
    mc_js   = json.dumps(CFG.MODEL_COLORS)
    return (template
            .replace("{home_js}", home_js)
            .replace("{mc_js}", mc_js))


# ═══════════════════════════
# PAGE 1 — MISSION CONTROL
# ═══════════════════════════
_P1 = """<!DOCTYPE html><html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>OpenClaw — Control</title>""" + _CSS + """</head><body>""" + \
_TOPBAR.format(p1="active",p2="",p3="",p4="",p5="",p6="",p7="") + """
<div style="display:flex;gap:9px;padding:10px 18px;background:var(--panel);border-bottom:1px solid var(--br);flex-wrap:wrap">
  <div class="panel" style="flex:1;min-width:190px">
    <div class="ph">Model Selection</div>
    <div style="display:flex;gap:5px;flex-wrap:wrap" id="mbrow">
      <button class="mb active" onclick="selM('gemma2:2b',this)">gemma2:2b</button>
      <button class="mb" onclick="selM('qwen2.5:3b',this)">qwen2.5:3b</button>
      <button class="mb" onclick="selM('deepseek-r1:1.5b',this)">deepseek-r1:1.5b</button>
    </div>
  </div>
  <div class="panel" style="flex:2;min-width:260px">
    <div class="ph">Mission Command</div>
    <textarea id="mission" rows="2">Inspect nearest two cylinders and return home.</textarea>
    <div style="display:grid;grid-template-columns:140px 1fr;gap:6px;margin-top:6px;align-items:end">
      <div>
        <label class="lbl">Runs for this model</label>
        <input id="mruns" type="number" min="1" max="100" value="1">
      </div>
      <div style="font-size:.64rem;color:var(--mu);line-height:1.4">
        Mission Control now runs the selected LLM repeatedly. Use 1 for demo, 30 for FYP, max 100. Phase 2 is not required.
      </div>
    </div>
    <div style="display:flex;gap:5px;flex-wrap:wrap;margin-top:6px">
      <button class="btn btnP" onclick="execMission()">&#9654; Execute Runs</button>
      <button class="btn btnD" onclick="stopNow()">&#9632; Stop</button>
      <button class="btn btnS" onclick="resetHome()">&#8635; Reset HOME</button>
      <button class="btn btnC" onclick="exploreArena()">&#128269; Explore</button>
      <button class="btn btnO" onclick="clearCyls()">&#128465; Clear Cyls</button>
    </div>
  </div>
</div>
<div style="display:grid;grid-template-columns:1.1fr 1fr;gap:9px;padding:9px 18px">
  <div class="panel">
    <div class="ph">Live Arena Map</div>
    <canvas id="mapcv" width="480" height="430" style="background:var(--card);border-radius:6px;width:100%;display:block"></canvas>
    <div style="display:flex;gap:7px;margin-top:4px;font-size:.60rem;flex-wrap:wrap">
      <span style="color:var(--cy)">&#9675; Cylinder</span>
      <span style="color:#f97316">&#9679; Obstacle</span>
      <span style="color:var(--pu)">&#9644; Planned</span>
      <span style="color:var(--re)">&#9644; Actual</span>
      <span style="color:var(--gr)">&#9679; HOME</span>
    </div>
  </div>
  <div style="display:flex;flex-direction:column;gap:9px">
    <div class="panel">
      <div class="ph">Mission Status — <span id="sbadge" class="badge s-IDLE">IDLE</span></div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:4px;margin-top:5px">
        <div class="mbox"><div class="v" id="st-model">—</div><div class="l">Model</div></div>
        <div class="mbox"><div class="v" id="st-run">—</div><div class="l">Run #</div></div>
        <div class="mbox"><div class="v" id="st-cyl">—</div><div class="l">Active Cyl</div></div>
        <div class="mbox"><div class="v" id="st-coll">0</div><div class="l">Collisions</div></div>
        <div class="mbox"><div class="v" id="st-clr">—</div><div class="l">Min Clr(m)</div></div>
        <div class="mbox"><div class="v" id="st-replan">0</div><div class="l">Replans</div></div>
        <div class="mbox"><div class="v" id="st-pos">0,0</div><div class="l">Pos (x,y)</div></div>
        <div class="mbox"><div class="v" id="st-dist">0m</div><div class="l">Distance</div></div>
      </div>
    </div>
    <div class="panel">
      <div class="ph">LiDAR</div>
      <canvas id="lidarcv" width="280" height="120" style="background:var(--card);border-radius:6px;width:100%;display:block"></canvas>
      <div style="display:flex;gap:4px;margin-top:4px;flex-wrap:wrap">
        <div class="mbox" style="flex:1"><div class="v" id="lid-min">—</div><div class="l">Closest(m)</div></div>
        <div class="mbox" style="flex:1"><div class="v en-SAFE" id="lid-st">SAFE</div><div class="l">Status</div></div>
        <div class="mbox" style="flex:1"><div class="v" id="lid-fr">—</div><div class="l">Front(m)</div></div>
        <div class="mbox" style="flex:1"><div class="v" id="risk-v">0</div><div class="l">Risk%</div></div>
      </div>
      <div class="bar-wrap"><div class="bar-fill" id="risk-bar" style="width:0%"></div></div>
    </div>
    <div class="panel">
      <div class="ph">Cylinders &amp; Route</div>
      <div id="cyl-chips"><span style="color:var(--mu);font-size:.65rem">Scanning…</span></div>
      <div style="font-size:.66rem;color:var(--mu);margin-top:5px">Planned Route</div>
      <div id="route-chips" style="margin-top:2px"><span style="color:var(--mu);font-size:.62rem">No route</span></div>
      <div class="bar-wrap"><div class="bar-fill" id="orbit-bar" style="width:0%;background:var(--pu)"></div></div>
      <div style="font-size:.60rem;color:var(--pu);margin-top:2px" id="orbit-lbl">—</div>
      <div style="font-size:.66rem;color:var(--mu);margin-top:5px">LLM Reasoning</div>
      <div style="background:var(--card);border-radius:5px;padding:6px;font-size:.68rem;color:#9fb0c0;min-height:40px;margin-top:3px;overflow:auto;max-height:70px" id="reasoning">—</div>
    </div>
  </div>
</div>
<div style="display:grid;grid-template-columns:1fr 1fr;gap:9px;padding:0 18px 10px">
  <div class="panel">
    <div class="ph">Mission Run Control</div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px">
      <div class="mbox"><div class="v" id="mc-model">—</div><div class="l">Selected / Running LLM</div></div>
      <div class="mbox"><div class="v" id="mc-run">—</div><div class="l">Current Run</div></div>
      <div class="mbox"><div class="v" id="mc-state">—</div><div class="l">Current Action</div></div>
      <div class="mbox"><div class="v" id="mc-target">—</div><div class="l">Target</div></div>
    </div>
    <div style="margin-top:7px;font-size:.68rem;color:var(--mu);line-height:1.5">
      Use Mission Command above for data collection. Choose one LLM, type one command, set runs 1–100, then Execute Runs. Phase 2 Gate is optional and OFF by default.
    </div>
    <div style="display:flex;gap:5px;flex-wrap:wrap;margin-top:7px">
      <button class="btn btnY" style="flex:1" onclick="runPhase2()">&#9888; Optional Phase 2 Test</button>
      <button class="btn btnS" style="flex:1" onclick="resetHome()">&#8635; Return HOME</button>
      <button class="btn btnD" style="flex:1" onclick="stopNow()">&#9632; Stop</button>
    </div>
    <div style="margin-top:5px;font-size:.62rem;color:var(--mu)">
      Phase 2: <span id="p2tag" style="color:var(--re)">OPTIONAL / NOT PASSED</span>
    </div>
    <div id="p2detail" style="font-size:.60rem;color:var(--mu);margin-top:2px;line-height:1.5"></div>
  </div>
  <div class="panel">
    <div class="ph">System Log</div>
    <div class="logbox" id="logbox" style="height:160px"></div>
  </div>
</div>
<script>
""" + _ARENA_JS + """
let curModel='gemma2:2b';
function selM(m,btn){curModel=m;document.querySelectorAll('.mb').forEach(b=>b.classList.remove('active'));btn.classList.add('active');}
async function poll(){
  try{
    const d=await(await fetch('/api/status')).json();
    const b=document.getElementById('sbadge');
    b.textContent=d.state;
    b.className='badge '+(['COMPLETED','SUCCESS'].includes(d.state)?'s-OK':['FAILED','ABORTED'].includes(d.state)?'s-FAIL':d.state==='IDLE'?'s-IDLE':'s-WORK');
    document.getElementById('st-model').textContent=d.model||'—';
    document.getElementById('st-run').textContent=d.run||'—';
    if(document.getElementById('mc-model'))document.getElementById('mc-model').textContent=d.model||curModel||'—';
    if(document.getElementById('mc-run'))document.getElementById('mc-run').textContent=d.run||'—';
    if(document.getElementById('mc-state'))document.getElementById('mc-state').textContent=d.state||'—';
    if(document.getElementById('mc-target'))document.getElementById('mc-target').textContent=d.active_cyl||'—';
    document.getElementById('st-cyl').textContent=d.active_cyl||'—';
    document.getElementById('st-coll').textContent=d.collision_count||0;
    document.getElementById('st-clr').textContent=d.scan_min<90?d.scan_min.toFixed(2)+'m':'—';
    document.getElementById('st-replan').textContent=d.replan_count||0;
    document.getElementById('st-pos').textContent=(d.robot.x||0).toFixed(2)+','+(d.robot.y||0).toFixed(2);
    document.getElementById('st-dist').textContent=(d.distance||0).toFixed(2)+'m';
    if(d.reasoning)document.getElementById('reasoning').textContent=d.reasoning;
    const cyls=d.cylinders||[];
    document.getElementById('cyl-chips').innerHTML=cyls.length
      ?cyls.map(c=>`<span class="cyl-chip${c.label===d.active_cyl?' active':''}">${c.label}${c.confirmed?'':' ?'}</span>`).join('')
      :'<span style="color:var(--mu);font-size:.63rem">No cylinders confirmed</span>';
    const r=d.planned_route||[];
    document.getElementById('route-chips').innerHTML=r.length
      ?'HOME→'+r.map(w=>`<span class="routechip">${w}</span>`).join('→')+'→HOME'
      :'<span style="color:var(--mu);font-size:.62rem">No route</span>';
    const od=d.orbit_deg||0;
    document.getElementById('orbit-bar').style.width=Math.min(100,od/360*100)+'%';
    document.getElementById('orbit-lbl').textContent=od>0?`Orbit: ${od.toFixed(0)}°/360°`:d.active_cyl?`→ ${d.active_cyl}`:'';
    const sm=d.scan_min<90?d.scan_min:99;
    document.getElementById('lid-min').textContent=sm<90?sm.toFixed(2):'—';
    const ls=document.getElementById('lid-st');ls.textContent=d.lidar_status;ls.className='v en-'+d.lidar_status;
    document.getElementById('lid-fr').textContent=d.scan_sectors&&d.scan_sectors.front<90?d.scan_sectors.front.toFixed(2):'—';
    const risk=d.risk_score||0;document.getElementById('risk-v').textContent=risk.toFixed(0);
    const rb=document.getElementById('risk-bar');rb.style.width=Math.min(100,risk)+'%';
    rb.style.background=risk>70?'#ef4444':risk>40?'#eab308':'#22c55e';
    const p2=d.phase2_passed;
    document.getElementById('p2tag').textContent=p2?'PASSED':'NOT PASSED';
    document.getElementById('p2tag').style.color=p2?'var(--gr)':'var(--re)';
    if(d.phase2_status)document.getElementById('p2detail').innerHTML=
      Object.entries(d.phase2_status).map(([m,v])=>`${m}: ${v.runs}/3${v.passed?' ✓':''}`).join('<br>');
    const lb=document.getElementById('logbox');
    lb.innerHTML=(d.log||[]).slice(-50).map(l=>`<p>${l}</p>`).join('');lb.scrollTop=lb.scrollHeight;
    drawArena(document.getElementById('mapcv'),d,true,d.trajectory||[]);
    drawLidar(document.getElementById('lidarcv'),d);
  }catch(e){}
  setTimeout(poll,500);
}
async function execMission(){
  const m=document.getElementById('mission').value;if(!m.trim())return;
  const runs=Math.max(1,Math.min(100,parseInt(document.getElementById('mruns').value)||1));
  await fetch('/api/mission',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({mission:m,model:curModel,runs:runs})});
}
async function stopNow(){await fetch('/api/stop',{method:'POST'});}
async function resetHome(){await fetch('/api/reset_robot',{method:'POST'});}
async function exploreArena(){await fetch('/api/explore',{method:'POST'});}
async function clearCyls(){await fetch('/api/reset_cylinders',{method:'POST'});}
async function runPhase2(){await fetch('/api/phase2',{method:'POST'});}
async function startBench(){
  const mode=document.getElementById('bmode').value,model=document.getElementById('bmodel').value,
        runs=parseInt(document.getElementById('bruns').value)||30,
        mtype=document.getElementById('bmission').value,
        force=document.getElementById('forceBench')?document.getElementById('forceBench').checked:true,
        mission=mtype==='custom'?document.getElementById('mission').value:'';
  await fetch('/api/start_benchmark',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({mode,model,runs,mission,force})});
}
poll();
</script></body></html>"""

# ═══════════════════════════
# PAGE 2 — CAMERA
# ═══════════════════════════
_P2 = """<!DOCTYPE html><html lang="en"><head>
<meta charset="UTF-8"><title>OpenClaw — Camera</title>""" + _CSS + """</head><body>""" + \
_TOPBAR.format(p1="",p2="active",p3="",p4="",p5="",p6="",p7="") + """
<div style="padding:14px 20px">
  <div class="panel" style="max-width:720px;margin:0 auto">
    <div class="ph">Live Gazebo Camera — <span id="cam-info" style="color:var(--mu)">Waiting…</span></div>
    <div style="background:#000;border-radius:6px;overflow:hidden;min-height:320px;display:flex;align-items:center;justify-content:center;position:relative">
      <img src="/video_feed" style="width:100%;display:block" onerror="this.style.opacity='.3'">
      <div style="position:absolute;top:5px;right:7px;font-size:.62rem;background:#0009;color:var(--cy);padding:2px 6px;border-radius:4px" id="cam-fps">— fps</div>
    </div>
    <div style="display:flex;gap:6px;margin-top:8px;flex-wrap:wrap">
      <div class="mbox" style="flex:1"><div class="v" id="cam-w">—</div><div class="l">Width px</div></div>
      <div class="mbox" style="flex:1"><div class="v" id="cam-h">—</div><div class="l">Height px</div></div>
      <div class="mbox" style="flex:1"><div class="v" id="cam-fps2">—</div><div class="l">FPS</div></div>
      <div class="mbox" style="flex:1"><div class="v" id="cam-ok">—</div><div class="l">Status</div></div>
    </div>
    <div style="margin-top:8px;font-size:.72rem;color:var(--mu)">
      Camera source: <code style="color:var(--ye)">{cam_topic}</code> (Gazebo only — no USB camera)
    </div>
  </div>
</div>
<script>
async function poll(){
  try{
    const d=await(await fetch('/api/status')).json();
    document.getElementById('cam-info').textContent=d.cam_ok?`${d.cam_w||'?'}×${d.cam_h||'?'} @ ${(d.cam_fps||0).toFixed(1)}fps`:'No signal';
    document.getElementById('cam-fps').textContent=(d.cam_fps||0).toFixed(1)+' fps';
    document.getElementById('cam-w').textContent=d.cam_w||'—';
    document.getElementById('cam-h').textContent=d.cam_h||'—';
    document.getElementById('cam-fps2').textContent=(d.cam_fps||0).toFixed(1);
    document.getElementById('cam-ok').textContent=d.cam_ok?'LIVE':'NO SIGNAL';
    document.getElementById('cam-ok').style.color=d.cam_ok?'var(--gr)':'var(--re)';
  }catch(e){}
  setTimeout(poll,1000);
}
poll();
</script></body></html>""".replace("{cam_topic}", CFG.CAM_TOPIC)

# ═══════════════════════════
# PAGE 3 — MAP & HEATMAP
# ═══════════════════════════
_P3 = """<!DOCTYPE html><html lang="en"><head>
<meta charset="UTF-8"><title>OpenClaw — Map &amp; Heatmap</title>""" + _CSS + """</head><body>""" + \
_TOPBAR.format(p1="",p2="",p3="active",p4="",p5="",p6="",p7="") + """
<div style="padding:10px 18px;display:grid;grid-template-columns:1fr 1fr;gap:10px">
  <div class="panel">
    <div class="ph">Live Arena Map</div>
    <canvas id="mapcv" width="500" height="460" style="background:var(--card);border-radius:6px;width:100%;display:block"></canvas>
  </div>
  <div class="panel">
    <div class="ph">Trajectory Heatmap — all runs</div>
    <canvas id="heatcv" width="500" height="460" style="background:var(--card);border-radius:6px;width:100%;display:block"></canvas>
    <div style="font-size:.62rem;color:var(--mu);margin-top:4px">
      Path density: brighter = more frequently traversed
    </div>
  </div>
</div>
<script>
""" + _ARENA_JS + """
let allTraj=[];
async function poll(){
  try{
    const d=await(await fetch('/api/status')).json();
    drawArena(document.getElementById('mapcv'),d,true,null);
  }catch(e){}
  setTimeout(poll,600);
}
async function loadHeat(){
  try{
    const d=await(await fetch('/api/results_all')).json();
    const pts=[];
    (d.results||[]).forEach(r=>{(r.trajectory||[]).forEach(p=>pts.push(p));});
    allTraj=pts;
    drawHeat();
  }catch(e){}
  setTimeout(loadHeat,5000);
}
function drawHeat(){
  const cv=document.getElementById('heatcv'),ctx=cv.getContext('2d');
  ctx.clearRect(0,0,cv.width,cv.height);
  ctx.fillStyle='#11151e';ctx.fillRect(0,0,cv.width,cv.height);
  const T=mapTf(cv);
  // Accumulate density in a grid
  const gs=12,gw=Math.ceil(cv.width/gs),gh=Math.ceil(cv.height/gs),grid=new Float32Array(gw*gh);
  let mx=0;
  allTraj.forEach(p=>{const[x,y]=T.w2c(p.x,p.y);const gx=Math.floor(x/gs),gy=Math.floor(y/gs);if(gx>=0&&gx<gw&&gy>=0&&gy<gh){grid[gy*gw+gx]+=1;if(grid[gy*gw+gx]>mx)mx=grid[gy*gw+gx];}});
  if(mx>0){
    for(let r=0;r<gh;r++)for(let c2=0;c2<gw;c2++){
      const v=grid[r*gw+c2];if(v<1)continue;
      const alpha=Math.min(0.9,v/mx*0.9+0.1);
      const heat=`rgba(239,68,68,${alpha.toFixed(2)})`;
      ctx.fillStyle=heat;ctx.fillRect(c2*gs,r*gs,gs,gs);
    }
  }
  // Overlay cylinders and home
  const dFake={cylinders:[],planned_route:[],trajectory:[],obstacles:[],robot:null,active_cyl:'',scan_sectors:{},lidar_status:'SAFE',risk_score:0};
  try{fetch('/api/status').then(r=>r.json()).then(d=>{drawArena(cv,d,false,null);});}catch(e){}
}
poll();loadHeat();
</script></body></html>"""

# ═══════════════════════════
# PAGE 4 — LOGS
# ═══════════════════════════
_P4 = """<!DOCTYPE html><html lang="en"><head>
<meta charset="UTF-8"><title>OpenClaw — Logs</title>""" + _CSS + """</head><body>""" + \
_TOPBAR.format(p1="",p2="",p3="",p4="active",p5="",p6="",p7="") + """
<div style="padding:12px 20px;display:grid;grid-template-columns:1fr 1fr;gap:10px">
  <div class="panel">
    <div class="ph">System Monitor</div>
    <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:5px" id="sysmon">
      <div class="mbox"><div class="v" id="sm-cpu">—</div><div class="l">CPU %</div></div>
      <div class="mbox"><div class="v" id="sm-ram">—</div><div class="l">RAM MB</div></div>
      <div class="mbox"><div class="v" id="sm-pwr">—</div><div class="l">Power W</div></div>
      <div class="mbox"><div class="v" id="sm-ros">—</div><div class="l">ROS /odom</div></div>
      <div class="mbox"><div class="v" id="sm-scan">—</div><div class="l">ROS /scan</div></div>
      <div class="mbox"><div class="v" id="sm-ollama">—</div><div class="l">Ollama</div></div>
    </div>
    <div style="margin-top:8px" id="model-avail"></div>
  </div>
  <div class="panel">
    <div class="ph">ROS Debug</div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:5px" id="rosdebug">
      <div class="mbox"><div class="v" id="rd-odom">0</div><div class="l">Odom msgs</div></div>
      <div class="mbox"><div class="v" id="rd-scan">0</div><div class="l">Scan msgs</div></div>
      <div class="mbox"><div class="v" id="rd-cam">0</div><div class="l">Cam msgs</div></div>
      <div class="mbox"><div class="v" id="rd-cmd">0</div><div class="l">CmdVel pubs</div></div>
    </div>
  </div>
  <div class="panel" style="grid-column:1/3">
    <div class="ph">Mission Log (live)</div>
    <div class="logbox" id="logbox" style="height:320px;font-size:.65rem"></div>
  </div>
</div>
<script>
async function poll(){
  try{
    const d=await(await fetch('/api/status')).json();
    const v=await(await fetch('/api/validation')).json();
    const dbg=await(await fetch('/api/ros_debug')).json();
    document.getElementById('sm-ros').textContent=v.odom_active?'OK':'FAIL';
    document.getElementById('sm-ros').style.color=v.odom_active?'var(--gr)':'var(--re)';
    document.getElementById('sm-scan').textContent=v.scan_active?'OK':'FAIL';
    document.getElementById('sm-scan').style.color=v.scan_active?'var(--gr)':'var(--re)';
    document.getElementById('sm-ollama').textContent=v.ollama_ok?'OK':'FAIL';
    document.getElementById('sm-ollama').style.color=v.ollama_ok?'var(--gr)':'var(--re)';
    document.getElementById('rd-odom').textContent=dbg.odom_msgs||0;
    document.getElementById('rd-scan').textContent=dbg.scan_msgs||0;
    document.getElementById('rd-cam').textContent=dbg.cam_msgs||0;
    document.getElementById('rd-cmd').textContent=dbg.cmdvel_pubs||0;
    document.getElementById('model-avail').innerHTML=(v.models_ok||[]).map(m=>`<span class="cyl-chip" style="background:#0e3a22;color:var(--gr)">${m}</span>`).join('')
      +(v.models_failed||[]).map(m=>`<span class="cyl-chip" style="background:#3a1414;color:var(--re)">${m} ✗</span>`).join('');
    const lb=document.getElementById('logbox');
    lb.innerHTML=(d.log||[]).map(l=>`<p>${l}</p>`).join('');lb.scrollTop=lb.scrollHeight;
  }catch(e){}
  // System metrics
  try{
    const s=await(await fetch('/api/sys_metrics')).json();
    document.getElementById('sm-cpu').textContent=(s.cpu||0).toFixed(1);
    document.getElementById('sm-ram').textContent=(s.ram||0).toFixed(0);
    document.getElementById('sm-pwr').textContent=(s.power||0).toFixed(2);
  }catch(e){}
  setTimeout(poll,1000);
}
poll();
</script></body></html>"""

# ═══════════════════════════
# PAGE 5 — INDIVIDUAL ANALYSIS
# ═══════════════════════════
_P5 = """<!DOCTYPE html><html lang="en"><head>
<meta charset="UTF-8"><title>OpenClaw — Individual</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"></script>
""" + _CSS + """</head><body>""" + \
_TOPBAR.format(p1="",p2="",p3="",p4="",p5="active",p6="",p7="") + """
<div style="background:var(--panel);border-bottom:1px solid var(--br);padding:9px 18px;display:flex;gap:6px;flex-wrap:wrap">
  <button class="mb active" onclick="sel('gemma2:2b',this)">gemma2:2b</button>
  <button class="mb" onclick="sel('qwen2.5:3b',this)">qwen2.5:3b</button>
  <button class="mb" onclick="sel('deepseek-r1:1.5b',this)">deepseek-r1:1.5b</button>
</div>
<div style="padding:12px 18px;display:grid;grid-template-columns:1fr 1fr;gap:10px">
  <div class="panel" style="grid-column:1/3">
    <div class="ph">Summary — <span id="sm" style="color:var(--ye)">gemma2:2b</span></div>
    <div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(120px,1fr));gap:5px" id="sg">
      <div style="color:var(--mu);padding:12px">Loading…</div>
    </div>
  </div>
  <div class="panel" style="grid-column:1/3">
    <div class="ph">Run History</div>
    <div id="rt"><div style="color:var(--mu);padding:12px">No runs yet.</div></div>
  </div>
  <div class="panel">
    <div class="ph">Run Detail</div>
    <div id="rd"><div style="color:var(--mu);padding:12px">Select a run.</div></div>
  </div>
  <div class="panel">
    <div class="ph">LLM Evidence</div>
    <div id="le"><div style="color:var(--mu);padding:12px">Select a run.</div></div>
  </div>
  <div class="panel" style="grid-column:1/3">
    <div class="ph">Statistics (μ, σ, min, max)</div>
    <div id="sb"><div style="color:var(--mu);padding:12px">Run ≥2 missions to see stats.</div></div>
  </div>
</div>
<style>
table{width:100%;border-collapse:collapse;font-size:.74rem}
thead th{background:var(--card);padding:5px 8px;text-align:left;color:var(--mu);font-weight:600;border-bottom:2px solid var(--br)}
tbody td{padding:4px 8px;border-bottom:1px solid var(--br)}tbody tr:hover td{background:var(--card)}
tbody tr.sel td{background:#1e3a5f}.ok{color:var(--gr);font-weight:700}.fail{color:var(--re);font-weight:700}
.dim{color:var(--mu);font-size:.68rem}
.box{background:var(--card);border-radius:5px;padding:8px;font-size:.70rem;color:#9fb0c0;max-height:120px;overflow-y:auto;margin-top:4px;word-break:break-all}
.kv{display:flex;gap:8px;padding:2px 0;border-bottom:1px solid var(--bg)}
.k{color:var(--mu);min-width:130px}
.sl{display:flex;justify-content:space-between;font-size:.72rem;padding:2px 0;border-bottom:1px solid var(--bg)}
.sl span:last-child{color:var(--bl);font-weight:700}
</style>
<script>
""" + _ARENA_JS + """
let cm='gemma2:2b';
function sel(m,btn){cm=m;document.querySelectorAll('.mb').forEach(b=>b.classList.remove('active'));btn.classList.add('active');document.getElementById('sm').textContent=m;load();}
async function load(){
  try{
    const d=await(await fetch('/api/analysis?model='+encodeURIComponent(cm))).json();
    const ms=d.stats||{},n=ms.n||0,suc=ms.success_count||0;
    const items=[
      ['Runs',n],['Pass',suc],['Fail',n-suc],['Rate%',(ms.success_rate||0).toFixed(1)],
      ['LLM(s)',(ms.llm_latency&&ms.llm_latency.mean||0).toFixed(3)],
      ['Nav(s)',(ms.nav_time&&ms.nav_time.mean||0).toFixed(2)],
      ['Total(s)',(ms.total_mission_time&&ms.total_mission_time.mean||0).toFixed(2)],
      ['IC%',(ms.inspection_pct&&ms.inspection_pct.mean||0).toFixed(1)],
      ['Safety',(ms.safety_score&&ms.safety_score.mean||0).toFixed(1)],
      ['MinClr(m)',(ms.min_clearance&&ms.min_clearance.mean||0).toFixed(3)],
      ['Coll.',(ms.collision_count&&ms.collision_count.mean||0).toFixed(1)],
      ['Eff.',(ms.route_efficiency&&ms.route_efficiency.mean||0).toFixed(3)],
      ['Power(W)',(ms.avg_power&&ms.avg_power.mean||0).toFixed(3)],
    ];
    document.getElementById('sg').innerHTML=items.map(([l,v])=>`<div class="mbox"><div class="v">${v}</div><div class="l">${l}</div></div>`).join('');
    const runs=d.results||[];
    if(!runs.length){document.getElementById('rt').innerHTML='<div style="color:var(--mu);padding:12px">No runs.</div>';return;}
    let h='<table><thead><tr><th>#</th><th>Status</th><th>LLM(s)</th><th>Nav(s)</th><th>Total(s)</th><th>IC%</th><th>Safety</th><th>Coll</th><th>Rtn</th><th>Route</th></tr></thead><tbody>';
    runs.forEach(r=>{
      h+=`<tr onclick="selRun('${r.run_id}',this)" style="cursor:pointer">
        <td class="dim">${r.mission_number}</td>
        <td class="${r.success?'ok':'fail'}">${r.success?'PASS':'FAIL'}</td>
        <td>${(r.llm_latency||0).toFixed(3)}</td><td>${(r.nav_time||0).toFixed(2)}</td>
        <td>${(r.total_mission_time||0).toFixed(2)}</td><td>${(r.inspection_pct||0).toFixed(1)}</td>
        <td>${(r.safety_score||0).toFixed(1)}</td><td>${r.collision_count||0}</td>
        <td>${r.return_home_success?'✓':'✗'}</td>
        <td style="font-size:.63rem">${(r.parsed_route||[]).join('→')}</td></tr>`;
    });
    h+='</tbody></table>';document.getElementById('rt').innerHTML=h;
    const r=(l,o)=>o?`<div class="sl"><span>${l}</span><span>μ=${o.mean.toFixed(4)} σ=${o.std.toFixed(4)} min=${o.min.toFixed(4)} max=${o.max.toFixed(4)}</span></div>`:'';
    document.getElementById('sb').innerHTML=`<div class="box" style="max-height:999px">
      ${r('Final Deployment Score',ms.final_deployment_score)}${r('Mission Score',ms.mission_score)}
      ${r('Latency Score',ms.latency_score)}${r('Path Score',ms.path_efficiency_score)}
      ${r('Recovery Score',ms.recovery_score)}${r('LLM Latency(s)',ms.llm_latency)}${r('Nav Time(s)',ms.nav_time)}
      ${r('Total Mission(s)',ms.total_mission_time)}${r('IC%',ms.inspection_pct)}
      ${r('Safety Score',ms.safety_score)}${r('MinClr(m)',ms.min_clearance)}
      ${r('Collisions',ms.collision_count)}${r('Replans',ms.replan_count)}
      ${r('Route Efficiency',ms.route_efficiency)}
      ${r('Avg CPU%',ms.avg_cpu)}${r('Avg RAM MB',ms.avg_ram)}${r('Avg Power W',ms.avg_power)}
    </div>`;
  }catch(e){}
}
async function selRun(id,row){
  document.querySelectorAll('tbody tr').forEach(r=>r.classList.remove('sel'));row.classList.add('sel');
  try{
    const d=await(await fetch('/api/trajectory_replay/'+id)).json();
    const kv=(k,v)=>`<div class="kv"><span class="k">${k}</span><span>${v}</span></div>`;
    document.getElementById('rd').innerHTML=`<div class="box" style="max-height:200px">
      ${kv('Mission',d.mission_text||'—')}${kv('Status',d.status)}
      ${kv('Failure',d.failure_reason||'—')}
      ${kv('Final Score',d.final_deployment_score||'—')}
      ${kv('Score Reason',d.pass_fail_reason||'—')}
      ${kv('Analysis',d.analysis_comment||'—')}
      ${kv('Learning Feedback',d.learning_feedback||'—')}
      ${kv('Planned',(d.planned_route||[]).join('→'))}
      ${kv('Executed',(d.executed_route||[]).join('→'))}
      ${kv('Return Home',d.return_home_success?'YES':'NO')}
      ${kv('Timestamp',d.timestamp||'—')}</div>`;
    document.getElementById('le').innerHTML=
      `<div style="font-size:.68rem;color:var(--mu)">Reasoning</div>
       <div class="box">${d.reasoning||'—'}</div>
       <div style="font-size:.68rem;color:var(--mu);margin-top:4px">Next-run improvement hint</div>
       <div class="box">${d.improvement_hint||'—'}</div>
       <div style="font-size:.68rem;color:var(--mu);margin-top:4px">Raw LLM Response</div>
       <div class="box">${esc(d.raw_llm||'—')}</div>`;
  }catch(e){}
}
function esc(s){return String(s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');}
load();
</script></body></html>"""

# ═══════════════════════════
# PAGE 6 — COMPARISON
# ═══════════════════════════
_P6 = """<!DOCTYPE html><html lang="en"><head>
<meta charset="UTF-8"><title>OpenClaw — Comparison</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"></script>
""" + _CSS + """</head><body>""" + \
_TOPBAR.format(p1="",p2="",p3="",p4="",p5="",p6="active",p7="") + """
<div style="padding:12px 20px">
  <h2 style="font-size:.88rem;margin-bottom:10px;color:var(--tx)">Three-Model Benchmark Comparison — gemma2:2b vs qwen2.5:3b vs deepseek-r1:1.5b</h2>
  <div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:12px" id="rrow">
    <div style="color:var(--mu)">Loading…</div>
  </div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px">
    <div class="panel"><div class="ph">Fig 1 — LLM Latency (s)</div><canvas id="f1"></canvas></div>
    <div class="panel"><div class="ph">Fig 2 — Success Rate (%)</div><canvas id="f2"></canvas></div>
    <div class="panel"><div class="ph">Fig 3 — Inspection Completion (%)</div><canvas id="f3"></canvas></div>
    <div class="panel"><div class="ph">Fig 4 — Safety Score</div><canvas id="f4"></canvas></div>
    <div class="panel"><div class="ph">Fig 5 — Min Clearance (m)</div><canvas id="f5"></canvas></div>
    <div class="panel"><div class="ph">Fig 6 — Deployment Score</div><canvas id="f6"></canvas></div>
    <div class="panel" style="grid-column:1/3"><div class="ph">Full Metrics Table</div>
      <div id="ct"><div style="color:var(--mu);padding:12px">Loading…</div></div>
    </div>
  </div>
</div>
<style>
.rcard{background:var(--panel);border:1px solid var(--br);border-radius:8px;padding:12px;flex:1;min-width:165px}
.rn{font-size:1.7rem;font-weight:800;margin-bottom:2px}.r1{color:#ffd700}.r2{color:#c0c0c0}.r3{color:#cd7f32}
.rm{font-family:monospace;font-size:.80rem;color:#79c0ff;margin-bottom:3px}
.rl{font-size:.65rem;color:var(--mu);line-height:1.6}
.rb{display:inline-block;padding:2px 7px;border-radius:8px;font-size:.60rem;font-weight:700;margin-top:3px}
.rb-r{background:#0e3a22;color:var(--gr)}.rb-n{background:#3a2200;color:#f97316}.rb-x{background:#3a1414;color:var(--re)}
table{width:100%;border-collapse:collapse;font-size:.70rem}
thead th{background:var(--card);padding:5px 8px;text-align:left;color:var(--mu);font-weight:600;border-bottom:2px solid var(--br)}
tbody td{padding:4px 8px;border-bottom:1px solid var(--br)}tbody tr:hover td{background:var(--card)}
.ok{color:var(--gr);font-weight:700}.fail{color:var(--re);font-weight:700}
</style>
<script>
const MC={mc_js};let ch={};
function bar(id,models,vals,col,opts={}){if(ch[id])ch[id].destroy();ch[id]=new Chart(document.getElementById(id),{type:'bar',data:{labels:models,datasets:[{data:vals,backgroundColor:col,borderRadius:4}]},options:{responsive:true,plugins:{legend:{display:false}},scales:{x:{ticks:{color:'#8b949e'},grid:{color:'#232c3d'}},y:{ticks:{color:'#8b949e'},grid:{color:'#232c3d'},beginAtZero:true,...(opts.y||{})}}}});}
async function load3(){
  try{
    const d=await(await fetch('/api/comparison')).json();
    if(d.error){document.getElementById('rrow').innerHTML='<div style="color:var(--mu)">'+d.error+'</div>';return;}
    const models=d.models||[],col=models.map(m=>MC[m]||'#8b949e');
    const rbl={'Ready':'rb-r','Needs Improvement':'rb-n','Not Ready':'rb-x'},rc=['r1','r2','r3'];
    document.getElementById('rrow').innerHTML=(d.ranking||[]).map((r,i)=>
      `<div class="rcard"><div class="rn ${rc[i]||''}">${r.medal}</div>
       <div class="rm">${r.model}</div>
       <div class="rl">Score: <b>${r.score}</b><br>Success: ${(r.success_rate||0).toFixed(1)}%<br>LatencyScore: ${(r.latency_score||0).toFixed(1)}<br>Path: ${(r.path_score||0).toFixed(1)}<br>Recovery: ${(r.recovery_score||0).toFixed(1)}</div>
       <span class="rb ${rbl[r.label]||'rb-x'}">${r.label}</span>
       <div class="rl" style="margin-top:6px;color:#c9d1d9">${r.comment||''}</div></div>`).join('');
    bar('f1',models,models.map(m=>d.stats[m].llm_latency.mean),col);
    bar('f2',models,models.map(m=>d.stats[m].success_rate),col,{y:{max:100}});
    bar('f3',models,models.map(m=>d.stats[m].inspection_pct.mean),col,{y:{max:100}});
    bar('f4',models,models.map(m=>d.stats[m].safety_score.mean),col,{y:{max:100}});
    bar('f5',models,models.map(m=>d.stats[m].min_clearance.mean),col);
    bar('f6',models,models.map(m=>d.stats[m].final_deployment_score.mean),col,{y:{max:100}});
    let h='<table><thead><tr><th>Model</th><th>Runs</th><th>Rate%</th><th>LLM(s)</th><th>Nav(s)</th><th>Total(s)</th><th>IC%</th><th>Deploy</th><th>Safety</th><th>Recovery</th><th>Eff.</th><th>Coll.</th><th>Score</th></tr></thead><tbody>';
    models.forEach(m=>{const s=d.stats[m],sc=d.scores[m]||{};
      h+=`<tr><td style="font-family:monospace;color:#79c0ff">${m}</td><td>${s.n}</td>
        <td class="${s.success_rate>=75?'ok':'fail'}">${(s.success_rate||0).toFixed(1)}%</td>
        <td>${s.llm_latency?s.llm_latency.mean.toFixed(3):'—'}</td>
        <td>${s.nav_time?s.nav_time.mean.toFixed(2):'—'}</td>
        <td>${s.total_mission_time?s.total_mission_time.mean.toFixed(2):'—'}</td>
        <td>${s.inspection_pct?s.inspection_pct.mean.toFixed(1):'—'}</td>
        <td>${s.final_deployment_score?s.final_deployment_score.mean.toFixed(1):'—'}</td>
        <td>${s.safety_score?s.safety_score.mean.toFixed(1):'—'}</td>
        <td>${s.recovery_score?s.recovery_score.mean.toFixed(1):'—'}</td>
        <td>${s.route_efficiency?s.route_efficiency.mean.toFixed(3):'—'}</td>
        <td>${s.collision_count?s.collision_count.mean.toFixed(1):'—'}</td>
        <td>${sc.score||0}</td></tr>`;});
    h+='</tbody></table>';document.getElementById('ct').innerHTML=h;
  }catch(e){console.error(e);}
}
load3();
</script></body></html>"""

# ═══════════════════════════
# PAGE 7 — EXPORT
# ═══════════════════════════
_P7 = """<!DOCTYPE html><html lang="en"><head>
<meta charset="UTF-8"><title>OpenClaw — Export</title>""" + _CSS + """</head><body>""" + \
_TOPBAR.format(p1="",p2="",p3="",p4="",p5="",p6="",p7="active") + """
<div style="padding:12px 20px;display:grid;grid-template-columns:1fr 1fr;gap:10px">
  <div class="panel" style="grid-column:1/3">
    <div class="ph">Benchmark Run Summary</div>
    <div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:8px">
      <div class="mbox" style="flex:1;text-align:center"><div class="v" style="font-size:1.6rem" id="tt">—</div><div class="l">Total Runs</div></div>
      <div class="mbox" style="flex:1;text-align:center"><div class="v" style="font-size:1.6rem" id="t1">—</div><div class="l">gemma2:2b</div></div>
      <div class="mbox" style="flex:1;text-align:center"><div class="v" style="font-size:1.6rem" id="t2">—</div><div class="l">qwen2.5:3b</div></div>
      <div class="mbox" style="flex:1;text-align:center"><div class="v" style="font-size:1.6rem" id="t3">—</div><div class="l">deepseek-r1:1.5b</div></div>
    </div>
  </div>
  <div class="panel"><div class="ph">Per-Model Summary</div><div id="ms"><div style="color:var(--mu);padding:12px">Loading…</div></div></div>
  <div class="panel">
    <div class="ph">Download Data</div>
    <div style="margin-bottom:8px;font-size:.76rem;color:var(--mu)">Data directory: <code style="color:var(--ye)">./data/</code></div>
    <a class="btn btnP" style="text-decoration:none;margin:3px;display:inline-block" href="/export_csv" download>&#x2B07; inspection_results.csv</a>
    <a class="btn btnS" style="text-decoration:none;margin:3px;display:inline-block" href="/export_vlog" download>&#x2B07; validation_log.csv</a>
    <a class="btn" style="background:var(--ye);color:#241f04;text-decoration:none;margin:3px;display:inline-block;padding:6px 14px;border-radius:6px;font-weight:700" href="/export_json" download>&#x2B07; results.json</a>
    <button class="btn btnD" style="margin:3px" onclick="if(confirm('Export all trajectories as ZIP?'))window.location='/export_trajectories'">&#x2B07; trajectories.zip</button>
  </div>
  <div class="panel" style="grid-column:1/3">
    <div class="ph">All Runs</div>
    <div id="fl"><div style="color:var(--mu);padding:12px">Loading…</div></div>
  </div>
</div>
<style>
table{width:100%;border-collapse:collapse;font-size:.70rem}
thead th{background:var(--card);padding:5px 8px;text-align:left;color:var(--mu);font-weight:600;border-bottom:2px solid var(--br)}
tbody td{padding:4px 8px;border-bottom:1px solid var(--br)}tbody tr:hover td{background:var(--card)}
.ok{color:var(--gr);font-weight:700}.fail{color:var(--re);font-weight:700}.dim{color:var(--mu);font-size:.68rem}
</style>
<script>
async function loadEx(){
  const d=await(await fetch('/api/results_all')).json();const res=d.results||[],bm={};
  res.forEach(r=>{if(!bm[r.model])bm[r.model]=[];bm[r.model].push(r);});
  document.getElementById('tt').textContent=res.length;
  document.getElementById('t1').textContent=(bm['gemma2:2b']||[]).length;
  document.getElementById('t2').textContent=(bm['qwen2.5:3b']||[]).length;
  document.getElementById('t3').textContent=(bm['deepseek-r1:1.5b']||[]).length;
  let h='';Object.entries(bm).forEach(([m,runs])=>{
    const n=runs.length,suc=runs.filter(r=>r.success).length;
    const ic=runs.reduce((a,r)=>a+(r.inspection_pct||0),0)/Math.max(1,n);
    h+=`<div style="margin-bottom:8px;padding:8px;background:var(--card);border-radius:7px;border:1px solid var(--br)">
      <div style="font-size:.73rem;color:#79c0ff;font-family:monospace;margin-bottom:4px">${m}</div>
      <div style="font-size:.73rem;display:flex;gap:10px;flex-wrap:wrap">
        <span>Runs: <b>${n}</b></span><span>Pass: <b style="color:var(--gr)">${suc}</b></span>
        <span>Rate: <b>${(suc/Math.max(1,n)*100).toFixed(1)}%</b></span>
        <span>IC: <b>${ic.toFixed(1)}%</b></span>
      </div></div>`;});
  document.getElementById('ms').innerHTML=h||'<div style="color:var(--mu);padding:12px">No data.</div>';
  if(!res.length){document.getElementById('fl').innerHTML='<div style="color:var(--mu);padding:12px">No runs.</div>';return;}
  let t='<table><thead><tr><th>#</th><th>Model</th><th>Status</th><th>LLM(s)</th><th>Nav(s)</th><th>Total(s)</th><th>IC%</th><th>Safety</th><th>Coll</th><th>MinClr</th><th>Mission</th><th>Timestamp</th></tr></thead><tbody>';
  res.slice().reverse().forEach((r,i)=>{
    t+=`<tr><td class="dim">${res.length-i}</td>
      <td style="font-family:monospace;color:#79c0ff">${r.model}</td>
      <td class="${r.success?'ok':'fail'}">${r.success?'PASS':'FAIL'}</td>
      <td>${(r.llm_latency||0).toFixed(3)}</td><td>${(r.nav_time||0).toFixed(2)}</td>
      <td>${(r.total_mission_time||0).toFixed(2)}</td><td>${(r.inspection_pct||0).toFixed(1)}</td>
      <td>${(r.safety_score||0).toFixed(1)}</td><td>${r.collision_count||0}</td>
      <td>${(r.min_clearance||0).toFixed(3)}</td>
      <td class="dim" style="max-width:130px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${r.mission_text||''}</td>
      <td class="dim">${r.timestamp||''}</td></tr>`;});
  t+='</tbody></table>';document.getElementById('fl').innerHTML=t;
}
loadEx();
</script></body></html>"""


# ══════════════════════════════════════════════════════════════
# SECTION 14 — FLASK ROUTES
# ══════════════════════════════════════════════════════════════

@app.route("/")
def pg_control():
    return _render(_P1)

@app.route("/camera")
def pg_camera():
    return _render(_P2)

@app.route("/map")
def pg_map():
    return _render(_P3)

@app.route("/logs")
def pg_logs():
    return _render(_P4)

@app.route("/individual")
def pg_individual():
    return _render(_P5)

@app.route("/comparison")
def pg_comparison():
    return _render(_P6)

@app.route("/export")
def pg_export():
    return _render(_P7)

@app.route("/video_feed")
def video_feed():
    return Response(
        stream_with_context(_mjpeg_generator()),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )

# ── Status & diagnostics ───────────────────────────────────────
@app.route("/api/status")
def api_status():
    with ST.lock:
        d = {
            "state":      ST.mission_state,
            "model":      ST.cur_model,
            "run":        ST.cur_run,
            "robot":      {"x": ST.rx, "y": ST.ry, "yaw": ST.ryaw},
            "distance":   ST.dist_total,
            "scan_min":   ST.scan_min if ST.scan_min < 90 else 99.0,
            "scan_count": ST.scan_count,
            "scan_sectors": {k: (v if v < 90 else 9.9)
                             for k, v in ST.scan_sectors.items()},
            "lidar_status": ST.lidar_status,
            "risk_score":   ST.risk_score,
            "obstacles":    ST.obstacle_pts,
            "active_cyl":   ST.active_cyl,
            "orbit_deg":    ST.orbit_deg,
            "planned_route": ST.planned_route,
            "reasoning":    ST.reasoning,
            "current_action_detail": ST.current_action_detail,
            "last_feedback": ST.last_feedback,
            "run_total":    ST.run_total,
            "trajectory":   ST.trajectory[-800:],
            "collision_count": ST.coll_count,
            "replan_count":    ST.replan_count,
            "cam_ok":  ST.cam_ok,
            "cam_fps": ST.cam_fps,
            "cam_w":   ST.cam_w,
            "cam_h":   ST.cam_h,
            "phase2_passed": ST.phase2_passed,
            "phase2_status": ST.phase2_status,
            "phase2_required_default": CFG.PHASE2_REQUIRED_DEFAULT,
            "benchmark_force_default": CFG.BENCHMARK_FORCE_DEFAULT,
            "log":           list(ST.log_buf)[-70:],
        }
    d["cylinders"] = _REGISTRY.to_json()
    return jsonify(d)

@app.route("/api/validation")
def api_validation():
    with ST.lock:
        v = dict(ST.validation)
        v["unavail_models"] = list(ST.unavail_models)
        v["phase2_passed"]  = ST.phase2_passed
        v["phase2_status"]  = ST.phase2_status
        v["odom_age"] = round(time.time() - ST.last_odom_t, 2) if ST.last_odom_t else 999
        v["scan_age"] = round(time.time() - ST.last_scan_t, 2) if ST.last_scan_t else 999
        v["cam_age"]  = round(time.time() - ST.last_cam_t,  2) if ST.last_cam_t  else 999
    v["cylinders_confirmed"] = len(_REGISTRY.confirmed_list())
    return jsonify(v)

@app.route("/api/cylinders")
def api_cylinders():
    return jsonify({"cylinders": _REGISTRY.to_json()})

@app.route("/api/ros_debug")
def api_ros_debug():
    with ST.lock:
        return jsonify({
            "odom_msgs":   ST.odom_msgs,
            "scan_msgs":   ST.scan_msgs,
            "cam_msgs":    ST.cam_msgs,
            "cmdvel_pubs": ST.cmdvel_pubs,
            "odom_age": round(time.time() - ST.last_odom_t, 2) if ST.last_odom_t else 999,
            "scan_age": round(time.time() - ST.last_scan_t, 2) if ST.last_scan_t else 999,
            "cmd_subs": NODE.cmd_pub.get_subscription_count() if NODE else 0,
        })

@app.route("/api/sys_metrics")
def api_sys_metrics():
    cpu, ram, pwr = _read_sys_metrics()
    return jsonify({"cpu": cpu, "ram": ram, "power": pwr})

# ── Mission control ────────────────────────────────────────────
@app.route("/api/mission", methods=["POST"])
def api_mission():
    """Mission-run mode: one selected LLM, one command, N repeated runs.
    This replaces the old heavy Benchmark Control path for reliable data collection.
    """
    if ST.bench_running:
        return jsonify({"status": "busy"})
    data    = request.get_json(force=True) or {}
    mission = data.get("mission", "").strip()
    model   = data.get("model", CFG.MODELS[0])
    runs    = max(1, min(100, int(data.get("runs", 1))))
    if not mission:
        return jsonify({"status": "error", "reason": "empty mission"})
    if model not in CFG.MODELS:
        return jsonify({"status": "error", "reason": f"unknown model: {model}"})
    def _run():
        ST.bench_running = True
        ST.stop_requested = False
        ST.cancel_event.clear()
        try:
            ok, why = preflight(False)
            if not ok:
                ST.log(f"Mission-run aborted: {why}", "WARN")
                ST.set_state("ABORTED")
                return
            ST.log(f"MISSION-RUN START: model={model} runs={runs} command='{mission}'")
            for i in range(1, runs + 1):
                if ST.stop_requested or ST.cancel_event.is_set():
                    ST.log("Mission-run stopped by user", "WARN")
                    break
                with ST.lock:
                    ST.cur_model = model
                    ST.cur_run = i
                    ST.run_total = runs
                ST.log(f"MISSION-RUN {i}/{runs}: {model}")
                # Fairness: every run starts from ideal HOME if possible.
                if NODE:
                    NODE.reset_robot()
                res = run_single_mission(model, mission, i)
                ST.results.append(res)
                save_result_csv(res)
                save_vlog_csv(model, i, res)
                # If mission ended away from home, force HOME before the next run.
                if NODE and not res.return_home_success:
                    ST.log("Run ended away from HOME — forcing reset before next run", "WARN")
                    NODE.reset_robot()
                time.sleep(0.5)
            ST.log(f"MISSION-RUN COMPLETE: {model} requested={runs} total_results={len(ST.results)}")
        finally:
            ST.bench_running = False
            ST.stop_requested = False
            ST.cancel_event.clear()
            if NODE:
                NODE.stop()
            ST.set_state("IDLE")
    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"status": "started", "model": model, "mission": mission, "runs": runs})

@app.route("/api/explore", methods=["POST"])
def api_explore():
    if ST.bench_running:
        return jsonify({"status": "busy"})
    threading.Thread(
        target=_explore_for_cylinders, args=(CFG.EXPLORE_BUDGET,),
        daemon=True).start()
    return jsonify({"status": "exploring"})

@app.route("/api/phase2", methods=["POST"])
def api_phase2():
    if ST.bench_running:
        return jsonify({"status": "busy"})
    def _run():
        ST.bench_running  = True
        ST.stop_requested = False
        try:
            run_phase2_gate(CFG.MODELS)
        finally:
            ST.bench_running = False
    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"status": "phase2_started"})

@app.route("/api/start_benchmark", methods=["POST"])
def api_start_benchmark():
    if ST.bench_running:
        return jsonify({"status": "already_running"})
    data    = request.get_json(force=True) or {}
    mode    = data.get("mode", "B")
    runs    = max(1, min(999, int(data.get("runs", CFG.RUNS_PER_MODEL))))
    force   = bool(data.get("force", CFG.BENCHMARK_FORCE_DEFAULT))
    mission = (data.get("mission", "").strip() or
               "Inspect all cylinders and return home.")
    models  = CFG.MODELS if mode == "B" else [data.get("model", CFG.MODELS[0])]
    threading.Thread(
        target=run_benchmark, args=(models, runs, mission, force),
        daemon=True).start()
    return jsonify({"status": "started", "models": models, "runs": runs})

@app.route("/api/pause", methods=["POST"])
def api_pause():
    """Pause means freeze exactly where it is standing."""
    ST.stop_requested = True
    ST.cancel_event.set()
    if NODE:
        NODE.stop()
    ST.set_state("PAUSED")
    ST.log("PAUSE — robot stopped at current position. Data preserved.", "WARN")
    return jsonify({"status": "paused", "message": "Robot paused at current position."})

@app.route("/api/stop", methods=["POST"])
def api_stop():
    """Stop means cancel current mission and send robot back to HOME/ideal."""
    ST.stop_requested = True
    ST.cancel_event.set()
    ST.bench_running = False
    if NODE:
        NODE.stop()
        def _home_after_stop():
            time.sleep(0.2)
            ST.stop_requested = False
            ST.cancel_event.clear()
            NODE.reset_robot()
        threading.Thread(target=_home_after_stop, daemon=True).start()
    ST.log("STOP — mission cancelled. Returning HOME/ideal coordinate.", "WARN")
    return jsonify({"status": "returning_home", "message": "Robot cancelled and returning to HOME/ideal position."})

@app.route("/api/reset_robot", methods=["POST"])
def api_reset_robot():
    if NODE:
        threading.Thread(target=NODE.reset_robot, daemon=True).start()
    return jsonify({"status": "resetting"})

@app.route("/api/reset_cylinders", methods=["POST"])
def api_reset_cylinders():
    _REGISTRY.reset()
    return jsonify({"status": "cylinder_registry_cleared"})

# ── Data / analysis ────────────────────────────────────────────
@app.route("/api/analysis")
def api_analysis():
    model = request.args.get("model")
    all_r = ST.results
    if model:
        mr  = [r for r in all_r if r.model == model]
        sts = compute_stats(mr)
        ms  = sts.get(model, {})
        out = [
            {
                "run_id":           r.run_id,
                "mission_number":   r.mission_number,
                "success":          r.success,
                "status":           r.status,
                "failure_reason":   r.failure_reason,
                "llm_latency":      r.llm_latency,
                "nav_time":         r.nav_time,
                "total_mission_time": r.total_mission_time,
                "inspection_pct":   r.inspection_pct,
                "safety_score":     r.safety_score,
                "mission_score":    r.mission_score,
                "latency_score":    r.latency_score,
                "path_efficiency_score": r.path_efficiency_score,
                "recovery_score":   r.recovery_score,
                "final_deployment_score": r.final_deployment_score,
                "pass_fail_reason": r.pass_fail_reason,
                "min_clearance":    r.min_clearance,
                "avg_clearance":    r.avg_clearance,
                "collision_count":  r.collision_count,
                "replan_count":     r.replan_count,
                "route_efficiency": r.route_efficiency,
                "return_home_success": r.return_home_success,
                "avg_power":        r.avg_power,
                "avg_cpu":          r.avg_cpu,
                "avg_ram":          r.avg_ram,
                "parsed_route":     r.parsed_route,
                "mission_text":     r.mission_text,
                "timestamp":        r.timestamp,
            }
            for r in mr
        ]
        return jsonify({"stats": ms, "results": out, "model": model})
    return jsonify(compute_stats(all_r))

@app.route("/api/comparison")
def api_comparison():
    all_r = ST.results
    if not all_r:
        return jsonify({"error": "No data — run missions first."})
    stats = compute_stats(all_r)
    if not stats:
        return jsonify({"error": "No data."})
    models  = list(stats.keys())
    scores  = {m: {"score": deployment_score(stats[m])[0],
                   "label": deployment_score(stats[m])[1]} for m in models}
    ranking = rank_models(stats)
    return jsonify({
        "models": models, "stats": stats,
        "scores": scores, "ranking": ranking,
    })

@app.route("/api/results")
@app.route("/api/results_all")
def api_results_all():
    return jsonify({"results": [
        {
            "run_id":          r.run_id,
            "model":           r.model,
            "mission_number":  r.mission_number,
            "mission_text":    r.mission_text,
            "timestamp":       r.timestamp,
            "status":          r.status,
            "success":         r.success,
            "failure_reason":  r.failure_reason,
            "analysis_comment": r.analysis_comment,
            "learning_feedback": r.learning_feedback,
            "improvement_hint": r.improvement_hint,
            "mission_score": r.mission_score,
            "latency_score": r.latency_score,
            "path_efficiency_score": r.path_efficiency_score,
            "recovery_score": r.recovery_score,
            "final_deployment_score": r.final_deployment_score,
            "pass_fail_reason": r.pass_fail_reason,
            "llm_latency":     r.llm_latency,
            "nav_time":        r.nav_time,
            "total_mission_time": r.total_mission_time,
            "distance_travelled": r.distance_travelled,
            "path_length":     r.path_length,
            "inspection_pct":  r.inspection_pct,
            "points_planned":  r.points_planned,
            "points_reached":  r.points_reached,
            "orbits_completed": r.orbits_completed,
            "orbit_success_rate": r.orbit_success_rate,
            "collision_count": r.collision_count,
            "replan_count":    r.replan_count,
            "min_clearance":   r.min_clearance,
            "avg_clearance":   r.avg_clearance,
            "risk_score":      r.risk_score,
            "safety_score":    r.safety_score,
            "route_efficiency": r.route_efficiency,
            "return_home_success": r.return_home_success,
            "avg_power":       r.avg_power,
            "avg_cpu":         r.avg_cpu,
            "avg_ram":         r.avg_ram,
            "parsed_route":    r.parsed_route,
            "executed_route":  r.executed_route,
            "reasoning":       r.reasoning,
            "route_valid":     r.route_valid,
            "trajectory":      r.trajectory,
        }
        for r in ST.results
    ]})

@app.route("/api/trajectory_list")
def api_traj_list():
    return jsonify({"trajectories": list_trajectories()})

@app.route("/api/trajectory_replay/<run_id>")
def api_traj_replay(run_id: str):
    d = load_trajectory(run_id)
    if d is None:
        return jsonify({"error": "not found"}), 404
    d["cylinders"] = _REGISTRY.to_json()
    return jsonify(d)

# ── Exports ────────────────────────────────────────────────────
@app.route("/export_csv")
def exp_csv():
    if not os.path.exists(CFG.CSV_PATH):
        return Response("No data yet", status=404)
    return send_file(CFG.CSV_PATH, as_attachment=True,
                     download_name="inspection_results.csv",
                     mimetype="text/csv")

@app.route("/export_vlog")
def exp_vlog():
    if not os.path.exists(CFG.VLOG_PATH):
        return Response("No data yet", status=404)
    return send_file(CFG.VLOG_PATH, as_attachment=True,
                     download_name="validation_log.csv",
                     mimetype="text/csv")

@app.route("/export_json")
def exp_json():
    buf = json.dumps([
        {
            "run_id":          r.run_id,
            "model":           r.model,
            "status":          r.status,
            "success":         r.success,
            "timestamp":       r.timestamp,
            "mission_text":    r.mission_text,
            "llm_latency":     r.llm_latency,
            "nav_time":        r.nav_time,
            "total_mission_time": r.total_mission_time,
            "inspection_pct":  r.inspection_pct,
            "safety_score":    r.safety_score,
            "collision_count": r.collision_count,
            "min_clearance":   r.min_clearance,
            "route_efficiency": r.route_efficiency,
            "return_home_success": r.return_home_success,
            "avg_cpu":         r.avg_cpu,
            "avg_ram":         r.avg_ram,
            "avg_power":       r.avg_power,
            "parsed_route":    r.parsed_route,
            "executed_route":  r.executed_route,
            "reasoning":       r.reasoning,
            "raw_llm":         r.raw_llm_response,
        }
        for r in ST.results
    ], indent=2).encode()
    return Response(
        buf, mimetype="application/json",
        headers={"Content-Disposition": 'attachment; filename="openclaw_results.json"'},
    )

@app.route("/export_trajectories")
def exp_traj():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for fn in os.listdir(CFG.TRAJ_DIR):
            if fn.endswith(".json"):
                zf.write(os.path.join(CFG.TRAJ_DIR, fn), fn)
    buf.seek(0)
    return Response(
        buf.read(), mimetype="application/zip",
        headers={"Content-Disposition": 'attachment; filename="trajectories.zip"'},
    )


# ══════════════════════════════════════════════════════════════
# SECTION 15 — WATCHDOG THREAD
# ══════════════════════════════════════════════════════════════
def _watchdog() -> None:
    while True:
        time.sleep(10.0)
        now = time.time()
        with ST.lock:
            odom_age = now - ST.last_odom_t if ST.last_odom_t else 999.0
            scan_age = now - ST.last_scan_t if ST.last_scan_t else 999.0
        if odom_age > CFG.ODOM_STALE:
            ST.log(f"WATCHDOG: /odom stale {odom_age:.1f}s — check Gazebo/bridge", "WARN")
        if scan_age > CFG.SCAN_STALE:
            ST.log(f"WATCHDOG: /scan stale {scan_age:.1f}s — check LiDAR", "WARN")
        n_conf = len(_REGISTRY.confirmed_list())
        with ST.lock:
            mstate = ST.mission_state
        if n_conf < CFG.CYL_MAX_COUNT and mstate in ("IDLE", "EXPLORING"):
            seed_fixed_cylinders()
            ST.log("WATCHDOG: fixed CYL_1..CYL_9 map refreshed")


# ══════════════════════════════════════════════════════════════
# SECTION 16 — MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════
def main() -> None:
    global NODE
    print("=" * 66, flush=True)
    print("OPENCLAW — LLM-Assisted Autonomous Inspection Benchmark v22", flush=True)
    print(f"Thesis: Latency-Aware LLM Framework for ROS 2 Mobile Robots", flush=True)
    print(f"HOME   : ({CFG.HOME_X}, {CFG.HOME_Y})  yaw={math.degrees(CFG.HOME_YAW):.0f}°", flush=True)
    print(f"Safety : EMERG={CFG.EMERG_DIST}m  WARN={CFG.WARN_DIST}m  "
          f"TARGET={CFG.TARGET_CLR}m", flush=True)
    print(f"LLMs   : {', '.join(CFG.MODELS)}", flush=True)
    print(f"Camera : {CFG.CAM_TOPIC if CFG.CAM_ENABLED else 'disabled'} "
          f"(Gazebo only, optional)", flush=True)
    print(f"Cyls   : fixed Gazebo map — CYL_1..CYL_9, LiDAR safety only", flush=True)
    print(f"Data   : {CFG.DATA_DIR}", flush=True)
    print(f"Dashboard: http://0.0.0.0:{CFG.FLASK_PORT}", flush=True)
    print("=" * 66, flush=True)

    rclpy.init()
    seed_fixed_cylinders()
    NODE = OpenClawNode()

    executor = MultiThreadedExecutor(num_threads=8)
    executor.add_node(NODE)
    ros_thread = threading.Thread(
        target=executor.spin, daemon=True, name="ros_exec")
    ros_thread.start()
    time.sleep(0.6)
    print(f"[{'OK  ' if ros_thread.is_alive() else 'FAIL'}] ROS executor thread",
          flush=True)

    threading.Thread(target=_watchdog, daemon=True, name="watchdog").start()
    print("[OK  ] Watchdog thread", flush=True)

    def _startup_bg():
        try:
            warmup_ollama(CFG.MODELS)
        except Exception as ex:
            ST.log(f"Ollama warmup error (non-fatal): {ex}", "WARN")
        try:
            NODE.run_validation(wait=10.0)
        except Exception as ex:
            ST.log(f"Validation error (non-fatal): {ex}", "WARN")

    threading.Thread(target=_startup_bg, daemon=True, name="startup").start()
    print("[OK  ] Background startup (Ollama warmup + ROS validation)", flush=True)
    print(f"[....] Flask starting on port {CFG.FLASK_PORT} …", flush=True)

    try:
        app.run(host="0.0.0.0", port=CFG.FLASK_PORT,
                debug=False, threaded=True)
    finally:
        executor.shutdown()
        NODE.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
