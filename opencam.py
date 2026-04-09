
import cv2
import sys
import time
import math
import collections
import numpy as np


# ═══════════════════════════════════════════════════════════
#  CONFIG  —  only edit this section
# ═══════════════════════════════════════════════════════════
WIDTH            = 320
HEIGHT           = 240
FPS              = 20           # higher FPS = smoother speed estimate
RTMP_URL         = "rtmp://10.159.87.99/live/testiot"

# Motion detection
MIN_AREA         = 600          # px²  — ignore contours smaller than this
DIFF_THRESHOLD   = 25           # 0-255 — pixel intensity change threshold

# Speed display
PIXELS_PER_METER = None         # Set e.g. 120.0 to also show km/h, else None
SPEED_SMOOTH_N   = 6            # number of frames to average speed over

# Danger thresholds  (in px/s)
SPEED_WARN       = 80           # yellow warning
SPEED_DANGER     = 160          # red danger

# Trail: how many past centroid positions to draw
TRAIL_LENGTH     = 20
# ═══════════════════════════════════════════════════════════


# ── GStreamer pipeline ──────────────────────────────────────────────────────────
def build_writer_pipeline(url, w, h, fps):
    """
    Pipeline string for cv2.VideoWriter with CAP_GSTREAMER.
    Explicit caps between elements stop GStreamer negotiation errors.
    """
    return (
        f"appsrc is-live=true block=true format=GST_FORMAT_TIME "
        f"caps=video/x-raw,format=BGR,width={w},height={h},framerate={fps}/1 ! "
        f"videoconvert ! "
        f"video/x-raw,format=I420 ! "
        f"x264enc tune=zerolatency bitrate=600 speed-preset=ultrafast ! "
        f"video/x-h264,profile=baseline ! "
        f"h264parse ! "
        f"flvmux streamable=true ! "
        f"rtmpsink location='{url} live=1'"
    )


# ── Startup checks ──────────────────────────────────────────────────────────────
def check_gstreamer():
    info = cv2.getBuildInformation()
    idx  = info.find("GStreamer")
    if idx == -1 or "YES" not in info[idx:idx+80]:
        sys.exit(
            "[FATAL] OpenCV is NOT compiled with GStreamer.\n"
            "  Install a GST-enabled build, e.g.:\n"
            "    sudo apt install python3-opencv        # Ubuntu/Debian\n"
            "  Or build OpenCV from source with -DWITH_GSTREAMER=ON"
        )
    print("[OK] GStreamer detected in OpenCV build.")


def open_camera():
    for backend in (cv2.CAP_V4L2, cv2.CAP_ANY):
        cap = cv2.VideoCapture(0, backend)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
            cap.set(cv2.CAP_PROP_FPS,          FPS)
            actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"[OK] Camera opened  {actual_w}x{actual_h} backend={backend}")
            return cap
    sys.exit("[FATAL] Cannot open camera index 0.")


def open_writer(pipeline):
    w = cv2.VideoWriter(pipeline, cv2.CAP_GSTREAMER, 0, FPS, (WIDTH, HEIGHT), True)
    if not w.isOpened():
        sys.exit(
            "[FATAL] VideoWriter (GStreamer) failed to open.\n"
            "  • Check RTMP_URL is reachable.\n"
            "  • Verify plugins:  gst-inspect-1.0 x264enc\n"
            "                     gst-inspect-1.0 rtmpsink\n"
            "                     gst-inspect-1.0 flvmux"
        )
    print("[OK] GStreamer VideoWriter opened — streaming started.")
    return w


# ── Speed tracker ───────────────────────────────────────────────────────────────
class SpeedTracker:
    """
    Tracks the centroid of the largest detected object across frames.
    Computes instantaneous speed (px/s) smoothed over a rolling window.
    """

    def __init__(self, history_n=SPEED_SMOOTH_N, trail_len=TRAIL_LENGTH):
        self.centroids   = collections.deque(maxlen=history_n)   # (x, y, timestamp)
        self.trail       = collections.deque(maxlen=trail_len)    # (x, y) for drawing
        self.speed_buf   = collections.deque(maxlen=history_n)    # px/s samples
        self.smooth_spd  = 0.0   # smoothed speed  (px/s)
        self.vx          = 0.0   # latest x-velocity component
        self.vy          = 0.0   # latest y-velocity component

    def update(self, cx, cy):
        """Call with the centroid of the current frame's best object."""
        now = time.monotonic()
        self.trail.append((cx, cy))

        if self.centroids:
            px, py, pt = self.centroids[-1]
            dt = now - pt
            if dt > 0:
                dx = cx - px
                dy = cy - py
                dist = math.hypot(dx, dy)           # Euclidean displacement (px)
                speed = dist / dt                    # px / s
                self.speed_buf.append(speed)
                self.vx = dx / dt
                self.vy = dy / dt

        self.centroids.append((cx, cy, now))

        # Smoothed speed = mean of recent samples
        self.smooth_spd = float(np.mean(self.speed_buf)) if self.speed_buf else 0.0

    def reset(self):
        """Call when no object is detected so old centroid doesn't jump."""
        self.centroids.clear()
        self.speed_buf.clear()
        self.smooth_spd = 0.0
        self.vx = 0.0
        self.vy = 0.0

    @property
    def speed_px(self):
        return self.smooth_spd

    @property
    def speed_kmh(self):
        if PIXELS_PER_METER is None:
            return None
        m_per_s = self.smooth_spd / PIXELS_PER_METER
        return m_per_s * 3.6

    def direction_arrow(self):
        """Returns a unicode arrow for dominant motion direction."""
        if abs(self.vx) < 5 and abs(self.vy) < 5:
            return "●"
        angle = math.degrees(math.atan2(-self.vy, self.vx))   # screen y is inverted
        dirs  = ["→","↗","↑","↖","←","↙","↓","↘"]
        idx   = round(angle / 45) % 8
        return dirs[idx]


# ── Drawing helpers ─────────────────────────────────────────────────────────────
def speed_color(speed_px):
    """Return BGR color that shifts green → yellow → red with speed."""
    if speed_px < SPEED_WARN:
        return (0, 220, 0)       # green
    elif speed_px < SPEED_DANGER:
        return (0, 200, 220)     # yellow
    else:
        return (0, 0, 255)       # red


def draw_speed_bar(frame, speed_px, x=10, y=200, bar_w=120, bar_h=10):
    """
    Horizontal speed bar in the bottom-left corner.
    Full scale = SPEED_DANGER * 1.5
    """
    full = SPEED_DANGER * 1.5
    fill = int(min(speed_px / full, 1.0) * bar_w)
    col  = speed_color(speed_px)
    cv2.rectangle(frame, (x, y), (x + bar_w, y + bar_h), (50, 50, 50), -1)
    if fill > 0:
        cv2.rectangle(frame, (x, y), (x + fill, y + bar_h), col, -1)
    cv2.rectangle(frame, (x, y), (x + bar_w, y + bar_h), (120, 120, 120), 1)


def draw_trail(frame, trail):
    """Draw fading centroid trail."""
    pts = list(trail)
    for i in range(1, len(pts)):
        alpha = i / len(pts)                          # fades older → newer
        thickness = max(1, int(alpha * 3))
        col = (int(255 * alpha), int(180 * alpha), 0)
        cv2.line(frame, pts[i - 1], pts[i], col, thickness, cv2.LINE_AA)


def draw_hud(frame, tracker, fps_live, direction_label):
    """Draws the full on-screen HUD: speed, bar, arrow, fps."""
    spd    = tracker.speed_px
    col    = speed_color(spd)
    arrow  = tracker.direction_arrow()

    # ── Speed readout ──────────────────────
    spd_text = f"{spd:5.1f} px/s"
    cv2.putText(frame, spd_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, col, 2, cv2.LINE_AA)

    # Optional km/h
    kmh = tracker.speed_kmh
    if kmh is not None:
        cv2.putText(frame, f"{kmh:.1f} km/h", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1, cv2.LINE_AA)

    # ── Direction label + arrow ────────────
    cv2.putText(frame, f"{arrow} {direction_label}", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 255), 1, cv2.LINE_AA)

    # ── Speed classification label ─────────
    if spd >= SPEED_DANGER:
        tag, tag_col = "!! FAST !!", (0, 0, 255)
    elif spd >= SPEED_WARN:
        tag, tag_col = "MOVING",     (0, 200, 220)
    else:
        tag, tag_col = "SLOW",       (0, 220, 0)

    cv2.putText(frame, tag, (10, 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, tag_col, 1, cv2.LINE_AA)

    # ── Speed bar ─────────────────────────
    draw_speed_bar(frame, spd)
    cv2.putText(frame, "SPD", (10, 225),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (160, 160, 160), 1)

    # ── Live FPS (top-right) ───────────────
    cv2.putText(frame, f"FPS {fps_live:.0f}", (WIDTH - 75, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 0), 1, cv2.LINE_AA)


# ── Main ────────────────────────────────────────────────────────────────────────
def main():
    check_gstreamer()

    pipeline = build_writer_pipeline(RTMP_URL, WIDTH, HEIGHT, FPS)
    print(f"[INFO] GStreamer pipeline:\n  {pipeline}\n")

    cap     = open_camera()
    writer  = open_writer(pipeline)
    tracker = SpeedTracker()

    # Morphological kernel — fills holes in motion mask
    kernel   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    prev_gray    = None
    fps_timer    = time.monotonic()
    frame_count  = 0
    fps_live     = 0.0

    print("[INFO] Streaming … press Ctrl+C to stop.\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.02)
                continue

            frame = cv2.resize(frame, (WIDTH, HEIGHT))

            # ── Preprocessing ────────────────────────────────────────────
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)

            if prev_gray is None:
                prev_gray = gray
                continue

            # ── Motion mask ──────────────────────────────────────────────
            diff   = cv2.absdiff(prev_gray, gray)
            _, thresh = cv2.threshold(diff, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            thresh = cv2.dilate(thresh, kernel, iterations=1)   # widen blobs

            # ── Find best (largest) contour ──────────────────────────────
            contours, _ = cv2.findContours(
                thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            max_area = 0
            best_box = None
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > MIN_AREA and area > max_area:
                    max_area = area
                    best_box = cv2.boundingRect(cnt)

            # ── Update tracker ───────────────────────────────────────────
            direction_label = "---"
            if best_box is not None:
                x, y, w, h = best_box
                cx = x + w // 2
                cy = y + h // 2

                tracker.update(cx, cy)

                # Bounding box + centroid dot
                box_col = speed_color(tracker.speed_px)
                cv2.rectangle(frame, (x, y), (x + w, y + h), box_col, 2)
                cv2.circle(frame,    (cx, cy), 4, (255, 255, 0), -1, cv2.LINE_AA)

                # Direction (uses centroid x-position on frame)
                if cx < WIDTH // 3:
                    direction_label = "LEFT"
                elif cx > 2 * WIDTH // 3:
                    direction_label = "RIGHT"
                else:
                    direction_label = "CENTER"

            else:
                tracker.reset()     # object gone — clear history so no stale jump

            # ── Trail ────────────────────────────────────────────────────
            draw_trail(frame, tracker.trail)

            # ── HUD ──────────────────────────────────────────────────────
            draw_hud(frame, tracker, fps_live, direction_label)

            # ── FPS calculation (updated every 15 frames) ─────────────────
            frame_count += 1
            if frame_count % 15 == 0:
                now      = time.monotonic()
                fps_live = 15 / (now - fps_timer + 1e-9)
                fps_timer = now

            prev_gray = gray

            # ── Write to RTMP ─────────────────────────────────────────────
            writer.write(frame)

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted — shutting down.")

    finally:
        cap.release()
        writer.release()
        print("[INFO] Camera and writer released. Done.")


if __name__ == "__main__":
    main()