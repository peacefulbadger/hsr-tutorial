"""Attack the selected enemy until combat ends, stopping on anything that needs thought.

The turn-based fight itself is not interesting to reason about: the skills are on a fixed
layout and the tutorial only ever asks for one of them, so a loop that presses the active
action and notices when the fight is over is enough. What matters is the *exit*: the loop
must stop on a reward screen, a blocking dialogue, or a dialogue with choices, because
those are the states where a blind press either skips story or picks an answer.

    python tools/fight.py --seconds 240
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adapters.game_hsr.perception import HsrPerception, RapidOcr  # noqa: E402

# The large action button bottom-right: basic attack, and the skill when it replaces it.
# This layout is the 1280x720 one measured off captured frames; the button sits slightly
# lower and further right than the naive "bottom right corner" guess.
ACTION_BUTTON = (1045, 590)
# The skill sits above the basic attack, bottom-right, and is the one that costs a point.
# The tutorial names it ("Switch to Skill") and highlights it, so it is worth distinguishing:
# pressing the attack into that gate does nothing at all.
SKILL_BUTTON = (1180, 500)
# Ultimate sits on the party portraits, bottom-left, and only glows at full energy. Which
# portrait glows changes with who is charged, so the point is measured, not fixed: the
# charged circle is the brightest blob in this region. (Hard-coding (230,590) pressed the
# first slot no matter which character was ready and did nothing for hundreds of seconds.)
ULTIMATE_REGION = (150, 460, 520, 220)
ULTIMATE_FALLBACK = (430, 570)
# Where tutorial tooltips open, mid-left. Their dismiss chevron sits inside the panel's
# lower-right corner; while one is open no other press registers.
TUTORIAL_TIP_REGION = (240, 85, 400, 215)
# The spoken line, for telling a real dialogue from combat's floating text.
DIALOGUE_BAND = (150, 520, 980, 190)
# Continue affordances: reward screens and dialogues advance on a click near the centre.
CONTINUE_POINT = (640, 400)
STOP_ON = {"dialogue", "dialogue_choice", "reward"}
# Combat frames are dimmed by tutorial overlays often enough that the classifier reports
# "unknown" while a fight is plainly running. Tapping through "unknown" is safe -- during
# an enemy turn the press is ignored -- and stopping there would stall the fight forever.
TAP_THROUGH = {"combat", "unknown"}
# Where the defeat panel's Retry button sits when its label cannot be read.
RETRY_FALLBACK = (640, 470)


def ultimate_point(frame: np.ndarray) -> tuple[int, int]:
    """Centre of the glowing ultimate button, or the middle of the left party row.

    A charged ultimate draws a bright cyan ring around its portrait. Ultimates can be used
    out of turn -- the tutorial asks for one *while the enemy is acting* -- so which
    character is charged is the only thing that decides where to press.
    """
    x, y, w, h = ULTIMATE_REGION
    crop = frame[y:y + h, x:x + w]
    if crop.size == 0:
        return ULTIMATE_FALLBACK
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    value, saturation = hsv[:, :, 2], hsv[:, :, 1]
    glow = ((value > 200) & (saturation > 90)).astype(np.uint8)
    glow = cv2.morphologyEx(glow, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    count, _labels, stats, centroids = cv2.connectedComponentsWithStats(glow, 8)
    best, best_area = None, 0
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area > best_area:
            best, best_area = index, area
    if best is None or best_area < 200:
        return ULTIMATE_FALLBACK
    cx, cy = centroids[best]
    return int(x + cx), int(y + cy)



def tutorial_tip_point(boxes, frame: np.ndarray) -> tuple[int, int] | None:
    """The dismiss arrow of a tutorial tooltip, if one is open.

    Tutorials show two different blocking things and they need different presses. A
    *banner* (top of the screen) names a button and waits for that button. A *tooltip*
    (a bordered panel mid-left) waits for its own down-chevron at the bottom-right, and
    while it is open nothing else registers -- the fight sits on "Enemy's Turn" forever,
    which looks exactly like a frozen game. The chevron sits just inside the panel's
    lower-right corner, and the panel hugs its text, so the text union locates it.
    """
    x, y, w, h = TUTORIAL_TIP_REGION
    inside = [box for box in boxes if _centre_in(box.bbox, (x, y, w, h))]
    if not inside:
        return None
    left = min(box.bbox[0] for box in inside)
    top = min(box.bbox[1] for box in inside)
    right = max(box.bbox[0] + box.bbox[2] for box in inside)
    bottom = max(box.bbox[1] + box.bbox[3] for box in inside)
    return min(620, right + 18), min(300, bottom + 22)


def _centre_in(bbox, region) -> bool:
    x, y, w, h = bbox
    rx, ry, rw, rh = region
    cx, cy = x + w / 2.0, y + h / 2.0
    return rx <= cx <= rx + rw and ry <= cy <= ry + rh


def retry_point(boxes) -> tuple[int, int] | None:
    """The Retry button on a defeat screen, if this is one.

    The standing policy is to retry rather than stop: these tutorial fights are scripted
    and winnable, and a wipe is a state to recover from, not a thing to report. The button
    is located by its own label instead of a fixed point, since the defeat panel lays out
    differently from the in-fight HUD.
    """
    for box in boxes:
        text = box.text.lower()
        if "retry" in text:
            x, y, w, h = box.bbox
            return x + w // 2, y + h // 2
    return None


def looks_like_defeat(boxes) -> bool:
    compact = "".join(ch for ch in " ".join(b.text for b in boxes).lower() if ch.isalnum())
    return any(keyword in compact for keyword in ("defeat", "challengefailed", "missionfailed"))


def tutorial_request(boxes) -> str | None:
    """What the tutorial banner is asking for, when it is asking for something.

    The tutorial gates input: it dims the UI and waits for one specific press, so an
    attack tap into that state is discarded and the fight silently stops progressing.
    Its banner says which press it wants in plain words, which is the only reliable
    signal -- the button it highlights is drawn differently every time it changes.

    The banner is also *sticky*: it keeps saying "activate your Ultimate" while the
    enemy is acting, because the ultimate is still ready. Pressing then does nothing (the
    button is inert), so the caller has to combine this with `whose_turn`.
    """
    for box in boxes:
        text = box.text.lower()
        if "activate your ultimate" in text or "ultimate now" in text:
            return "ultimate"
    compact = "".join(ch for ch in " ".join(b.text for b in boxes).lower() if ch.isalnum())
    if "switchtoskill" in compact or "switchtoyourskill" in compact:
        return "skill"
    return None


def whose_turn(boxes) -> str:
    """'ours', 'enemy', or 'unknown', from the turn banner.

    Tapping during the enemy's turn is a no-op at best; the ultimate button in particular
    sits on the party portraits, so repeated presses there risk switching characters.
    """
    compact = "".join(ch for ch in " ".join(b.text for b in boxes).lower() if ch.isalnum())
    if "enemysturn" in compact:
        return "enemy"
    if "yourturn" in compact:
        return "ours"
    return "unknown"


def fetch(base: str) -> np.ndarray:
    with urllib.request.urlopen(f"{base}/frame", timeout=10) as response:
        return cv2.imdecode(np.frombuffer(response.read(), np.uint8), cv2.IMREAD_COLOR)


def act(base: str, body: dict) -> None:
    request = urllib.request.Request(
        f"{base}/action", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    urllib.request.urlopen(request, timeout=15).read()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fight until the game asks for input")
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    parser.add_argument("--seconds", type=float, default=240.0)
    parser.add_argument("--interval", type=float, default=1.1)
    parser.add_argument("--out", type=Path, default=Path("data/phase1"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    base = args.url.rstrip("/")
    args.out.mkdir(parents=True, exist_ok=True)
    perception = HsrPerception(RapidOcr())

    deadline = time.monotonic() + args.seconds
    last = ""
    seen_combat = False
    stop_confirmations = 0
    losses = 0
    while time.monotonic() < deadline:
        frame = fetch(base)
        screen, confidence, _features, boxes = perception.analyze(frame)
        screen = "dialogue" if screen == "dialogue_bark" else screen

        if screen != last:
            print(f"{time.strftime('%H:%M:%S')} {screen} ({confidence:.2f})")
            last = screen

        if looks_like_defeat(boxes):
            point = retry_point(boxes) or RETRY_FALLBACK
            print(f"{time.strftime('%H:%M:%S')} defeat screen -> retry at {point}")
            if not args.dry_run:
                act(base, {"type": "tap", "x": point[0], "y": point[1]})
            losses += 1
            time.sleep(args.interval * 2)
            continue

        if screen in TAP_THROUGH:
            seen_combat = True
            tip = tutorial_tip_point(boxes, frame)
            if tip is not None:
                print(f"{time.strftime('%H:%M:%S')} tutorial tooltip open -> dismiss {tip}")
                if not args.dry_run:
                    act(base, {"type": "tap", "x": tip[0], "y": tip[1]})
                time.sleep(args.interval)
                continue
            turn = whose_turn(boxes)
            request = tutorial_request(boxes)
            if request == "ultimate":
                # Ultimates interrupt, so this is the one press that is wanted during the
                # enemy's turn. Anything else can wait for ours.
                point = ultimate_point(frame)
                print(f"{time.strftime('%H:%M:%S')} tutorial asks for the ultimate "
                      f"({turn}) -> tap {point}")
                if not args.dry_run:
                    act(base, {"type": "tap", "x": point[0], "y": point[1]})
                time.sleep(args.interval)
                continue
            if turn == "enemy":
                time.sleep(args.interval)
                continue
            point = SKILL_BUTTON if request == "skill" else ACTION_BUTTON
            if request == "skill":
                print(f"{time.strftime('%H:%M:%S')} tutorial asks for the skill -> tap {point}")
            if not args.dry_run:
                act(base, {"type": "tap", "x": point[0], "y": point[1]})
            time.sleep(args.interval)
            continue

        if screen in STOP_ON and seen_combat:
            # "dialogue" mid-fight is usually a lie: damage numbers, an ultimate name, or
            # the character close-up of an ultimate animation all read as subtitle text.
            # A real blocking dialogue has a spoken line, so ask for one.
            subtitle = " ".join(box.text for box in boxes
                                if _centre_in(box.bbox, DIALOGUE_BAND)).strip()
            if screen == "dialogue" and len(subtitle) < 8:
                if not args.dry_run:
                    act(base, {"type": "tap", "x": ACTION_BUTTON[0], "y": ACTION_BUTTON[1]})
                time.sleep(args.interval)
                continue
            # Combat frames also flicker, so demand the stop state twice before believing it.
            stop_confirmations += 1
            if stop_confirmations < 2:
                time.sleep(args.interval)
                continue
            cv2.imwrite(str(args.out / f"fight-stopped-{screen}.jpg"), frame)
            print(f"stopped on {screen}; saved fight-stopped-{screen}.jpg")
            return 0
        stop_confirmations = 0

        if screen == "reward":
            if not args.dry_run:
                act(base, {"type": "tap", "x": CONTINUE_POINT[0], "y": CONTINUE_POINT[1]})
            time.sleep(args.interval)
            continue

        # Loading, transition, menus between rounds: just wait for the next state.
        time.sleep(args.interval)

    print("time budget exhausted")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
