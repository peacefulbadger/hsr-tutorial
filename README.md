# HSR Agent

Portable, pixels-in/actions-out game automation. Phase 0 (infrastructure and
foothold) is what exists in this repo today.

Two machines:

| Host | Address | Runs |
|---|---|---|
| Windows PC | `10.0.0.1` | MuMuPlayer (HSR 1280x720 EN) + `sidecar_windows/` |
| Linux laptop | `10.0.0.2` | Mica, perception, orchestrator, evaluator |

The Windows side is deliberately stupid: a camera and a finger behind HTTP. All
game knowledge and every decision lives elsewhere.

## Endpoints

`GET /frame` — latest frame as `image/jpeg` (or `image/png` if configured), with timing headers:
timing headers:

| Header | Meaning |
|---|---|
| `X-Frame-Seq` | monotonic frame counter |
| `X-Capture-Epoch-Ms` | sidecar clock at capture *start* (pessimistic) |
| `X-Response-Epoch-Ms` | sidecar clock when the response was built |
| `X-Server-Age-Ms` | `response - capture`, computed on one clock |
| `X-Capture-Duration-Ms` | cost of the adb screencap |

A client that cannot trust clock sync should compute an upper bound on frame age:

```
age_upper_ms = server_age_ms + rtt_ms
```

`server_age_ms` is exact (same clock, both sides of the sidecar). `rtt_ms` is
measured locally. The bound is pessimistic by at most one network leg — if it
passes your staleness budget, the truth passes too. `tools/check_lan.py` does
exactly this.

`POST /action` — `{"type": "tap", "x": 640, "y": 680}`, or
`{"type": "swipe", "x":..,"y":..,"x2":..,"y2":..,"duration_ms":..}`,
`{"type": "back"}`, `{"type": "wait", "duration_ms": ..}`.
Coordinates are framebuffer pixels (1280x720), which is also the space
`input tap` uses on this emulator build — verified by tapping and watching the
dialogue advance.

`GET /health`, `GET /frame/meta`, `GET /stats` — diagnostics.

## Run it

Use a project venv: the global interpreter on this box carries an
`opentelemetry` pin that recent `fastapi` releases fight with.

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --no-cache-dir -r requirements.txt
.\.venv\Scripts\python.exe -m sidecar_windows.server --config configs\sidecar.toml
```

(`--no-cache-dir` avoids a pip cache-lock hang seen when two installs overlap.)

Then from the Linux laptop:

```bash
python tools/check_lan.py --url http://10.0.0.1:8765 --n 20
```

Tests need no emulator, no adb, no network: `.\.venv\Scripts\python.exe -m pytest`

## Measured on this rig

MuMuPlayerGlobal 12.0 (SM_A528B), HSR `com.HoYoverse.hkrpgoversea`, 1280x720,
2026-10-29.

| Capture path | Latency | Frame size |
|---|---|---|
| `screencap -p` (device-side PNG) | 315 ms | 1.23 MB |
| `screencap` raw framebuffer | 132 ms | 3.69 MB |
| raw + local JPEG q80 (**default**) | 142 ms | 127 KB |

End-to-end through HTTP, 15 samples:

| Target rate | Effective | Frame age p50 / p95 | Bandwidth |
|---|---|---|---|
| 4 Hz | 4.00 Hz | 250 / 359 ms | ~4.1 Mbps |
| 6 Hz (**default**) | 5.95 Hz | 219 / 297 ms | ~6 Mbps |
| 7 Hz | 6.94 Hz | 203 / 281 ms | ~7 Mbps |

Two conclusions that contradict the plan's assumptions:

1. **PNG is the wrong transport.** The plan assumed 1–4 Mbps and did not budget
   encode latency; on-device PNG compression costs 315 ms/frame and caps capture
   at 3 Hz. Fetching raw pixels and encoding JPEG here is 2.4x faster and 10x
   smaller per frame. `--frame-format png` exists if the literal contract matters.
2. **Input latency is not the bottleneck, capture is.** A tap dispatches in 25 ms.
   Frame freshness is bounded below by one screencap (~132 ms), so no amount of
   tick rate gets below ~200 ms p50. Plan settle delays around 300–400 ms.

## Phase 0 gate status

Gate: *a script resets the account, launches the client, captures a frame over
LAN, taps the first screen, and captures the next.*

`tools/gate0.py --config configs/gate0.hsr.json` runs the whole sequence. Last
run: **12/12 checks in 98.7 s**, from this host.

- [x] adb connected, `screencap` + `input` verified at 1280x720
- [x] sidecar serves frames with timestamps, dispatches taps
- [x] LAN reachable at `http://10.0.0.1:8765` (python.exe inbound rules present)
- [x] relaunch -> title -> "Start Game" -> "Tap to Start" -> in-game, then
      capture -> tap -> capture with the pixels changing (12/12)
- [x] `EmulatorController.reset_app()`: force-stop, relaunch, process up in ~1.1 s
- [ ] **account reset** -- the script relaunches the client, which is *not* an
      account reset: tutorial progress is checkpointed server-side and survives a
      relaunch (verified: it resumed on the same dialogue line). Needs the
      private server's admin path, or an explicit "relaunch is enough".
- [ ] **measured from 10.0.0.2** -- no SSH key here. One command on the laptop:
      `python tools/check_lan.py --url http://10.0.0.1:8765`

### Reset cost, measured

Relaunch to playable takes **~99 s**: ~1 s to relaunch, ~15 s of black while the
client boots, ~40 s before the SDK accepts input on the title screen (the script
retries the tap until brightness jumps), ~25 s to load the scene. Anything that
resets per episode pays this, so the "reset flow too slow for iteration" risk in
the plan is real, and an emulator snapshot path matters.

### Traps found while writing the gate

- The client has no readable "ready" state; brightness is the cheapest proxy
  (title ~47, "Tap to Start" ~120, in-game ~97). Pixel diff is useless on the
  title screen because its background drifts, which produced a false positive
  and cost a whole failing run.
- A single tap is unreliable: the SDK appears ~40 s before it accepts input.

## Phase 1 gate status

Gate: *on a recorded tutorial run, >=95% of frames get a correct screen type;
choices and interactive-element labels are extractable with confidence sufficient
for fuzzy matching against the reference.*

Screen-type classification scores **43/45 = 95.6%** on hand-labelled frames across
seven classes, including a real `reward` screen captured by fighting the tutorial
battle to "Battle Over".

| class | n | recall |
|---|---|---|
| dialogue | 9 | 100% |
| dialogue_bark | 1 | 100% |
| combat | 29 | 97% |
| menu | 3 | 100% |
| loading | 1 | 100% |
| reward | 1 | 100% |
| transition | 1 | 0% |

The taxonomy is **nine classes, one more than the plan lists**. The extra one is
`dialogue_bark`: a subtitle that plays while the character keeps walking, which is
visually a dialogue and functionally nothing at all. It is a separate class
because folding it in would make the policy shell tap "continue" on every ambient
line, forever. Barks are told apart by whether the movement HUD survives — the
sprint/attack cluster measures 0.038 near-white during a bark and <= 0.0075 on all
nine blocking-dialogue frames — and a bark deliberately exposes *no* interactive
elements, because pressing something is exactly what it does not ask for.

**The gate is still not met**, on two counts:

1. **`dialogue_choice` has no recorded example.** `choices` extraction is
   unimplemented, so that half of the gate is missing, not merely unmeasured.
   The other seven classes are now exercised.
2. **Thresholds and regions were tuned on the same frames they are scored on.**
   95.5% is an optimistic bound, not an estimate.

The independent check on unlabelled data still holds: classifying all of
`prologue-01` flips `dialogue` -> `combat` at 68s/73s, exactly where the session
changed.

### Note on exploration

Walking around resolves to `unknown`, and that is a plan gap rather than a
classifier failure: the taxonomy has no `exploration` class although it is the
state the agent spends most of its time in. Recorded as a test so it stays
visible.

### The two misses, and why one of them is structural

- `combat -> dialogue` on an attack-animation frame: for ~1s in every attack the
  teal ring is gone and the tutorial prompt is not drawn, so the "combat" signals
  genuinely are absent. Fixing it needs a temporal prior (the screen was combat a
  frame ago), which the orchestrator has and a stateless classifier does not.
- `transition -> loading` is not a bug: a single black frame cannot distinguish a
  scene transition from a load. That distinction is a *duration*, so it belongs
  in the orchestrator's state machine. The taxonomy keeps both labels; perception
  reports the ambiguous one and the test asserts the ambiguity rather than
  pretending otherwise.

### What perception actually does

`frame -> features -> (maybe OCR) -> GameState`

1. Near-black frame -> `loading`, no OCR at all.
2. Reward keyword ("Battle Over", ...) -> `reward`, checked *before* the combat
   fast lane because the result screen keeps the combat HUD behind it.
3. Teal action ring bottom-right -> `combat` (one pixel pass).
4. Otherwise OCR once and read text: combat prompt above the subtitle strip ->
   `combat`; title keywords -> `menu`; a subtitle line -> `dialogue`.
5. Dark and detail-free -> `transition`, else `unknown`.

### Four things that were wrong on the first attempt

- **Region brightness does not classify HSR screens.** The first classifier used
  only pixel statistics and scored 33%. The scene behind the HUD dominates:
  the dialogue band is 93% dark on one frame and 71% on another, and combat loses
  its teal ring entirely during attack animations. Text is the stable signal.
- **The tutorial banner moves.** It sits at y~290 next to the target reticle and
  y~558 beside the action button, so a fixed banner region fails silently on half
  the combat steps.
- **OCR output needs normalising.** It returns `Start.Game` and `Use BasicATKon`,
  so exact substring matching misses both. Compare alphanumerics only.
- **The combat button layout swaps.** The active action is the large button at
  bottom-right; after "Switch to your Skill" the skill moves into that slot and
  the basic attack becomes the small greyed one. Tapping a fixed point for
  "attack" silently does nothing for long stretches.

### OCR engine

The plan specifies PaddleOCR. PaddlePaddle will not install on this box: it
tries to downgrade numpy, which breaks the pinned opencv build, then dies on a
locked `cv2.pyd`. RapidOCR runs the same PP-OCR models on onnxruntime and installs
in 7 s. `OcrEngine` is a Protocol, so swapping PaddleOCR in later is one class.

## Navigating the world (what the markers mean)

Three gold diamonds exist and they are not interchangeable. Getting this wrong
cost several failed attempts:

| Marker | Where | What it tells you |
|---|---|---|
| world pip | in the 3D scene, with an `Nm` label | the objective's position. Moves with character position *and* camera orientation, so its screen x is a fine steering error once it is visible |
| minimap pip | inside the minimap circle, player is the cyan arrow | a *world* bearing to the objective. Clamped to the rim beyond minimap range, so the direction survives when the distance does not |
| quest panel pip | fixed UI, left side | decoration; it never moves |

Two things about the minimap had to be measured rather than assumed, because the
naive reading of it is wrong:

- **The disc does not rotate when the camera free-looks.** Rotating the camera with
  a right-half drag and matching the disc against itself gives a best-match rotation
  of 0.0 deg, clearly beating +-1 deg, while the 3D scene visibly turns. So a pip
  offset is a bearing in the world, not a "turn the camera this much" instruction.
  Using it as one is a bug that shows up whenever the camera and the character
  disagree, which near a wall is most of the time.
- **The cyan arrow is the character's facing, and it is readable.** The icon is a
  dart: solid nose, V-notch in the tail. Filling the convex hull and taking the
  largest uncovered piece finds the notch, and the nose points from the notch
  through the body. Measured against a dart pointing straight down it reads
  172.8 / 176.4 / 173.0 deg where a principal-axis fit reads 215 / 208 / 215.
  `adapters/game_hsr/minimap.py` does this; the notch estimator is primary and an
  axis-profile fallback covers the notch-less case.

With facing known, the world bearing becomes a controllable error, because pushing
the stick turns the character to face the push. Holding the camera still, a push at
stick angle `s` makes her face world angle `yaw + s`, so correcting `s` by
`bearing - facing` converges without ever knowing `yaw`. `tools/walk_to_marker.py`
runs that as an explicit heading phase, then a camera-sweep phase to bring the world
pip on screen, then screen-space steering. Measured: 177 deg of heading error in two
pushes, then the pip found after two sweeps, then 39 m to 8 m of walking.

The two live markers are still a coarse/fine pair -- the minimap says *which way to
face*, the world pip says *how to walk* once you can see it -- but the coarse half
is the arrow's facing, not a camera correction in disguise.

Trap: the pip's `Nm` distance reads as an arrival when it is not. Standing at 5 m
with the tutorial banner *"Attack selected enemy to trigger combat"* up, the stick
does nothing: the game is gated on a tap, and the loop that only walks reports a
stuck character. Check for a tutorial instruction before concluding the walk failed.


Interaction is a labelled button, named after the object: `Investigate` at one
console, `Controller` at another. Since the label is arbitrary, detection is
structural — a short, confidently-read text box inside the prompt region, not a
keyword list. A keyword search for "investigate"/"interact"/"read" missed
`Controller` for fourteen sweep attempts and looked exactly like "no prompt".

The prompt only renders when the character is in range *and* facing the object,
so a failed search means "turn further", not "nothing here".

Trap: the world pip's `Nm` label is near the objective, not necessarily the thing
to press. At 1 m the pip can be marking a doorway or a trigger volume, in which
case there is no interaction prompt at all and walking through it is the answer.

## Controls that actually move the character

- Left stick is a virtual analogue stick at (250, 570); a swipe from there moves.
  Camera-relative, and the camera follows the heading, so repeated "up" swipes
  curve rather than go straight.
- Camera drags on the right half (around (950, 360)) rotate the view. Required to
  face an interactable -- the `Investigate` prompt only appears when in range
  *and* facing it.
- Sprint toggles at (1180, 600).

## Layout

```
sidecar_windows/     runs on Windows: adb -> HTTP
  adb.py             adb wrapper (auto-detects MuMu's adb + serial)
  capture.py         capture thread, frame timestamps
  imaging.py         raw framebuffer -> BGR -> JPEG
  server.py          FastAPI app
emulator_control/    force-stop / relaunch the client
tools/check_lan.py   latency + freshness probe, run from the laptop
tools/gate0.py       the Phase 0 gate runner
tools/record.py      replay recorder (frames + JSONL index)
tools/evaluate_perception.py   screen-type scoring against labels
tools/classify_replay.py       timeline over a whole replay
tools/label_sample.py          labelling sheet generator
core/                game-agnostic contracts (state, action)
adapters/game.py     the GameAdapter protocol
adapters/game_hsr/   everything HSR-specific
  regions.py         UI geometry, read off real frames
  features.py        cheap per-frame signals
  perception.py      classifier + OCR + GameState assembly
  render.py          GameState -> Mica text
  resolve.py         action ids -> coordinates
configs/             runtime config
tests/               no emulator, no adb, no network
```
