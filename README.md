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

`GET /frame` 窶・latest frame as `image/jpeg` (or `image/png` if configured), with
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
measured locally. The bound is pessimistic by at most one network leg 窶・if it
passes your staleness budget, the truth passes too. `tools/check_lan.py` does
exactly this.

`POST /action` 窶・`{"type": "tap", "x": 640, "y": 680}`, or
`{"type": "swipe", "x":..,"y":..,"x2":..,"y2":..,"duration_ms":..}`,
`{"type": "back"}`, `{"type": "wait", "duration_ms": ..}`.
Coordinates are framebuffer pixels (1280x720), which is also the space
`input tap` uses on this emulator build 窶・verified by tapping and watching the
dialogue advance.

`GET /health`, `GET /frame/meta`, `GET /stats` 窶・diagnostics.

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

1. **PNG is the wrong transport.** The plan assumed 1窶・ Mbps and did not budget
   encode latency; on-device PNG compression costs 315 ms/frame and caps capture
   at 3 Hz. Fetching raw pixels and encoding JPEG here is 2.4x faster and 10x
   smaller per frame. `--frame-format png` exists if the literal contract matters.
2. **Input latency is not the bottleneck, capture is.** A tap dispatches in 25 ms.
   Frame freshness is bounded below by one screencap (~132 ms), so no amount of
   tick rate gets below ~200 ms p50. Plan settle delays around 300窶・00 ms.

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

## Layout

```
sidecar_windows/     runs on Windows: adb -> HTTP
  adb.py             adb wrapper (auto-detects MuMu's adb + serial)
  capture.py         capture thread, frame timestamps
  imaging.py         raw framebuffer -> BGR -> JPEG
  server.py          FastAPI app
emulator_control/    force-stop / relaunch the client
tools/check_lan.py   latency + freshness probe, run from the laptop
configs/             runtime config
tests/               fake-adb tests, no emulator needed
```
