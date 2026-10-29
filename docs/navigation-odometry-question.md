# Question to a planning model: unblock exploration navigation in the HSR tutorial agent

I need a detailed project plan for one subsystem of an existing, working-at-the-edges agent.
The repository is at `C:\Users\thekl\Documents\00\hsr-tutorial` (branch `master`, HEAD
`f8a4f41`). Read the code before proposing anything: I am asking for a plan that fits the
code that exists, not a greenfield design.

The system already wins the scripted tutorial fights, advances dialogue, picks dialogue
options, and survives lost fights. It cannot reliably **walk to a quest marker in the
exploration sections**, and walking is the only thing between us and the primary objective.

My working hypothesis for the next step is **visual odometry on the minimap disc plus a
persistent room map** (§4). I want your judgement on that, and if you agree, a phased plan
to build it. If you think it is the wrong investment, say so and propose what to do instead.

---

## 1. What the system is

- **Goal (from the original plan, attached separately as `hsr-tutorial-v1.md`):** complete the
  HSR tutorial end to end, no human intervention, pixels in / actions out, no SDK.
- **Portability contract:** all game-specific code lives in `adapters/game_hsr/`; `core/` must
  stay game-agnostic. `core/` currently holds only `core/state.py` and `core/action.py` (the
  generic `GameState` / `Action` schemas from plan §5).
- **Windows PC** runs MuMuPlayer plus a dumb sidecar (`sidecar_windows/server.py`, FastAPI):
  `GET /frame` (JPEG 1280x720), `POST /action`, `GET /health`, driven over ADB
  (`127.0.0.1:16384`). Capture runs ~6 Hz. Frames carry `X-Frame-Seq`,
  `X-Capture-Epoch-Ms`, `X-Server-Age-Ms`. Roughly 70k frames and 1800 dispatched actions so far.
- **Perception** (`adapters/game_hsr/perception.py`, `regions.py`, `features.py`) is OCR
  (RapidOCR) plus structural rules, classifying into
  `loading | dialogue | dialogue_bark | dialogue_choice | combat | reward | menu | transition | unknown`.
  Exploration frames land in `unknown`.
- **Minimap reader** (`adapters/game_hsr/minimap.py`) and **route planner**
  (`adapters/game_hsr/navigation.py`) exist and are unit-tested; they are the subsystem in
  question.
- **Policies are local heuristics**, dispatched by `tools/play.py` to `tools/fight.py`,
  `tools/advance_dialogue.py`, `tools/walk_to_marker.py`. **Mica was never integrated** — that
  was a deliberate reversal of plan §4 (decisions stay local). Flag if you think that reversal
  is now the problem; I do not.
- **Tests:** 112 passing, none require an emulator (`tests/fixtures/` holds one frame per screen
  type). `data/` is gitignored and holds ~4400 captured frames.

## 2. Where we are against the original plan

| Plan phase | State | Evidence |
|---|---|---|
| **0. Infrastructure** | Mostly done | sidecar + ADB + LAN check (`tools/gate0.py`, `tools/check_lan.py`); client pinned 1280x720 EN; `emulator_control/reset_app.py` force-stop+relaunch. **Not done:** Mica on the 3080 Ti, server-admin account reset, live-endpoint firewall assertion. |
| **1. Perception adapter** | Done and beyond the plan | OCR + regions + screen classifier + replay harness (`tools/record.py`, `classify_replay.py`, `evaluate_perception.py`). Taxonomy gained `dialogue_bark`. Gate 1 (>=95% screen-type accuracy on a recorded run) was never formally measured, and **there is no labelled reference trace**. |
| **2. Decision layer with Mica** | Deliberately replaced | No `core/decision.py`, `policy_shell.py`, `mica_adapter.py`, no orchestrator, no episode logger/evaluator. Replaced by per-screen-type local policies in `tools/`. **Gate 2 as written is unsatisfiable in this design and needs re-scoping.** |
| **3. Full tutorial completion (Gate 3)** | Blocked | Combat and dialogue pass; exploration navigation does not. The "Ralph loop" is running manually: run, observe, fix. |
| **4. Portability validation** | Not started | — |

So: the unplanned part is winning. The plan specifies perception and decision; it says nothing
about moving a character through a 3D room, which is what the tutorial actually requires.

## 3. The blocking issue, with measurements

The goal: `Chaos In the Deep — "Assist Silver Wolf with investigating the terminals 0/2 in the
Monitoring Room"`. Two terminals, each with a gold diamond on the minimap disc and a labelled
gold diamond in the 3D view ("`16m`"). The character must walk to one and press its interact
prompt. She starts a few metres away and **cannot close the last 5-14 m**.

Measured facts about the minimap (all from live frames, all in `minimap.py` / `navigation.py`):

- Disc is 140x140 px at radius 70, centre `(105,130)` in frame space; scale ~0.3 m/px.
- **Camera free-look does not rotate the disc** (best-match rotation 0.0 deg, beating +-1 deg), so
  a pip offset is a world bearing, not a camera rotation. Camera yaw is a separate unknown.
- Disc content translates when the stick moves, i.e. it tracks the character.
- White lines are geometry. Brightness does not separate them from floor (wall V 46-75, floor
  V 78-141) but **saturation does** (lines S 0-28, floor S 39-84). Threshold in use:
  `sat < 45 and val > 150`, ~21% of the inner disc reads as wall.
- The player arrow is a ~17-20 px cyan dart; facing comes from the notch in its tail. It is drawn
  **over** the map, so it hides ~5 m of floor, mostly ahead of her.
- A pip beyond disc range is **clamped to the rim**, so it names a direction and not a place.

Five distinct failure mechanisms, each reproduced:

1. **Pip ambiguity.** The 0/2 objective puts two gold diamonds on the disc (measured areas 97 and
   134, ~200 deg apart). "Take the largest" flips between them frame to frame, which is a 200 deg
   change of objective; the walker turns round and walks back.
2. **Rim-clamped goal.** Planning to the rim pixel asks "reach that one wall tile". Routes set off
   the wrong way and bend up to 178 deg to come back.
3. **Plan instability.** The grid is re-derived from the frame every time, and two routes round the
   room are within a cell or two of each other in cost, so pixel noise picks between them.
   Successive frames gave detours of `+52`, `-161`, `+151` deg and waypoint bearings swinging
   +-150 deg. A 30-step run driven by that plan was a spin on the spot, 0 m of progress.
4. **The blind spot under the arrow.** A desk standing in front of her is entirely under the nose
   of the dart, so "everything that is not a line is floor" declares it walkable and the route goes
   straight into it. She then stalls (distance frozen for 20+ steps) while the disc shows nothing
   in the way.
5. **Arrival is not interaction.** She reaches the terminal, no prompt is detected, and the walk
   reports success. The prompt plate is drawn in at least two places (a `Controller` fixture at
   x~822 on the right, and the choice band x 380-1180 y 250-630), and the detector only looks at
   the first.

I have already tried the obvious patches and **reverted them**, so you can read the failures
without them in the way. What was tried, and why it was not enough:

- **A: treat the floor under the arrow as unknown** (third cell state, one-off toll to enter, so
  routes prefer visible floor). This fixed nothing on its own: it cannot tell a desk from empty
  floor it merely cannot see, and a per-cell penalty made the route orbit the dart instead.
- **B: a bump map** — a stall is evidence, the facing bearing at the stall is where the solid
  thing is, blocked for a few frames, and the walker commits to one side (hold 5 steps, grow
  55 to 80 deg, flip after 3). This *works at contact level* (log: `stuck at 16m, bump at
  311 deg; detouring 55 deg right` then `1m closer after the detour`) but is reactive and
  forgets.
- Also in that reverted commit: a pip-candidate selector (all candidates, previous bearing picks
  between them), the disc-crop fix (planner cropped at r=66 while the minimap reports at r=70, so
  every wall was one cell off), the rim-arc goal, and the widened prompt region. **All of it is
  available in git tag `ab-try`** (`3088132`; A+B in `f270e13`, prompt in `3088132`). Your plan
  should say which of those to reinstate and in what order.

Behavioural symptoms that make this expensive: a stall loop burns game time (~1.5 m per 0.75 s
push plus ~0.3 s per decision), the fights auto-retry, and input has died silently once for
~25 minutes (frames fine, actions dispatched, nothing registered) and only a package relaunch
fixed it.

## 4. Proposed next step: visual odometry plus a persistent room map

The idea. If consecutive discs are registered against each other, the registration *is* the
character's displacement, and the frames can be merged into a grid that belongs to the room rather
than to this frame. Then plan on the merged grid.

The property that makes it worth building: **it does not require resolving whether the disc is a
player-centred window or a world-fixed one.** In both cases the disc's origin is anchored to the
character, so the disc content slides by exactly her displacement between frames. Camera free-look
provably does not rotate it, so there is no rotation term. `cv2.phaseCorrelate` on the smoothed
line masks should give a (dx, dy) in disc pixels; 0.3 m/px turns it into metres.

Three pieces:

1. **Odometry.** Phase-correlate consecutive line masks. Reject implausible steps (>10 px per push)
   and flat correlations (uniform regions, the arrow's own footprint).
2. **Room map.** An occupancy grid in the *room* frame, ~0.5 m/cell, that each frame's lines are
   merged into and never un-learned within a visit. Memory fills the arrow's blind spot; a desk
   seen from one side stays on the map when she walks round it.
3. **Plan on the map.** Search the merged grid rather than the frame's, and let the live frame
   override it where it disagrees (lines are more trustworthy than memory from a different angle).

How that maps onto the five failures: (1) and (2) are unaffected but become irrelevant, because the
route no longer depends on which diamond is the pip or on where the rim is — it depends on the room,
which is stable; (3) disappears because the grid stops changing under noise; (4) shrinks as memory
fills it; (5) is orthogonal and I will fix it separately.

What is genuinely unresolved, and needs measurement before design commitment:

- **Is the disc content actually redundant frame to frame?** If the renderer redraws the room at a
  different scale, or crops the room to the disc rather than showing a window of a fixed map, then
  registration will be inconsistent and this approach dies. This is a one-hour measurement: walk a
  known distance, correlate, compare against the pip's own displacement and against the 0.3 m/px
  scale.
- **Ambiguity.** Corridors and repeated furniture give flat correlation surfaces. What is the
  measured failure rate, and what should the fallback be?
- **Drift.** How far can we walk before the map is useless, and does anything cheap give loop
  closure (the zone label `Base Zone` in the top-left is OCR-able)?
- **When to throw the map away** — on zone change, on a load screen, on a failed correlation run?
- **Persisted or per-visit.** I am inclined to keep it in memory for one visit first, because
  re-entry needs place recognition, and the last thing this project needs is another
  half-working localiser.

## 5. Constraints the plan must respect

- **Portability:** all of it inside `adapters/game_hsr/`; `core/` untouched. If a piece is
  genuinely generic (an occupancy grid, a phase-correlation odometer), say so and place it
  accordingly rather than assuming.
- **Tests must run without an emulator**, from `tests/fixtures/` and synthetic discs. Prefer
  synthetic sequences with known ground-truth displacement — the odometer is exactly the kind
  of thing that can be unit-tested to a number.
- **Unattended stretches are ~20 minutes**; a wrong plan costs seconds, not minutes, so caps on
  any loop are part of the design.
- **Local decisions, no Mica** — keep it that way, but if that reversal makes Gate 2's definition
  incoherent, say what the honest gate should be.
- **No live-server interaction, ever.**

## 6. What I need from you

1. A **verdict** on §4: build it, build something else (a one-off survey of the game's own map
   screen, or a topological route memo instead of a metric map), or stop and fix the simpler
   things first. With reasoning against the failure list in §3.
2. A **measurement plan** for the unresolved questions in §4, ordered so that the cheapest
   decisive one comes first, with the numeric thresholds that would make you abandon the design.
3. A **module and interface design** that fits the existing code (`minimap.MinimapView`,
   `navigation.plan`/`Route`, `plan_preview.py` as the debug surface), including what the
   walker's control loop should look like when it is steering by a map instead of by a pip.
4. A **phased implementation plan** with gates, each gate a measurable acceptance criterion
   (e.g. "closes a 10 m gap in a cluttered room within N steps, 5 runs in a row"), and an explicit
   stop point where you would abandon and fall back.
5. A **test strategy**: synthetic disc sequences, fixtures to record, the numbers that constitute
   a pass, and how to test the failure paths (flat correlation, drift, zone change).
6. A **risk register** specific to this work, and what each risk costs if it lands late.
7. A clear answer on the **reverted work**: which of the pip selector, disc-crop fix, rim-arc
   goal, prompt region, and A/B deserve to come back, in what order, and which should stay dead.
8. The **first week of work**, in order, with the point at which you would report back rather
   than continue.
