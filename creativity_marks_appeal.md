Subject: Request for Review — Creativity Component (2 pts), Assignment 2: Pathfinding and MCTS

To the Course Coordinator / Teaching Staff,
Games and Artificial Intelligence Techniques

Student: Huynh Thai Duong
Student ID: s3978955
Assignment: Assignment 2 — Pathfinding and MCTS
Component in question: Creativity (2 pts)
Result received: 0 / 2

I am writing to request a review of the mark awarded for the Creativity component of Assignment 2.
I received 0 out of 2 for this criterion, but my submitted code contains multiple concrete,
functioning features that map directly onto the three creativity dot-points listed in the
assignment brief ("Design and color choice", "AI Agents smart features added", "Any other
interesting non-mentioned idea"). I believe these may not have been visible during marking —
possibly because they only surface during live gameplay rather than a static code read, or because
the submitted ZIP/demo did not make them obvious — and I would appreciate the chance to have them
reconsidered, or to walk through them directly if that is more useful to you.

Below is a specific, file-and-line-referenced list of what is actually implemented, organised by
the three creativity dot-points from the brief.

## 1. Design and colour choice

- **Shared debug/stat panel design system** (`src/debug_ui.py`) — one `Panel` builder, one colour
  palette, one font scale, and one toggle-button primitive, reused identically across the A* panel,
  the FSM monitor panel, and both Connect 4 stats panels, rather than each screen improvising its
  own HUD.
- **A dedicated dark theme with a deliberate palette** (`src/settings.py`) — warm-coral/amber-gold
  discs and a charcoal/slate board in Connect 4, not default pygame red/yellow-on-white.
- **Colorblind-safe disc palette**, toggle key **C** (`set_colorblind_palette()`,
  `src/connect4.py:126`) — swaps to the standard blue/orange pairing safe under deuteranopia,
  protanopia, and tritanopia.
- **Two-colour A\* path rendering** — the raw A\* grid path (green) and the smoothed/string-pulled
  path the frog actually follows (gold) are drawn as two distinct, clearly labelled lines rather
  than one, so the difference between "what A\* searched" and "what the frog follows" is visible,
  not just described.
- **Gapped, rounded explored-cell tiles** with cost labels, and highlighted tiles along the actual
  chosen path cells (`render_explored_overlay`, `src/part1_frog_astar.py`), rather than flat
  full-bleed squares.
- **A generated maze arena** (`generate_maze()`, `src/part1_frog_astar.py:163`) — a real
  recursive-backtracker maze (rooms + corridors carved directly on the A\* grid, with braided loop
  connections) with a checker-tile floor and beveled wall slabs, replacing a plain open field with
  a few floating obstacles.
- **Goal-reached pulse** — an expanding, fading ring animation the instant the frog completes an
  A\* path (`main.py:194`, `part1_frog_astar.py:435`).

## 2. AI Agents — smart features added

- **Weighted A\*** (`HEURISTIC_WEIGHT_PRESETS`, `src/settings.py:108`) — a real, textbook technique
  (Pohl 1970) implemented as `f = g + weight·h`, cycled live with **W** through Optimal/Balanced/Fast
  presets that visibly trade path optimality for a smaller explored set. This is a genuine algorithm
  extension, not a cosmetic setting — proven by dedicated tests
  (`test_weighted_astar_reduces_exploration`, `test_weighted_astar_finds_valid_path`).
- **Biased MCTS rollout policy** (`biased_rollout_policy()`, `src/connect4.py:288`) — during
  simulation, the AI takes an immediate winning move if one exists, else blocks the opponent's
  immediate win, else plays randomly. Toggle key **B** switches between this and the pure-random
  baseline live, so the two can be A/B'd directly in front of the marker.
- **Live, non-blocking MCTS search** (`class LiveSearch`, `src/connect4.py:439`) — the search runs
  in time-sliced chunks across real frames instead of one blocking call, driving a genuine animated
  spinner and live iteration/time counter instead of a frozen "AI is thinking" screen.
- **AI reasoning readout** (`describe_ai_reasoning()`, `src/connect4.py:508`) — a plain-English
  sentence explaining the AI's chosen move (e.g. confidence vs. the runner-up column, or "won —
  4 in a row"), computed directly from the same MCTS statistics already on screen, not decorative
  text.
- **Human hint system** (Task 1 only, **T** key, `src/connect4.py:1021`) — an optional, independent
  MCTS "peek" for the human player, at its own separate (smaller) search budget.
- **Difficulty presets** (**1/2/3** keys) — Fast/Normal/Strong MCTS time/iteration budgets, switching
  the AI's actual strength, not just a label.
- **Session stats ticker** (Part 1, `main.py:201`) — running totals across a session: paths computed,
  real distance travelled, cheapest path cost seen.

## 3. Other interesting ideas not explicitly mentioned in the brief

- **Live diagonal-vs-cardinal A\* toggle (N key)** — reproduces the assignment brief's own Figure 2
  comparison (clunky staircase vs. natural diagonal line) as a real, re-runnable search mode on the
  same map on demand, rather than a static picture.
- **F1 debug overlay** (all three entry points) — a separate raw-diagnostics view (FPS/frame time,
  live mouse→cell/column mapping, raw internal state) distinct from the graded panels, off by
  default.
- **Regenerate-on-demand maze (R key)** — a new random maze layout every press, for Part 1.

---

Given the density and functional depth of what is listed above — spanning a real second algorithmic
technique in each part (weighted A\*, biased rollout), a shared design system, and several
independently toggleable live features — I don't believe 0/2 reflects what was actually submitted,
and I would like to request that the Creativity component be re-checked against the code above
(and, if useful, demonstrated live) before a final mark is confirmed.

I'm happy to walk through any of the above in person or over a call, or to point to the exact line
in the demo recording where each feature is shown, whichever is more convenient.

Thank you for your time and consideration.

Kind regards,
Huynh Thai Duong
s3978955
