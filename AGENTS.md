<!-- Tool-neutral project memory. Never put local paths, costs or personal data here. -->
# Swarm Clock

## Project goal

A small always-on-top analog clock for the Windows desktop whose particle ring (thousands of GPU-simulated particles) swarms to whatever audio is playing.

## Stack

- Language / framework: JavaScript, Next.js 16 (static export to `out/`) + React 18, Electron, electron-builder (Windows installer); Node.js ≥ 22.12
- Services: none (local only; a WebSocket hub on `127.0.0.1:8080` links the windows)
- Test: `npx next build` (no automated tests; the build is the check, then run `npm run start:all` and play some audio)

## File layout

- `electron/main.js` — transparent clock window, hidden audio-engine window, tray menu, WebSocket hub
- `electron/security.js` — the single check for which origins count as the app (WebSocket `verifyClient`, media permission handler, navigation blocking)
- `app/audio-engine/page.jsx` — captures audio, runs an `AnalyserNode` and sends `{ bass, frequencies }` to `/engine`
- `app/page.js` — the clock window: connects as a viewer and draws the clock and the GPU particle ring (WebGL2)
- `docs/DEVELOPMENT.md` — commands, structure and notes

## Constraints

- The WebSocket hub binds to `127.0.0.1` only, relays only messages from the `/engine` path and rejects payloads over 64 KB; origin checks go through `electron/security.js`, never a second copy.
- Keep the zero-gain routing of the analyser, the `setTimeout` audio loop and `backgroundThrottling: false`: without them the hidden engine window is throttled or the FFT data comes back as zeros.
- Path-type constraints are mirrored in `kit.toml [protected]`; keep the two in sync (one glob per line).

## Conventions

- Conventional commits; run `npx next build` before committing.
- GitHub Actions: top-level `permissions: contents: read`, raised per job only when needed.
- GitHub Actions: pin every `uses:` to a 40-character commit SHA with a trailing `# vX.Y.Z` comment.
- `README.md` (Traditional Chinese, default) and `README.en.md` (English) mirror each other; update both in the same commit.

## Where to look

- `docs/DEVELOPMENT.md` — commands, structure, audio and rendering notes
