# Android Memory Visualizer

A local web dashboard that streams live `/proc/meminfo` data from a connected Android device via ADB and renders it as interactive charts.

## Features

- **Live line chart** — tracks Used, Available, Cached, and Free memory over the last 60 samples
- **Doughnut chart** — real-time memory breakdown (Used / Cached / Buffers / Free)
- **Segmented progress bar** — visual memory distribution at a glance
- **6 stat cards** — Total RAM, Used, Available, Cached, Buffers, Swap
- **Raw data table** — every `/proc/meminfo` field with inline bar
- **Auto-reconnect** — WebSocket reconnects automatically if the server restarts
- **Error display** — clear messages when ADB is unavailable

## Requirements

- Node.js ≥ 14
- `adb` in your PATH (`brew install android-platform-tools` / `apt install adb`)
- Android device connected with USB debugging enabled

## Quick start

```bash
cd memory-viz
npm install
npm start
```

Then open **http://localhost:3000** in your browser.

## Configuration

| Env var  | Default | Description                        |
|----------|---------|------------------------------------|
| `PORT`   | `3000`  | HTTP / WebSocket server port       |
| `POLL_MS`| `2000`  | ADB poll interval in milliseconds  |

```bash
POLL_MS=1000 npm start   # 1-second updates
PORT=8080   npm start   # run on port 8080
```

## How it works

```
Android device
  └─ adb shell cat /proc/meminfo   (polled every POLL_MS)
        │
        ▼
  Node.js server  (Express + ws)
        │  WebSocket broadcast
        ▼
  Browser  (Chart.js dashboard)
```
