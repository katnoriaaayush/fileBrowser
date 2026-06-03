const express = require('express');
const WebSocket = require('ws');
const http = require('http');
const { exec } = require('child_process');
const path = require('path');

const app = express();
const server = http.createServer(app);
const wss = new WebSocket.Server({ server });

const PORT = parseInt(process.env.PORT || '3000', 10);
const POLL_MS = parseInt(process.env.POLL_MS || '2000', 10);
const APPS_POLL_MS = parseInt(process.env.APPS_POLL_MS || '5000', 10);

app.use(express.static(path.join(__dirname, 'public')));

function parseMeminfo(raw) {
  const result = {};
  for (const line of raw.trim().split('\n')) {
    const m = line.match(/^([\w()]+):\s+(\d+)(?:\s+kB)?/);
    if (m) result[m[1]] = parseInt(m[2], 10);
  }
  return result;
}

function parseTopApps(raw) {
  const apps = [];
  const lines = raw.split('\n');
  let inSection = false;
  for (const line of lines) {
    if (/Total PSS by process:/.test(line)) { inSection = true; continue; }
    if (inSection) {
      if (!line.trim() || /^Total PSS by/.test(line.trim())) break;
      const m = line.match(/^\s+([\d,]+)K:\s+(.+?)\s+\(pid\s+(\d+)/);
      if (m) apps.push({
        name: m[2].trim(),
        pss: parseInt(m[1].replace(/,/g, ''), 10),
        pid: parseInt(m[3], 10)
      });
    }
  }
  return apps.slice(0, 15);
}

function fetchMeminfo(cb) {
  exec('adb shell cat /proc/meminfo', { timeout: 5000 }, (err, stdout) => {
    if (err) { cb({ error: err.message || 'ADB command failed. Is a device connected?' }); return; }
    const data = parseMeminfo(stdout);
    if (!data.MemTotal) { cb({ error: 'Empty or unrecognised data from device.' }); return; }
    cb({ data, timestamp: Date.now() });
  });
}

let appsData = [];

function refreshApps() {
  exec('adb shell dumpsys meminfo', { timeout: 10000 }, (err, stdout) => {
    if (!err && stdout) appsData = parseTopApps(stdout);
  });
}

function broadcast(payload) {
  const msg = JSON.stringify(payload);
  for (const c of wss.clients) {
    if (c.readyState === WebSocket.OPEN) c.send(msg);
  }
}

wss.on('connection', (ws) => {
  console.log('[ws] client connected  total=%d', wss.clients.size);
  fetchMeminfo((result) => {
    if (ws.readyState === WebSocket.OPEN)
      ws.send(JSON.stringify({ ...result, apps: appsData }));
  });
  ws.on('close', () => console.log('[ws] client disconnected  total=%d', wss.clients.size));
});

setInterval(() => {
  if (wss.clients.size > 0) fetchMeminfo((r) => broadcast({ ...r, apps: appsData }));
}, POLL_MS);

setInterval(refreshApps, APPS_POLL_MS);
refreshApps();

server.listen(PORT, () => {
  console.log(`\n  Android Memory Visualizer`);
  console.log(`  http://localhost:${PORT}`);
  console.log(`  meminfo poll: ${POLL_MS} ms   apps poll: ${APPS_POLL_MS} ms\n`);
  console.log('  Tip: POLL_MS=1000 APPS_POLL_MS=3000 npm start\n');
});
