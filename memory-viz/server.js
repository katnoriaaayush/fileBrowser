const express = require('express');
const WebSocket = require('ws');
const http = require('http');
const { exec } = require('child_process');
const path = require('path');

const app = express();
const server = http.createServer(app);
const wss = new WebSocket.Server({ server });

const PORT = process.env.PORT || 3000;
const POLL_MS = parseInt(process.env.POLL_MS || '2000', 10);

app.use(express.static(path.join(__dirname, 'public')));

function parseMeminfo(raw) {
  const result = {};
  for (const line of raw.trim().split('\n')) {
    const m = line.match(/^([\w()]+):\s+(\d+)(?:\s+kB)?/);
    if (m) result[m[1]] = parseInt(m[2], 10);
  }
  return result;
}

function fetchMeminfo(cb) {
  exec('adb shell cat /proc/meminfo', { timeout: 5000 }, (err, stdout) => {
    if (err) {
      cb({ error: err.message || 'ADB command failed. Is a device connected?' });
      return;
    }
    const data = parseMeminfo(stdout);
    if (!data.MemTotal) {
      cb({ error: 'Received empty or unrecognised data from device.' });
      return;
    }
    cb({ data, timestamp: Date.now() });
  });
}

function broadcast(payload) {
  const msg = JSON.stringify(payload);
  for (const client of wss.clients) {
    if (client.readyState === WebSocket.OPEN) client.send(msg);
  }
}

wss.on('connection', (ws) => {
  console.log('[ws] client connected  total=%d', wss.clients.size);
  fetchMeminfo((payload) => {
    if (ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(payload));
  });
  ws.on('close', () => console.log('[ws] client disconnected  total=%d', wss.clients.size));
});

setInterval(() => {
  if (wss.clients.size > 0) fetchMeminfo(broadcast);
}, POLL_MS);

server.listen(PORT, () => {
  console.log(`\n  Android Memory Visualizer`);
  console.log(`  http://localhost:${PORT}`);
  console.log(`  Polling /proc/meminfo every ${POLL_MS} ms\n`);
  console.log('  Make sure your Android device is connected via ADB.');
  console.log('  Tip: set POLL_MS=1000 for 1-second updates.\n');
});
