// Renders scene.html frame by frame with headless Chrome, then encodes with ffmpeg.
// Resumable: frames already on disk are skipped, and Chrome is relaunched every
// BATCH frames to keep memory flat. Usage: node render.mjs [durationSeconds=40] [fps=30]
import puppeteer from 'puppeteer-core';
import { spawn } from 'node:child_process';
import { existsSync, mkdirSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const here = path.dirname(fileURLToPath(import.meta.url));
const duration = Number(process.argv[2] ?? 40);
const fps = Number(process.argv[3] ?? 30);
const BATCH = 200;
const out = path.join(here, 'out');
const frames = path.join(out, 'frames');
mkdirSync(frames, { recursive: true });
const frameFile = (i) => path.join(frames, `f${String(i).padStart(5, '0')}.png`);
const total = Math.round(duration * fps);

for (let start = 0; start < total; start += BATCH) {
  const todo = [];
  for (let i = start; i < Math.min(start + BATCH, total); i++) if (!existsSync(frameFile(i))) todo.push(i);
  if (!todo.length) continue;
  const browser = await puppeteer.launch({
    executablePath: process.env.CHROME_PATH ?? '/usr/bin/google-chrome',
    args: ['--no-sandbox', '--hide-scrollbars'],
    headless: 'new',
  });
  const page = await browser.newPage();
  await page.setViewport({ width: 960, height: 540, deviceScaleFactor: 2 });
  await page.goto('file://' + path.join(here, 'scene.html'));
  for (const i of todo) {
    await page.evaluate((t) => window.render(t), i / fps);
    await page.screenshot({ path: frameFile(i) });
  }
  await browser.close();
  console.log('frames', Math.min(start + BATCH, total), '/', total);
}

const ff = spawn('ffmpeg', ['-y', '-loglevel', 'error', '-framerate', String(fps), '-i', path.join(frames, 'f%05d.png'),
  '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '16', path.join(out, 'overview.mp4')], { stdio: 'inherit' });
await new Promise((res, rej) => ff.on('exit', (c) => (c === 0 ? res() : rej(new Error('ffmpeg ' + c)))));
console.log('wrote', path.join(out, 'overview.mp4'));
