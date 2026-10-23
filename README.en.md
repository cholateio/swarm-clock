# Swarm Clock

[繁體中文](README.md) | **English**

A small always-on-top analog clock for your Windows desktop that moves to the music you are playing. Run it, play anything, and thousands of particles ring the clock face and swarm with the sound.

![The orange particle ring clock on the desktop with the tray menu open: Lock, Show the clock, Hide the clock, Close](https://github.com/user-attachments/assets/711fbbd3-0bb1-4d4d-b9d4-44c76bfa33d4)

- **Reacts to sound**: a ring of particles around the clock bulges and ripples with 64 frequency bands from your system audio.
- **Color follows the bass**: the swarm shifts from yellow toward orange and red as the low end gets louder.
- **Stays out of the way**: a 250x250 transparent window with no border, always on top, and hidden from the taskbar.
- **Click-through**: the tray menu switches between **Lock (mouse pass-through)** and **Unlock (allows dragging)**, so you can place it once and click through it afterwards.

The particle simulation runs on the GPU with WebGL2, so it needs a graphics driver that supports it.

## How it works

```
System audio (Stereo Mix or mic) ──► hidden audio window: FFT
        └─► WebSocket 127.0.0.1:8080 ──► clock window: particles + hands
```

The audio is analyzed in a hidden window inside the app and only the 128-bin spectrum numbers travel to the clock window; there is no recording, no server, and no network use beyond that local socket.

## Install

Needs Windows and Node.js 22.12 or newer. The clock listens to what your PC plays through the Windows **Stereo Mix** device, or to a microphone if you have no Stereo Mix.

1. Turn on Stereo Mix: **Sound settings → Recording** (or **More sound settings → Recording**), right-click an empty area, choose **Show Disabled Devices**, then right-click **Stereo Mix** and choose **Enable**. Make it the default recording device.
2. In a terminal, run `npm install`, then `npm run build`. The installer is written to the `release` folder.
3. Run the installer and start **Swarm Clock**. Drag the clock where you want it, then right-click its tray icon and choose **Lock (mouse pass-through)**.

To try it without installing, run `npm run start:all`.

## Privacy & security

Audio never leaves your machine. The app reads it from the default recording device, turns it into spectrum numbers in memory, and sends only those numbers to its own clock window over a WebSocket bound to `127.0.0.1`, so other computers cannot connect. The socket also rejects any connection whose Origin is not the app itself, so web pages in your browser cannot read it. Microphone permission is granted only to the app's own pages, for audio only. Nothing is saved or uploaded.

## License

[MIT](LICENSE)
