# Notetaker: recording with a teammate

One computer is the **server** (it has the GPU and does the transcribing and note-writing).
Everyone else runs the small **recorder** on their own computer. It records their calls and
sends each one to the server when the call ends. Notes appear for everyone at the server's
address a few minutes later.

The server and recorders talk over **Tailscale**, a free private network: only devices you
invite can reach the server, nothing is exposed to the internet. There is no sign-in yet, so
everyone who can reach the server sees all meetings.

---

## Part 1: the server (the PC with the GPU)

1. **Install Tailscale** from https://tailscale.com/download and sign in.
2. **Share the PC with your teammate**: in the Tailscale admin page (Machines), open this PC's
   menu → *Share...* → enter their email. They only get access to this one machine.
3. **Allow the notetaker through Windows Firewall, for Tailscale only.** In PowerShell *as
   Administrator*:
   ```powershell
   New-NetFirewallRule -DisplayName "Notetaker (Tailscale)" -Direction Inbound -Protocol TCP `
     -LocalPort 8000 -InterfaceAlias "Tailscale" -Action Allow
   ```
   (Tied to the Tailscale adapter, not to an address range: some hotel and mobile networks
   also hand out 100.64.x.x addresses.)
4. **Start the notetaker** as usual: `powershell -ExecutionPolicy Bypass -File scripts\start.ps1`.
   The app is now at http://localhost:8000 (and at `http://<this PC's Tailscale name>:8000`
   for your teammate).
5. **Make the recorder package**: `.venv/Scripts/python scripts/make_recorder_bundle.py`.
   It writes `dist/notetaker-recorder.zip` with this PC's Tailscale address inside. Send
   the zip to your teammate.
6. Keep the PC **on and awake** while people record (Settings → System → Power → Sleep:
   Never, at least when plugged in). If it's off, the recorder keeps the audio and offers to
   send it again later; nothing is lost.

---

## Part 2: the recorder on a Mac (about 15 minutes, once)

### a. Join the network
1. Install **Tailscale** (Mac App Store or https://tailscale.com/download), sign in, and
   accept the share invitation you got by email.
2. Check: open the address in `server.txt` (inside the zip, e.g. `http://100.101.102.103:8000`)
   in your browser. The notetaker should load.

### b. Let the recorder hear the call: BlackHole
macOS doesn't let apps record what the computer plays, so a free virtual sound device,
BlackHole, gets a copy of the call audio.

1. Install **BlackHole 2ch**: `brew install blackhole-2ch` (with Homebrew), or the installer
   from https://existential.audio/blackhole/ (free).
2. Open **Audio MIDI Setup** (Spotlight: "Audio MIDI"). Click **+** (bottom left) →
   **Create Multi-Output Device**. Tick your speakers (or headphones) *first*, then
   **BlackHole 2ch**. Tick *Drift Correction* for BlackHole. Rename it, e.g.
   "Speakers + Notetaker".
   - If you switch between speakers and headphones, make one of these for each (e.g. also
     "AirPods + Notetaker") and pick the matching one.
3. **System Settings → Sound → Output**: choose your Multi-Output Device.
   (The volume keys don't work on a Multi-Output Device; set the volume in the call app.)
4. In **Zoom**: Settings → Audio → Speaker: *Same as System*. In **Teams**: Settings →
   Devices → Speaker: your Multi-Output Device. Leave the microphone as it is.

### c. Start the recorder
1. Unzip `notetaker-recorder.zip` somewhere permanent (e.g. Documents).
2. **Right-click** `Start Notetaker Recorder.command` → **Open** → **Open** (the first time
   only; macOS asks because it's not from the App Store).
3. A Terminal window opens. The first start downloads what it needs (about a minute). Then
   the browser opens **http://localhost:8765**: that's the notetaker, with recording.
4. macOS asks whether Terminal may use the microphone: **Allow**.
5. On the page: **Record a call** shows your microphone and "BlackHole 2ch". Enter your name
   and tick "My team knows our calls are recorded" (asked once).

### d. Test it (2 minutes)
1. Play any video with speech, press **Start recording**, and say a few sentences.
2. Both meters at the top should move: **You** when you talk, **Others** with the video.
   If "Others" stays flat, the Mac's output isn't the Multi-Output Device (step b3).
3. Press **Stop and take notes**. The meeting opens; transcript and notes follow in a few
   minutes.

### Every day
Double-click `Start Notetaker Recorder.command` before a call (keep the Terminal window open
while you record), keep the output on your Multi-Output Device, and use **http://localhost:8765**.
Zoom calls are detected ("Zoom call detected. Record it?") and recording stops by itself when
the Zoom call ends; for Teams on a Mac, press Stop yourself.

### If something goes wrong
Copy the text in the Terminal window and send it to whoever runs the server. A recording that
couldn't be sent stays on your Mac and shows up as "hasn't been turned into notes yet", with
a button to send it again.

---

## Recording on Windows (teammates with a PC)
No BlackHole needed: Windows can record what it plays. Install Tailscale (part 2a), unzip,
and double-click `Start Notetaker Recorder.cmd`. Zoom and Teams calls are detected and
stopped automatically.
