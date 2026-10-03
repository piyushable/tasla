
"""
Smart Autonomous Vehicle Path Planner & 3D Cockpit Simulator

Features:
- Real-time 3D WebGL Frontend with multiple camera views:
  * 🏎️ Cockpit View (First-Person Driver POV) with steering wheel animation & AR windshield HUD
  * 🎥 Chase Cam (Smooth 3rd-person follow)
  * 🚁 Drone View (High-angle tactical overhead)
  * 🌐 Free 3D Orbit (Full 360° mouse inspection)
  * 🗺️ 2D Tactical LiDAR (Top-down precision map)
- Interactive 3D obstacle placement: click directly on the road while driving!
- Moving obstacle physics with velocity vectors & edge bouncing
- Time-aware A* neural path planning with dynamic re-routing & safe gap waiting
- 2D & 3D synchronized telemetry & analysis
"""
import os
import heapq
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import streamlit.components.v1 as components
from streamlit.errors import StreamlitAPIException

# newer Streamlit replaced components.html and use_container_width; keep older installs working too
NEW_EMBED_API = hasattr(st, "iframe")
CHART_WIDTH = {"width": "stretch"} if NEW_EMBED_API else {"use_container_width": True}

# ---- Embedded Standalone 3D HTML Frontend ----
EMBEDDED_3D_HTML = '<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n<meta name="viewport" content="width=device-width, initial-scale=1.0">\n<title>AutoNav 3D // Smart Autonomous Vehicle Path Planner & Cockpit</title>\n<link rel="preconnect" href="https://fonts.googleapis.com">\n<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>\n<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&family=Orbitron:wght@500;700;800;900&display=swap" rel="stylesheet">\n<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>\n<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>\n<style>\n:root{--bg:#080c14;--panel:rgba(13,20,36,.78);--border:rgba(0,240,255,.22);--cyan:#00f0ff;--glow:rgba(0,240,255,.45);--emerald:#00ff9d;--amber:#ffaa00;--red:#ff3366;--text:#f0f4f8;--muted:#8fa0b5}\n*{box-sizing:border-box;margin:0;padding:0;user-select:none}\nbody,html{width:100%;height:100%;overflow:hidden;background:var(--bg);font-family:\'Inter\',-apple-system,sans-serif;color:var(--text)}\n#canvas-container,#cockpit-hud,.cockpit-overlay,#warning-strobe{position:absolute;top:0;left:0;width:100%;height:100%}\n#canvas-container{z-index:1}\n#cockpit-hud{pointer-events:none;z-index:5;display:none}\n.cockpit-overlay{pointer-events:none;z-index:4;display:none;background:radial-gradient(circle at 50% 40%,transparent 60%,rgba(5,10,20,.85) 100%)}\n#warning-strobe{pointer-events:none;z-index:6;border:4px solid var(--red);opacity:0;transition:opacity .2s;box-shadow:inset 0 0 50px rgba(255,51,102,.5)}\n#ui-layer{position:absolute;inset:0;z-index:10;pointer-events:none;display:flex;flex-direction:column;justify-content:space-between;padding:14px 18px}\n.interactive{pointer-events:auto}\n.glass{background:var(--panel);backdrop-filter:blur(16px);-webkit-backdrop-filter:blur(16px);border:1px solid var(--border);border-radius:14px;box-shadow:0 8px 32px rgba(0,0,0,.55),inset 0 1px 1px rgba(255,255,255,.1)}\n.header-bar{display:flex;align-items:center;justify-content:space-between;gap:10px 14px;padding:9px 16px;flex-wrap:wrap}\n.brand{display:flex;align-items:center;gap:10px}\n.logo{width:36px;height:36px;border-radius:10px;background:linear-gradient(135deg,#00f0ff,#7928ca);display:flex;align-items:center;justify-content:center;font-size:19px;box-shadow:0 0 16px var(--glow)}\n.brand-title{font-family:\'Orbitron\',monospace;font-size:14px;font-weight:800;letter-spacing:1.5px;background:linear-gradient(90deg,#fff,var(--cyan));-webkit-background-clip:text;-webkit-text-fill-color:transparent}\n.brand-sub{font-size:11px;color:var(--muted)}\n.status-pill{display:flex;align-items:center;gap:8px;padding:6px 14px;border-radius:30px;background:rgba(0,240,255,.1);border:1px solid rgba(0,240,255,.3);font-family:\'Orbitron\',sans-serif;font-size:11px;font-weight:700;letter-spacing:.5px;max-width:340px}\n#status-text{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}\n.status-dot{flex:none;width:8px;height:8px;border-radius:50%;background:var(--cyan);box-shadow:0 0 10px var(--cyan);animation:pulse 1.5s infinite}\n@keyframes pulse{0%,100%{opacity:1;transform:scale(1)}50%{opacity:.4;transform:scale(.85)}}\n.seg{display:flex;background:rgba(8,14,26,.7);padding:4px;border-radius:12px;border:1px solid rgba(255,255,255,.08);gap:4px}\n.seg-btn{background:transparent;border:none;color:var(--muted);padding:7px 11px;border-radius:8px;font-size:12px;font-weight:600;cursor:pointer;font-family:\'Inter\',sans-serif;transition:all .2s}\n.seg-btn:hover{color:#fff;background:rgba(255,255,255,.06)}\n.seg-btn.active{background:linear-gradient(135deg,rgba(0,240,255,.25),rgba(121,40,202,.35));outline:1px solid var(--cyan);color:#fff}\n.quick{display:flex;align-items:center;gap:8px}\n.icon-btn{background:rgba(255,255,255,.05);border:1px solid rgba(255,255,255,.1);color:var(--text);min-width:36px;height:36px;border-radius:10px;display:flex;align-items:center;justify-content:center;cursor:pointer;font-size:15px;transition:all .2s}\n.icon-btn:hover{background:rgba(0,240,255,.15);border-color:var(--cyan);box-shadow:0 0 10px var(--glow)}\n#btn-panels{display:none}\n.middle{display:flex;justify-content:space-between;align-items:flex-start;margin:10px 0;flex:1;min-height:0;pointer-events:none;gap:10px}\n.side-card{width:310px;max-height:100%;overflow-y:auto;padding:14px;display:flex;flex-direction:column;gap:12px;pointer-events:auto}\n.section-title{font-family:\'Orbitron\',sans-serif;font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:1.2px;color:var(--cyan);display:flex;align-items:center;gap:8px}\n.paths{display:flex;flex-direction:column;gap:8px}\n.path-card{background:rgba(17,27,46,.6);border:1px solid rgba(255,255,255,.08);border-radius:10px;padding:9px 12px;cursor:pointer;display:flex;align-items:center;justify-content:space-between;transition:all .2s}\n.path-card:hover{border-color:rgba(0,240,255,.4);background:rgba(0,240,255,.06)}\n.path-card.selected{border-color:var(--cyan);background:linear-gradient(90deg,rgba(0,240,255,.15),rgba(0,240,255,.02));box-shadow:0 0 12px rgba(0,240,255,.2)}\n.path-name{font-family:\'Orbitron\',monospace;font-size:13px;font-weight:700;display:block}\n.path-meta{font-size:11px;color:var(--muted)}\n.badge{font-size:10px;padding:3px 8px;border-radius:20px;font-weight:700;border:1px solid}\n.badge.clear{background:rgba(0,255,157,.15);border-color:var(--emerald);color:var(--emerald)}\n.badge.blocked{background:rgba(255,51,102,.15);border-color:var(--red);color:var(--red)}\n.spawner{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}\n.sp-item{background:rgba(17,27,46,.6);border:1px solid rgba(255,255,255,.08);border-radius:10px;padding:9px 6px;display:flex;flex-direction:column;align-items:center;gap:5px;cursor:pointer;transition:all .2s}\n.sp-item:hover{border-color:var(--cyan);background:rgba(0,240,255,.08)}\n.sp-item.active{border-color:var(--cyan);background:rgba(0,240,255,.2);box-shadow:0 0 12px rgba(0,240,255,.25)}\n.sp-emoji{font-size:22px}.sp-label{font-size:10px;font-weight:600;text-align:center}\n.toggle{display:flex;background:rgba(8,14,26,.7);padding:3px;border-radius:8px;gap:3px}\n.tg-btn{flex:1;padding:6px 4px;background:transparent;border:1px solid transparent;color:var(--muted);border-radius:6px;font-size:11px;font-weight:600;cursor:pointer;transition:all .2s;text-align:center}\n.tg-btn.active{background:rgba(0,240,255,.2);color:#fff;border-color:var(--cyan)}\n.tip{font-size:11px;color:var(--cyan);background:rgba(0,240,255,.08);border:1px dashed rgba(0,240,255,.3);padding:8px 10px;border-radius:8px;text-align:center;line-height:1.4}\n.dock{display:flex;align-items:center;justify-content:space-between;gap:10px 16px;padding:10px 18px;flex-wrap:wrap}\n.main-controls,.telemetry{display:flex;align-items:center;gap:10px}\n.telemetry{gap:20px;flex-wrap:wrap}\n.drive-btn,.top-start-btn{background:linear-gradient(135deg,#00f0ff,#0077ff);color:#040914;border:none;border-radius:10px;font-family:\'Orbitron\',sans-serif;font-weight:800;letter-spacing:1px;cursor:pointer;display:flex;align-items:center;gap:6px;box-shadow:0 0 16px rgba(0,240,255,.5);transition:all .2s}\n.drive-btn{padding:11px 24px;font-size:14px}.top-start-btn{padding:8px 16px;font-size:12px}\n.drive-btn:hover,.top-start-btn:hover{transform:translateY(-2px);box-shadow:0 0 26px rgba(0,240,255,.8)}\n.drive-btn.stopped,.top-start-btn.stopped{background:linear-gradient(135deg,#ff3366,#ff7700);color:#fff;box-shadow:0 0 16px rgba(255,51,102,.5)}\n.t-item{display:flex;flex-direction:column;gap:2px;min-width:68px}\n.t-label{font-size:10px;font-weight:600;color:var(--muted);letter-spacing:.8px}\n.t-val{font-family:\'Orbitron\',monospace;font-size:17px;font-weight:800;display:flex;align-items:baseline;gap:4px}\n.t-unit{font-size:11px;color:var(--cyan);font-weight:500}\n.progress-box{width:200px;display:flex;flex-direction:column;gap:6px}\n.progress-track{width:100%;height:8px;background:rgba(255,255,255,.1);border-radius:4px;overflow:hidden}\n.progress-fill{height:100%;width:0;background:linear-gradient(90deg,var(--cyan),var(--emerald));box-shadow:0 0 10px var(--cyan)}\n#manual-dock{position:absolute;bottom:92px;right:22px;z-index:18;display:grid;grid-template-columns:repeat(3,46px);grid-template-rows:repeat(2,46px);gap:6px;transition:opacity .3s}\n#manual-dock.hidden{opacity:0;pointer-events:none}\n.d-btn{background:rgba(13,20,36,.85);border:1px solid var(--cyan);color:var(--cyan);border-radius:10px;font-size:17px;cursor:pointer;touch-action:none}\n.d-btn:active,.d-btn.pressed{background:var(--cyan);color:#040914}\n#toast-msg{position:absolute;top:78px;left:50%;transform:translateX(-50%);background:rgba(13,20,36,.92);border:1px solid var(--cyan);padding:9px 20px;border-radius:30px;font-size:13px;font-weight:600;box-shadow:0 0 20px var(--glow);opacity:0;transition:opacity .3s;z-index:20;pointer-events:none;max-width:90vw;text-align:center}\n#toast-msg.show{opacity:1}\n@media(max-width:1100px){.side-card{width:240px}.brand-sub{display:none}}\n@media(max-width:760px){\n  #btn-panels{display:flex}\n  .side-card{display:none}\n  body.panels-open .middle{flex-direction:column;overflow-y:auto}\n  body.panels-open .side-card{display:flex;width:100%;max-height:none}\n  .progress-box{width:140px}.status-pill{max-width:200px}\n}\n</style>\n</head>\n<body>\n<div id="canvas-container"></div>\n<canvas id="cockpit-hud"></canvas>\n<div class="cockpit-overlay" id="cockpit-overlay"></div>\n<div id="warning-strobe"></div>\n<div id="toast-msg">Notification</div>\n\n<div id="ui-layer">\n  <header class="header-bar glass interactive">\n    <div class="brand">\n      <div class="logo">🏎️</div>\n      <div><div class="brand-title">AUTONAV // 3D</div><div class="brand-sub">Autonomous Path Planner &amp; Cockpit Simulator</div></div>\n    </div>\n    <div class="status-pill"><div class="status-dot" id="status-dot"></div><span id="status-text">AUTOPILOT READY</span></div>\n    <button class="top-start-btn" id="btn-top-drive" onclick="startDriving()">▶ Start driving</button>\n    <div class="seg">\n      <button class="seg-btn active" id="mode-auto" onclick="setDriveMode(\'auto\')">🤖 Autopilot</button>\n      <button class="seg-btn" id="mode-manual" onclick="setDriveMode(\'manual\')">🎮 Manual</button>\n    </div>\n    <div class="seg">\n      <button class="seg-btn active" id="btn-cam-cockpit" onclick="setCameraMode(\'cockpit\')">🏎️ Cockpit</button>\n      <button class="seg-btn" id="btn-cam-chase" onclick="setCameraMode(\'chase\')">🎥 Chase</button>\n      <button class="seg-btn" id="btn-cam-drone" onclick="setCameraMode(\'drone\')">🚁 Drone</button>\n      <button class="seg-btn" id="btn-cam-orbit" onclick="setCameraMode(\'orbit\')">🌐 Orbit</button>\n      <button class="seg-btn" id="btn-cam-tactical" onclick="setCameraMode(\'tactical\')">🗺️ 2D</button>\n    </div>\n    <div class="quick">\n      <button class="icon-btn" title="Lighting" onclick="toggleEnvironment()"><span id="env-icon">🌃</span></button>\n      <button class="icon-btn" title="Sound" onclick="toggleAudio()"><span id="sound-icon">🔊</span></button>\n      <button class="icon-btn" title="Reset camera" onclick="resetCameraOrientation()">🎯</button>\n      <button class="icon-btn" id="btn-panels" title="Panels" onclick="document.body.classList.toggle(\'panels-open\')">☰</button>\n    </div>\n  </header>\n\n  <div class="middle">\n    <aside class="side-card glass">\n      <div class="section-title"><span>🛣️ Planned Trajectories</span><span style="margin-left:auto;font-size:10px;color:var(--muted)" id="path-count-lbl">3 PATHS</span></div>\n      <div style="display:flex;gap:6px;align-items:center;font-size:12px;color:var(--muted)">\n        <span>Candidates:</span>\n        <input type="range" min="1" max="5" value="3" id="path-slider" style="flex:1" oninput="updatePathCount(this.value)">\n        <span id="path-slider-val" style="font-family:\'Orbitron\';font-weight:700;color:#fff">3</span>\n      </div>\n      <div class="paths" id="paths-list"></div>\n      <div style="font-size:11px;color:var(--muted);line-height:1.4;border-top:1px solid rgba(255,255,255,.08);padding-top:8px">\n        The autopilot follows a time-aware A* route and re-plans when predicted obstacle motion intersects it. Path badges are re-evaluated live from the car\'s current position.\n      </div>\n    </aside>\n\n    <aside class="side-card glass">\n      <div class="section-title"><span>⚠️ Obstacle Spawner</span><span style="margin-left:auto;font-size:10px;color:var(--amber)" id="obs-count-badge">0 PLACED</span></div>\n      <div class="spawner">\n        <div class="sp-item active" data-kind="🧍 Pedestrian"><span class="sp-emoji">🧍</span><span class="sp-label">Pedestrian</span></div>\n        <div class="sp-item" data-kind="🐄 Cow"><span class="sp-emoji">🐄</span><span class="sp-label">Cow</span></div>\n        <div class="sp-item" data-kind="🛒 Pushcart"><span class="sp-emoji">🛒</span><span class="sp-label">Pushcart</span></div>\n        <div class="sp-item" data-kind="🚗 Vehicle"><span class="sp-emoji">🚗</span><span class="sp-label">Vehicle</span></div>\n        <div class="sp-item" data-kind="🕳️ Pothole"><span class="sp-emoji">🕳️</span><span class="sp-label">Pothole</span></div>\n        <div class="sp-item" id="sp-random" title="Randomize"><span class="sp-emoji">🎲</span><span class="sp-label">Randomize</span></div>\n      </div>\n      <div class="section-title">Motion Trajectory</div>\n      <div class="toggle" id="motion-toggle">\n        <button class="tg-btn active" data-motion="Static">Static</button>\n        <button class="tg-btn" data-motion="Crossing the road">Crossing</button>\n        <button class="tg-btn" data-motion="Moving ahead">Moving Ahead</button>\n      </div>\n      <div class="tip">👆 <strong>Click the 3D road</strong> to drop obstacles, even while driving. (Dragging to look around won\'t place anything.)</div>\n      <div style="display:flex;gap:8px">\n        <button class="icon-btn" style="flex:1;height:32px;font-size:12px" onclick="undoLastObstacle()">↩️ Undo</button>\n        <button class="icon-btn" style="flex:1;height:32px;font-size:12px" onclick="clearAllObstacles()">🗑️ Clear All</button>\n      </div>\n    </aside>\n  </div>\n\n  <footer class="dock glass interactive">\n    <div class="main-controls">\n      <button class="drive-btn" id="btn-drive" onclick="startDriving()">▶ Start driving</button>\n      <button class="icon-btn" title="Reset (R)" onclick="resetSimulation()" style="width:42px;height:42px">⏹</button>\n      <div style="display:flex;flex-direction:column;gap:2px;margin-left:6px">\n        <span style="font-size:10px;color:var(--muted);font-weight:600">SIM SPEED</span>\n        <div class="toggle" id="speed-toggle" style="padding:2px">\n          <button class="tg-btn" data-speed="0.5">0.5x</button>\n          <button class="tg-btn active" data-speed="1">1.0x</button>\n          <button class="tg-btn" data-speed="2">2.0x</button>\n        </div>\n      </div>\n    </div>\n    <div class="telemetry">\n      <div class="t-item"><span class="t-label">SPEED</span><div class="t-val"><span id="tel-speed">0.0</span><span class="t-unit">m/s</span></div></div>\n      <div class="t-item"><span class="t-label">STEER</span><div class="t-val"><span id="tel-steer">0.0</span><span class="t-unit">°</span></div></div>\n      <div class="t-item"><span class="t-label">REPLANS</span><div class="t-val" style="color:var(--cyan)"><span id="tel-replans">0</span></div></div>\n      <div class="t-item"><span class="t-label">DISTANCE</span><div class="t-val"><span id="tel-dist">0.0</span><span class="t-unit">/ 40m</span></div></div>\n      <div class="progress-box">\n        <div style="display:flex;justify-content:space-between;font-size:10px;color:var(--muted);font-weight:600"><span>START</span><span id="tel-pct">0%</span><span>GOAL</span></div>\n        <div class="progress-track"><div class="progress-fill" id="progress-fill"></div></div>\n      </div>\n    </div>\n  </footer>\n</div>\n\n<div id="manual-dock" class="hidden">\n  <span></span><button class="d-btn" data-key="arrowup">▲</button><span></span>\n  <button class="d-btn" data-key="arrowleft">◀</button><button class="d-btn" data-key="arrowdown">▼</button><button class="d-btn" data-key="arrowright">▶</button>\n</div>\n\n<script>\n/* ===== 1. CONSTANTS ===== */\nconst ROAD_HALF = 6.0, X_END = 40.0, VEH_LEN = 4.2, HALF_W = 0.9, RES = 0.25;\nconst HX0 = VEH_LEN / 2, HY0 = HALF_W;\nconst HX = HX0 + 0.5, HY = HY0 + 0.6;       // planner collision margins\nconst VX = HX0 + 0.3, VY = HY0 + 0.4;       // route-validity margins (looser than planner => no replan thrash)\nconst LIM = ROAD_HALF - 0.8;\nconst V_NOM = 4.0;                           // autopilot speed in sim-seconds: 40 m = exactly 10 s at 1x\nconst LAT_RATE = 0.95;                       // max lateral speed as a fraction of forward speed\n\nconst OBJ_PROPS = {\n  "🧍 Pedestrian": { L: 0.8, W: 0.8, spd: 1.2 },\n  "🐄 Cow":        { L: 2.0, W: 0.9, spd: 0.7 },\n  "🛒 Pushcart":   { L: 1.6, W: 1.0, spd: 0.9 },\n  "🚗 Vehicle":    { L: 4.2, W: 1.8, spd: 1.5 },\n  "🕳️ Pothole":    { L: 1.2, W: 1.2, spd: 0.0 },\n};\nconst XS = [], YS = [];\nfor (let x = 0; x <= X_END + 0.001; x += RES) XS.push(+x.toFixed(3));\nfor (let y = -ROAD_HALF; y <= ROAD_HALF + 0.001; y += RES) YS.push(+y.toFixed(3));\nconst clamp = (v, a, b) => Math.min(Math.max(v, a), b);\n\nfunction disposeTree(o) {\n  o.traverse(c => {\n    if (c.geometry) c.geometry.dispose();\n    if (c.material) (Array.isArray(c.material) ? c.material : [c.material]).forEach(m => m.dispose());\n  });\n}\n\n/* ===== 2. SCENE ===== */\nconst container = document.getElementById(\'canvas-container\');\nconst scene = new THREE.Scene();\nscene.background = new THREE.Color(0x060913);\nscene.fog = new THREE.FogExp2(0x060913, 0.015);\nconst camera = new THREE.PerspectiveCamera(65, innerWidth / innerHeight, 0.1, 1000);\nconst renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });\nrenderer.setSize(innerWidth, innerHeight);\nrenderer.setPixelRatio(Math.min(devicePixelRatio, 2));\nrenderer.shadowMap.enabled = true;\nrenderer.shadowMap.type = THREE.PCFSoftShadowMap;\nrenderer.toneMapping = THREE.ACESFilmicToneMapping;\nrenderer.toneMappingExposure = 1.1;\ncontainer.appendChild(renderer.domElement);\n\nconst controls = new THREE.OrbitControls(camera, renderer.domElement);\ncontrols.enableDamping = true;\ncontrols.dampingFactor = 0.05;\ncontrols.maxPolarAngle = Math.PI / 2 - 0.02;\ncontrols.minDistance = 2;\ncontrols.maxDistance = 80;\n\nconst hudCanvas = document.getElementById(\'cockpit-hud\');\nconst hudCtx = hudCanvas.getContext(\'2d\');\nfunction resizeHud() { hudCanvas.width = innerWidth; hudCanvas.height = innerHeight; }\nresizeHud();\n\nconst ambientLight = new THREE.AmbientLight(0x2a3b5c, 0.9);\nscene.add(ambientLight);\nconst dirLight = new THREE.DirectionalLight(0x8bc34a, 0.8);\ndirLight.position.set(20, 40, -30);\ndirLight.castShadow = true;\ndirLight.shadow.mapSize.set(2048, 2048);\nObject.assign(dirLight.shadow.camera, { near: 0.5, far: 120, left: -30, right: 50, top: 20, bottom: -20 });\nscene.add(dirLight);\n\nlet currentEnv = \'night\';\nfunction setEnvironment(env) {\n  currentEnv = env;\n  const cfg = {\n    night:  { bg: 0x060913, amb: 0x1a2844, ai: 0.8, dir: 0x38bdf8, di: 0.4, icon: \'🌃\' },\n    sunset: { bg: 0x1e1026, amb: 0xff7700, ai: 0.9, dir: 0xffaa44, di: 1.2, icon: \'🌅\' },\n    day:    { bg: 0x87ceeb, amb: 0xffffff, ai: 1.1, dir: 0xfff5e6, di: 1.4, icon: \'☀️\' },\n  }[env];\n  scene.background.set(cfg.bg); scene.fog.color.set(cfg.bg);\n  ambientLight.color.set(cfg.amb); ambientLight.intensity = cfg.ai;\n  dirLight.color.set(cfg.dir); dirLight.intensity = cfg.di;\n  document.getElementById(\'env-icon\').innerText = cfg.icon;\n}\nfunction toggleEnvironment() { setEnvironment({ night: \'sunset\', sunset: \'day\', day: \'night\' }[currentEnv]); }\nsetEnvironment(\'night\');\n\n/* ===== 3. ROAD ===== */\nconst worldGroup = new THREE.Group();\nscene.add(worldGroup);\nconst ground = new THREE.Mesh(new THREE.PlaneGeometry(160, 160), new THREE.MeshStandardMaterial({ color: 0x0c121e, roughness: .9, metalness: .1 }));\nground.rotation.x = -Math.PI / 2; ground.position.set(20, -0.05, 0); ground.receiveShadow = true;\nworldGroup.add(ground);\nconst roadMesh = new THREE.Mesh(new THREE.PlaneGeometry(60, ROAD_HALF * 2), new THREE.MeshStandardMaterial({ color: 0x181c24, roughness: .8, metalness: .2 }));\nroadMesh.rotation.x = -Math.PI / 2; roadMesh.position.set(20, 0, 0); roadMesh.receiveShadow = true;\nworldGroup.add(roadMesh);\n\nfunction roadLine(x, z, len, w, color, dashed = false) {\n  const mk = (l, px) => {\n    const m = new THREE.Mesh(new THREE.PlaneGeometry(l, w), new THREE.MeshBasicMaterial({ color }));\n    m.rotation.x = -Math.PI / 2; m.position.set(px, 0.01, z); return m;\n  };\n  if (!dashed) return mk(len, x);\n  const g = new THREE.Group(), n = Math.floor(len / 4);\n  for (let i = 0; i < n; i++) g.add(mk(2, x - len / 2 + i * 4 + 1));\n  return g;\n}\nworldGroup.add(roadLine(20, -5.8, 60, .2, 0xffffff), roadLine(20, 5.8, 60, .2, 0xffffff));\nworldGroup.add(roadLine(20, 0, 60, .18, 0xfacc15, true), roadLine(20, -3, 60, .12, 0xaaaaaa, true), roadLine(20, 3, 60, .12, 0xaaaaaa, true));\nconst startLine = roadLine(0, 0, .4, ROAD_HALF * 2, 0x00f0ff); startLine.rotation.z = 0;\nworldGroup.add(startLine, roadLine(40, 0, .6, ROAD_HALF * 2, 0x00ff9d));\n[-6.2, 6.2].forEach(z => {\n  const c = new THREE.Mesh(new THREE.BoxGeometry(60, .25, .4), new THREE.MeshStandardMaterial({ color: 0x334155, roughness: .6 }));\n  c.position.set(20, .1, z); worldGroup.add(c);\n});\nconst poleMat = new THREE.MeshStandardMaterial({ color: 0x475569, metalness: .8 });\n[-6, 6, 18, 30, 42].forEach(x => [-7.2, 7.2].forEach(z => {\n  const pole = new THREE.Group(); pole.position.set(x, 0, z);\n  const mast = new THREE.Mesh(new THREE.CylinderGeometry(.1, .15, 6, 8), poleMat); mast.position.y = 3; pole.add(mast);\n  const dir = z > 0 ? -1 : 1;\n  const arm = new THREE.Mesh(new THREE.BoxGeometry(.1, .1, 1.8), poleMat); arm.position.set(0, 5.9, dir * .8); pole.add(arm);\n  const bulb = new THREE.Mesh(new THREE.SphereGeometry(.2, 8, 8), new THREE.MeshBasicMaterial({ color: 0xfff0aa })); bulb.position.set(0, 5.8, dir * 1.6); pole.add(bulb);\n  if (z > 0) { const l = new THREE.PointLight(0xffe899, 1.1, 16, 1.8); l.position.copy(bulb.position); pole.add(l); } // one light per pair keeps the GPU happy\n  worldGroup.add(pole);\n}));\n[10, 20, 30, 40].forEach(d => {\n  const g = new THREE.Group(); g.position.set(d, 0, 0);\n  const m = new THREE.MeshStandardMaterial({ color: 0x334155, metalness: .7 });\n  const bar = new THREE.Mesh(new THREE.BoxGeometry(.2, .2, 13), m); bar.position.y = 5.2; g.add(bar);\n  const p1 = new THREE.Mesh(new THREE.CylinderGeometry(.12, .12, 5.2), m); p1.position.set(0, 2.6, -6.4); g.add(p1);\n  const p2 = p1.clone(); p2.position.z = 6.4; g.add(p2);\n  const sign = new THREE.Mesh(new THREE.BoxGeometry(.05, .8, 3.2), new THREE.MeshBasicMaterial({ color: d === 40 ? 0x00ff9d : 0x00f0ff })); sign.position.set(0, 4.8, 0); g.add(sign);\n  worldGroup.add(g);\n});\n\n/* ===== 4. EGO CAR (+x forward, +z to the car\'s right; driver sits on the RIGHT for left-hand traffic) ===== */\nconst carGroup = new THREE.Group();\nscene.add(carGroup);\nconst bodyMat = new THREE.MeshStandardMaterial({ color: 0x1d4ed8, roughness: .25, metalness: .85 });\nconst carbonMat = new THREE.MeshStandardMaterial({ color: 0x111827, roughness: .7, metalness: .3 });\nconst glassMat = new THREE.MeshPhysicalMaterial({ color: 0x0284c7, transparent: true, opacity: .45, roughness: .1, metalness: .1 });\nconst chassis = new THREE.Mesh(new THREE.BoxGeometry(VEH_LEN, .65, HALF_W * 2), bodyMat);\nchassis.position.y = .5; chassis.castShadow = true; carGroup.add(chassis);\nconst cabin = new THREE.Mesh(new THREE.BoxGeometry(2.3, .65, HALF_W * 1.8), glassMat); cabin.position.set(-.25, 1.05, 0); carGroup.add(cabin);\nconst roof = new THREE.Mesh(new THREE.BoxGeometry(2.1, .08, HALF_W * 1.7), bodyMat); roof.position.set(-.25, 1.4, 0); carGroup.add(roof);\n\nconst tireGeo = new THREE.CylinderGeometry(.36, .36, .28, 20), tireMat = new THREE.MeshStandardMaterial({ color: 0x0f172a, roughness: .8 });\nconst rimGeo = new THREE.CylinderGeometry(.24, .24, .29, 12), spokeGeo = new THREE.BoxGeometry(.46, .04, .31);\nconst rimMat = new THREE.MeshStandardMaterial({ color: 0xe2e8f0, metalness: .9, roughness: .2 });\nfunction createWheel() {           // axle runs along z; spin about z. Spokes make the rolling visible\n  const g = new THREE.Group();\n  const tire = new THREE.Mesh(tireGeo, tireMat); tire.rotation.x = Math.PI / 2; tire.castShadow = true; g.add(tire);\n  const rim = new THREE.Mesh(rimGeo, rimMat); rim.rotation.x = Math.PI / 2; g.add(rim);\n  for (let i = 0; i < 3; i++) { const s = new THREE.Mesh(spokeGeo, carbonMat); s.rotation.z = i * Math.PI / 3; g.add(s); }\n  return g;\n}\nconst frontLeftGroup = new THREE.Group(); frontLeftGroup.position.set(1.3, .36, -HALF_W - .05);\nconst frontRightGroup = new THREE.Group(); frontRightGroup.position.set(1.3, .36, HALF_W + .05);\nconst flWheel = createWheel(), frWheel = createWheel(), rlWheel = createWheel(), rrWheel = createWheel();\nfrontLeftGroup.add(flWheel); frontRightGroup.add(frWheel);\nrlWheel.position.set(-1.3, .36, -HALF_W - .05); rrWheel.position.set(-1.3, .36, HALF_W + .05);\ncarGroup.add(frontLeftGroup, frontRightGroup, rlWheel, rrWheel);\nconst allWheels = [flWheel, frWheel, rlWheel, rrWheel];\n\n[-.6, .6].forEach(z => {\n  const hl = new THREE.SpotLight(0xffffff, 2.5, 30, Math.PI / 5, .3);\n  hl.position.set(2.1, .6, z); hl.target.position.set(12, 0, z);\n  carGroup.add(hl, hl.target);\n});\nconst fBar = new THREE.Mesh(new THREE.BoxGeometry(.05, .08, 1.4), new THREE.MeshBasicMaterial({ color: 0x00f0ff })); fBar.position.set(2.11, .62, 0); carGroup.add(fBar);\nconst tBar = new THREE.Mesh(new THREE.BoxGeometry(.05, .08, 1.6), new THREE.MeshBasicMaterial({ color: 0xff1144 })); tBar.position.set(-2.11, .68, 0); carGroup.add(tBar);\nconst underglow = new THREE.PointLight(0x00f0ff, 1.2, 4); underglow.position.set(0, .15, 0); carGroup.add(underglow);\n\nconst DRIVER_Z = 0.38;                                   // right-hand drive\nconst dash = new THREE.Mesh(new THREE.BoxGeometry(.7, .35, 1.55), carbonMat); dash.position.set(.65, .8, 0); carGroup.add(dash);\nconst steerGroup = new THREE.Group();\nsteerGroup.position.set(.42, .85, DRIVER_Z); steerGroup.rotation.y = -Math.PI / 2; steerGroup.rotation.x = -.35;\nconst swMat = new THREE.MeshStandardMaterial({ color: 0x1f2937, roughness: .5 });\nsteerGroup.add(new THREE.Mesh(new THREE.TorusGeometry(.2, .024, 12, 28), swMat));\nsteerGroup.add(new THREE.Mesh(new THREE.BoxGeometry(.38, .03, .015), swMat));\nconst hub = new THREE.Mesh(new THREE.CylinderGeometry(.06, .06, .03, 16), new THREE.MeshStandardMaterial({ color: 0x00f0ff, metalness: .9 })); hub.rotation.x = Math.PI / 2; steerGroup.add(hub);\ncarGroup.add(steerGroup);\nconst cluster = new THREE.Mesh(new THREE.PlaneGeometry(.28, .12), new THREE.MeshBasicMaterial({ color: 0x00f0ff }));\ncluster.position.set(.58, .94, DRIVER_Z); cluster.rotation.y = -Math.PI / 2; cluster.rotation.x = -.2; carGroup.add(cluster);\nconst tablet = new THREE.Mesh(new THREE.BoxGeometry(.02, .26, .38), new THREE.MeshStandardMaterial({ color: 0x0f172a, roughness: .2 }));\ntablet.position.set(.48, .88, -.12); tablet.rotation.y = .3; carGroup.add(tablet);\nconst driverSeatPos = new THREE.Vector3(-.05, 1.1, DRIVER_Z);\n\n/* ===== 5. OBSTACLES ===== */\nlet obstacles = [];\nconst obstacleMeshes = [];\n\nfunction makeObstacleData(kind, motion, x, y) {\n  const p = OBJ_PROPS[kind];\n  let vx = 0, vy = 0;\n  if (kind !== "🕳️ Pothole") {\n    if (motion === "Crossing the road") vy = y >= 0 ? -p.spd : p.spd;\n    else if (motion === "Moving ahead") vx = p.spd;\n  }\n  return { type: kind, motion, x, y, x0: x, y0: y, vx, vy, vy0: vy, L: p.L, W: p.W, animTime: Math.random() * 10 };\n}\n\nfunction createObstacle3D(o) {\n  const g = new THREE.Group();\n  g.position.set(o.x, 0, o.y);\n  const M = (c, extra = {}) => new THREE.MeshStandardMaterial(Object.assign({ color: c, roughness: .7 }, extra));\n  const add = (geo, mat, x, y, z) => { const m = new THREE.Mesh(geo, mat); m.position.set(x, y, z); g.add(m); return m; };\n  if (o.type === "🧍 Pedestrian") {\n    const skin = M(0xfbcfe8), cloth = M(0x0284c7), pants = M(0x1e293b, { roughness: .8 });\n    add(new THREE.SphereGeometry(.18, 12, 12), skin, 0, 1.55, 0);\n    add(new THREE.CylinderGeometry(.2, .22, .65, 8), cloth, 0, 1.05, 0);\n    const lg = new THREE.CylinderGeometry(.08, .07, .65, 8);\n    g.userData.legs = [add(lg, pants, 0, .35, -.12), add(lg, pants, 0, .35, .12)];\n    g.userData.kind = "pedestrian";\n  } else if (o.type === "🐄 Cow") {\n    const white = M(0xf1f5f9), black = M(0x0f172a), horn = M(0xd97706);\n    add(new THREE.BoxGeometry(1.6, .8, .7), white, 0, .85, 0);\n    add(new THREE.BoxGeometry(.6, .5, .72), black, .1, .88, 0);\n    add(new THREE.BoxGeometry(.5, .45, .4), white, .9, 1.1, 0);\n    const hg = new THREE.ConeGeometry(.04, .2, 6);\n    add(hg, horn, .85, 1.35, -.15).rotation.z = -.3; add(hg, horn, .85, 1.35, .15).rotation.z = -.3;\n    const lg = new THREE.CylinderGeometry(.08, .08, .55, 8);\n    g.userData.legs = [add(lg, white, .55, .28, -.25), add(lg, white, .55, .28, .25), add(lg, white, -.55, .28, -.25), add(lg, white, -.55, .28, .25)];\n    g.userData.kind = "cow";\n  } else if (o.type === "🛒 Pushcart") {\n    const wood = M(0x92400e, { roughness: .8 }), metal = M(0x64748b, { metalness: .8 });\n    add(new THREE.BoxGeometry(1.4, .5, .9), wood, 0, .65, 0);\n    const wg = new THREE.CylinderGeometry(.3, .3, .1, 16);\n    add(wg, metal, 0, .3, -.52).rotation.x = Math.PI / 2; add(wg, metal, 0, .3, .52).rotation.x = Math.PI / 2;\n    add(new THREE.BoxGeometry(.6, .06, .8), metal, -.9, .85, 0);\n  } else if (o.type === "🚗 Vehicle") {\n    add(new THREE.BoxGeometry(3.8, .7, 1.7), M(0xe11d48, { metalness: .8, roughness: .3 }), 0, .55, 0);\n    add(new THREE.BoxGeometry(2.1, .6, 1.5), glassMat.clone(), -.2, 1.05, 0);\n    add(new THREE.BoxGeometry(.05, .1, 1.4), new THREE.MeshBasicMaterial({ color: 0xfef08a }), 1.91, .58, 0);\n  } else {\n    add(new THREE.CylinderGeometry(.6, .6, .08, 18), M(0x05070c, { roughness: 1 }), 0, .01, 0);\n    const ring = add(new THREE.RingGeometry(.58, .75, 24), new THREE.MeshBasicMaterial({ color: 0xf59e0b, side: THREE.DoubleSide }), 0, .02, 0);\n    ring.rotation.x = -Math.PI / 2;\n  }\n  if (Math.abs(o.vx) + Math.abs(o.vy) > .01) {\n    g.add(new THREE.ArrowHelper(new THREE.Vector3(o.vx, 0, o.vy).normalize(), new THREE.Vector3(0, .04, 0), 2.0, 0xfacc15, .6, .4));\n  }\n  g.traverse(c => { if (c.isMesh) c.castShadow = true; });\n  scene.add(g);\n  return g;\n}\n\nfunction syncObstacleMeshes() {\n  obstacleMeshes.forEach(m => { scene.remove(m); disposeTree(m); });   // free GPU memory\n  obstacleMeshes.length = 0;\n  obstacles.forEach(o => obstacleMeshes.push(createObstacle3D(o)));\n  document.getElementById(\'obs-count-badge\').innerText = `${obstacles.length} PLACED`;\n}\n\n/* ===== 6. TIME-AWARE A* ===== */\nfunction fold(y, lim = LIM) {\n  const p = 4 * lim, m = (((y + lim) % p) + p) % p;\n  return -lim + (m <= 2 * lim ? m : p - m);\n}\nfunction predict(o, t) {\n  return { x: Math.min(o.x + o.vx * t, X_END + 8), y: o.vy !== 0 ? fold(o.y + o.vy * t) : o.y };\n}\nfunction densify(pts, step = RES) {\n  if (!pts || !pts.length) return [];\n  const out = [pts[0]];\n  for (let i = 0; i < pts.length - 1; i++) {\n    const a = pts[i], b = pts[i + 1], n = Math.max(Math.ceil(Math.hypot(b[0] - a[0], b[1] - a[1]) / step), 1);\n    for (let k = 1; k <= n; k++) out.push([a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n]);\n  }\n  return out;\n}\n// Is the straight corridor still clear when the car arrives, starting from its CURRENT x?\nfunction analysePath(pts, obs, fromX = 0) {\n  const dense = densify([[0, 0], ...pts]);\n  for (const p of dense) {\n    if (p[0] < fromX - 0.01) continue;\n    const t = Math.max(p[0] - fromX, 0) / V_NOM;\n    if (Math.abs(p[1]) + HALF_W + .3 > ROAD_HALF) return { blocked: true, firstBlockX: p[0] };\n    for (const o of obs) {\n      const q = predict(o, t);\n      if (Math.abs(p[0] - q.x) < o.L / 2 + HX && Math.abs(p[1] - q.y) < o.W / 2 + HY) return { blocked: true, firstBlockX: p[0] };\n    }\n  }\n  return { blocked: false, firstBlockX: null };\n}\nfunction makePaths(n) {\n  const out = {};\n  for (let i = 0; i < n; i++) {\n    const y = n === 1 ? 0 : 3.8 - (7.6 / (n - 1)) * i;\n    out[`Path ${i + 1}`] = [[8, y], [X_END, y]];\n  }\n  return out;\n}\n\nclass MinHeap {\n  constructor() { this.h = []; }\n  size() { return this.h.length; }\n  push(it) { const h = this.h; h.push(it); let i = h.length - 1; while (i > 0) { const p = (i - 1) >> 1; if (h[i].f < h[p].f) { [h[i], h[p]] = [h[p], h[i]]; i = p; } else break; } }\n  pop() {\n    const h = this.h, top = h[0], last = h.pop();\n    if (h.length) { h[0] = last; let i = 0; for (;;) { let l = 2 * i + 1, r = l + 1, b = i; if (l < h.length && h[l].f < h[b].f) b = l; if (r < h.length && h[r].f < h[b].f) b = r; if (b === i) break; [h[i], h[b]] = [h[b], h[i]]; i = b; } }\n    return top;\n  }\n}\n\nlet devCache = { key: \'\', arr: null };\nfunction deviationField(nominalPath) {\n  const key = nominalPath.map(p => p.join(\',\')).join(\'|\');\n  if (devCache.key === key) return devCache.arr;\n  const dev = new Float32Array(XS.length * YS.length), nom = densify([[0, 0], ...nominalPath]);\n  for (let i = 0; i < XS.length; i++) for (let j = 0; j < YS.length; j++) {\n    let m = 999;\n    for (let k = 0; k < nom.length; k += 4) { const d = Math.hypot(XS[i] - nom[k][0], YS[j] - nom[k][1]); if (d < m) m = d; }\n    dev[i * YS.length + j] = m;\n  }\n  devCache = { key, arr: dev };\n  return dev;\n}\n\nfunction planRoute(goal, obs, start, v, nominalPath, tight = false) {\n  const hxm = tight ? HX0 + .4 : HX, hym = tight ? HY0 + .4 : HY;   // tight mode: smaller safety margins (used with reduced speed)\n  const NY = YS.length, [sx, sy] = start;\n  const si = Math.round(sx / RES);\n  let sj = 0, md = 999;\n  for (let j = 0; j < NY; j++) { const d = Math.abs(YS[j] - sy); if (d < md) { md = d; sj = j; } }\n  if (si >= XS.length - 1) return densify([start, [X_END, sy]]);\n  const dev = deviationField(nominalPath);\n  const blocked = new Uint8Array(XS.length * NY), vel = Math.max(v, .3);\n  for (let i = 0; i < XS.length; i++) {\n    const t = (Math.max(XS[i] - sx, 0) / vel) * 1.1;\n    const preds = obs.map(o => ({ o, q: predict(o, t), unc: tight ? .1 : Math.min(.2 + .15 * t * (Math.abs(o.vx) + Math.abs(o.vy)), 1.2) }));\n    for (let j = 0; j < NY; j++) {\n      const y = YS[j];\n      if (Math.abs(y) + HALF_W + .3 > ROAD_HALF) { blocked[i * NY + j] = 1; continue; }\n      for (const { o, q, unc } of preds) {\n        if (Math.abs(XS[i] - q.x) < o.L / 2 + hxm + unc && Math.abs(y - q.y) < o.W / 2 + hym + unc) { blocked[i * NY + j] = 1; break; }\n      }\n    }\n  }\n  for (let i = Math.max(si - 2, 0); i <= Math.min(si + 5, XS.length - 1); i++)\n    for (let j = Math.max(sj - 4, 0); j <= Math.min(sj + 4, NY - 1); j++) blocked[i * NY + j] = 0;\n\n  let gi = -1, gj = -1, best = 1e9;\n  for (let i = XS.length - 1; i >= XS.length - 5 && gi === -1; i--)\n    for (let j = 0; j < NY; j++) if (!blocked[i * NY + j]) { const d = Math.hypot(XS[i] - goal[0], YS[j] - goal[1]); if (d < best) { best = d; gi = i; gj = j; } }\n  if (gi === -1) return null;\n\n  const g = new Float64Array(XS.length * NY).fill(Infinity), parent = new Int32Array(XS.length * NY).fill(-1);\n  g[si * NY + sj] = 0;\n  const heap = new MinHeap();\n  heap.push({ f: RES * (gi - si), g: 0, i: si, j: sj });\n  let endI = -1, endJ = -1;\n  while (heap.size()) {\n    const c = heap.pop(), ci = c.i * NY + c.j;\n    if (c.g > g[ci] + 1e-5) continue;\n    if (c.i === gi && Math.abs(c.j - gj) <= 1) { endI = c.i; endJ = c.j; break; }\n    for (let dj = -1; dj <= 1; dj++) {\n      const ni = c.i + 1, nj = c.j + dj;\n      if (ni >= XS.length || nj < 0 || nj >= NY) continue;\n      const nIdx = ni * NY + nj;\n      if (blocked[nIdx]) continue;\n      const ng = c.g + Math.hypot(1, dj) * RES * (1 + .8 * dev[nIdx]) + .02 * Math.abs(dj);\n      if (ng < g[nIdx]) { g[nIdx] = ng; parent[nIdx] = ci; heap.push({ f: ng + RES * (gi - ni) + Math.abs(gj - nj) * RES * .5, g: ng, i: ni, j: nj }); }\n    }\n  }\n  if (endI === -1) return null;\n\n  const raw = [];\n  for (let cur = endI * NY + endJ; cur !== -1; cur = parent[cur]) {\n    const ci = Math.floor(cur / NY), cj = cur % NY;\n    raw.push([XS[ci], YS[cj]]);\n    if (ci === si && cj === sj) break;\n  }\n  raw.reverse();\n  raw[0] = [sx, sy];\n  const k = 7, half = 3;\n  if (raw.length <= k) return densify(raw);\n  const sm = raw.map((_, i) => {\n    let ax = 0, ay = 0;\n    for (let w = -half; w <= half; w++) { const p = raw[clamp(i + w, 0, raw.length - 1)]; ax += p[0]; ay += p[1]; }\n    return [ax / k, ay / k];\n  });\n  sm[0] = raw[0]; sm[sm.length - 1] = raw[raw.length - 1];\n  // smoothing must not cut a corner into a blocked cell; otherwise fall back to the raw route\n  const clipped = sm.slice(5).some(p => {\n    const ci = clamp(Math.round(p[0] / RES), 0, XS.length - 1), cj = clamp(Math.round((p[1] + ROAD_HALF) / RES), 0, NY - 1);\n    return blocked[ci * NY + cj];\n  });\n  return densify(clipped ? raw : sm);\n}\n\n/* ===== 7. SIM STATE ===== */\nlet pathCount = 3, paths = makePaths(3), chosenPathName = "Path 1";\nlet selectedObstacleType = "🧍 Pedestrian", selectedObstacleMotion = "Static";\nlet simSpeedMultiplier = 1.0, driveMode = \'auto\';\n\nconst sim = {\n  running: false, pos: [0, 0], heading: 0, steerAngle: 0, velocity: 0, route: [], idx: 0, trail: [[0, 0]],\n  replans: 0, tight: false, status: \'ready\', tick: 0, simTime: 0, lastReplanT: 0, collided: false, inContact: false,\n};\n\nfunction toast(text, ms = 2400) {\n  const t = document.getElementById(\'toast-msg\');\n  t.innerText = text; t.classList.add(\'show\');\n  clearTimeout(toast._t); toast._t = setTimeout(() => t.classList.remove(\'show\'), ms);\n}\nlet lastStatusText = \'\';\nfunction updateStatusDisplay(text, color) {\n  if (text === lastStatusText) return;\n  lastStatusText = text;\n  const st = document.getElementById(\'status-text\'), dot = document.getElementById(\'status-dot\');\n  st.innerText = text; st.style.color = color; dot.style.background = color; dot.style.boxShadow = `0 0 10px ${color}`;\n}\nfunction triggerWarningStrobe() {\n  const w = document.getElementById(\'warning-strobe\');\n  w.style.opacity = \'1\'; setTimeout(() => (w.style.opacity = \'0\'), 250);\n}\nfunction setDriveButtons(running) {\n  [\'btn-drive\', \'btn-top-drive\'].forEach(id => {\n    const b = document.getElementById(id);\n    b.innerText = running ? \'⏸ Pause driving\' : \'▶ Start driving\';\n    b.classList.toggle(\'stopped\', running);\n  });\n}\n\n/* --- 3D path ribbons (rebuilt only on edits; recoloured live) --- */\nconst pathLineGroup = new THREE.Group(); scene.add(pathLineGroup);\nconst pathTubes = {};\nfunction render3DPaths() {\n  while (pathLineGroup.children.length) { const c = pathLineGroup.children[0]; pathLineGroup.remove(c); disposeTree(c); }\n  Object.keys(pathTubes).forEach(k => delete pathTubes[k]);\n  Object.entries(paths).forEach(([name, pts]) => {\n    const full = densify([[0, 0], ...pts]);\n    const curve = new THREE.CatmullRomCurve3(full.filter((_, i) => i % 4 === 0 || i === full.length - 1).map(p => new THREE.Vector3(p[0], .08, p[1])));\n    const sel = name === chosenPathName;\n    const tube = new THREE.Mesh(new THREE.TubeGeometry(curve, 48, sel ? .12 : .05, 8, false), new THREE.MeshBasicMaterial({ transparent: true, opacity: sel ? .95 : .45 }));\n    pathLineGroup.add(tube); pathTubes[name] = tube;\n  });\n  colorPathTubes();\n}\nfunction colorPathTubes() {\n  Object.entries(paths).forEach(([name, pts]) => {\n    const tube = pathTubes[name]; if (!tube) return;\n    const blocked = analysePath(pts, obstacles, sim.pos[0]).blocked;\n    tube.material.color.set(blocked ? 0xff3366 : (name === chosenPathName ? 0x00f0ff : 0x334155));\n  });\n}\nfunction updatePathCards() {\n  const list = document.getElementById(\'paths-list\');\n  list.innerHTML = \'\';\n  Object.entries(paths).forEach(([name, pts]) => {\n    const s = analysePath(pts, obstacles, sim.pos[0]);\n    const card = document.createElement(\'div\');\n    card.className = `path-card ${name === chosenPathName ? \'selected\' : \'\'}`;\n    card.onclick = () => selectPath(name);\n    card.innerHTML = `<div><span class="path-name">${name}</span><span class="path-meta">${s.blocked ? `Blocked at ${s.firstBlockX.toFixed(1)}m` : \'Clear corridor\'}</span></div>\n      <span class="badge ${s.blocked ? \'blocked\' : \'clear\'}">${s.blocked ? \'⛔ BLOCKED\' : \'✅ CLEAR\'}</span>`;\n    list.appendChild(card);\n  });\n  colorPathTubes();\n}\nfunction updatePathUI() { updatePathCards(); render3DPaths(); }\n\n/* --- active trajectory (disposed on each rebuild) + preallocated tyre trail --- */\nlet activeTrajectoryMesh = null;\nfunction renderActiveTrajectory() {\n  if (activeTrajectoryMesh) { scene.remove(activeTrajectoryMesh); disposeTree(activeTrajectoryMesh); activeTrajectoryMesh = null; }\n  if (driveMode !== \'auto\' || !sim.running || sim.route.length < sim.idx + 3) return;\n  const rem = sim.route.slice(sim.idx).filter((_, i, a) => i % 4 === 0 || i === a.length - 1);\n  if (rem.length < 2) return;\n  const curve = new THREE.CatmullRomCurve3(rem.map(p => new THREE.Vector3(p[0], .12, p[1])));\n  activeTrajectoryMesh = new THREE.Mesh(new THREE.TubeGeometry(curve, Math.min(rem.length * 2, 60), .15, 6, false), new THREE.MeshBasicMaterial({ color: 0x00ff9d, transparent: true, opacity: .9 }));\n  scene.add(activeTrajectoryMesh);\n}\nconst TRAIL_MAX = 4000;\nconst trailGeo = new THREE.BufferGeometry();\ntrailGeo.setAttribute(\'position\', new THREE.BufferAttribute(new Float32Array(TRAIL_MAX * 3), 3));\ntrailGeo.setDrawRange(0, 0);\nconst trailLine = new THREE.Line(trailGeo, new THREE.LineBasicMaterial({ color: 0x00f0ff, transparent: true, opacity: .6 }));\ntrailLine.frustumCulled = false; scene.add(trailLine);\nfunction updateTrail() {\n  const a = trailGeo.attributes.position.array, n = Math.min(sim.trail.length, TRAIL_MAX);\n  for (let i = 0; i < n; i++) { a[3 * i] = sim.trail[i][0]; a[3 * i + 1] = .03; a[3 * i + 2] = sim.trail[i][1]; }\n  trailGeo.attributes.position.needsUpdate = true; trailGeo.setDrawRange(0, n);\n}\n\n/* --- UI handlers --- */\nfunction selectPath(name) {\n  chosenPathName = name; updatePathUI();\n  if (sim.running) triggerReplan();\n}\nfunction updatePathCount(val) {\n  pathCount = parseInt(val);\n  document.getElementById(\'path-slider-val\').innerText = val;\n  document.getElementById(\'path-count-lbl\').innerText = `${val} PATHS`;\n  paths = makePaths(pathCount);\n  if (!paths[chosenPathName]) chosenPathName = Object.keys(paths)[0];\n  updatePathUI();\n  if (sim.running) triggerReplan();\n}\ndocument.querySelectorAll(\'.sp-item[data-kind]\').forEach(el => el.addEventListener(\'click\', () => {\n  selectedObstacleType = el.dataset.kind;\n  document.querySelectorAll(\'.sp-item\').forEach(i => i.classList.remove(\'active\'));\n  el.classList.add(\'active\');\n}));\ndocument.getElementById(\'sp-random\').addEventListener(\'click\', randomizeObstacles);\ndocument.querySelectorAll(\'#motion-toggle .tg-btn\').forEach(b => b.addEventListener(\'click\', () => {   // scoped: no longer clears the speed buttons\n  selectedObstacleMotion = b.dataset.motion;\n  document.querySelectorAll(\'#motion-toggle .tg-btn\').forEach(x => x.classList.remove(\'active\'));\n  b.classList.add(\'active\');\n}));\ndocument.querySelectorAll(\'#speed-toggle .tg-btn\').forEach(b => b.addEventListener(\'click\', () => {\n  simSpeedMultiplier = parseFloat(b.dataset.speed);\n  document.querySelectorAll(\'#speed-toggle .tg-btn\').forEach(x => x.classList.remove(\'active\'));\n  b.classList.add(\'active\');\n}));\n\n/* ===== 8. AUDIO ===== */\nlet audioCtx = null, audioEnabled = true, engineOsc = null, engineGain = null;\nfunction initAudio() {\n  if (audioCtx) { if (audioCtx.state === \'suspended\') audioCtx.resume(); return; }\n  try {\n    audioCtx = new (window.AudioContext || window.webkitAudioContext)();\n    engineOsc = audioCtx.createOscillator(); engineGain = audioCtx.createGain();\n    engineOsc.type = \'sawtooth\'; engineGain.gain.value = 0;\n    const f = audioCtx.createBiquadFilter(); f.type = \'lowpass\'; f.frequency.value = 300;\n    engineOsc.connect(f); f.connect(engineGain); engineGain.connect(audioCtx.destination); engineOsc.start();\n  } catch (e) { console.warn(\'Audio unavailable\', e); }\n}\nfunction playChime(freq = 587, type = \'sine\', dur = .25, vol = .15) {\n  if (!audioEnabled || !audioCtx) return;\n  try {\n    const o = audioCtx.createOscillator(), g = audioCtx.createGain();\n    o.type = type; o.frequency.value = freq; g.gain.setValueAtTime(vol, audioCtx.currentTime);\n    g.gain.exponentialRampToValueAtTime(.001, audioCtx.currentTime + dur);\n    o.connect(g); g.connect(audioCtx.destination); o.start(); o.stop(audioCtx.currentTime + dur);\n  } catch (e) {}\n}\nfunction setEngine(v) {            // v < 0 silences the hum (paused / stopped / finished)\n  if (!audioCtx || !engineOsc) return;\n  const t = audioCtx.currentTime;\n  engineGain.gain.setTargetAtTime(audioEnabled && v >= 0 ? (v > 0 ? .05 : .015) : 0, t, .05);\n  if (v >= 0) engineOsc.frequency.setTargetAtTime(45 + v * 25, t, .05);\n}\nfunction toggleAudio() {\n  audioEnabled = !audioEnabled;\n  document.getElementById(\'sound-icon\').innerText = audioEnabled ? \'🔊\' : \'🔇\';\n  setEngine(audioEnabled && sim.running ? sim.velocity : -1);\n}\n\n/* ===== 9. CONTROLLERS ===== */\nfunction willCollide(x, y, obs, buf = .35) {\n  for (const o of obs) if (Math.abs(x - o.x) < o.L / 2 + HX0 + buf && Math.abs(y - o.y) < o.W / 2 + HY0 + buf) return o;\n  return null;\n}\nconst curSpeed = () => V_NOM * (sim.tight ? .5 : 1);\nfunction replan(count = true) {\n  const nominal = paths[chosenPathName] || Object.values(paths)[0];\n  const start = sim.pos[0] < .5 ? [0, 0] : [sim.pos[0], sim.pos[1]];\n  const goal = nominal[nominal.length - 1];\n  let r = null;\n  try { r = planRoute(goal, obstacles, start, V_NOM, nominal); sim.tight = false; } catch (e) { console.warn(\'planRoute failed\', e); }\n  if (!r) {                                       // no route with full margins: try tighter margins at half speed\n    try { r = planRoute(goal, obstacles, start, V_NOM * .5, nominal, true); sim.tight = !!r; } catch (e) { console.warn(\'tight planRoute failed\', e); }\n  }\n  sim.lastReplanT = sim.simTime;\n  if (!r || r.length < 2) { sim.route = []; sim.status = \'waiting\'; sim.tight = false; return false; }\n  sim.route = r; sim.idx = 0; sim.status = \'driving\'; sim.waitSince = 0;\n  if (count) sim.replans++;\n  renderActiveTrajectory();\n  return true;\n}\nfunction triggerReplan() {                       // was called everywhere but never defined\n  if (sim.running && driveMode === \'auto\') replan(true);\n  updatePathUI();\n}\nfunction routeOk() {\n  const r = sim.route;\n  if (r.length < 2) return false;\n  for (let k = sim.idx + 4; k < Math.min(r.length, sim.idx + 64); k++) {\n    const p = r[k], t = Math.max(p[0] - sim.pos[0], 0) / curSpeed();\n    for (const o of obstacles) {\n      const q = predict(o, t);\n      if (Math.abs(p[0] - q.x) < o.L / 2 + VX && Math.abs(p[1] - q.y) < o.W / 2 + VY) return false;\n    }\n  }\n  return true;\n}\n\nfunction turnHeading(newHeading, dt, speed) {\n  const dH = newHeading - sim.heading;\n  sim.heading += dH * Math.min(1, dt * 8);\n  const yawRate = dt > 0 ? (dH * Math.min(1, dt * 8)) / dt : 0;\n  const target = speed > .3 ? clamp(Math.atan(2.6 * yawRate / speed), -.6, .6) : 0;\n  sim.steerAngle += (target - sim.steerAngle) * Math.min(1, dt * 10);\n}\n\nfunction stepAuto(dt) {\n  if (sim.status === \'waiting\') {\n    sim.velocity = 0;\n    if (sim.simTime - sim.lastReplanT > .3) replan(false);\n    if (!sim.waitSince) sim.waitSince = sim.simTime;\n    updateStatusDisplay(sim.simTime - sim.waitSince > 3 ? \'⏸ Road blocked. Undo or clear an obstacle.\' : \'⏸ No safe gap. Waiting for clearance...\', \'#ffaa00\');\n    if (sim.status === \'waiting\') return;\n  } else if (!routeOk()) {\n    replan(true);\n    if (sim.status === \'waiting\') return;\n  }\n  const r = sim.route;\n  while (sim.idx < r.length - 2 && r[sim.idx + 1][0] <= sim.pos[0]) sim.idx++;\n  const targetY = r[Math.min(sim.idx + 3, r.length - 1)][1];\n  const [ox, oy] = sim.pos, lim = ROAD_HALF - .9, spd = curSpeed();\n  let advance = 1.0, hit = null, nx, ny;\n  for (const [f, lat] of [[1, 1], [1, 0], [.3, 1], [.3, 0], [0, 0]]) {   // steer+go, straight, steer+creep, straight creep, hold\n    const maxLat = LAT_RATE * spd * dt * f * lat;                      // a stopped car cannot slide sideways\n    ny = clamp(oy + clamp(targetY - oy, -maxLat, maxLat), -lim, lim);\n    nx = ox + spd * dt * f;\n    hit = willCollide(nx, ny, obstacles, .3);\n    advance = f;\n    if (!hit) break;\n  }\n  if (advance === 0) {                           // blocked this tick: brake and ask for a fresh route\n    sim.velocity = 0;\n    updateStatusDisplay(`🛑 BRAKING: ${hit ? hit.type.split(\' \')[1] : \'obstacle\'} ahead. Re-routing...`, \'#ffaa00\');\n    triggerWarningStrobe();\n    if (sim.simTime - sim.lastReplanT > .25) replan(true);\n    return;\n  }\n  sim.pos = [nx, ny];\n  sim.velocity = spd * advance;\n  turnHeading(Math.atan2(ny - oy, nx - ox), dt, sim.velocity);\n  if (sim.tick % 4 === 0) sim.trail.push([nx, ny]);\n  updateStatusDisplay(sim.tight ? \'🐢 Tight gap: squeezing through at half speed\' : advance < 1 ? \'⚠️ Slowing for obstacle...\' : `Driving ${chosenPathName}. A* route active.`, sim.tight || advance < 1 ? \'#ffaa00\' : \'#00f0ff\');\n}\n\nfunction stepManual(dt) {\n  const k = keys;\n  const up = k[\'w\'] || k[\'arrowup\'], down = k[\'s\'] || k[\'arrowdown\'], left = k[\'a\'] || k[\'arrowleft\'], right = k[\'d\'] || k[\'arrowright\'];\n  let v = sim.velocity;\n  v += (up ? 4.5 : down ? -9 : -1.5) * dt;\n  v = clamp(v, 0, 9);\n  const steerIn = ((right ? 1 : 0) - (left ? 1 : 0)) * .5;       // +z is the car\'s right\n  sim.steerAngle += (steerIn - sim.steerAngle) * Math.min(1, dt * 6);\n  sim.heading = clamp(sim.heading + (v / 2.6) * Math.tan(sim.steerAngle) * dt, -.9, .9);\n  if (steerIn === 0) sim.heading *= (1 - Math.min(1, dt * 1.5));  // light self-centring\n  let nx = Math.max(sim.pos[0] + v * Math.cos(sim.heading) * dt, 0), ny = sim.pos[1] + v * Math.sin(sim.heading) * dt;\n  const lim = ROAD_HALF - .9;\n  if (Math.abs(ny) > lim) { ny = clamp(ny, -lim, lim); v *= .97; }\n  const hit = willCollide(nx, ny, obstacles, 0);\n  if (hit) { v = 0; nx = sim.pos[0]; ny = sim.pos[1]; }            // crash stops the car (audit below flags it)\n  sim.pos = [nx, ny]; sim.velocity = v;\n  if (sim.tick % 4 === 0) sim.trail.push([nx, ny]);\n  updateStatusDisplay(`MANUAL // ${v.toFixed(1)} m/s  [W/S] throttle  [A/D] steer`, \'#00ff9d\');\n}\n\nfunction finishRun() {\n  sim.running = false; sim.status = \'done\'; sim.velocity = 0;\n  setDriveButtons(false); setEngine(-1);\n  playChime(784, \'triangle\', .6, .3);\n  if (sim.collided) {\n    toast(\'🏁 Reached the goal, but a collision occurred.\');\n    updateStatusDisplay(\'🏁 Goal reached (collision recorded)\', \'#ffaa00\');\n  } else {\n    toast(`🏁 Arrived safely with zero collisions! ${driveMode === \'auto\' ? `(${sim.replans} re-plans)` : \'\'}`);\n    updateStatusDisplay(\'🏁 Arrived at goal with ZERO collisions!\', \'#00ff9d\');\n  }\n}\n\nfunction simStep(dt) {\n  sim.simTime += dt; sim.tick++;\n  obstacles.forEach(o => {\n    let nx = Math.min(o.x + o.vx * dt, X_END + 8), ny = o.y + o.vy * dt, nvy = o.vy;\n    if (Math.abs(ny) > LIM) { ny = Math.sign(ny) * (2 * LIM - Math.abs(ny)); nvy = -o.vy; }\n    // pedestrians/cows/carts stop rather than walk into a stationary car (potholes never move anyway)\n    const stepsInto = Math.abs(sim.pos[0] - nx) < o.L / 2 + HX0 + .15 && Math.abs(sim.pos[1] - ny) < o.W / 2 + HY0 + .15;\n    if (!stepsInto) { o.x = nx; o.y = ny; o.vy = nvy; }\n    o.animTime += dt * 4;\n  });\n  if (driveMode === \'auto\') stepAuto(dt); else stepManual(dt);\n  const hit = willCollide(sim.pos[0], sim.pos[1], obstacles, 0);\n  if (hit && !sim.inContact) {\n    sim.collided = true; sim.inContact = true;\n    triggerWarningStrobe(); playChime(150, \'sawtooth\', .2, .3);\n    toast(`💥 Collision with ${hit.type.split(\' \')[1]}!`);\n  } else if (!hit) sim.inContact = false;\n  if (sim.pos[0] >= X_END - .1 && sim.status !== \'done\') finishRun();\n}\n\nlet lastTime = performance.now();\nfunction stepWorld() {\n  const now = performance.now(), dt = Math.min((now - lastTime) / 1000, .05);\n  lastTime = now;\n  if (sim.running) {\n    const sdt = dt * simSpeedMultiplier, n = Math.max(1, Math.ceil(sdt / .02));\n    for (let s = 0; s < n && sim.running; s++) simStep(sdt / n);\n    if (sim.running) {\n      if (sim.tick % 30 === 0) updatePathCards();            // badges stay truthful while driving\n      if (driveMode === \'auto\' && sim.tick % 12 === 0) renderActiveTrajectory();\n      if (sim.tick % 8 === 0) updateTrail();\n      setEngine(sim.velocity);\n    } else updateTrail();\n  }\n  // visuals\n  carGroup.position.set(sim.pos[0], 0, sim.pos[1]);\n  carGroup.rotation.y = -sim.heading;\n  frontLeftGroup.rotation.y = frontRightGroup.rotation.y = -sim.steerAngle;\n  steerGroup.rotation.z = -sim.steerAngle * 4.5;\n  const spin = (sim.velocity * dt * simSpeedMultiplier) / .36;\n  allWheels.forEach(w => (w.rotation.z -= spin));            // spin about the axle (z); minus = rolling forward\n  obstacles.forEach((o, i) => {\n    const m = obstacleMeshes[i]; if (!m) return;\n    m.position.set(o.x, 0, o.y);\n    const legs = m.userData.legs;\n    if (legs && (o.vx || o.vy)) {\n      const sw = Math.sin(o.animTime) * (m.userData.kind === \'cow\' ? .3 : .4);\n      legs.forEach((l, k) => (l.rotation.z = k % 2 ? -sw : sw));\n    }\n  });\n  if (sim.tick % 3 === 0 || !sim.running) updateTelemetry();\n}\nfunction updateTelemetry() {\n  document.getElementById(\'tel-speed\').innerText = sim.velocity.toFixed(1);\n  document.getElementById(\'tel-steer\').innerText = (sim.steerAngle * 180 / Math.PI).toFixed(1);\n  document.getElementById(\'tel-replans\').innerText = sim.replans;\n  document.getElementById(\'tel-dist\').innerText = sim.pos[0].toFixed(1);\n  const pct = Math.min(Math.round(sim.pos[0] / X_END * 100), 100);\n  document.getElementById(\'tel-pct\').innerText = pct + \'%\';\n  document.getElementById(\'progress-fill\').style.width = pct + \'%\';\n}\n\n/* ===== 10. START / PAUSE / RESET / MODE ===== */\nfunction startDriving() {\n  initAudio(); playChime(659, \'triangle\', .3, .15);\n  if (sim.running) { pauseDriving(); return; }\n  if (sim.status === \'done\' || sim.pos[0] >= X_END - .5) resetSimulation();\n  sim.running = true;\n  sim.status = \'driving\';\n  if (driveMode === \'auto\') {\n    if (sim.pos[0] < .5) { sim.pos = [0, 0]; sim.heading = 0; sim.trail = [[0, 0]]; sim.replans = 0; sim.collided = false; sim.inContact = false; }\n    replan(false);\n    sim.velocity = sim.status === \'waiting\' ? 0 : curSpeed();\n    toast(`▶ Driving ${chosenPathName} (goal in ~10 s at 1x)`);\n  } else toast(\'🎮 Manual: W/S throttle & brake, A/D steer\');\n  setDriveButtons(true);\n  lastTime = performance.now();\n}\nfunction pauseDriving() {\n  sim.running = false; sim.velocity = 0;\n  setDriveButtons(false); setEngine(-1);\n  renderActiveTrajectory();\n  updateStatusDisplay("PAUSED // press Start to resume", \'#ffaa00\');\n}\nfunction resetSimulation() {\n  Object.assign(sim, { running: false, status: \'ready\', pos: [0, 0], heading: 0, steerAngle: 0, velocity: 0, idx: 0, route: [], trail: [[0, 0]],\n    replans: 0, tight: false, tick: 0, simTime: 0, lastReplanT: 0, collided: false, inContact: false });\n  obstacles.forEach(o => { o.x = o.x0; o.y = o.y0; o.vy = o.vy0; });\n  Object.keys(keys).forEach(k => (keys[k] = false));\n  syncObstacleMeshes(); renderActiveTrajectory(); updateTrail(); updatePathUI(); updateTelemetry();\n  setDriveButtons(false); setEngine(-1);\n  updateStatusDisplay(driveMode === \'auto\' ? "AUTOPILOT READY // press Start" : "MANUAL READY // press W to go", \'#00f0ff\');\n}\nfunction setDriveMode(mode) {\n  if (sim.running) pauseDriving();\n  driveMode = mode;\n  document.getElementById(\'mode-auto\').classList.toggle(\'active\', mode === \'auto\');\n  document.getElementById(\'mode-manual\').classList.toggle(\'active\', mode === \'manual\');\n  document.getElementById(\'manual-dock\').classList.toggle(\'hidden\', mode !== \'manual\');\n  renderActiveTrajectory();\n  if (sim.pos[0] > 0.5 && sim.status !== \'done\') { toast(\'Mode changed. Press Start to continue from here.\'); }\n  updateStatusDisplay(mode === \'auto\' ? "AUTOPILOT READY // press Start" : "MANUAL READY // press W to go", \'#00f0ff\');\n}\n\n/* ===== 11. CAMERAS & HUD ===== */\nlet currentCameraMode = \'cockpit\';\nconst cockpitLook = { yaw: 0, pitch: 0 };\nfunction setCameraMode(mode) {\n  currentCameraMode = mode;\n  document.querySelectorAll(\'[id^="btn-cam-"]\').forEach(b => b.classList.remove(\'active\'));\n  document.getElementById(`btn-cam-${mode}`).classList.add(\'active\');\n  const cockpit = mode === \'cockpit\';\n  controls.enabled = mode === \'orbit\';\n  hudCanvas.style.display = cockpit ? \'block\' : \'none\';\n  document.getElementById(\'cockpit-overlay\').style.display = cockpit ? \'block\' : \'none\';\n  camera.up.set(mode === \'tactical\' ? 0 : 0, mode === \'tactical\' ? 0 : 1, mode === \'tactical\' ? -1 : 0);   // straight-down view needs a non-parallel up vector\n  if (mode === \'orbit\') { camera.position.set(sim.pos[0] - 8, 7, sim.pos[1] + 8); controls.target.set(sim.pos[0], .8, sim.pos[1]); }\n  if (cockpit) toast(\'🏎️ Cockpit view: drag to look around\');\n}\nfunction resetCameraOrientation() {\n  cockpitLook.yaw = cockpitLook.pitch = 0;\n  if (currentCameraMode === \'orbit\') { camera.position.set(sim.pos[0] - 8, 7, sim.pos[1] + 8); controls.target.set(sim.pos[0], .8, sim.pos[1]); }\n}\naddEventListener(\'mousemove\', e => {\n  if (currentCameraMode === \'cockpit\' && e.buttons === 1) {\n    cockpitLook.yaw = clamp(cockpitLook.yaw - e.movementX * .003, -.6, .6);\n    cockpitLook.pitch = clamp(cockpitLook.pitch - e.movementY * .003, -.3, .3);\n  }\n});\nfunction updateCameraRig() {\n  const cp = carGroup.position, h = sim.heading;\n  const fwd = new THREE.Vector3(Math.cos(h), 0, Math.sin(h)), right = new THREE.Vector3(-Math.sin(h), 0, Math.cos(h));\n  if (currentCameraMode === \'cockpit\') {\n    const eye = cp.clone().add(fwd.clone().multiplyScalar(driverSeatPos.x)).add(new THREE.Vector3(0, driverSeatPos.y, 0)).add(right.clone().multiplyScalar(driverSeatPos.z));\n    camera.position.copy(eye);\n    camera.lookAt(eye.clone().add(fwd.multiplyScalar(10)).add(right.multiplyScalar(cockpitLook.yaw * 8)).add(new THREE.Vector3(0, cockpitLook.pitch * 6, 0)));\n  } else if (currentCameraMode === \'chase\') {\n    camera.position.lerp(new THREE.Vector3(cp.x - Math.cos(h) * 6.2, 2.6, cp.z - Math.sin(h) * 6.2), .12);\n    camera.lookAt(cp.clone().add(new THREE.Vector3(Math.cos(h) * 4, 1, Math.sin(h) * 4)));\n  } else if (currentCameraMode === \'drone\') {\n    camera.position.lerp(new THREE.Vector3(cp.x - 5, 16, cp.z), .08);\n    camera.lookAt(cp.x + 8, 0, cp.z);\n  } else if (currentCameraMode === \'orbit\') {\n    controls.target.lerp(new THREE.Vector3(cp.x, .8, cp.z), .1);\n    controls.update();\n  } else {\n    camera.position.set(20, 36, 0);\n    camera.lookAt(20, 0, 0);\n  }\n}\nfunction renderCockpitHUD() {\n  if (currentCameraMode !== \'cockpit\') return;\n  const w = hudCanvas.width, h = hudCanvas.height;\n  hudCtx.clearRect(0, 0, w, h);\n  obstacles.forEach(o => {\n    const dist = Math.hypot(o.x - sim.pos[0], o.y - sim.pos[1]);\n    if (dist >= 32 || o.x <= sim.pos[0] - 1) return;\n    const sp = new THREE.Vector3(o.x, .8, o.y).project(camera);\n    if (sp.z >= 1 || Math.abs(sp.x) > 1.2) return;\n    const sx = (sp.x * .5 + .5) * w, sy = (-sp.y * .5 + .5) * h, b = Math.max(24, Math.min(100, 480 / dist)), hb = b / 2, bl = b * .25;\n    hudCtx.strokeStyle = dist < 8 ? \'#ff3366\' : \'#00f0ff\'; hudCtx.lineWidth = 2;\n    hudCtx.beginPath();\n    [[-1, -1], [1, -1], [-1, 1], [1, 1]].forEach(([cx, cy]) => {\n      hudCtx.moveTo(sx + cx * hb, sy + cy * (hb - bl)); hudCtx.lineTo(sx + cx * hb, sy + cy * hb); hudCtx.lineTo(sx + cx * (hb - bl), sy + cy * hb);\n    });\n    hudCtx.stroke();\n    hudCtx.fillStyle = \'#fff\'; hudCtx.font = \'10px Orbitron, monospace\'; hudCtx.textAlign = \'left\';\n    hudCtx.fillText(`${o.type.split(\' \')[1]} ${dist.toFixed(1)}m`, sx - hb, sy - hb - 6);\n  });\n  const cx = w * .5, cy = h * .68;\n  hudCtx.save();\n  hudCtx.shadowColor = \'#00f0ff\'; hudCtx.shadowBlur = 10;\n  hudCtx.strokeStyle = \'rgba(0,240,255,.35)\'; hudCtx.lineWidth = 1.5;\n  hudCtx.beginPath(); hudCtx.moveTo(cx - 70, cy); hudCtx.lineTo(cx - 20, cy); hudCtx.moveTo(cx + 20, cy); hudCtx.lineTo(cx + 70, cy); hudCtx.stroke();\n  hudCtx.fillStyle = \'#00f0ff\'; hudCtx.beginPath(); hudCtx.arc(cx, cy, 3, 0, Math.PI * 2); hudCtx.fill();\n  hudCtx.textAlign = \'center\'; hudCtx.font = \'bold 24px Orbitron, monospace\';\n  hudCtx.fillText(`${sim.velocity.toFixed(1)} m/s`, cx, cy + 32);\n  hudCtx.font = \'11px Orbitron, monospace\'; hudCtx.fillStyle = \'#00ff9d\';\n  hudCtx.fillText(driveMode === \'auto\' ? `⚡ AUTOPILOT // ${chosenPathName}` : \'🎮 MANUAL\', cx, cy + 48);\n  hudCtx.restore();\n}\n\n/* ===== 12. OBSTACLE PLACEMENT (click, never drag) ===== */\nconst raycaster = new THREE.Raycaster(), mouse = new THREE.Vector2();\nlet downPos = null;\naddEventListener(\'pointerdown\', e => { downPos = [e.clientX, e.clientY]; window.focus(); });\naddEventListener(\'click\', e => {\n  if (e.target.closest(\'#ui-layer, #manual-dock\')) return;\n  if (downPos && Math.hypot(e.clientX - downPos[0], e.clientY - downPos[1]) > 5) return;   // it was a look-around / orbit drag\n  mouse.set((e.clientX / innerWidth) * 2 - 1, -(e.clientY / innerHeight) * 2 + 1);\n  raycaster.setFromCamera(mouse, camera);\n  const hit = raycaster.intersectObject(roadMesh)[0];\n  if (!hit) return;\n  const px = Math.round(hit.point.x * 10) / 10, py = Math.round(hit.point.z * 10) / 10;\n  if (px < 1 || px > X_END + 2) return toast(\'Place obstacles on the roadway (1 m to 40 m).\');\n  if (Math.abs(py) > ROAD_HALF - .5) return toast(\'Place inside the road boundaries.\');\n  if (sim.running && px < sim.pos[0] + 3) return toast(\'⚠️ Too close to the vehicle. Drop it further ahead!\');\n  obstacles.push(makeObstacleData(selectedObstacleType, selectedObstacleMotion, px, py));\n  syncObstacleMeshes();\n  playChime(523, \'sine\', .15, .2);\n  spawnRipple(px, py);\n  toast(`Dropped ${selectedObstacleType} (${selectedObstacleMotion}) at ${px} m`);\n  triggerReplan();\n});\nfunction spawnRipple(x, z) {\n  const mat = new THREE.MeshBasicMaterial({ color: 0x00f0ff, side: THREE.DoubleSide, transparent: true, opacity: 1 });\n  const rip = new THREE.Mesh(new THREE.RingGeometry(.1, .25, 24), mat);\n  rip.rotation.x = -Math.PI / 2; rip.position.set(x, .05, z); scene.add(rip);\n  let s = 1;\n  const id = setInterval(() => {\n    s += .4; rip.scale.set(s, s, s); mat.opacity -= .08;\n    if (mat.opacity <= 0) { clearInterval(id); scene.remove(rip); disposeTree(rip); }\n  }, 30);\n}\nfunction undoLastObstacle() {\n  if (!obstacles.length) return;\n  obstacles.pop(); syncObstacleMeshes(); toast(\'↩️ Undid last obstacle.\'); triggerReplan();\n}\nfunction clearAllObstacles() {\n  obstacles.length = 0; syncObstacleMeshes(); toast(\'🗑️ Cleared all obstacles!\'); triggerReplan();\n}\nfunction randomizeObstacles() {\n  const kinds = Object.keys(OBJ_PROPS), motions = ["Static", "Crossing the road", "Moving ahead"];\n  const nom = paths[chosenPathName] || Object.values(paths)[0];\n  const start = sim.running ? [sim.pos[0], sim.pos[1]] : [0, 0];\n  const generate = count => {\n    obstacles.length = 0;\n    for (let i = 0, tries = 0; i < count && tries < 200; tries++) {\n      const x = Math.round((8 + Math.random() * 26) * 10) / 10, y = Math.round((-4.2 + Math.random() * 8.4) * 10) / 10;\n      if (obstacles.some(o => Math.hypot(o.x - x, o.y - y) < 3.5)) continue;\n      if (sim.running && x < sim.pos[0] + 6) continue;\n      obstacles.push(makeObstacleData(kinds[Math.floor(Math.random() * kinds.length)], motions[Math.floor(Math.random() * 3)], x, y)); i++;\n    }\n  };\n  // only hand out layouts the planner can actually solve (random walls of obstacles made the car wait forever)\n  search: for (let count = 5; count >= 0; count--) for (let attempt = 0; attempt < 15; attempt++) {\n    generate(count);\n    let ok = count === 0;\n    if (!ok) { try { ok = !!planRoute(nom[nom.length - 1], obstacles, start, V_NOM, nom); } catch (e) { ok = false; } }\n    if (ok) break search;\n  }\n  syncObstacleMeshes(); toast(`🎲 Spawned ${obstacles.length} random obstacles`); triggerReplan();\n}\n\n/* ===== 13. INPUT ===== */\nconst keys = {};\nconst DRIVE_KEYS = [\'w\', \'a\', \'s\', \'d\', \'arrowup\', \'arrowdown\', \'arrowleft\', \'arrowright\'];\naddEventListener(\'keydown\', e => {\n  const k = e.key.toLowerCase();\n  keys[k] = true;\n  if (DRIVE_KEYS.includes(k) || k === \' \') e.preventDefault();\n  if (e.repeat) return;\n  if (k === \' \') startDriving();\n  else if (k === \'r\') resetSimulation();\n  else if (\'12345\'.includes(k) && k.length === 1) setCameraMode([\'cockpit\', \'chase\', \'drone\', \'orbit\', \'tactical\'][+k - 1]);\n  else if (driveMode === \'manual\' && DRIVE_KEYS.includes(k) && !sim.running) startDriving();   // WASD only starts the car in manual mode\n});\naddEventListener(\'keyup\', e => (keys[e.key.toLowerCase()] = false));\naddEventListener(\'blur\', () => Object.keys(keys).forEach(k => (keys[k] = false)));\ndocument.querySelectorAll(\'.d-btn\').forEach(b => {\n  const k = b.dataset.key;\n  const set = on => { keys[k] = on; b.classList.toggle(\'pressed\', on); if (on && !sim.running && driveMode === \'manual\') startDriving(); };\n  b.addEventListener(\'pointerdown\', e => { e.preventDefault(); set(true); });\n  [\'pointerup\', \'pointerleave\', \'pointercancel\'].forEach(ev => b.addEventListener(ev, () => set(false)));\n});\naddEventListener(\'resize\', () => {\n  camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix();\n  renderer.setSize(innerWidth, innerHeight); resizeHud();\n});\n\n/* ===== 14. MAIN LOOP ===== */\nfunction animate() {\n  requestAnimationFrame(animate);\n  stepWorld(); updateCameraRig(); renderer.render(scene, camera); renderCockpitHUD();\n}\nupdatePathUI();\nrandomizeObstacles();\nsetCameraMode(\'cockpit\');\nupdateStatusDisplay(\'AUTOPILOT READY // press Start\', \'#00f0ff\');\nwindow.__autonav = { sim, get obstacles() { return obstacles; }, startDriving, resetSimulation, setDriveMode, simStep, keys };   // debug hook\nanimate();\n</script>\n</body>\n</html>\n'

# ---- Page Configuration & Dark Cyber Theme ----
st.set_page_config(
    page_title="AutoNav 3D // Smart Autonomous Path Planner",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-Tech Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&family=Inter:wght@300;400;600;700&display=swap');
    
    .stApp {
        background: #080c14;
        color: #f1f5f9;
        font-family: 'Inter', sans-serif;
    }
    
    /* Header Gradient */
    .hero-header {
        background: linear-gradient(135deg, rgba(0, 240, 255, 0.12), rgba(121, 40, 202, 0.15));
        border: 1px solid rgba(0, 240, 255, 0.25);
        border-radius: 16px;
        padding: 20px 26px;
        margin-bottom: 20px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        backdrop-filter: blur(12px);
    }
    
    .hero-title {
        font-family: 'Orbitron', monospace;
        font-size: 26px;
        font-weight: 800;
        letter-spacing: 2px;
        background: linear-gradient(90deg, #ffffff, #00f0ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    
    .hero-subtitle {
        font-size: 13px;
        color: #94a3b8;
        margin-top: 4px;
    }
    
    .tech-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 14px;
        border-radius: 20px;
        background: rgba(0, 240, 255, 0.15);
        border: 1px solid #00f0ff;
        color: #00f0ff;
        font-family: 'Orbitron', sans-serif;
        font-size: 11px;
        font-weight: 700;
    }
    
    /* Metrics Card */
    div[data-testid="stMetric"] {
        background: rgba(15, 23, 42, 0.65);
        border: 1px solid rgba(0, 240, 255, 0.2);
        border-radius: 12px;
        padding: 12px 16px;
    }
    
    div[data-testid="stMetricLabel"] {
        font-family: 'Orbitron', sans-serif;
        font-size: 11px;
        color: #94a3b8;
    }
    
    div[data-testid="stMetricValue"] {
        font-family: 'Orbitron', monospace;
        color: #00f0ff;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(15, 23, 42, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 8px 18px;
        color: #94a3b8;
        font-family: 'Orbitron', sans-serif;
        font-size: 13px;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background: rgba(0, 240, 255, 0.15) !important;
        border-color: #00f0ff !important;
        color: #ffffff !important;
        box-shadow: 0 0 15px rgba(0, 240, 255, 0.25);
    }
</style>
""", unsafe_allow_html=True)

# ---- Fixed Physical Model Settings ----
HALF_W, ROAD_HALF = 0.9, 6.0
VEH_LEN, X_END, RES, DT = 4.2, 40.0, 0.25, 0.2
HX0, HY0 = VEH_LEN / 2, HALF_W
HX, HY = HX0 + 0.5, HY0 + 0.6
VX, VY = HX0 + 0.3, HY0 + 0.4
V_NOM = 2.5
MAX_YAW = 1.5  # rad/s
OBJ = {
    "🧍 Pedestrian": (0.8, 0.8, 1.2),
    "🐄 Cow": (2.0, 0.9, 0.7),
    "🛒 Pushcart": (1.6, 1.0, 0.9),
    "🚗 Vehicle": (4.2, 1.8, 1.5),
    "🕳️ Pothole": (1.2, 1.2, 0.0),
}
MOTIONS = ["Static", "Crossing the road", "Moving ahead"]
LIM = ROAD_HALF - 0.8

XS = np.arange(0, X_END + RES, RES)
YS = np.arange(-ROAD_HALF, ROAD_HALF + RES, RES)
GX, GY = np.meshgrid(XS, YS, indexing="ij")


def make_obstacle(kind, motion, x, y):
    L, W, spd = OBJ[kind]
    vx = vy = 0.0
    if kind != "🕳️ Pothole":
        if motion == "Crossing the road":
            vy = -spd if y >= 0 else spd
        elif motion == "Moving ahead":
            vx = spd
    return dict(type=kind, x=x, y=y, x0=x, y0=y, vx=vx, vy=vy, vy0=vy, L=L, W=W)


def fold(y, lim=LIM):
    p = 4 * lim
    m = np.mod(np.asarray(y, float) + lim, p)
    return -lim + np.where(m <= 2 * lim, m, p - m)


def predict(o, t):
    t = np.asarray(t, float)
    x = np.minimum(o["x"] + o["vx"] * t, X_END + 8)
    y = fold(o["y"] + o["vy"] * t) if o["vy"] else np.full_like(t, o["y"])
    return x, y


def restore(obs):
    for o in obs:
        o["x"], o["y"], o["vy"] = o["x0"], o["y0"], o["vy0"]


def overlap(px, py, o, hx=HX0, hy=HY0, eps=0.05):
    return abs(px - o["x"]) < o["L"] / 2 + hx + eps and abs(py - o["y"]) < o["W"] / 2 + hy + eps


def move_obstacles(obs, ego):
    for o in obs:
        nx = min(o["x"] + o["vx"] * DT, X_END + 8)
        ny, nvy = o["y"] + o["vy"] * DT, o["vy"]
        if abs(ny) > LIM:
            ny, nvy = np.sign(ny) * (2 * LIM - abs(ny)), -o["vy"]
        old = (o["x"], o["y"])
        o["x"], o["y"] = nx, ny
        if overlap(ego[0], ego[1], o):
            o["x"], o["y"] = old
        else:
            o["vy"] = nvy


def densify(pts, step=RES):
    pts = np.asarray(pts, float)
    out = [pts[0]]
    for a, b in zip(pts[:-1], pts[1:]):
        n = max(int(np.ceil(np.linalg.norm(b - a) / step)), 1)
        for t in np.linspace(0, 1, n + 1)[1:]:
            out.append(a + (b - a) * t)
    return np.array(out)


def analyse_path(pts, obs):
    dense = densify([(0.0, 0.0)] + list(pts))
    t = np.arange(len(dense)) * RES / V_NOM
    bad = np.abs(dense[:, 1]) + HALF_W + 0.3 > ROAD_HALF
    for o in obs:
        ox, oy = predict(o, t)
        bad |= (np.abs(dense[:, 0] - ox) < o["L"] / 2 + HX) & (np.abs(dense[:, 1] - oy) < o["W"] / 2 + HY)
    return dict(blocked=bool(bad.any()), first_block_x=float(dense[bad.argmax(), 0]) if bad.any() else None)


def path_dev(chosen):
    cells = np.column_stack([GX.ravel(), GY.ravel()])
    dev = np.empty(len(cells))
    for s in range(0, len(cells), 4000):
        dev[s:s + 4000] = np.min(np.linalg.norm(cells[s:s + 4000, None, :] - chosen[None, :, :], axis=2), axis=1)
    return dev.reshape(GX.shape)


def plan_route(goal, obs, start, v, dev, steep=False, heading=0.0):
    """Time-aware A*. Lane changes are gentle (one cell sideways per two forward) unless steep is set.
    heading is the car's direction of travel; a replanned route leaves along it."""
    sx, sy = start
    t = np.maximum(XS - sx, 0) / max(v, 0.3) * 1.1
    blocked = np.abs(GY) + HALF_W + 0.3 > ROAD_HALF
    for o in obs:
        ox, oy = predict(o, t)
        unc = np.minimum(0.2 + 0.15 * t * (abs(o["vx"]) + abs(o["vy"])), 1.2)[:, None]
        blocked |= (np.abs(GX - ox[:, None]) < o["L"] / 2 + HX + unc) & (np.abs(GY - oy[:, None]) < o["W"] / 2 + HY + unc)
    si, sj = int(round(sx / RES)), int(np.argmin(np.abs(YS - sy)))
    if si >= len(XS) - 1:
        return densify([start, (X_END, sy)])
    blocked[max(si - 2, 0):si + 5, max(sj - 4, 0):sj + 5] = False
    free = np.argwhere(~blocked)
    if len(free) == 0:
        return None
    gi, gj = free[np.argmin(np.hypot(XS[free[:, 0]] - goal[0], YS[free[:, 1]] - goal[1]))]

    g, parent = {(si, sj): 0.0}, {}
    heap = [(RES * (gi - si), 0.0, si, sj)]
    done = False
    while heap:
        _, gc, i, j = heapq.heappop(heap)
        if gc > g.get((i, j), np.inf):
            continue
        if i >= gi and abs(j - gj) <= 1:
            done = True
            break
        for dj in (-1, 0, 1):
            di = 2 if dj and not steep else 1
            ni, nj = i + di, j + dj
            if ni >= len(XS) or not (0 <= nj < len(YS)) or blocked[ni, nj]:
                continue
            if di == 2 and (blocked[i + 1, j] or blocked[i + 1, nj]):
                continue
            ng = gc + np.hypot(di, dj) * RES * (1 + 0.8 * dev[ni, nj]) + 0.02 * abs(dj)
            if ng < g.get((ni, nj), np.inf):
                g[(ni, nj)], parent[(ni, nj)] = ng, (i, j)
                heapq.heappush(heap, (ng + RES * (gi - ni), ng, ni, nj))
    if not done:
        return None
    node, out = (i, j), []
    while True:
        out.append((XS[node[0]], YS[node[1]]))
        if node == (si, sj):
            break
        node = parent[node]
    raw = densify(np.array(out[::-1]))
    raw[0] = start
    # Round the grid's kinks with repeated moving averages, clamping each point into the free stretch of
    # its grid column so smoothing can slide along an obstacle's edge but never cut into it.
    ci = np.clip(np.round(raw[:, 0] / RES).astype(int), 0, len(XS) - 1)
    cj = np.clip(np.round((raw[:, 1] + ROAD_HALF) / RES).astype(int), 0, len(YS) - 1)
    lo, hi = raw[:, 1].copy(), raw[:, 1].copy()
    for k, (i, j) in enumerate(zip(ci, cj)):
        if blocked[i, j]:
            continue
        a = b = j
        while a > 0 and not blocked[i, a - 1]:
            a -= 1
        while b < len(YS) - 1 and not blocked[i, b + 1]:
            b += 1
        lo[k], hi[k] = YS[a], YS[b]
    lo[-2:], hi[-2:] = raw[-2:, 1], raw[-2:, 1]  # keep the goal pinned
    # An obstacle's edge is a sudden step in these limits, which would put a corner in the smoothed route.
    # Taper each step into a ramp no steeper than the route itself may be; the raw route still fits inside.
    slope = 1.0 if steep else 0.5
    dist = np.abs(raw[:, 0][:, None] - raw[:, 0][None, :]) * slope

    def tapered(start_y, lead):
        a, b = lo.copy(), hi.copy()
        a[lead], b[lead] = start_y[lead], start_y[lead]
        return np.max(a[None, :] - dist, axis=1), np.min(b[None, :] + dist, axis=1)

    # On a replan the first half metre carries on the way the car is already going, so the new route
    # doesn't kink where it joins the old one. Longer than that and it overshoots when the car was
    # mid lane change. If the route has to leave the other way it can't fit, and only the start is pinned.
    ahead = raw[:, 0] - start[0]
    pinned = np.arange(len(raw)) < 2
    if start[0] > 0.5:
        lo2, hi2 = tapered(start[1] + np.clip(np.tan(heading), -slope, slope) * ahead, ahead <= 0.5)
    if start[0] <= 0.5 or np.any(lo2 > hi2 + 1e-9):
        lo2, hi2 = tapered(raw[:, 1], pinned)
    lo, hi = lo2, hi2
    y = raw[:, 1].copy()
    if len(y) > 9:
        for _ in range(12):
            y = np.clip(np.convolve(np.pad(y, 4, mode="edge"), np.ones(9) / 9, mode="valid"), lo, hi)
    return np.column_stack([raw[:, 0], y])


def make_paths(n):
    ys = [0.0] if n == 1 else list(np.linspace(3.8, -3.8, n))
    return {f"Path {i + 1}": [(8.0, float(y)), (X_END, float(y))] for i, y in enumerate(ys)}


def random_obstacles(n):
    rng = np.random.default_rng()
    out = []
    for _ in range(500):
        if len(out) >= n:
            break
        x, y = float(rng.uniform(8, 34)), float(rng.uniform(-4.8, 4.8))
        if all(np.hypot(x - o["x"], y - o["y"]) > 3 for o in out):
            out.append(make_obstacle(str(rng.choice(list(OBJ))), str(rng.choice(MOTIONS)), round(x, 1), round(y, 1)))
    return out


def heading_at(route, idx):
    a, b = route[max(idx - 1, 0)], route[min(idx + 1, len(route) - 1)]
    return float(np.arctan2(b[1] - a[1], b[0] - a[0]))


def new_sim(choice, paths, obs, v):
    if not choice or choice not in paths:
        choice = list(paths.keys())[0]
    chosen = densify([(0.0, 0.0)] + paths[choice])
    sim = dict(choice=choice, goal=paths[choice][-1], dev=path_dev(chosen), route=chosen, idx=0, pos=(0.0, 0.0),
               heading=0.0, trail=[(0.0, 0.0)], running=True, status="driving", replans=0, tick=0, collided=False)
    replan(sim, obs, v, count=False)
    return sim


def replan(sim, obs, v, count=True):
    # gentle lane changes first; the old 45 degree steps only when nothing gentle fits
    # the direction the car is actually moving (its body heading lags behind on purpose)
    (ax, ay), (bx, by) = sim["trail"][-2:] if len(sim["trail"]) > 1 else ((0.0, 0.0), (1.0, 0.0))
    travel = float(np.arctan2(by - ay, bx - ax))
    route = plan_route(sim["goal"], obs, sim["pos"], v, sim["dev"], heading=travel)
    if route is None:
        route = plan_route(sim["goal"], obs, sim["pos"], v, sim["dev"], steep=True, heading=travel)
    if route is None:
        sim["status"] = "waiting"
        return False
    sim.update(route=route, idx=0, status="driving", replans=sim["replans"] + (1 if count else 0))
    return True


def route_ok(sim, obs, v):
    seg = sim["route"][sim["idx"] + 4: sim["idx"] + 64]
    if len(seg) == 0:
        return True
    t = (np.arange(len(seg)) + 4) * RES / max(v, 0.3)
    bad = np.zeros(len(seg), bool)
    for o in obs:
        ox, oy = predict(o, t)
        bad |= (np.abs(seg[:, 0] - ox) < o["L"] / 2 + VX) & (np.abs(seg[:, 1] - oy) < o["W"] / 2 + VY)
    return not bad.any()


def step_world(sim, obs, inc):
    v = inc * RES / DT
    sim["tick"] += 1
    move_obstacles(obs, sim["pos"])
    if sim["status"] == "waiting":
        if sim["tick"] % 3 == 0:
            replan(sim, obs, v)
    elif not route_ok(sim, obs, v):
        replan(sim, obs, v)
    if sim["status"] == "driving":
        r = sim["route"]
        nidx = min(sim["idx"] + inc, len(r) - 1)
        cand = r[nidx]
        if not any(overlap(cand[0], cand[1], o, eps=0.1) for o in obs):
            sim["idx"], sim["pos"] = nidx, tuple(cand)
            # turn the body toward the route, but no faster than MAX_YAW so a replan never snaps it round
            turn = np.clip(heading_at(r, nidx) - sim["heading"], -MAX_YAW * DT, MAX_YAW * DT)
            sim["heading"] = float(sim["heading"] + turn)
            sim["trail"].append(sim["pos"])
        if sim["idx"] >= len(r) - 1:
            sim["running"], sim["status"] = False, "done"
    if any(overlap(sim["pos"][0], sim["pos"][1], o, eps=-0.01) for o in obs):
        sim["collided"] = True


def poly_rect(cx, cy, length, width, heading=0.0):
    c, s = np.cos(heading), np.sin(heading)
    k = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]]) * [length / 2, width / 2]
    return cx + k[:, 0] * c - k[:, 1] * s, cy + k[:, 0] * s + k[:, 1] * c


def make_figure(obs, paths, status, sim=None):
    fig = go.Figure()
    # Road Asphalt
    fig.add_trace(go.Scatter(x=[-6, X_END + 3, X_END + 3, -6, -6],
                             y=[-ROAD_HALF, -ROAD_HALF, ROAD_HALF, ROAD_HALF, -ROAD_HALF],
                             fill="toself", fillcolor="#181c24", line=dict(color="#00f0ff", width=2),
                             hoverinfo="skip", showlegend=False))
    # Lane divider dash
    for y_lane in [-3.0, 0.0, 3.0]:
        fig.add_trace(go.Scatter(x=[-5, X_END + 2], y=[y_lane, y_lane], mode="lines",
                                 line=dict(color="#334155" if y_lane != 0 else "#facc15", dash="dash", width=1.5),
                                 hoverinfo="skip", showlegend=False))

    for name, pts in paths.items():
        P = np.array([(0.0, 0.0)] + list(pts))
        blocked = status[name]["blocked"]
        chosen = sim is not None and sim["choice"] == name
        fig.add_trace(go.Scatter(
            x=P[:, 0], y=P[:, 1], mode="lines+text", hoverinfo="skip",
            line=dict(color="#ff3366" if blocked else "#00ff9d", dash="dash", width=5 if chosen else 2.5),
            text=[""] * (len(P) - 1) + [f"🏁 {name}"], textposition="middle right",
            textfont=dict(color="white", family="Orbitron"), showlegend=False))

    for o in obs:
        moving = abs(o["vx"]) + abs(o["vy"]) > 0
        bx, by = poly_rect(o["x"], o["y"], o["L"], o["W"])
        fig.add_trace(go.Scatter(x=bx, y=by, fill="toself", fillcolor="#f59e0b" if moving else "#ef4444",
                                 line=dict(color="white", width=1), hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=[o["x"]], y=[o["y"] + o["W"] / 2 + 0.6], mode="text", text=[o["type"].split()[0]],
                                 textfont=dict(size=18), hoverinfo="skip", showlegend=False))
        if moving:
            d = np.array([o["vx"], o["vy"]]) / np.hypot(o["vx"], o["vy"]) * 2.2
            fig.add_annotation(x=o["x"] + d[0], y=o["y"] + d[1], ax=o["x"], ay=o["y"], xref="x", yref="y",
                               axref="x", ayref="y", showarrow=True, arrowhead=3, arrowwidth=2.5, arrowcolor="#00f0ff")
    px, py, ph = 0.0, 0.0, 0.0
    if sim is not None:
        T = np.array(sim["trail"])
        fig.add_trace(go.Scatter(x=T[:, 0], y=T[:, 1], mode="lines", hoverinfo="skip", showlegend=False,
                                 line=dict(color="rgba(0, 240, 255, 0.5)", width=3)))
        R = sim["route"][sim["idx"]:]
        if sim["status"] == "driving" and len(R) > 1:
            fig.add_trace(go.Scatter(x=R[:, 0], y=R[:, 1], mode="lines", hoverinfo="skip", showlegend=False,
                                     line=dict(color="#00ff9d", width=5)))
        (px, py), ph = sim["pos"], sim["heading"]

    vx, vy = poly_rect(px, py, VEH_LEN, 2 * HALF_W, ph)
    fig.add_trace(go.Scatter(x=vx, y=vy, fill="toself", fillcolor="#2563eb", line=dict(color="#00f0ff", width=2),
                             hoverinfo="skip", showlegend=False))

    gx, gy = np.meshgrid(np.arange(5, 36.5, 0.5), np.arange(-5.5, 5.6, 0.5))
    fig.add_trace(go.Scatter(x=gx.ravel(), y=gy.ravel(), mode="markers", showlegend=False,
                             marker=dict(size=13, color="rgba(255,255,255,0.01)"), hoverinfo="none"))

    fig.update_layout(height=420, margin=dict(l=5, r=5, t=10, b=5), showlegend=False,
                      plot_bgcolor="#0c121e", paper_bgcolor="#0c121e",
                      xaxis=dict(visible=False, range=[-6, X_END + 6]),
                      yaxis=dict(visible=False, range=[-ROAD_HALF - 0.5, ROAD_HALF + 0.5], scaleanchor="x"))
    return fig


# ---- Header Banner ----
st.markdown("""
<div class="hero-header">
    <div>
        <h1 class="hero-title">⚡ AUTONAV 3D // SMART PATH PLANNER</h1>
        <div class="hero-subtitle">Real-Time Autonomous Driving • Time-Aware A* Obstacle Prediction • Interactive Cockpit HUD</div>
    </div>
    <div style="display: flex; gap: 10px; align-items: center;">
        <span class="tech-pill">60 FPS WebGL</span>
        <span class="tech-pill" style="border-color: #00ff9d; color: #00ff9d;">Neural A*</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Tabs: 3D Cockpit Simulation vs 2D Tactical Analysis
tab_3d, tab_2d = st.tabs(["🎮 3D COCKPIT SIMULATOR", "📊 2D TACTICAL LIDAR RADAR"])

with tab_3d:
    html_3d_file = os.path.join(os.path.dirname(__file__), "vehicle_path_3d.html")
    if os.path.exists(html_3d_file):
        with open(html_3d_file, "r", encoding="utf-8") as f:
            html_content = f.read()
    else:
        html_content = EMBEDDED_3D_HTML

    st.markdown("""
<div style="background: rgba(0, 240, 255, 0.1); border: 1px solid #00f0ff; border-radius: 10px; padding: 10px 16px; margin-bottom: 10px; display: flex; align-items: center; justify-content: space-between;">
    <div style="display: flex; align-items: center; gap: 8px;">
        <span style="font-size: 18px;">🏎️</span>
        <span style="font-size: 13px; font-weight: 600; color: #f1f5f9;">
            Click <strong>[▶ Start driving]</strong> to drive the car forward towards the goal within 10 seconds with dynamic obstacle avoidance!
        </span>
    </div>
    <span style="font-family: monospace; font-size: 11px; color: #00f0ff; background: rgba(0,240,255,0.15); padding: 4px 10px; border-radius: 6px;">[Spacebar] Drive / Pause</span>
</div>
""", unsafe_allow_html=True)
    if NEW_EMBED_API:
        st.iframe(html_content, height=740)
    else:
        components.html(html_content, height=740, scrolling=False)

    # GitHub's "> [!TIP]" alert syntax isn't Streamlit markdown and showed up literally
    st.info("""**Cockpit Controls**
- Switch views on the top bar: **🏎️ Cockpit (FPV)**, **🎥 Chase Cam**, **🚁 Drone 3D**, **🌐 Orbit 3D**, **🗺️ Tactical 2D**.
- **Click anywhere on the 3D road** to drop pedestrians, cows, carts, cars, or potholes in real time while the car is driving!
- Toggle environments: **🌃 Cyberpunk Night**, **🌅 Sunset**, **☀️ Daylight**.
- Press **[Space]** to start/pause autopilot, **[R]** to reset, or number keys **[1-5]** to switch cameras instantly.""", icon="💡")

with tab_2d:
    ss = st.session_state
    ss.setdefault("obs", [])
    ss.setdefault("ck", 0)
    ss.setdefault("sim", None)
    ss.setdefault("speed", "Normal")

    c1, c2 = st.columns(2)
    n_paths = c1.slider("Number of candidate paths", 1, 5, 3)
    n_obs = c2.slider("Number of obstacles to place", 0, 10, 4)
    ss.obs = ss.obs[:n_obs]
    paths = make_paths(n_paths)

    mode = st.radio("Step", ["1️⃣ Build the road", "2️⃣ Drive"], horizontal=True, label_visibility="collapsed")
    p1, p2 = st.columns([1, 2])
    kind = p1.selectbox("What do you want to place?", list(OBJ))
    motion = p2.radio("How does it move?", MOTIONS, horizontal=True)

    def handle_click(point, driving_x=0.0):
        if len(ss.obs) >= n_obs:
            return "All obstacles are placed. Raise the obstacle slider to add more."
        if point["x"] < driving_x + 5:
            return "Too close to the vehicle. Drop it further ahead."
        ss.obs.append(make_obstacle(kind, motion, round(float(point["x"]), 1), round(float(point["y"]), 1)))
        return ""

    if mode.startswith("1"):
        ss.sim = None
        restore(ss.obs)
        k1, k2, k3 = st.columns(3)
        if k1.button("🎲 Random obstacles"):
            ss.obs = random_obstacles(n_obs)
        if k2.button("↩️ Undo last") and ss.obs:
            ss.obs.pop()
        if k3.button("🗑️ Clear all"):
            ss.obs = []
        if n_obs == 0:
            st.info("Move the obstacle slider up to start placing obstacles.")
        elif len(ss.obs) < n_obs:
            st.info(f"👆 Click anywhere on the road to place a {kind}  ({len(ss.obs)}/{n_obs} placed)")
        else:
            st.success(f"All {n_obs} obstacles placed. Switch to **2️⃣ Drive** above.")
        status = {n: analyse_path(p, ss.obs) for n, p in paths.items()}
        event = st.plotly_chart(make_figure(ss.obs, paths, status), key=f"road_{ss.ck}",
                                on_select="rerun", selection_mode="points", **CHART_WIDTH)
        clicked = event["selection"]["points"] if event else []
        if clicked:
            msg = handle_click(clicked[0])
            ss.ck += 1
            if msg:
                st.toast(msg)
            st.rerun()

    else:
        status = {n: analyse_path(p, ss.obs) for n, p in paths.items()}
        choice = st.radio("🚗 Which path should I take?", list(paths), index=0, horizontal=True,
                          format_func=lambda n: f"{n}  {'⛔ blocked' if status[n]['blocked'] else '✅ clear'}")
        if ss.sim and ss.sim["choice"] != choice:
            ss.sim = None

        b1, b2, b3, b4 = st.columns([1, 1, 1, 2])
        b4.select_slider("Speed", ["Slow", "Normal", "Fast"], key="speed")
        inc = {"Slow": 1, "Normal": 2, "Fast": 4}[ss.speed]
        v = inc * RES / DT
        if b1.button("▶ Start driving", disabled=choice is None):
            restore(ss.obs)
            ss.sim = new_sim(choice, paths, ss.obs, v)
        if b2.button("↩️ Undo last obstacle") and ss.obs:
            ss.obs.pop()
            if ss.sim and ss.sim["running"]:
                replan(ss.sim, ss.obs, v)
        if b3.button("⏹ Reset"):
            restore(ss.obs)
            ss.sim = None

        running = bool(ss.sim and ss.sim["running"])

        @st.fragment(run_every=DT if running else None)
        def live():
            sim = ss.sim
            if sim and sim["running"]:
                step_world(sim, ss.obs, inc)
            stat = {n: analyse_path(p, ss.obs) for n, p in paths.items()}

            if sim is None:
                st.info("The vehicle is waiting. Pick a path and press **▶ Start driving**.")
            elif sim["status"] == "driving":
                st.success(f"Driving {sim['choice']}. Dynamic obstacle avoidance active.")
            elif sim["status"] == "waiting":
                st.warning("⏸ No safe gap right now. Vehicle is waiting for clearance.")
            elif sim["collided"]:
                st.error("💥 A collision happened.")
            else:
                st.success(f"🏁 Arrived safely with zero collisions. Re-planned {sim['replans']} time(s).")

            event = st.plotly_chart(make_figure(ss.obs, paths, stat, sim), key=f"live_{ss.ck}",
                                    on_select="rerun", selection_mode="points", **CHART_WIDTH)
            clicked = event["selection"]["points"] if event else []
            if clicked:
                msg = handle_click(clicked[0], sim["pos"][0] if sim else 0.0)
                ss.ck += 1
                if msg:
                    st.toast(msg)
                elif sim and sim["running"]:
                    replan(sim, ss.obs, v)
                try:
                    st.rerun(scope="fragment")
                except StreamlitAPIException:
                    st.rerun()
            if sim and not sim["running"] and running:
                st.rerun()

        live()
