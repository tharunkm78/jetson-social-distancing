const canvas = document.getElementById('overlay-canvas');
const ctx = canvas.getContext('2d');
const peopleCountEl = document.getElementById('people-count');
const violationCountEl = document.getElementById('violation-count');
const statusTextEl = document.getElementById('status-text');
const deviceInfoEl = document.getElementById('device-info');
const predictionListEl = document.getElementById('prediction-list');
const logStreamEl = document.getElementById('log-stream');
const desyncOverlayEl = document.getElementById('desync-overlay');
const shutterEl = document.getElementById('eagle-shutter');

let isDetectionActive = true;
let activeViolationPairs = new Set(); // To track logged violations

// Shutter/Blink effect logic
function triggerBlink() {
    if (!shutterEl) return;
    shutterEl.classList.add('blink');
    setTimeout(() => {
        shutterEl.classList.remove('blink');
    }, 400);
}

// Random blinks
setInterval(() => {
    if (Math.random() > 0.92 && isDetectionActive) triggerBlink();
}, 1000);

async function updateStatus() {
    if (!isDetectionActive) {
        drawSuspendedOverlay();
        return;
    }

    try {
        const response = await fetch('/api/status');
        const data = await response.json();

        if (data.error) return;

        drawOverlays(data);
        updateDashboard(data);
    } catch (err) {
        console.error("API Error:", err);
    }
}

function drawOverlays(data) {
    const width = data.width || 640;
    const height = data.height || 480;
    
    if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
    }

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    const skeleton = [[5,6],[5,11],[6,12],[11,12],[5,7],[7,9],[6,8],[8,10],[11,13],[13,15],[12,14],[14,16]];

    if (!data.midpoints) return;

    data.midpoints.forEach((mid, idx) => {
        const kps = data.raw_detections ? data.raw_detections[idx] : null;
        const isViolating = data.violations.some(v => v[0] === idx || v[1] === idx);
        const color = isViolating ? '#8b0000' : '#c5a059'; // Warm Origins Gold
        
        // Draw Radius
        ctx.beginPath();
        ctx.strokeStyle = color;
        ctx.lineWidth = 1;
        ctx.globalAlpha = 0.15;
        const radius = data.radii ? data.radii[idx] : (data.threshold / 2 || 75);
        ctx.arc(mid[0], mid[1], radius, 0, Math.PI * 2);
        ctx.stroke();
        ctx.globalAlpha = 1.0;

        // Draw Skeleton
        if (kps) {
            ctx.beginPath();
            ctx.strokeStyle = color;
            ctx.lineWidth = 2;
            skeleton.forEach(([s, e]) => {
                const start = kps[s]; const end = kps[e];
                if (start && end && start.confidence > 0.1 && end.confidence > 0.1) {
                    ctx.moveTo(start.x, start.y);
                    ctx.lineTo(end.x, end.y);
                }
            });
            ctx.stroke();

            // Dots
            ctx.fillStyle = isViolating ? '#ff4d4d' : '#ffd700';
            kps.forEach(kp => {
                if (kp && kp.confidence > 0.1) {
                    ctx.beginPath();
                    ctx.arc(kp.x, kp.y, 3, 0, Math.PI * 2);
                    ctx.fill();
                }
            });
        }

        // Label
        ctx.fillStyle = color;
        ctx.font = "12px Cinzel";
        const personId = data.ids ? data.ids[idx] : idx;
        ctx.fillText(`TGT ${personId}`, mid[0] + 10, mid[1] - 10);
    });

    // Violation connections
    data.violations.forEach(v => {
        const p1 = data.midpoints[v[0]]; const p2 = data.midpoints[v[1]];
        if (p1 && p2) {
            ctx.beginPath();
            ctx.strokeStyle = '#8b0000';
            ctx.setLineDash([5, 5]);
            ctx.moveTo(p1[0], p1[1]);
            ctx.lineTo(p2[0], p2[1]);
            ctx.stroke();
            ctx.setLineDash([]);
        }
    });
}

function updateDashboard(data) {
    peopleCountEl.innerText = data.num_people;
    violationCountEl.innerText = data.violations.length;
    
    if (data.device && deviceInfoEl) {
        deviceInfoEl.innerText = data.device.toUpperCase();
    }

    const currentViolationIds = new Set();
    data.violations.forEach(v => {
        const id1 = data.ids ? data.ids[v[0]] : v[0];
        const id2 = data.ids ? data.ids[v[1]] : v[1];
        const pairId = `${id1}-${id2}`;
        currentViolationIds.add(pairId);
        
        if (!activeViolationPairs.has(pairId)) {
            addLog(`>> ALERT: Tactical Synchronization Error [TGT ${id1} | TGT ${id2}]`);
            activeViolationPairs.add(pairId);
        }
    });

    // Clean up inactive violations
    activeViolationPairs.forEach(id => {
        if (!currentViolationIds.has(id)) activeViolationPairs.delete(id);
    });

    if (data.violations.length > 0) {
        violationCountEl.parentElement.classList.add('active');
        desyncOverlayEl.classList.add('active');
        statusTextEl.innerText = 'DESYNCHRONIZED';
        statusTextEl.style.color = '#8b0000';
    } else {
        violationCountEl.parentElement.classList.remove('active');
        desyncOverlayEl.classList.remove('active');
        statusTextEl.innerText = 'SYNCHRONIZED';
        statusTextEl.style.color = '#ffffff';
    }

    predictionListEl.innerHTML = '';
    if (data.ids) {
        data.ids.forEach((id, i) => {
            const action = data.actions ? data.actions[i] : "UNKNOWN";
            const li = document.createElement('li');
            li.className = 'prediction-item';
            li.innerHTML = `<span>TGT ${id}</span> <span>${action.toUpperCase()}</span>`;
            predictionListEl.appendChild(li);
        });
    }
}

function addLog(msg) {
    const p = document.createElement('p');
    p.innerText = `> ${new Date().toLocaleTimeString()} | ${msg}`;
    logStreamEl.appendChild(p);
    logStreamEl.parentElement.scrollTop = logStreamEl.parentElement.scrollHeight;
    if (logStreamEl.children.length > 30) logStreamEl.firstChild.remove();
}

function drawSuspendedOverlay() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "rgba(0, 0, 0, 0.8)";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.font = "20px Cinzel";
    ctx.fillStyle = "#ffffff";
    ctx.textAlign = "center";
    ctx.fillText("VISION SUSPENDED", canvas.width / 2, canvas.height / 2);
}

// Toggle Protocol
document.getElementById('toggle-detection').addEventListener('click', (e) => {
    isDetectionActive = !isDetectionActive;
    e.target.innerText = isDetectionActive ? 'PROTOCOL: ACTIVE' : 'PROTOCOL: SUSPENDED';
    addLog(`Eagle Vision Protocol ${isDetectionActive ? 'Synchronized' : 'Desynchronized'}`);
    if (!isDetectionActive) {
        statusTextEl.innerText = 'OFFLINE';
        statusTextEl.style.color = '#444';
        desyncOverlayEl.classList.remove('active');
        violationCountEl.parentElement.classList.remove('active');
    } else {
        triggerBlink();
    }
});

addLog("Initializing Eagle Vision Interface...");
addLog("Establishing Animus Connection...");
addLog("Eagle Eye Sync Complete.");

setInterval(updateStatus, 100);
