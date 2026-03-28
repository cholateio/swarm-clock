// app/page.js
'use client';

import { useEffect, useRef } from 'react';

// ==========================================
// WebGL 著色器字串 (Shaders)
// ==========================================
// 頂點著色器：負責全螢幕 Quad 的繪製，用於 GPGPU 的 Ping-Pong 運算
const PHYSICS_VERT = `#version 300 es
in vec2 aPosition;
void main() { gl_Position = vec4(aPosition, 0.0, 1.0); }
`;

// 片段著色器：預計算 FBM (Fractional Brownian Motion) 噪聲紋理
// 這裡的參數通常不需要調整，它只是在初始階段產生一張靜態的 256x256 噪聲圖供後續讀取，以節省每一幀的 GPU 算力。
const NOISE_GEN_FRAG = `#version 300 es
precision highp float;
out vec4 fragColor;
float hash(vec2 p) {
  vec3 p3 = fract(vec3(p.xyx) * 0.13);
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}
float noise(vec2 x) {
  vec2 i = floor(x); vec2 f = fract(x);
  float a = hash(i); float b = hash(i + vec2(1.0, 0.0));
  float c = hash(i + vec2(0.0, 1.0)); float d = hash(i + vec2(1.0, 1.0));
  vec2 u = f * f * (3.0 - 2.0 * f);
  return mix(a, b, u.x) + (c - a) * u.y * (1.0 - u.x) + (d - b) * u.x * u.y;
}
float fbm(vec2 p) {
  float value = 0.0; float amplitude = 0.5;
  // 迴圈次數越多決定噪聲的細節豐富度(Octaves)
  for (int i = 0; i < 4; i++) { value += amplitude * noise(p); p *= 2.0; amplitude *= 0.5; }
  return value;
}
void main() {
  vec2 uv = gl_FragCoord.xy / 256.0;
  vec2 p = uv * 6.0; const float e = 0.1;
  // 計算偏導數 (Gradient) 生成 Curl Noise 所需的向量場
  float a = (fbm(p + vec2(0.0, e)) - fbm(p - vec2(0.0, e))) / (2.0 * e);
  float b = (fbm(p + vec2(e, 0.0)) - fbm(p - vec2(e, 0.0))) / (2.0 * e);
  fragColor = vec4(a, -b, 0.0, 1.0);
}
`;

// 片段著色器：核心物理引擎 (GPGPU Physics Pass)
// 負責讀取上一幀的狀態，結合音訊 Uniforms 計算出新的一幀的座標與速度向量。
const PHYSICS_FRAG = `#version 300 es
precision highp float;

uniform sampler2D uState;   // 儲存上一幀的 [pos.x, pos.y, vel.x, vel.y]
uniform sampler2D uOrigin;  // 儲存初始的 [origin.x, origin.y, size, opacity]
uniform sampler2D uNoise;   // 預計算的噪聲紋理
uniform vec2 uResolution;   // 畫布解析度向量
uniform float uBass;        // 平滑化後的重低音純量 (0.0 ~ 1.0)
uniform float uFreqs[64];   // 64 頻段的頻譜陣列
uniform float uTime;        // 系統時間純量 (秒)

out vec4 outState;          // 輸出計算結果至下一幀的 Texture

// 靜態雜湊函數：根據粒子的二維座標生成一個 0.0~1.0 的偽隨機常數 pRand
float hash(vec2 p) {
  vec3 p3 = fract(vec3(p.xyx) * 0.13);
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}

void main() {
  ivec2 coord = ivec2(gl_FragCoord.xy);
  vec4 state = texelFetch(uState, coord, 0);
  vec4 origin = texelFetch(uOrigin, coord, 0);

  vec2 pos = state.xy;          // 當前位置向量
  vec2 vel = state.zw;          // 當前速度向量
  vec2 originPos = origin.xy;   // 粒子在時鐘刻度上的初始錨點向量

  if (originPos.x < -9000.0) {
    outState = state;
    return;
  }

  float pRand = hash(vec2(float(coord.x), float(coord.y))); // 套用前面的 hash 來讓粒子產生隨機性
  vec2 center = uResolution * 0.5;
  float clockSize = min(uResolution.x, uResolution.y);

  // ==========================================
  // 1. 頻譜波浪形變計算
  // ==========================================
  vec2 originToCenter = originPos - center;
  float angle = atan(originToCenter.y, originToCenter.x);
  
  // 🌟【參數調整區 1.1：頻譜自轉角速度】
  // 0.15 為角速度純量(rad/s)。從粒子極座標角度中減去此純量，使整體波浪目標點順時針旋轉。
  // 範圍建議: 0.0 (靜止) ~ 0.5 (快速旋轉)
  float rotatedAngle = angle - uTime * 0.15;

  // 將角度正規化為 0.0 ~ 1.0，並鏡像映射至 128 個採樣點以對應 64 個頻段 (這邊將 64 頻段對到 180 度來實現時鐘對秤)
  float normalizedAngle = fract((rotatedAngle + 3.14159265359) / 6.28318530718);
  float floatIndex = normalizedAngle * 128.0;
  if (floatIndex >= 64.0) floatIndex = 127.0 - floatIndex;
  floatIndex = clamp(floatIndex, 0.0, 63.0);
  
  // 線性插值 (mix)：平滑相鄰頻段的離散數值，避免波浪產生階梯狀方塊感
  int idx1 = int(floatIndex);
  int idx2 = min(idx1 + 1, 63);
  float blend = fract(floatIndex);
  float freqVal = mix(uFreqs[idx1], uFreqs[idx2], blend);

  // 🌟【參數調整區 1.2：頻譜指數曲線】
  // 蟲子對聲音起伏的反應曲線。數值越高，蟲子對背景小雜音越「無感」，但對大聲的節拍會反應更劇烈。
  // 建議: 1.0 (線性，一點雜音就跟著起伏) ~ 3.0 (高冷，只有重拍才會讓波浪凸起)
  freqVal = pow(freqVal, 2.5);

  vec2 pushDirOrigin = normalize(originToCenter);
  
  // 🌟【參數調整區 1.3：徑向最大位移乘數】
  // 當被聲音激怒時，蟲群最多能往外圍「衝出/攀爬」多遠。0.16 代表時鐘大小的 16%。
  // 範圍建議: 0.05 (微幅起伏) ~ 0.30 (誇張延伸)
  float waveAmplitude = 0.16; 
  vec2 targetPos = originPos + pushDirOrigin * (freqVal * clockSize * waveAmplitude);

  // ==========================================
  // 2. 蟲群分化與微擾向量運算 (讓極少數亂飛，大部分遵循輪廓)
  // ==========================================
  vec2 noiseUV = pos * 0.005 + uTime * 0.2 + pRand * 5.0;
  vec2 curl = texture(uNoise, noiseUV).xy; // 取得 Curl Noise 向量

  // 🌟 參數調整區 2.1：散開指數】
  // 當 freqVal < 0.1 (波浪很平) 時，dispersion = 0.0 (完全不散開)
  // 當 freqVal > 0.6 (波浪很高) 時，dispersion = 1.0 (最大程度散開)
  float dispersion = smoothstep(0.1, 0.6, freqVal);

  // 🌟【參數調整區 2.2：游離權重映射閾值】
  // 安靜時：閾值為 0.95 (只有 5% 的蟲子能飛)
  // 高潮時：閾值降為 0.40 (高達 60% 的蟲子會脫離輪廓一起飛舞)
  float dynamicFreedomThreshold = mix(0.95, 0.40, dispersion);
  float freedom = smoothstep(dynamicFreedomThreshold - 0.1, dynamicFreedomThreshold, pRand);

  // 🌟【參數調整區 2.3：重低音動態閘門】
  // 建立與 uBass 綁定的開關純量。低於 0.02 (幾乎無聲) 時輸出 0.0。
  // 確保在音樂暫停或純靜音時，切斷後續所有的擾動向量運算。
  // 範圍建議: 若對微小聲音也要有反應，可降為 smoothstep(0.005, 0.05, uBass)
  float audioPresence = smoothstep(0.02, 0.08, uBass);
  
  // 🌟【參數調整區 2.4：擾動加速度純量】
  // 計算疊加至速度向量 (vel) 的純量乘數，僅對 freedom > 0.0 的游離粒子(允許亂飛蟲子)生效。
  // 0.002 決定了 Curl Noise 向量的力道大小 (單位: 像素/幀² 的比例)。
  // 範圍建議: 0.001 (微幅蠕動) ~ 0.01 (劇烈抖動)
  float swarmChaos = 0.002 * audioPresence * dispersion; 
  vel += curl * clockSize * swarmChaos * freedom;

  // 🌟【參數調整區 2.5：速度阻尼係數 (摩擦力)】
  // 每一幀將當前速度向量乘以 0.85 進行衰減。
  // 範圍建議: 0.70 (高阻尼，運動生硬但精準) ~ 0.95 (低阻尼，產生滑行與冰面感)
  vel *= 0.85;
  pos += vel;

  // 🌟【參數調整區 2.6：目標座標離散半徑】
  // 改變游離粒子彈簧拉扯的最終錨點。0.01 代表目標點會在 targetPos 周圍 1% 的半徑內飄動。
  // 範圍建議: 0.01 (緊貼輪廓) ~ 0.05 (形成極厚的雲霧環繞層)
  float scatterRadius = clockSize * 0.01 * audioPresence* dispersion; 
  vec2 finalTarget = targetPos + curl * scatterRadius * freedom;

  // ==========================================
  // 3. 彈簧物理插值 (Lerp)
  // ==========================================
  // 🌟【參數調整區 3.1：目標回歸線性插值權重】
  // 決定當前座標 (pos) 更新為目標座標 (finalTarget) 的插值百分比。
  // 核心粒子 (freedom=0.0): 權重 0.8 -> 強制向目標收斂 80% 的距離 (死死貼著波浪)。
  // 游離粒子 (freedom=1.0): 權重 0.05 -> 僅向目標收斂 5% 的距離 (保留大量歷史速度軌跡產生拖尾)。
  // 範圍建議: 核心 (0.5 ~ 0.95)，游離 (0.01 ~ 0.1)
  float springStrength = mix(0.8, 0.05, freedom);
  pos = mix(pos, finalTarget, springStrength);

  outState = vec4(pos, vel);
}
`;

// ==========================================
// 渲染著色器：蟲群視覺化 (Render Pass)
// ==========================================
const RENDER_VERT = `#version 300 es
precision highp float;

uniform sampler2D uState;      // 讀取物理引擎算好的最新狀態 (位置、速度)
uniform sampler2D uOrigin;     // 讀取蟲子的天生屬性 (初始位置、大小、透明度)
uniform vec2 uResolution;      // 畫布解析度
uniform float uTextureSize;    // 狀態紋理的邊長 (用於計算 1D 陣列索引轉 2D UV 座標)
uniform float uTime;           // 系統時間
uniform float uPointScale;     // 裝置像素比 (DPR)，用於 Retina 螢幕的高清縮放

out float vOpacity;            // 傳遞給片段著色器的變數：蟲子基礎透明度
out float vSpeed;              // 傳遞給片段著色器的變數：蟲子當下的動能係數

void main() {
  // 利用當前頂點的 ID (gl_VertexID)，推算出這隻蟲子在紋理矩陣 (Texture) 中的 2D 座標
  int index = gl_VertexID;
  ivec2 texCoord = ivec2(index % int(uTextureSize), index / int(uTextureSize));

  // 從 GPU 記憶體中精準抽取該蟲子的向量資料
  vec4 state = texelFetch(uState, texCoord, 0);
  vec4 origin = texelFetch(uOrigin, texCoord, 0);

  if (origin.x < -9000.0) {
    gl_Position = vec4(-2.0, -2.0, 0.0, 1.0);
    gl_PointSize = 0.0;
    return;
  }

  vec2 pos = state.xy;        // 蟲子的絕對座標 (Pixel)
  vec2 vel = state.zw;        // 蟲子的速度向量
  float size = origin.z;      // 蟲子的天生體型
  float opacity = origin.w;   // 蟲子的天生透明度

  // 🌟【參數調整區：動能發光閾值映射】
  // 計算速度向量的長度 (純量)，並將其映射為 0.0 ~ 1.0 的發光係數。
  // 0.03 代表當蟲子每幀移動超過「畫布大小的 3%」時，發光係數即達到頂峰 1.0。
  // 建議範圍: 0.01 (極容易發亮) ~ 0.05 (只有被猛烈炸飛時才會發亮)
  float speed = length(vel);
  float clockSize = min(uResolution.x, uResolution.y);
  float speedFactor = smoothstep(0.0, clockSize * 0.03, speed);

  // 將像素座標 (0 ~ Width) 轉換為 WebGL 的標準化設備座標 (NDC, -1.0 ~ +1.0)
  // WebGL 的 Y 軸朝上，而 Canvas 2D 的 Y 軸朝下，因此需要做 clipPos.y = -clipPos.y 反轉
  vec2 clipPos = (pos / uResolution) * 2.0 - 1.0;
  clipPos.y = -clipPos.y;

  gl_Position = vec4(clipPos, 0.0, 1.0);
  
  // 🌟 【參數調整區：單顆粒子大小】
  // 如果覺得蟲子太密集或太粗，可以把 2.0 調小成 1.5 或 1.0
  gl_PointSize = size * 2.0 * uPointScale;

  vOpacity = opacity;
  vSpeed = speedFactor;
}
`;

const RENDER_FRAG = `#version 300 es
precision highp float;

uniform vec3 uColor;
in float vOpacity;
in float vSpeed;
out vec4 fragColor;

void main() {
  float dist = length(gl_PointCoord - 0.5) * 2.0;
  if (dist > 1.0) discard;

  // 讓方形點變成邊緣柔和的圓點
  float alpha = smoothstep(1.0, 0.4, dist) * vOpacity;

  fragColor = vec4(uColor, alpha);
}
`;

// ==========================================
// WebGL 與色彩輔助函式 (Helpers)
// ==========================================
// 原本的色相轉換邏輯完全保留
function hslToRgb(h, s, l) {
    let r, g, b;
    if (s === 0) {
        r = g = b = l;
    } else {
        const hue2rgb = (p, q, t) => {
            if (t < 0) t += 1;
            if (t > 1) t -= 1;
            if (t < 1 / 6) return p + (q - p) * 6 * t;
            if (t < 1 / 2) return q;
            if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6;
            return p;
        };
        const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
        const p = 2 * l - q;
        r = hue2rgb(p, q, h + 1 / 3);
        g = hue2rgb(p, q, h);
        b = hue2rgb(p, q, h - 1 / 3);
    }
    return [r, g, b];
}

function compileShader(gl, type, source) {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) return null;
    return shader;
}

function linkProgram(gl, vertSource, fragSource) {
    const vert = compileShader(gl, gl.VERTEX_SHADER, vertSource);
    const frag = compileShader(gl, gl.FRAGMENT_SHADER, fragSource);
    if (!vert || !frag) return null;
    const program = gl.createProgram();
    gl.attachShader(program, vert);
    gl.attachShader(program, frag);
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) return null;
    return program;
}

function createFloatTexture(gl, size, data) {
    const texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, size, size, 0, gl.RGBA, gl.FLOAT, data);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    return texture;
}

// ==========================================
// 3. 蟲群圓環取樣函式
// ==========================================
// 這個函式的核心邏輯是：在記憶體中偷偷建立一個 Canvas (離屏渲染)，
// 畫出我們想要的圖形 (這裡是一個圓環)，然後逐像素檢查，
// 有畫到白色的地方，就生成一顆「蟲卵」(包含初始 X, Y, 大小, 透明度)，
// 最後將這些蟲卵資料打包成 Float32Array 交給 GPU。
function sampleClockParticleData(width, height) {
    if (width === 0 || height === 0) return null;

    const clockSize = Math.min(width, height);
    const isSmall = clockSize < 400;

    // 🌟 【參數調整區：蟲群密度】
    // particleGap 決定 for 迴圈讀取像素陣列的步長 (跨距)。
    // 數字越小，讀到的像素越多，蟲子越密集，GPU 運算負擔呈「平方級」增加。
    // 建議: 小視窗用 1.0 (極密)，大螢幕用 1.5 ~ 2.0 (維持效能並保留顆粒感)。
    const particleGap = isSmall ? 1.0 : 1.5;
    const particleSize = isSmall ? 1.0 : 1.2;

    const offscreen = document.createElement('canvas');
    offscreen.width = width;
    offscreen.height = height;
    const ctx = offscreen.getContext('2d');
    if (!ctx) return null;

    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, width, height);

    const centerX = width / 2;
    const centerY = height / 2;
    // 基準圓環半徑 (與你原本 0.35 * 0.6 比例吻合)
    const radius = clockSize * 0.21;

    ctx.save();
    ctx.translate(centerX, centerY);

    ctx.beginPath();
    ctx.arc(0, 0, radius, 0, Math.PI * 2);
    ctx.strokeStyle = '#FFF';
    ctx.lineWidth = Math.max(2, clockSize * 0.015);
    ctx.stroke();
    ctx.restore();

    // 讀取剛剛畫好的離屏畫布像素
    const imageData = ctx.getImageData(0, 0, width, height);
    const { data } = imageData;
    const tempList = [];

    for (let y = 0; y < height; y += particleGap) {
        for (let x = 0; x < width; x += particleGap) {
            // 計算一維陣列的 Index (每個像素佔 RGBA 4 個位置)
            const index = (Math.floor(y) * width + Math.floor(x)) * 4;
            const r = data[index];

            // 只要紅色通道 > 50 (代表不是全黑的背景)，就記錄為蟲子位置
            if (r > 50) {
                const size = particleSize * (0.5 + Math.random() * 0.8);
                const opacity = 0.5 + Math.random() * 0.5;
                tempList.push(x, y, size, opacity);
            }
        }
    }

    const count = tempList.length / 4;
    if (count === 0) return null;

    // 將資料分成兩份：
    // originData: 蟲子永遠不會忘記的老家 (Origin)
    // stateData: 蟲子當下的狀態 (State，初始時與老家相同)
    const originData = new Float32Array(tempList);
    const stateData = new Float32Array(count * 4);
    for (let i = 0; i < count; i++) {
        stateData[i * 4] = originData[i * 4];
        stateData[i * 4 + 1] = originData[i * 4 + 1];
    }

    return { originData, stateData, count };
}

// ==========================================
// React 元件 (主程式)
// ==========================================
export default function ClockTerminal() {
    const webglCanvasRef = useRef(null);
    const d2CanvasRef = useRef(null);

    // 在這種每秒 60 幀運作的視覺特效中，嚴禁使用 useState 來存放音訊資料，否則會觸發 React 每秒 60 次的元件重繪 (Re-render) 導致卡頓。
    // 使用 useRef 可以跨幀保存資料且不觸發重繪。
    const audioDataRef = useRef({ bass: 0, frequencies: [] });
    const smoothedFreqsRef = useRef(new Array(64).fill(0));
    const reqAnimRef = useRef(null);
    const wsRef = useRef(null);

    const glRefs = useRef({
        gl: null,
        particleCount: 0,
        textureSize: 0,
        originTexture: null,
        stateTextureList: [],
        framebufferList: [],
        noiseTexture: null,
        quadVao: null,
        particleVao: null,
        physicsProgram: null,
        renderProgram: null,
        physicsUniforms: {},
        renderUniforms: {},
        currentStateIndex: 0,
    });

    useEffect(() => {
        const glCanvas = webglCanvasRef.current;
        const d2Canvas = d2CanvasRef.current;
        if (!glCanvas || !d2Canvas) return;

        // 初始化 WebGL2 Context (開啟 alpha 以支援 Electron 背景透明)
        const gl = glCanvas.getContext('webgl2', { premultipliedAlpha: false, alpha: true });
        const ctx2d = d2Canvas.getContext('2d');

        // GPGPU 的核心依賴：必須支援將浮點數 (Float) 寫入顏色緩衝區，否則無法記錄座標的精確小數點
        if (!gl || !gl.getExtension('EXT_color_buffer_float')) return;
        glRefs.current.gl = gl;

        // 畫布尺寸與 WebGL 資源初始化
        const initCanvasAndWebGL = () => {
            const dpr = window.devicePixelRatio || 1;
            const w = window.innerWidth;
            const h = window.innerHeight;
            const logicalWidth = Math.floor(w * dpr);
            const logicalHeight = Math.floor(h * dpr);

            if (logicalWidth === 0 || logicalHeight === 0) return;
            if (glCanvas.width === logicalWidth && glCanvas.height === logicalHeight) return;

            // 雙層畫布的尺寸同步
            glCanvas.width = logicalWidth;
            glCanvas.height = logicalHeight;
            glCanvas.style.width = `${w}px`;
            glCanvas.style.height = `${h}px`;
            d2Canvas.width = logicalWidth;
            d2Canvas.height = logicalHeight;
            d2Canvas.style.width = `${w}px`;
            d2Canvas.style.height = `${h}px`;

            gl.viewport(0, 0, logicalWidth, logicalHeight);

            ctx2d.resetTransform();
            ctx2d.scale(dpr, dpr);

            const particleData = sampleClockParticleData(logicalWidth, logicalHeight);
            if (!particleData) return;

            const particleCount = particleData.count;
            const textureSize = Math.ceil(Math.sqrt(particleCount));
            const totalPixels = textureSize * textureSize;

            const paddedOrigin = new Float32Array(totalPixels * 4);
            const paddedState = new Float32Array(totalPixels * 4);
            paddedOrigin.set(particleData.originData);
            paddedState.set(particleData.stateData);

            for (let i = particleCount; i < totalPixels; i++) {
                paddedOrigin[i * 4] = -99999;
                paddedOrigin[i * 4 + 1] = -99999;
                paddedState[i * 4] = -99999;
                paddedState[i * 4 + 1] = -99999;
            }

            // Ping-Pong Buffer Design：
            // 在 WebGL 中，你不能「同時」從一張紋理讀取資料又寫入資料。
            // 因此我們建立兩張一模一樣的 stateTexture (A 和 B)，
            // 第一幀：讀 A -> 運算 -> 寫 B。
            // 第二幀：讀 B -> 運算 -> 寫 A。來回交替(Ping-Pong)。
            const originTexture = createFloatTexture(gl, textureSize, paddedOrigin);
            const stateTextureList = [
                createFloatTexture(gl, textureSize, paddedState),
                createFloatTexture(gl, textureSize, paddedState),
            ];

            const framebufferList = stateTextureList.map((texture) => {
                const fb = gl.createFramebuffer();
                gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
                gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
                return fb;
            });
            gl.bindFramebuffer(gl.FRAMEBUFFER, null);

            const quadVao = gl.createVertexArray();
            gl.bindVertexArray(quadVao);
            const quadBuffer = gl.createBuffer();
            gl.bindBuffer(gl.ARRAY_BUFFER, quadBuffer);
            gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
            gl.bindVertexArray(null);

            const noiseProgram = linkProgram(gl, PHYSICS_VERT, NOISE_GEN_FRAG);
            const noiseTexture = gl.createTexture();
            gl.bindTexture(gl.TEXTURE_2D, noiseTexture);
            gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, 256, 256, 0, gl.RGBA, gl.FLOAT, null);
            const noiseFilter = gl.getExtension('OES_texture_float_linear') ? gl.LINEAR : gl.NEAREST;
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, noiseFilter);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, noiseFilter);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.REPEAT);

            const noiseFb = gl.createFramebuffer();
            gl.bindFramebuffer(gl.FRAMEBUFFER, noiseFb);
            gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, noiseTexture, 0);
            gl.viewport(0, 0, 256, 256);
            gl.useProgram(noiseProgram);
            gl.bindVertexArray(quadVao);

            const posLoc = gl.getAttribLocation(noiseProgram, 'aPosition');
            gl.enableVertexAttribArray(posLoc);
            gl.bindBuffer(gl.ARRAY_BUFFER, quadBuffer);
            gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 0, 0);

            gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
            gl.bindFramebuffer(gl.FRAMEBUFFER, null);
            gl.deleteProgram(noiseProgram);
            gl.deleteFramebuffer(noiseFb);

            const physicsProgram = linkProgram(gl, PHYSICS_VERT, PHYSICS_FRAG);
            const renderProgram = linkProgram(gl, RENDER_VERT, RENDER_FRAG);

            if (!physicsProgram || !renderProgram) return;

            const physicsUniforms = {
                state: gl.getUniformLocation(physicsProgram, 'uState'),
                origin: gl.getUniformLocation(physicsProgram, 'uOrigin'),
                noise: gl.getUniformLocation(physicsProgram, 'uNoise'),
                resolution: gl.getUniformLocation(physicsProgram, 'uResolution'),
                bass: gl.getUniformLocation(physicsProgram, 'uBass'),
                freqs: gl.getUniformLocation(physicsProgram, 'uFreqs'),
                time: gl.getUniformLocation(physicsProgram, 'uTime'),
            };

            const renderUniforms = {
                state: gl.getUniformLocation(renderProgram, 'uState'),
                origin: gl.getUniformLocation(renderProgram, 'uOrigin'),
                resolution: gl.getUniformLocation(renderProgram, 'uResolution'),
                textureSize: gl.getUniformLocation(renderProgram, 'uTextureSize'),
                time: gl.getUniformLocation(renderProgram, 'uTime'),
                pointScale: gl.getUniformLocation(renderProgram, 'uPointScale'),
                color: gl.getUniformLocation(renderProgram, 'uColor'),
            };

            const particleVao = gl.createVertexArray();

            Object.assign(glRefs.current, {
                particleCount,
                textureSize,
                originTexture,
                stateTextureList,
                framebufferList,
                noiseTexture,
                quadVao,
                particleVao,
                physicsProgram,
                renderProgram,
                physicsUniforms,
                renderUniforms,
                currentStateIndex: 0,
            });

            gl.viewport(0, 0, logicalWidth, logicalHeight);
        };

        window.addEventListener('resize', initCanvasAndWebGL);
        requestAnimationFrame(initCanvasAndWebGL);

        const connectWebSocket = () => {
            wsRef.current = new WebSocket('ws://localhost:8080');
            wsRef.current.onmessage = (event) => {
                try {
                    audioDataRef.current = JSON.parse(event.data);
                } catch (err) {
                    console.error(err);
                }
            };
        };
        connectWebSocket();

        // 60 fps render loop
        const renderLoop = () => {
            const refs = glRefs.current;
            if (!refs.gl || !refs.physicsProgram || !refs.renderProgram) {
                reqAnimRef.current = requestAnimationFrame(renderLoop);
                return;
            }

            const w = window.innerWidth;
            const h = window.innerHeight;
            const dpr = window.devicePixelRatio || 1;
            const logicalWidth = Math.floor(w * dpr);
            const logicalHeight = Math.floor(h * dpr);
            const time = performance.now() * 0.001;

            const rawBass = audioDataRef.current.bass || 0;
            const rawFreqs = audioDataRef.current.frequencies || [];

            // 使用原本的 0.15 Lerp 平滑係數
            for (let i = 0; i < 64; i++) {
                const targetFreq = (rawFreqs[i] || 0) / 255.0;
                smoothedFreqsRef.current[i] += (targetFreq - smoothedFreqsRef.current[i]) * 0.15;
            }

            const readIndex = refs.currentStateIndex;
            const writeIndex = 1 - readIndex;

            // --- Physics Pass ---
            gl.bindFramebuffer(gl.FRAMEBUFFER, refs.framebufferList[writeIndex]);
            gl.viewport(0, 0, refs.textureSize, refs.textureSize);
            gl.useProgram(refs.physicsProgram);

            gl.activeTexture(gl.TEXTURE0);
            gl.bindTexture(gl.TEXTURE_2D, refs.stateTextureList[readIndex]);
            gl.uniform1i(refs.physicsUniforms.state, 0);

            gl.activeTexture(gl.TEXTURE1);
            gl.bindTexture(gl.TEXTURE_2D, refs.originTexture);
            gl.uniform1i(refs.physicsUniforms.origin, 1);

            gl.activeTexture(gl.TEXTURE2);
            gl.bindTexture(gl.TEXTURE_2D, refs.noiseTexture);
            gl.uniform1i(refs.physicsUniforms.noise, 2);

            gl.uniform2f(refs.physicsUniforms.resolution, logicalWidth, logicalHeight);
            gl.uniform1f(refs.physicsUniforms.bass, rawBass / 255.0);
            gl.uniform1fv(refs.physicsUniforms.freqs, new Float32Array(smoothedFreqsRef.current));
            gl.uniform1f(refs.physicsUniforms.time, time);

            gl.bindVertexArray(refs.quadVao);
            gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

            // --- Render Pass ---
            gl.bindFramebuffer(gl.FRAMEBUFFER, null);
            gl.viewport(0, 0, logicalWidth, logicalHeight);
            gl.clearColor(0.0, 0.0, 0.0, 0.0);
            gl.clear(gl.COLOR_BUFFER_BIT);

            gl.enable(gl.BLEND);
            gl.blendFuncSeparate(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA, gl.ONE, gl.ONE_MINUS_SRC_ALPHA);

            gl.useProgram(refs.renderProgram);

            gl.activeTexture(gl.TEXTURE0);
            gl.bindTexture(gl.TEXTURE_2D, refs.stateTextureList[writeIndex]);
            gl.uniform1i(refs.renderUniforms.state, 0);

            gl.activeTexture(gl.TEXTURE1);
            gl.bindTexture(gl.TEXTURE_2D, refs.originTexture);
            gl.uniform1i(refs.renderUniforms.origin, 1);

            gl.uniform2f(refs.renderUniforms.resolution, logicalWidth, logicalHeight);
            gl.uniform1f(refs.renderUniforms.textureSize, refs.textureSize);
            gl.uniform1f(refs.renderUniforms.time, time);
            gl.uniform1f(refs.renderUniforms.pointScale, dpr);

            const bassRatio = Math.min(rawBass / 255.0, 1.0);

            // 安靜時為 60 (黃色)，重低音最強時減少 60 變成 0 (紅色)
            const dynamicHue = 60 - bassRatio * 60;
            const dynamicLightness = 50 + bassRatio * 15;
            const [rColor, gColor, bColor] = hslToRgb(dynamicHue / 360, 0.8, dynamicLightness / 100);
            gl.uniform3f(refs.renderUniforms.color, rColor, gColor, bColor);

            gl.bindVertexArray(refs.particleVao);
            gl.drawArrays(gl.POINTS, 0, refs.particleCount);

            gl.disable(gl.BLEND);
            refs.currentStateIndex = writeIndex;

            // --- 繪製頂層 2D 時鐘指針 ---
            ctx2d.clearRect(0, 0, w, h);
            ctx2d.save();
            ctx2d.translate(w / 2, h / 2);
            ctx2d.rotate(-Math.PI / 2);

            const clockSize = h * 0.35;
            const now = new Date();
            const sec = now.getSeconds() + now.getMilliseconds() / 1000;
            const min = now.getMinutes() + sec / 60;
            const hr = (now.getHours() % 12) + min / 60;

            const secAngle = (sec / 60) * (Math.PI * 2);
            const minAngle = (min / 60) * (Math.PI * 2);
            const hrAngle = (hr / 12) * (Math.PI * 2);

            const drawHand = (angle, length, color, weight) => {
                ctx2d.save();
                ctx2d.rotate(angle);
                ctx2d.strokeStyle = color;
                ctx2d.lineWidth = weight;
                ctx2d.lineCap = 'round';
                ctx2d.beginPath();
                ctx2d.moveTo(0, 0);
                ctx2d.lineTo(length, 0);
                ctx2d.stroke();
                ctx2d.restore();
            };

            drawHand(secAngle, clockSize * 0.8, '#D3D3D3', 2);
            drawHand(minAngle, clockSize * 0.65, '#808080', 4);
            drawHand(hrAngle, clockSize * 0.4, '#778899', 6);

            ctx2d.fillStyle = '#FFFFFF';
            ctx2d.beginPath();
            ctx2d.arc(0, 0, 4, 0, Math.PI * 2);
            ctx2d.fill();

            ctx2d.restore();

            reqAnimRef.current = requestAnimationFrame(renderLoop);
        };
        renderLoop();

        // ==========================================
        // 元件解除安裝 (Unmount) 的清理機制
        // ==========================================
        return () => {
            window.removeEventListener('resize', initCanvasAndWebGL);
            if (reqAnimRef.current) cancelAnimationFrame(reqAnimRef.current);
            if (wsRef.current) wsRef.current.close();

            // 釋放所有向 GPU 借用的資源
            const refs = glRefs.current;
            if (refs.gl) {
                refs.framebufferList.forEach((fb) => refs.gl.deleteFramebuffer(fb));
                refs.stateTextureList.forEach((t) => refs.gl.deleteTexture(t));
                if (refs.originTexture) refs.gl.deleteTexture(refs.originTexture);
                if (refs.noiseTexture) refs.gl.deleteTexture(refs.noiseTexture);
                if (refs.quadVao) refs.gl.deleteVertexArray(refs.quadVao);
                if (refs.particleVao) refs.gl.deleteVertexArray(refs.particleVao);
                if (refs.physicsProgram) refs.gl.deleteProgram(refs.physicsProgram);
                if (refs.renderProgram) refs.gl.deleteProgram(refs.renderProgram);
            }
        };
    }, []);

    return (
        <main className="relative w-screen h-screen overflow-hidden bg-transparent" style={{ WebkitAppRegion: 'drag' }}>
            <canvas ref={webglCanvasRef} className="absolute top-0 left-0 w-full h-full z-0 block" />
            <canvas ref={d2CanvasRef} className="absolute top-0 left-0 w-full h-full z-10 block pointer-events-none" />
        </main>
    );
}
