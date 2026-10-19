// electron/main.js
const { app, BrowserWindow, session, Tray, Menu } = require('electron');
const path = require('path');
const WebSocket = require('ws');
const { isAppUrl, isEngineRequest, createWsServerOptions } = require('./security');

// 解決背景 AudioContext 無法自動播放的問題 (繞過使用者互動限制)
app.commandLine.appendSwitch('autoplay-policy', 'no-user-gesture-required');

let mainWindow;
let audioWorkerWindow; // engine window (hide default)
let wss; // WebSocket Server
let tray = null;
let isLocked = false;
const isDev = !app.isPackaged;

function createWebSocketServer() {
    try {
        // 只綁定 127.0.0.1，且 verifyClient 只接受本 App 的 Origin
        wss = new WebSocket.Server(createWsServerOptions(isDev));
        wss.on('error', (err) => {
            console.error('WebSocket Server Error:', err);
        });
        wss.on('connection', (ws, req) => {
            ws.isEngine = isEngineRequest(req);
            ws.on('message', (message) => {
                // 只轉發來自音訊引擎 (/engine) 的資料，忽略其他 client 送來的訊息
                if (!ws.isEngine) return;
                // Pub/Sub 轉發邏輯：將收到來自 Worker 的音訊資料，廣播給所有連線的 Client (主視覺)
                wss.clients.forEach((client) => {
                    if (client !== ws && !client.isEngine && client.readyState === WebSocket.OPEN) {
                        client.send(message.toString());
                    }
                });
            });

            ws.on('error', (err) => {
                console.error('WebSocket Error:', err);
            });
        });
    } catch (error) {
        console.error('Failed to start WebSocket Server:', error);
    }
}

function updateTrayMenu() {
    if (!tray) return;
    const contextMenu = Menu.buildFromTemplate([
        {
            label: isLocked ? '🔓 Unlock (allows dragging)' : '🔒 Lock (mouse pass-through)',
            click: () => {
                isLocked = !isLocked;
                if (mainWindow) {
                    // 核心魔法：當 isLocked 為 true 時，視窗忽略滑鼠事件，滑鼠直接點擊到背後的桌面
                    mainWindow.setIgnoreMouseEvents(isLocked, { forward: true });
                }
                updateTrayMenu(); // 點擊後重新渲染選單文字
            },
        },
        { type: 'separator' },
        { label: '👁️ Show the clock', click: () => mainWindow?.show() },
        { label: '👻 Hide the clock', click: () => mainWindow?.hide() },
        { type: 'separator' },
        {
            label: '❌ Close',
            click: () => {
                app.isQuiting = true;
                app.quit();
            },
        },
    ]);
    tray.setContextMenu(contextMenu);
}

function createWindows() {
    // 視窗 A: 主視覺終端 (透明、無邊框、永遠置頂)
    mainWindow = new BrowserWindow({
        width: 250,
        height: 250,
        transparent: true, // 透明背景
        frame: false, // 無邊框
        alwaysOnTop: true, // 永遠置頂
        skipTaskbar: true,
        resizable: false,
        webPreferences: {
            nodeIntegration: false,
            contextIsolation: true,
        },
    });

    mainWindow.on('closed', () => {
        app.quit();
    });

    mainWindow.setIgnoreMouseEvents(isLocked, { forward: true });

    // 視窗 B: 音訊引擎 Worker (隱藏、不降速)
    audioWorkerWindow = new BrowserWindow({
        width: 400,
        height: 400,
        show: false, // 隱藏於背景
        webPreferences: {
            nodeIntegration: false,
            contextIsolation: true,
            backgroundThrottling: false, // 關鍵防呆：防止 Chrome 在背景降速 setTimeout 或 setInterval
        },
    });

    // 關鍵防呆 1：自動允許麥克風權限，避免擾人彈窗 (僅限本 App 頁面、僅限音訊)
    session.defaultSession.setPermissionRequestHandler((webContents, permission, callback, details = {}) => {
        const isOwnPage =
            isAppUrl(webContents.getURL(), isDev) && (!details.requestingUrl || isAppUrl(details.requestingUrl, isDev));
        const audioOnly = (details.mediaTypes || []).every((type) => type === 'audio');
        callback(permission === 'media' && isOwnPage && audioOnly);
    });

    // [Performance Optimization] 根據環境決定載入來源
    // 開發時依賴 Next.js server，生產環境直接讀取靜態編譯出來的 HTML
    if (isDev) {
        mainWindow.loadURL('http://localhost:3000');
        audioWorkerWindow.loadURL('http://localhost:3000/audio-engine');
    } else {
        // Next.js 'export' 會輸出到 out 資料夾
        mainWindow.loadURL('app://-/');
        audioWorkerWindow.loadURL('app://-/audio-engine.html');
    }
}

async function bootstrap() {
    try {
        // [Minimum Viable] 在應用程式準備就緒前，使用動態 import() 載入 ESM 模組
        if (!isDev) {
            const serve = (await import('electron-serve')).default;
            serve({ directory: 'out' });
        }

        // 等待 Electron 引擎準備完畢
        await app.whenReady();

        createWebSocketServer();
        createWindows();

        const iconPath = path.join(__dirname, 'clock.ico');
        tray = new Tray(iconPath);
        tray.setToolTip('Swarm Clock');

        updateTrayMenu();

        app.on('activate', () => {
            if (BrowserWindow.getAllWindows().length === 0) createWindows();
        });
    } catch (err) {
        console.error('App initialization failed:', err);
    }
}

// 防止任何視窗被導向 App 以外的網址或開啟新視窗
app.on('web-contents-created', (_event, contents) => {
    contents.setWindowOpenHandler(() => ({ action: 'deny' }));
    const blockForeignNavigation = (event, url) => {
        if (!isAppUrl(url, isDev)) event.preventDefault();
    };
    contents.on('will-navigate', blockForeignNavigation);
    contents.on('will-redirect', blockForeignNavigation);
});

bootstrap();

app.on('window-all-closed', () => {
    app.quit();
});
