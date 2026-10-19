// electron/security.js
// 集中管理「哪些來源算是本 App」的判斷，供 WebSocket、權限、導航防護共用。

const WS_HOST = '127.0.0.1';
const WS_PORT = 8080;
// 音訊引擎連線用的路徑；只有從這個路徑連進來的 client 訊息會被轉發
const ENGINE_PATH = '/engine';
const VIEWER_PATH = '/';
// 頻譜訊息約 1KB，限制單則訊息大小避免被灌爆記憶體
const WS_MAX_PAYLOAD = 64 * 1024;

const DEV_ORIGIN = 'http://localhost:3000'; // next dev
const PROD_ORIGIN = 'app://-'; // electron-serve (scheme: app, hostname: -)

function getAppOrigins(isDev) {
    return isDev ? [DEV_ORIGIN] : [PROD_ORIGIN];
}

// 注意：Node 的 URL 對非 special scheme (app://) 的 .origin 會回傳 'null'，所以自行組合
function originOf(url) {
    try {
        const u = new URL(url);
        return `${u.protocol}//${u.host}`;
    } catch {
        return null;
    }
}

function isAppUrl(url, isDev) {
    const origin = originOf(url);
    return origin !== null && getAppOrigins(isDev).includes(origin);
}

function requestPath(req) {
    try {
        return new URL(req.url || '/', 'ws://localhost').pathname;
    } catch {
        return null;
    }
}

function isEngineRequest(req) {
    return requestPath(req) === ENGINE_PATH;
}

// ws 的 verifyClient：只接受 Origin 為本 App 的連線 (沒帶 Origin 的一律拒絕)
function createVerifyClient(isDev) {
    const allowed = getAppOrigins(isDev);
    return (info) => {
        const path = requestPath(info.req);
        if (path !== ENGINE_PATH && path !== VIEWER_PATH) return false;
        return typeof info.origin === 'string' && allowed.includes(info.origin);
    };
}

function createWsServerOptions(isDev) {
    return {
        host: WS_HOST,
        port: WS_PORT,
        maxPayload: WS_MAX_PAYLOAD,
        verifyClient: createVerifyClient(isDev),
    };
}

module.exports = {
    getAppOrigins,
    isAppUrl,
    isEngineRequest,
    createVerifyClient,
    createWsServerOptions,
};
