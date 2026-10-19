// app/audio-engine/page.jsx
'use client';

import { useEffect, useRef, useState } from 'react';

export default function AudioEngine() {
    const [status, setStatus] = useState('Initializing...');
    // 保持對 source 節點的參照，防止被 Garbage Collector 回收
    const sourceRef = useRef(null);
    const wsRef = useRef(null);
    const audioCtxRef = useRef(null);
    const loopTimeoutRef = useRef(null);

    useEffect(() => {
        let analyser;
        let dataArray;

        const initEngine = async () => {
            try {
                // 連接 WebSocket Server
                wsRef.current = new WebSocket('ws://127.0.0.1:8080/engine');
                wsRef.current.onopen = () => setStatus('WebSocket Connected');
                wsRef.current.onerror = (err) => console.error('WS Error', err);

                // 取得麥克風或立體聲混音權限
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
                // [Minimum Viable] 設備遺失防呆機制：監聽音軌狀態
                stream.getTracks().forEach((track) => {
                    track.onended = () => {
                        console.warn('Audio device lost or track ended');
                        if (wsRef.current?.readyState === WebSocket.OPEN) {
                            // 發送 Error 訊號給主視覺
                            wsRef.current.send(JSON.stringify({ type: 'error', status: 'device_lost' }));
                        }
                        setStatus('Error: Audio Device Lost');
                    };
                });
                // 初始化 Web Audio API
                const AudioContext = window.AudioContext || window.webkitAudioContext;
                audioCtxRef.current = new AudioContext();

                // 處理 context 被休眠的問題
                if (audioCtxRef.current.state === 'suspended') {
                    await audioCtxRef.current.resume();
                }

                const audioCtx = audioCtxRef.current;
                analyser = audioCtx.createAnalyser();
                analyser.fftSize = 256;
                const bufferLength = analyser.frequencyBinCount;
                dataArray = new Uint8Array(bufferLength);

                sourceRef.current = audioCtx.createMediaStreamSource(stream);

                // 致命坑點防護 1：建立 Dummy GainNode 並設為 0，確保流向 Destination，避免 FFT 陣列全 0
                const dummyGain = audioCtx.createGain();
                dummyGain.gain.value = 0;

                // 路由：Source -> Analyser -> DummyGain -> Destination
                sourceRef.current.connect(analyser);
                analyser.connect(dummyGain);
                dummyGain.connect(audioCtx.destination);

                setStatus('Audio Engine Running');

                // 背景休眠防護：使用 setTimeout 模擬 60FPS，不使用 requestAnimationFrame
                const processAudio = () => {
                    if (!analyser || wsRef.current?.readyState !== WebSocket.OPEN) {
                        loopTimeoutRef.current = setTimeout(processAudio, 16);
                        return;
                    }

                    // [Minimum Viable] 改回取得頻率域資料，確保沒有負數與頻繁的零交叉(Zero-crossing)
                    analyser.getByteFrequencyData(dataArray);

                    let bassSum = 0;
                    for (let i = 0; i < 5; i++) {
                        bassSum += dataArray[i];
                    }
                    const bass = bassSum / 5;

                    const frequencies = Array.from(dataArray);

                    wsRef.current.send(JSON.stringify({ bass, frequencies }));

                    loopTimeoutRef.current = setTimeout(processAudio, 16);
                };

                processAudio();
            } catch (error) {
                console.error('Engine Initialization Error:', error);
                setStatus(`Error: ${error.message}`);
            }
        };

        initEngine();

        // Cleanup function
        return () => {
            clearTimeout(loopTimeoutRef.current);
            if (wsRef.current) wsRef.current.close();
            if (audioCtxRef.current) audioCtxRef.current.close();
            if (sourceRef.current) sourceRef.current.disconnect();
        };
    }, []);

    return (
        <div style={{ color: 'white', padding: '20px' }}>
            <h1>Audio Engine Worker</h1>
            <p>Status: {status}</p>
            <p>This window operates in the background.</p>
        </div>
    );
}
