/*
 * demo-eventgpt.js — EventGPT 走讀互動 demo
 * 「event camera 模擬器:RGB 幀 vs 事件流」
 *
 * 介面:window.initEventgptDemo(containerElement)
 *  - 樣式只注入一次(<style id="egd-eventgpt-demo-style">),class 一律 egd- 前綴
 *  - 無外部資源(純 canvas + CSS + emoji);容器最窄 ~350px 可用(auto-fit 自動上下疊)
 *  - prefers-reduced-motion:自動暫停動畫、改渲染靜態示意 + 說明,仍可手動按 ▶ 播放
 *  - 重複對同一容器呼叫會先清掉舊實例(rAF / observer / listener 皆會釋放)
 *
 * 論文事實來源:papers/eventgpt.md(所有數字皆取自該檔背景節與方法核心節)
 * 物理模擬部分為「概念示意」:以亮度差閾值化產生 ON/OFF 事件,非真實 DVS 感光物理。
 */
(function () {
  'use strict';

  var STYLE_ID = 'egd-eventgpt-demo-style';

  // ---- 模擬常數 ----
  var CW = 480, CH = 270;   // 顯示 canvas 內部解析度(CSS 縮放至容器寬,響應式)
  var GW = 120, GH = 68;    // 概念「感光格點」解析度(事件產生與資料量示意以此計)
  var PIX = GW * GH;
  var BALL_R = 9;           // 球半徑(格點單位)
  var DIFF_TH = 22;         // 亮度變化閾值(0–255):|ΔL| 超過才吐事件(示意)
  var MAX_FPS = 60;
  var C_ON = '#08b94e';     // ON(變亮)= 綠
  var C_OFF = '#ec7500';    // OFF(變暗)= 橙

  // 三階段訓練(取材自 eventgpt.md「方法核心 ③ 三階段訓練」)
  var STAGES = [
    {
      name: '視覺-語言對齊(暖身)',
      html: '<b>Stage 1|視覺-語言對齊</b> — 先用 LLaVA-Pretrain <b>558k</b> 影像-文字資料暖身' +
        '(image-text 的 gap 較小),只訓練 Linear Projector,其餘模組全部凍結。'
    },
    {
      name: '事件-語言對齊(主 gap)',
      html: '<b>Stage 2|事件-語言對齊</b> — 用 <b>1M</b> 合成 event-文字資料 N-ImageNet-Chat,' +
        '訓練時空聚合器(Spatio-Temporal Aggregator)與 Event-Language Adapter,橋接 event↔language 的主要 domain gap。'
    },
    {
      name: '指令微調(實戰)',
      html: '<b>Stage 3|指令微調</b> — 用 <b>120k</b> 真實世界指令資料 Event-Chat' +
        '(caption/VQA/推理,含低光、高速、隧道場景),全參數解凍(含 LLM Vicuna v1.5)微調。'
    }
  ];

  var CSS = [
    '.egd-wrap{border:1px solid #d0d7de;border-radius:14px;background:linear-gradient(180deg,#fbfcfd,#fff);overflow:hidden;margin:1.4em 0;color:#1f2328;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Noto Sans TC","PingFang TC",sans-serif;line-height:1.6}',
    '.egd-head{padding:9px 14px;font-size:.9rem;font-weight:700;color:var(--accent,#0969da);background:rgba(9,105,218,.06);border-bottom:1px solid #d0d7de}',
    '.egd-body{padding:12px}',
    '.egd-rmnote{font-size:.78rem;color:#6b5218;background:#fff8ec;border:1px solid #f0d9ac;border-radius:8px;padding:8px 10px;margin:0 0 10px}',
    '.egd-rmnote[hidden]{display:none}',
    '.egd-panels{display:grid;grid-template-columns:repeat(auto-fit,minmax(228px,1fr));gap:10px}',
    '.egd-panel{border:1px solid #d0d7de;border-radius:10px;background:#fff;padding:8px;min-width:0}',
    '.egd-ptitle{display:flex;justify-content:space-between;align-items:baseline;gap:6px;margin:0 0 6px;font-size:.86rem;font-weight:700}',
    '.egd-sub{font-weight:400;color:#57606a;font-size:.72rem;white-space:nowrap}',
    '.egd-cv{display:block;width:100%;height:auto;background:#000;border-radius:8px}',
    '.egd-legend{font-size:.72rem;color:#57606a;margin-top:5px}',
    '.egd-mk-on{color:var(--green,#08b94e)}',
    '.egd-mk-off{color:var(--orange,#ec7500)}',
    '.egd-bar-row{display:flex;align-items:center;gap:7px;margin-top:7px}',
    '.egd-bar-label{font-size:.72rem;color:#57606a;white-space:nowrap}',
    '.egd-track{flex:1 1 auto;min-width:36px;height:10px;background:#f6f8fa;border:1px solid #d0d7de;border-radius:999px;overflow:hidden}',
    '.egd-fill{height:100%;width:0;border-radius:999px;transition:width .15s linear}',
    '.egd-fill-rgb{background:var(--accent,#0969da)}',
    '.egd-fill-ev{background:var(--green,#08b94e);min-width:2px}',
    '.egd-bar-val{font-size:.7rem;color:#57606a;white-space:nowrap;font-variant-numeric:tabular-nums}',
    '.egd-controls{display:flex;flex-wrap:wrap;align-items:center;gap:8px 14px;margin-top:10px;padding:9px 12px;background:#f6f8fa;border:1px solid #d0d7de;border-radius:10px}',
    '.egd-btn{appearance:none;font-family:inherit;font-size:.84rem;font-weight:600;color:#1f2328;background:#fff;border:1.5px solid #d0d7de;border-radius:999px;padding:5px 14px;cursor:pointer;transition:border-color .15s,color .15s;flex:none}',
    '.egd-btn:hover{border-color:var(--accent,#0969da);color:var(--accent,#0969da)}',
    '.egd-ctl{display:flex;align-items:center;gap:7px;font-size:.78rem;color:#1f2328;flex:1 1 150px;min-width:145px}',
    '.egd-ctl input[type=range]{flex:1 1 60px;min-width:56px;accent-color:var(--accent,#0969da);margin:0}',
    '.egd-val{color:var(--accent,#0969da);font-weight:700;font-size:.76rem;min-width:3.4em;text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}',
    '.egd-facts{display:flex;flex-wrap:wrap;gap:6px;margin-top:12px}',
    '.egd-chip{font-size:.72rem;font-weight:600;border-radius:999px;padding:3px 10px;background:rgba(9,105,218,.07);border:1px solid rgba(9,105,218,.25);color:#0757a8}',
    '.egd-oneliner{margin:9px 0 0;font-size:.8rem;color:#39424d;background:#f6f8fa;border-left:4px solid var(--green,#08b94e);border-radius:0 8px 8px 0;padding:7px 11px}',
    '.egd-note{margin-top:8px;font-size:.72rem;color:#6b5218;background:#fff8ec;border:1px solid #f0d9ac;border-radius:8px;padding:6px 10px}',
    '.egd-stepper{margin-top:14px;border-top:1px dashed #d0d7de;padding-top:11px}',
    '.egd-steptitle{font-size:.84rem;font-weight:700;margin:0 0 8px}',
    '.egd-steps{display:flex;align-items:center;gap:6px;margin:0 0 8px}',
    '.egd-dot{appearance:none;flex:none;width:30px;height:30px;border-radius:50%;border:2px solid #d0d7de;background:#fff;color:#57606a;font-family:inherit;font-size:.8rem;font-weight:700;cursor:pointer;transition:background .15s,border-color .15s,color .15s;padding:0;line-height:1}',
    '.egd-dot:hover{border-color:var(--accent,#0969da);color:var(--accent,#0969da)}',
    '.egd-dot.egd-on{background:var(--accent,#0969da);border-color:var(--accent,#0969da);color:#fff}',
    '.egd-conn{flex:0 1 56px;min-width:10px;height:2px;background:#d0d7de;border-radius:1px}',
    '.egd-stepname{font-size:.76rem;color:#57606a;margin-left:4px;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}',
    '.egd-stage{font-size:.8rem;color:#39424d;background:#f6f8fa;border-left:4px solid var(--accent,#0969da);border-radius:0 8px 8px 0;padding:8px 12px}',
    '.egd-stage b{color:var(--accent,#0969da)}',
    '.egd-btn:focus-visible,.egd-dot:focus-visible,.egd-ctl input:focus-visible{outline:2px solid var(--accent,#0969da);outline-offset:2px}',
    '@media (prefers-reduced-motion: reduce){.egd-fill,.egd-btn,.egd-dot{transition:none!important}}'
  ].join('\n');

  var HTML =
    '<div class="egd-wrap">' +
      '<div class="egd-head">🎮 互動走讀|為什麼 event 是『只看變化』的相機</div>' +
      '<div class="egd-body">' +
        '<div class="egd-rmnote" hidden>⚙️ 偵測到系統「減少動態效果」設定:動畫已自動暫停,以下為靜態示意 —— ' +
          '左圖是 RGB 相機某次快門拍到的整幀;右圖是同一瞬間 event 相機的輸出:只有球邊緣「亮度有變化」的像素吐出事件' +
          '(前緣變亮 = 綠色 ON、後緣變暗 = 橙色 OFF),靜止背景完全沒有輸出。想觀看動畫可按「▶ 播放」。</div>' +
        '<div class="egd-panels">' +
          '<div class="egd-panel">' +
            '<div class="egd-ptitle"><span>📷 RGB 相機</span>' +
              '<span class="egd-sub">快門 <span class="egd-fpstag">12</span> fps・整幀同步曝光</span></div>' +
            '<canvas class="egd-cv egd-cv-rgb" width="480" height="270" role="img" ' +
              'aria-label="RGB 相機模擬:以固定幀率取樣移動的亮球,球速快時幀間跳躍明顯"></canvas>' +
            '<div class="egd-legend">快門瞬間才更新畫面 —— 球快時明顯「跳格」</div>' +
            '<div class="egd-bar-row"><span class="egd-bar-label">資料量</span>' +
              '<div class="egd-track"><div class="egd-fill egd-fill-rgb"></div></div>' +
              '<span class="egd-bar-val egd-val-rgb">每幀傳整張影像</span></div>' +
          '</div>' +
          '<div class="egd-panel">' +
            '<div class="egd-ptitle"><span>⚡ Event 相機</span>' +
              '<span class="egd-sub">每像素獨立・異步</span></div>' +
            '<canvas class="egd-cv egd-cv-ev" width="480" height="270" role="img" ' +
              'aria-label="Event 相機模擬:只在亮度變化的像素輸出事件,前緣綠色 ON、後緣橙色 OFF,靜止背景無輸出"></canvas>' +
            '<div class="egd-legend"><span class="egd-mk-on">●</span> ON(變亮) <span class="egd-mk-off">●</span> OFF(變暗)・靜止 = 無輸出</div>' +
            '<div class="egd-bar-row"><span class="egd-bar-label">資料量</span>' +
              '<div class="egd-track"><div class="egd-fill egd-fill-ev"></div></div>' +
              '<span class="egd-bar-val egd-val-ev">只傳變化像素</span></div>' +
          '</div>' +
        '</div>' +
        '<div class="egd-controls">' +
          '<button type="button" class="egd-btn egd-play" aria-label="暫停或播放動畫">⏸ 暫停</button>' +
          '<label class="egd-ctl">球速<input type="range" class="egd-speed" min="1" max="10" step="0.5" value="3" aria-label="球速">' +
            '<span class="egd-val egd-speedval">3.0×</span></label>' +
          '<label class="egd-ctl">RGB fps<input type="range" class="egd-fps" min="5" max="60" step="1" value="12" aria-label="RGB 相機幀率">' +
            '<span class="egd-val egd-fpsval">12 fps</span></label>' +
        '</div>' +
        '<div class="egd-facts">' +
          '<span class="egd-chip">🕸️ 稀疏:只傳變化</span>' +
          '<span class="egd-chip">⏱️ 微秒級延遲(RGB 毫秒級)</span>' +
          '<span class="egd-chip">🌗 高動態範圍 &gt;120 dB(RGB ~60 dB)</span>' +
          '<span class="egd-chip">🔋 低功耗(neuromorphic)</span>' +
        '</div>' +
        '<p class="egd-oneliner">💡 一句話:event 相機<b>只記變化、不記靜止</b> —— 每個像素獨立、異步地在' +
          '「亮度變化超過閾值」時吐出事件 (x, y, t, p);靜止場景幾乎無輸出,資料天生稀疏。</p>' +
        '<div class="egd-note">⚠️ 本模擬為概念示意(亮度差閾值化 + 舊事件快速衰減),<b>非真實感光物理</b>;' +
          '資料量條亦為相對示意。特性與數字出自本站〈EventGPT〉走讀的背景節。</div>' +
        '<div class="egd-stepper">' +
          '<div class="egd-steptitle">🧩 延伸:EventGPT 怎麼讓 LLM 看懂這種事件流?—— 三階段訓練(點圓點切換)</div>' +
          '<div class="egd-steps" role="group" aria-label="EventGPT 三階段訓練切換">' +
            '<button type="button" class="egd-dot egd-on" data-egd-stage="0" aria-label="第 1 階段">1</button>' +
            '<span class="egd-conn" aria-hidden="true"></span>' +
            '<button type="button" class="egd-dot" data-egd-stage="1" aria-label="第 2 階段">2</button>' +
            '<span class="egd-conn" aria-hidden="true"></span>' +
            '<button type="button" class="egd-dot" data-egd-stage="2" aria-label="第 3 階段">3</button>' +
            '<span class="egd-stepname"></span>' +
          '</div>' +
          '<div class="egd-stage" role="status"></div>' +
        '</div>' +
      '</div>' +
    '</div>';

  function injectStyle() {
    if (document.getElementById(STYLE_ID)) return;
    var s = document.createElement('style');
    s.id = STYLE_ID;
    s.textContent = CSS;
    document.head.appendChild(s);
  }

  window.initEventgptDemo = function (container) {
    if (!container) return;
    if (typeof container.__egdCleanup === 'function') container.__egdCleanup();
    injectStyle();
    container.innerHTML = HTML;

    function $(sel) { return container.querySelector(sel); }

    var rgbCv = $('.egd-cv-rgb'), evCv = $('.egd-cv-ev');
    var rctx = rgbCv.getContext('2d');
    var evctx = evCv.getContext('2d');
    // 隱藏的「真實場景」canvas:以格點解析度渲染,供逐像素亮度差分
    var scene = document.createElement('canvas');
    scene.width = GW; scene.height = GH;
    var sctx = scene.getContext('2d', { willReadFrequently: true });

    var playBtn = $('.egd-play');
    var speedIn = $('.egd-speed'), fpsIn = $('.egd-fps');
    var speedVal = $('.egd-speedval'), fpsVal = $('.egd-fpsval'), fpsTag = $('.egd-fpstag');
    var fillRGB = $('.egd-fill-rgb'), fillEV = $('.egd-fill-ev');
    var valRGB = $('.egd-val-rgb'), valEV = $('.egd-val-ev');
    var rmNote = $('.egd-rmnote');
    var stageBox = $('.egd-stage'), stepName = $('.egd-stepname');
    var dots = container.querySelectorAll('.egd-dot');

    var st = {
      phase: 0.9,                          // 球的相位(正弦來回)
      speed: parseFloat(speedIn.value),
      fps: parseInt(fpsIn.value, 10),
      playing: false,
      visible: true,
      destroyed: false,
      acc: 0,                              // 快門累計時間
      uiAcc: 0,                            // UI(資料條)更新累計
      frame: 0,                            // RGB 幀計數
      evRate: 0                            // 事件率 EMA(事件/秒)
    };

    var prevL = new Uint8Array(PIX), curL = new Uint8Array(PIX);
    var onIdx = new Int32Array(PIX), offIdx = new Int32Array(PIX);
    var rafId = 0, lastT = 0;
    var SX = CW / GW, SY = CH / GH;

    function ballX(phase) {
      var amp = GW / 2 - BALL_R - 4;
      return GW / 2 + amp * Math.sin(phase);
    }

    // 灰階場景:黑背景 + 亮球(邊緣柔化,讓「變化帶」自然出現)
    function drawSceneAt(x) {
      sctx.fillStyle = '#000';
      sctx.fillRect(0, 0, GW, GH);
      var y = GH / 2;
      var g = sctx.createRadialGradient(x, y, 1, x, y, BALL_R);
      g.addColorStop(0, '#ffffff');
      g.addColorStop(0.62, '#f4f4f4');
      g.addColorStop(1, 'rgba(255,255,255,0)');
      sctx.fillStyle = g;
      sctx.beginPath();
      sctx.arc(x, y, BALL_R, 0, Math.PI * 2);
      sctx.fill();
    }

    function readLuma(dst) {
      var d = sctx.getImageData(0, 0, GW, GH).data;
      for (var i = 0; i < PIX; i++) dst[i] = d[i << 2]; // 灰階場景取 R 通道即可
    }

    // 事件產生(概念示意):ΔL > 閾值 → ON(綠)、ΔL < -閾值 → OFF(橙)
    // 回傳本 tick 事件數。fade=true 時先讓舊事件快速衰減(半透明黑覆蓋)。
    function emitEvents(fade) {
      var nOn = 0, nOff = 0, i;
      for (i = 0; i < PIX; i++) {
        var dl = curL[i] - prevL[i];
        if (dl > DIFF_TH) onIdx[nOn++] = i;
        else if (dl < -DIFF_TH) offIdx[nOff++] = i;
      }
      if (fade) {
        evctx.fillStyle = 'rgba(0,0,0,0.34)';
        evctx.fillRect(0, 0, CW, CH);
      } else {
        evctx.fillStyle = '#000';
        evctx.fillRect(0, 0, CW, CH);
      }
      var k, p;
      evctx.fillStyle = C_ON;
      for (k = 0; k < nOn; k++) {
        p = onIdx[k];
        evctx.fillRect((p % GW) * SX + 1, ((p / GW) | 0) * SY + 1, 2.5, 2.5);
      }
      evctx.fillStyle = C_OFF;
      for (k = 0; k < nOff; k++) {
        p = offIdx[k];
        evctx.fillRect((p % GW) * SX + 1, ((p / GW) | 0) * SY + 1, 2.5, 2.5);
      }
      return nOn + nOff;
    }

    // RGB 相機:只有快門瞬間才呼叫(幀間畫面保持不動 → 高速時跳格)
    function drawRGBFrame(x) {
      rctx.fillStyle = '#000';
      rctx.fillRect(0, 0, CW, CH);
      var cx = x * SX, cy = CH / 2, r = BALL_R * SX * 1.25;
      var g = rctx.createRadialGradient(cx, cy, 1, cx, cy, r);
      g.addColorStop(0, '#ffffff');
      g.addColorStop(0.55, '#f2f2f2');
      g.addColorStop(0.8, 'rgba(255,255,255,.35)');
      g.addColorStop(1, 'rgba(255,255,255,0)');
      rctx.fillStyle = g;
      rctx.beginPath();
      rctx.arc(cx, cy, r, 0, Math.PI * 2);
      rctx.fill();
      rctx.fillStyle = '#8b949e';
      rctx.font = '15px ui-monospace,SFMono-Regular,Menlo,monospace';
      rctx.fillText('幀 #' + st.frame, 9, CH - 10);
    }

    // 資料量示意條:同一線性刻度(滿格 = 60fps 整幀),文字顯示誠實的相對比例
    function updateBars() {
      var rgbRate = st.fps * PIX;            // 概念單位:像素/秒
      var maxRate = MAX_FPS * PIX;
      var evPct = Math.min(100, st.evRate / maxRate * 100);
      fillRGB.style.width = (st.fps / MAX_FPS * 100).toFixed(1) + '%';
      fillEV.style.width = Math.max(evPct, 0.6).toFixed(2) + '%';
      valRGB.textContent = '整張 ' + GW + '×' + GH + ' × ' + st.fps + ' fps';
      var ratio = rgbRate > 0 ? (st.evRate / rgbRate * 100) : 0;
      valEV.textContent = '≈ RGB 的 ' + (ratio < 0.05 ? '0.0' : ratio.toFixed(1)) + '%';
    }

    function tick(now) {
      rafId = requestAnimationFrame(tick);
      var dt = (now - lastT) / 1000;
      lastT = now;
      if (dt <= 0) return;
      if (dt > 0.05) dt = 0.05;              // 分頁切回時避免大跳

      var omega = 0.55 * st.speed;           // 角速度(示意)
      st.phase += omega * dt;
      var x = ballX(st.phase);

      // — Event 相機:每個顯示 tick 都連續差分(近似「異步、微秒級」) —
      drawSceneAt(x);
      var tmp = prevL; prevL = curL; curL = tmp;
      readLuma(curL);
      var n = emitEvents(true);
      var a = 1 - Math.exp(-dt / 0.35);      // 事件率 EMA(約 0.35s 時間常數)
      st.evRate += (n / dt - st.evRate) * a;

      // — RGB 相機:只有快門到時才更新畫面 —
      st.acc += dt;
      var itv = 1 / st.fps;
      if (st.acc >= itv) {
        st.acc %= itv;
        st.frame++;
        drawRGBFrame(x);
      }

      st.uiAcc += dt;
      if (st.uiAcc >= 0.12) { st.uiAcc = 0; updateBars(); }
    }

    // 靜態替代(reduced-motion 或手動暫停時的滑桿調整):
    // 同一瞬間的「RGB 快門幀」vs「事件輸出」(位移量隨球速滑桿變化)
    function renderStatic() {
      var x1 = ballX(st.phase);
      var x2 = ballX(st.phase + 0.028 * st.speed + 0.02);
      drawSceneAt(x1); readLuma(prevL);
      drawSceneAt(x2); readLuma(curL);
      var n = emitEvents(false);
      st.frame = Math.max(1, st.frame);
      drawRGBFrame(x2);
      st.evRate = n * MAX_FPS;               // 以 1/60 秒位移估事件率(示意)
      updateBars();
    }

    function syncLoop() {
      var should = st.playing && st.visible && !st.destroyed;
      if (should && !rafId) {
        lastT = performance.now();
        rafId = requestAnimationFrame(tick);
      } else if (!should && rafId) {
        cancelAnimationFrame(rafId);
        rafId = 0;
      }
    }

    function setPlaying(p, fromUser) {
      st.playing = p;
      playBtn.textContent = p ? '⏸ 暫停' : '▶ 播放';
      if (p && fromUser) rmNote.hidden = true; // 使用者主動播放 → 收起靜態說明
      syncLoop();
      if (!p) updateBars();
    }

    // ---- 控制列 ----
    playBtn.addEventListener('click', function () { setPlaying(!st.playing, true); });
    speedIn.addEventListener('input', function () {
      st.speed = parseFloat(speedIn.value);
      speedVal.textContent = st.speed.toFixed(1) + '×';
      if (!st.playing) renderStatic();
    });
    fpsIn.addEventListener('input', function () {
      st.fps = parseInt(fpsIn.value, 10);
      fpsVal.textContent = st.fps + ' fps';
      fpsTag.textContent = String(st.fps);
      if (!st.playing) renderStatic(); else updateBars();
    });

    // ---- 三階段 stepper ----
    function setStage(idx) {
      for (var k = 0; k < dots.length; k++) {
        var on = k === idx;
        dots[k].classList.toggle('egd-on', on);
        if (on) dots[k].setAttribute('aria-current', 'step');
        else dots[k].removeAttribute('aria-current');
      }
      stepName.textContent = STAGES[idx].name;
      stageBox.innerHTML = STAGES[idx].html;
    }
    for (var di = 0; di < dots.length; di++) {
      dots[di].addEventListener('click', function (e) {
        setStage(parseInt(e.currentTarget.getAttribute('data-egd-stage'), 10));
      });
    }
    setStage(0);

    // ---- prefers-reduced-motion:暫停動畫 + 靜態替代與說明 ----
    var mq = window.matchMedia ? window.matchMedia('(prefers-reduced-motion: reduce)') : null;
    function onMqChange() {
      if (mq && mq.matches) {
        rmNote.hidden = false;
        setPlaying(false, false);
        renderStatic();
      }
    }
    if (mq) {
      if (mq.addEventListener) mq.addEventListener('change', onMqChange);
      else if (mq.addListener) mq.addListener(onMqChange);
    }

    // ---- 不在視窗內時自動暫停迴圈(省電;回到視窗自動續播) ----
    var io = null;
    if (typeof IntersectionObserver !== 'undefined') {
      io = new IntersectionObserver(function (entries) {
        st.visible = entries[0].isIntersecting;
        syncLoop();
      });
      io.observe(container);
    }

    // ---- 初始畫面:先建立「無變化 → 無事件」基準,再各畫一幀 ----
    drawSceneAt(ballX(st.phase));
    readLuma(curL);
    prevL.set(curL);
    evctx.fillStyle = '#000';
    evctx.fillRect(0, 0, CW, CH);
    st.frame = 1;
    drawRGBFrame(ballX(st.phase));
    updateBars();

    if (mq && mq.matches) {
      rmNote.hidden = false;
      renderStatic();
      setPlaying(false, false);
    } else {
      setPlaying(true, false);
    }

    // ---- 清理(同容器重複 init 時自動呼叫) ----
    container.__egdCleanup = function () {
      st.destroyed = true;
      st.playing = false;
      if (rafId) { cancelAnimationFrame(rafId); rafId = 0; }
      if (io) io.disconnect();
      if (mq) {
        if (mq.removeEventListener) mq.removeEventListener('change', onMqChange);
        else if (mq.removeListener) mq.removeListener(onMqChange);
      }
      container.innerHTML = '';
      container.__egdCleanup = null;
    };
  };
})();
