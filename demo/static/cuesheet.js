// CueSheet demo: replay precomputed per-second segment track + real
// 4-class PANN posterior over the source audio (or video). This page
// shows only what the CueSheet pipeline actually emits.

const SEGMENT_CLASS_MAP = {
  Pre_Concert: "pre",
  Performance: "perf",
  MC_Talk: "mc",
  Applause: "applause",
  Intermission: "intermission",
  Ambient: "ambient",
};

const SEGMENT_DISPLAY = {
  Pre_Concert: "Pre-concert",
  Performance: "Performance",
  MC_Talk: "MC talk",
  Applause: "Applause",
  Intermission: "Intermission",
  Ambient: "Ambient",
};

const audioEl = document.getElementById("audio");
const videoEl = document.getElementById("video");
const audiosetTopEl = document.getElementById("audioset-top");
const pipeOutDetail = document.getElementById("pipe-out-detail");
const pipeStages = ["pipe-audio", "pipe-pann", "pipe-group", "pipe-heur", "pipe-out"]
  .map(id => document.getElementById(id));
const showSelect = document.getElementById("show");
const genreBadge = document.getElementById("genre");
const segmentNow = document.getElementById("segment-now");
const timeNow = document.getElementById("time-now");
const posteriorList = document.getElementById("posterior-list");
const ribbon = document.getElementById("ribbon");
const playhead = document.getElementById("playhead");
const uploadInput = document.getElementById("upload");
const uploadStatus = document.getElementById("upload-status");
const liveBtn = document.getElementById("live-btn");
const liveWaveformEl = document.getElementById("live-waveform");
const livePlaybackEl = document.getElementById("live-playback");
const themeBtn = document.getElementById("theme-btn");

// ===== App mode ===============
// The demo covers the audio pipeline. body.mm-mode CSS keeps the audio
// chrome (the lone Record button, audio-only upload accept) and is left in
// place as a harmless layout toggle.
const APP_MODE = "mm";
document.body.classList.toggle("mm-mode", APP_MODE === "mm");
if (APP_MODE === "mm" && uploadInput) {
  uploadInput.setAttribute("accept", "audio/*");
}
// Initial idle label paint for the live buttons — the HTML carries
// English defaults via data-i18n spans, but in MM mode we want the lone
// audio button to read "● Record" instead of "● Audio" before any
// language switch has a chance to repaint.
if (APP_MODE === "mm" && liveBtn) {
  liveBtn.textContent = "● Record";
}
// Header tag carries the public attribution.
{
  const tagEl = document.getElementById("header-tag");
  if (tagEl) {
    tagEl.textContent = "Yamaha Corporation · CONCERT-10 dataset";
  }
}

let currentShow = null;
let catalog = [];
let lastReportedSec = -1;
let mediaEl = audioEl;   // whichever element is currently driving the playhead

function useMediaElement(kind) {
  // The video element lives in the header as a small monitor; the audio
  // player sits in the #media-wrap band. Show whichever the source needs
  // and collapse the audio band when a video is playing.
  const wrap = document.getElementById("media-wrap");
  // video-mode enlarges the header text so it fills the height the video
  // monitor adds, instead of leaving an empty band under the title.
  document.body.classList.toggle("video-mode", kind === "video");
  if (kind === "video") {
    audioEl.pause(); audioEl.style.display = "none";
    videoEl.style.display = "block";
    if (wrap) wrap.style.display = "none";
    mediaEl = videoEl;
  } else {
    videoEl.pause(); videoEl.style.display = "none";
    audioEl.style.display = "block";
    if (wrap) wrap.style.display = "";
    mediaEl = audioEl;
  }
}

// ===== Fullscreen ==========================================================
const fsBtn = document.getElementById("fs-btn");
function toggleFullscreen() {
  const root = document.documentElement;
  if (!document.fullscreenElement) {
    (root.requestFullscreen || root.webkitRequestFullscreen || (() => {}))
      .call(root);
  } else {
    (document.exitFullscreen || document.webkitExitFullscreen || (() => {}))
      .call(document);
  }
}
if (fsBtn) fsBtn.addEventListener("click", toggleFullscreen);
document.addEventListener("keydown", (e) => {
  if ((e.key === "f" || e.key === "F") &&
      !/^(INPUT|SELECT|TEXTAREA)$/.test(e.target.tagName || "")) {
    toggleFullscreen();
  }
});
document.addEventListener("fullscreenchange", () => {
  if (fsBtn) {
    fsBtn.textContent = document.fullscreenElement
      ? t("btn.fs.exit") : t("btn.fs.enter");
  }
});

// ===== Scale-to-fit ========================================================
// The whole demo is one fixed-design stage (1400 px wide). We scale it as a
// single unit so it always fills exactly one screen: larger in fullscreen,
// smaller in a small window, never scrolling and never leaving dead space.
const stage = document.getElementById("stage");
function fitStage() {
  if (!stage) return;
  // The user picks between "width" (scale to window width, vertical
  // scroll allowed, content grows with the window) and "screen" (also
  // cap by height so the whole page fits in one viewport without
  // scrolling). Choice persists in localStorage and applies whether or
  // not the user is currently fullscreen. naturalH is measured with
  // transform cleared so the scaled height doesn't feed back.
  const mode = (typeof getFitMode === "function") ? getFitMode() : "width";
  let s = window.innerWidth / 1400;
  if (mode === "screen") {
    stage.style.transform = "";
    const naturalH = stage.offsetHeight || 1;
    s = Math.min(s, window.innerHeight / naturalH);
  }
  stage.style.transform = `scale(${s})`;
  stage.style.transformOrigin = "top center";
}
window.addEventListener("resize", fitStage);
document.addEventListener("fullscreenchange", () => setTimeout(fitStage, 80));
if (window.ResizeObserver && stage) {
  new ResizeObserver(() => fitStage()).observe(stage);
}
fitStage();

// Re-draw the ribbon when the browser finally knows the media's true
// duration, otherwise ribbon segments stay scaled to the streaming
// frontier and the playhead drifts.
function onMetadataLoaded() {
  try { drawRibbon(); }
  catch (e) { console.error("[CueSheet] redraw on loadedmetadata failed:", e); }
}
audioEl.addEventListener("loadedmetadata", onMetadataLoaded);
videoEl.addEventListener("loadedmetadata", onMetadataLoaded);
audioEl.addEventListener("durationchange", onMetadataLoaded);
videoEl.addEventListener("durationchange", onMetadataLoaded);

async function loadCatalog() {
  const resp = await fetch("data/catalog.json");
  catalog = await resp.json();
  for (const item of catalog) {
    const opt = document.createElement("option");
    opt.value = item.show_id;
    opt.textContent = `${item.label} (${item.genre}, ${(item.n_seconds / 60).toFixed(0)} min)`;
    showSelect.appendChild(opt);
  }
  if (catalog.length > 0) {
    const params = new URLSearchParams(window.location.search);
    const wanted = params.get("show");
    const initial = wanted && catalog.some(c => c.show_id === wanted)
      ? wanted : catalog[0].show_id;
    showSelect.value = initial;
    await loadShow(initial);
  }
}

function _renderShowWf1() {
  const el = document.getElementById("show-wf1");
  if (!el) return;
  const enc = currentEncoder || "efficientat";
  const s = ENCODER_SCORES[enc] || {};
  const per = (s.per_show_wf1_pct || {})[currentShow && currentShow.show_id];
  el.textContent = per != null
    ? `${ENCODER_META[enc].title} on this show · wF1 ${per}%`
    : "";
}

async function loadShow(showId) {
  const suffix = currentEncoder && currentEncoder !== "efficientat"
    ? `_${currentEncoder}` : "";
  let resp = await fetch(`data/${showId}${suffix}.json`);
  if (!resp.ok && suffix) {
    resp = await fetch(`data/${showId}.json`);
  }
  if (!resp.ok) {
    uploadStatus.textContent = `could not load show data for ${showId}`;
    return;
  }
  if (modePre) modePre.textContent = "Precomputed";
  currentShow = await resp.json();
  _renderShowWf1();
  genreBadge.textContent = currentShow.genre;
  // Clear any leftover upload / live-session status from a previous source —
  // unless the upload chooser is open (a slow show JSON fetch resolving after
  // the user already picked a file was erasing the chooser buttons).
  if (!uploadChooserOpen) uploadStatus.textContent = "";
  // A new show resets the inference mode to the precomputed track.
  resetToPrecomputed();
  const entry = catalog.find(c => c.show_id === showId);
  const audioName = entry && entry.audio_filename ? entry.audio_filename : `${showId}.mp3`;
  // Carry the audible-excerpt metadata (used by the ribbon overlay) from
  // the catalog entry onto currentShow so drawRibbon can find it.
  if (entry && entry.excerpt_start_sec != null) {
    currentShow.excerpt_start_sec = entry.excerpt_start_sec;
    currentShow.excerpt_duration_sec = entry.excerpt_duration_sec;
  }
  // Audible-window text hint next to the audio player. The displayed
  // start/end seconds are the *player time* (== audio file time) — the
  // silenced mp3 has its audible region at exactly these timestamps
  // in its own timeline, so scrubbing the native audio player to MM:SS
  // jumps to the music immediately.
  const hintEl = document.getElementById("audible-hint");
  const hintRangeEl = document.getElementById("audible-hint-range");
  const noteEl = document.getElementById("excerpt-note");
  if (noteEl) {
    // The copyright note only applies to gallery shows with a silenced
    // excerpt window; uploads and live capture are fully audible.
    noteEl.hidden = !(currentShow.excerpt_start_sec != null
                      && currentShow.excerpt_duration_sec);
  }
  if (hintEl && hintRangeEl) {
    if (currentShow.excerpt_start_sec != null && currentShow.excerpt_duration_sec) {
      const s = currentShow.excerpt_start_sec;
      const e = s + currentShow.excerpt_duration_sec;
      const fmt = (t) => `${Math.floor(t/60)}:${String(t%60).padStart(2,"0")}`;
      hintRangeEl.textContent =
        `player time ${fmt(s)} – ${fmt(e)}  `
      + `(scrub here to hear the music; rest of the track is silenced for copyright safety)`;
      hintEl.hidden = false;
    } else {
      hintEl.hidden = true;
    }
  }
  // Runtime sanity check: log if the catalog excerpt range falls
  // outside the loaded audio's actual duration. Helps catch the case
  // where someone updates n_seconds without re-encoding the mp3.
  const _onAudioLoaded = () => {
    if (!isFinite(audioEl.duration) || !currentShow) return;
    const dur = audioEl.duration;
    const es  = currentShow.excerpt_start_sec;
    const ee  = es != null
                ? es + (currentShow.excerpt_duration_sec || 0) : null;
    if (es != null && (es < 0 || ee > dur + 0.5)) {
      console.warn(`[CueSheet] excerpt [${es}, ${ee}]s is outside audio duration ${dur.toFixed(2)}s for ${currentShow.show_id}`);
    }
    audioEl.removeEventListener("loadedmetadata", _onAudioLoaded);
  };
  audioEl.addEventListener("loadedmetadata", _onAudioLoaded);
  // Show the source-of-record button in the picker row (always visible,
  // regardless of audio/video mode, so a reader can verify provenance
  // with one click). The HF Space gallery is YouTube-only so every
  // entry renders as a clickable red badge.
  const sourceBtn = document.getElementById("source-btn");
  const sourceBtnIcon = document.getElementById("source-btn-icon");
  const sourceBtnText = document.getElementById("source-btn-text");
  if (sourceBtn && sourceBtnIcon && sourceBtnText) {
    if (entry && entry.source_url) {
      sourceBtn.href = entry.source_url;
      sourceBtn.classList.add("source-btn-link");
      sourceBtn.title = `Open original on ${entry.source_host || "source"}`
        + (entry.source_channel ? ` — ${entry.source_channel}` : "")
        + " (new tab)";
      sourceBtnText.textContent = "Source";
      sourceBtnIcon.innerHTML = "&nearr;";   // ↗ external link
      sourceBtn.hidden = false;
    } else {
      sourceBtn.hidden = true;
    }
  }
  // Use the source video when the catalog includes one AND we are not in
  // MM mode — the demo is audio-only, so even when a
  // video is available we play the mp3 in mm-mode to keep the demo
  // matched to the public gallery. f_7ntJHYAmc (Georgia Tech jazz) is
  // deliberately audio-only in the catalog because it is in the visual
  // probe training set.
  const videoName = (APP_MODE !== "mm" && entry && entry.video_filename)
                    ? entry.video_filename : null;
  if (livePlaybackEl) livePlaybackEl.hidden = true;
  if (videoName) {
    useMediaElement("video");
    videoEl.style.display = "block";
    videoEl.src = `video/${videoName}`;
    videoEl.load();
  } else {
    useMediaElement("audio");
    audioEl.style.display = "";        // re-show after a live-mic session
    audioEl.src = `audio/${audioName}`;
    audioEl.load();
  }
  drawRibbon();
  drawPosteriorRows();
  lastReportedSec = -1;
  // Kick off the causal cache pre-warm in the background. By the time
  // the user clicks "Causal from playhead", the show audio is decoded
  // in memory and per-second pumps drop from ~140 ms to ~30 ms.
  fetch(`/api/causal_warm?show=${encodeURIComponent(showId)}`,
        {method: "POST"}).catch(() => {});
}

function ribbonDuration() {
  // The audio file is the source of truth for wall-clock time — the
  // playhead syncs with mediaEl.currentTime so the ribbon must scale
  // to mediaEl.duration for the playhead and the audible-window box
  // to align. When the audio hasn't loaded yet we fall back to the
  // longest data array we have so the visualization stays sensible.
  // The HF Space includes silenced mp3s that are typically ~10 s longer
  // than the precomputed segment_int (mp3 encoder tail padding); using
  // mediaEl.duration here is what keeps the audible-window overlay
  // pinned to the actual audible seconds in the player.
  if (mediaEl && isFinite(mediaEl.duration) && mediaEl.duration > 0) {
    return mediaEl.duration;
  }
  const piLen = (currentShow.segment_int || []).length;
  const gtLen = Array.isArray(currentShow.gt_labels)
                ? currentShow.gt_labels.length : 0;
  return Math.max(1, currentShow.n_seconds || 1, piLen, gtLen);
}

function drawRibbon() {
  if (!currentShow) return;
  ribbon.innerHTML = "";
  // Pick the right source for the ribbon:
  //   1. causal mode → causalSegmentInt (live overlay)
  //   2. otherwise → currentShow.segment_int (audio-derived precomputed)
  let pi;
  if (causalActive) {
    pi = causalSegmentInt;
  } else {
    pi = currentShow.segment_int || [];
  }
  // In precomputed mode with no data yet (fresh upload mid-stream), no
  // ribbon to draw. In causal mode we still want the frontier marker at
  // causalStart even before the first prediction lands.
  if (!causalActive && pi.length === 0) return;
  const total = ribbonDuration();
  // If the per-show JSON bakes in `gt_labels` + `gt_classes`, split the
  // ribbon into two equal-height rows: top = hand-labeled GT, bottom =
  // model prediction. Left-edge "GT" / "PRED" badges + a solid divider
  // keep the two from reading as one continuous track of the same
  // color palette. Unequal split was confusing — the bigger row read
  // as "the main view" and the smaller row read as a side legend.
  const hasGt = Array.isArray(currentShow.gt_labels)
                && currentShow.gt_labels.length > 0;
  const GT_TOP = "0%", GT_HEIGHT = "35%";
  const PRED_TOP = hasGt ? "37%" : "0%";
  const PRED_HEIGHT = hasGt ? "63%" : "100%";
  const emit = (start, end, ph, source, dim, top, height, kind) => {
    const segmentName = source.segment_classes[ph];
    const cls = segmentName ? SEGMENT_CLASS_MAP[segmentName] : "ambient";
    const seg = document.createElement("div");
    const gtFlag = (kind === "ground truth") ? " gt" : "";
    seg.className = `seg bg-${cls || "ambient"}${dim ? " dim" : ""}${gtFlag}`;
    seg.style.position = "absolute";
    seg.style.left = `${(start / total) * 100}%`;
    seg.style.width = `${((end - start) / total) * 100}%`;
    seg.style.top = top;
    seg.style.height = height;
    seg.title = `${kind || "prediction"} · ${segmentName || "unknown"}  ${start}s → ${end}s`
      + (dim ? "  (precomputed history)" : "");
    ribbon.appendChild(seg);
  };
  const emitRuns = (arr, lo, hi, source, dim, top, height, kind) => {
    if (hi <= lo) return;
    let runStart = lo, runSeg = arr[lo];
    for (let t = lo + 1; t < hi; t++) {
      if (arr[t] !== runSeg) {
        emit(runStart, t, runSeg, source, dim, top, height, kind);
        runStart = t; runSeg = arr[t];
      }
    }
    emit(runStart, hi, runSeg, source, dim, top, height, kind);
  };
  if (hasGt) {
    const gtSrc = {segment_classes: currentShow.gt_classes
                                  || currentShow.segment_classes};
    const gtHi = Math.min(currentShow.gt_labels.length,
                          Math.floor(total) + 1);
    emitRuns(currentShow.gt_labels, 0, gtHi,
             gtSrc, false, GT_TOP, GT_HEIGHT, "ground truth");
    if (gtHi < total) {
      const noGt = document.createElement("div");
      noGt.className = "ribbon-no-gt";
      noGt.style.left = `${(gtHi / total) * 100}%`;
      noGt.style.width = `${((total - gtHi) / total) * 100}%`;
      noGt.style.top = GT_TOP;
      noGt.style.height = GT_HEIGHT;
      noGt.title = `No hand-labeled GT past t = ${gtHi} s`;
      ribbon.appendChild(noGt);
    }
    const divider = document.createElement("div");
    divider.className = "ribbon-divider";
    ribbon.appendChild(divider);
    const gtBadge = document.createElement("div");
    gtBadge.className = "ribbon-label ribbon-label-gt";
    gtBadge.textContent = "GT";
    gtBadge.title = "Hand-labeled ground truth (top strip)";
    ribbon.appendChild(gtBadge);
    const predBadge = document.createElement("div");
    predBadge.className = "ribbon-label ribbon-label-pred";
    predBadge.textContent = "PRED";
    predBadge.title = "Model prediction (bottom strip)";
    ribbon.appendChild(predBadge);
  }
  let frontier;
  if (causalActive) {
    // Nothing before the switch-on point is shown — causal mode must
    // not leak any precomputed memory into the visualization. Only the
    // live causal output between causalStart and the computed frontier
    // (capped at the playhead) is drawn; everything else stays blank.
    const head = (lastReportedSec >= 0) ? lastReportedSec : causalStart;
    const liveHi = Math.max(causalStart, Math.min(pi.length, causalNext, head + 1));
    emitRuns(pi, causalStart, liveHi, currentShow, false,
             PRED_TOP, PRED_HEIGHT, "causal");
    frontier = liveHi;
  } else {
    emitRuns(pi, 0, pi.length, currentShow, false,
             PRED_TOP, PRED_HEIGHT, "prediction");
    frontier = pi.length;
  }
  // If the prediction track ends before the audio's full duration
  // (HF Space mp3s are ~10 s longer than n_seconds due to encoder
  // tail padding) draw a striped filler so the playhead doesn't
  // enter visually-empty space near the right edge.
  if (frontier < total) {
    const noPred = document.createElement("div");
    noPred.className = "ribbon-no-pred";
    noPred.style.left = `${(frontier / total) * 100}%`;
    noPred.style.width = `${((total - frontier) / total) * 100}%`;
    noPred.style.top = PRED_TOP;
    noPred.style.height = PRED_HEIGHT;
    noPred.title = `No precomputed prediction past t = ${frontier} s`;
    ribbon.appendChild(noPred);
  }
  // Inference frontier marker — how far the predictions reach.
  const front = document.createElement("div");
  front.className = "infer-frontier";
  front.style.left = `${(frontier / total) * 100}%`;
  front.title = `inference frontier · t = ${frontier} s`;
  ribbon.appendChild(front);
  // Audible-window marker (static HF Space deployment: audio outside
  // this 30-second region is silenced for copyright safety). Drawn
  // as two thick cyan vertical lines pinned to the music-start and
  // music-end seconds: the START line's center aligns with the
  // playhead at the exact moment the user first hears sound, and
  // the END line's center aligns with the moment the audio goes
  // silent. A box overlay covering [es, ee] was confusing because
  // its center fell mid-audible — users expected the marker's center
  // to be the audible-onset moment, not the midpoint of the window.
  if (currentShow.excerpt_start_sec != null && currentShow.excerpt_duration_sec) {
    const es = currentShow.excerpt_start_sec;
    const ee = es + currentShow.excerpt_duration_sec;
    const fmtMS = (sec) => `${Math.floor(sec/60)}:${String(Math.floor(sec%60)).padStart(2,'0')}`;
    const tip  = `Audible region: ${fmtMS(es)} – ${fmtMS(ee)} `
               + `(player time = file time, identical)`;
    const startLine = document.createElement("div");
    startLine.className = "audible-line audible-line-start";
    startLine.style.left = `${(es / total) * 100}%`;
    startLine.title = `Music starts at ${fmtMS(es)} — ${tip}`;
    ribbon.appendChild(startLine);
    const endLine = document.createElement("div");
    endLine.className = "audible-line audible-line-end";
    endLine.style.left = `${(ee / total) * 100}%`;
    endLine.title = `Music ends at ${fmtMS(ee)} — ${tip}`;
    ribbon.appendChild(endLine);
    // Soft cyan tint between the two lines so the user can still
    // visually scan "this is the audible band" even though the two
    // lines, not the box, are the load-bearing alignment marks.
    const band = document.createElement("div");
    band.className = "audible-band";
    band.style.left  = `${(es / total) * 100}%`;
    band.style.width = `${(currentShow.excerpt_duration_sec / total) * 100}%`;
    band.title = tip;
    ribbon.appendChild(band);
    // Floating "♪ 25:53–26:23" label centered above the audible band.
    const winLabel = document.createElement("div");
    winLabel.className = "audible-window-label";
    const centerPct = ((es + currentShow.excerpt_duration_sec / 2) / total) * 100;
    winLabel.style.left = `${centerPct}%`;
    winLabel.textContent = `♪ ${fmtMS(es)}–${fmtMS(ee)}`;
    winLabel.title = tip;
    ribbon.appendChild(winLabel);
  }
}

// Build a static row template per posterior class. The set of classes
// comes from the payload's posterior_classes (PANN's 4-class projection:
// Performance, MC_Talk, Applause, Ambient). Pre_Concert and Intermission
// are heuristic-only so they intentionally never appear as bars.
// Six segment rows total, matching the CueSheet taxonomy. The first 4
// (Performance, MC_Talk, Applause, Ambient) get real PANN posterior
// values per second. The remaining 2 (Pre_Concert, Intermission) come
// from temporal heuristics — we render them so users can see all
// six classes, but we mark them with a "heuristic"
// badge and only light them up when the final segment_int label matches.
const SIX_SEGMENT_DISPLAY_ORDER = [
  "Performance", "MC_Talk", "Applause", "Ambient",
  "Pre_Concert", "Intermission",
];
const HEURISTIC_ONLY = new Set(["Pre_Concert", "Intermission"]);
// Shown on hover over the `heuristic` tag — explains why these two segments
// carry no encoder posterior and how the temporal rule assigns them.
const HEURISTIC_TIP = "Heuristic-only segment: no AudioSet class exists for "
  + "it, so the encoder emits no posterior. It is assigned by a temporal "
  + "rule over the encoder output — pre-concert is the lead-in before the "
  + "first sustained run of Performance; intermission is a low-activity "
  + "gap between Performance segments.";

function drawPosteriorRows() {
  if (!currentShow) return;
  posteriorList.innerHTML = "";
  for (const segmentName of SIX_SEGMENT_DISPLAY_ORDER) {
    const row = document.createElement("div");
    row.className = "posterior-row";
    row.dataset.segment = segmentName;
    const cls = SEGMENT_CLASS_MAP[segmentName] || "ambient";
    const isHeuristic = HEURISTIC_ONLY.has(segmentName);
    const label = (SEGMENT_DISPLAY[segmentName] || segmentName)
      + (isHeuristic
         ? ` <span class="heuristic-tag" title="${HEURISTIC_TIP}">heuristic</span>`
         : "");
    row.innerHTML = `
      <div class="pname">${label}</div>
      <div class="ptrack"><div class="pfill bg-${cls}"></div></div>
      <div class="pval">${isHeuristic ? "—" : "0.00"}</div>
    `;
    posteriorList.appendChild(row);
  }
}

function tick() {
  if (!currentShow || !mediaEl) {
    requestAnimationFrame(tick);
    return;
  }
  // Pin playback time to the media element if metadata has loaded;
  // otherwise treat the element as paused-at-0 and let URL ?t= /
  // streaming-seed state stand. We still keep ticking so a later
  // canplaythrough / seek picks up.
  const haveDuration = isFinite(mediaEl.duration) && mediaEl.duration > 0;
  if (!haveDuration) {
    requestAnimationFrame(tick);
    return;
  }
  const t = Math.floor(mediaEl.currentTime);
  if (t !== lastReportedSec) {
    lastReportedSec = t;
    updateWidgets(t);
    triggerFlowPulse();          // one data packet per new playback second
    window.__cuesheet_last_t = t;
    // Causal mode: the playhead just reached a new second, so ask the
    // backend to predict it (one on-demand call, no precompute) and
    // reveal one more ribbon cell — never ahead of "now".
    if (causalActive) {
      pumpCausal();
      try { drawRibbon(); } catch (e) {}
    }
  }
  const total = ribbonDuration();
  const frac = Math.min(1, mediaEl.currentTime / Math.max(total, 1));
  playhead.style.left = `${frac * 100}%`;
  // Keep the debug scrubber slider + readout in sync with whatever
  // drives currentTime — clicking the native audio/video seek bar
  // moves currentTime directly, so the slider would otherwise lag.
  if (scrubber) scrubber.value = String(frac * 100);
  if (scrubberReadout) scrubberReadout.textContent = `t = ${t} s`;
  requestAnimationFrame(tick);
}

const pipelineEl = document.getElementById("pipeline");

function setPipelineActive(active) {
  for (const el of pipeStages) {
    if (!el) continue;
    if (active) el.classList.add("active");
    else el.classList.remove("active");
  }
}

// Emit a single pipeline flow pulse — one packet of data travels left to
// right through the stages and disappears. Called from real data events
// (streaming row received, or playback advancing one second).
let lastPulseAt = 0;
function triggerFlowPulse() {
  const now = performance.now();
  // Throttle: at most one new pulse per 200 ms. PANN can emit ~10–20
  // rows per real second during streaming on CPU, but spawning that
  // many overlapping pulses just becomes visual noise.
  if (now - lastPulseAt < 200) return;
  lastPulseAt = now;
  for (const dot of document.querySelectorAll(".pipe-arrow .pipe-dot")) {
    // Force the CSS animation to restart by toggling the class.
    dot.classList.remove("pulse");
    void dot.offsetWidth;          // force a reflow so the next add re-runs the keyframes
    dot.classList.add("pulse");
  }
}

function drawAudiosetTop(top) {
  if (!audiosetTopEl) return;
  audiosetTopEl.innerHTML = "";
  if (!top || top.length === 0) {
    audiosetTopEl.innerHTML = '<div class="muted" style="font-size:13px">no data for this second</div>';
    return;
  }
  for (const item of top) {
    const row = document.createElement("div");
    row.className = "posterior-row";
    const p = Math.max(0, Math.min(1, item.p));
    row.innerHTML = `
      <div class="pname" title="${item.name}">${item.name}</div>
      <div class="ptrack"><div class="pfill" style="width:${(p*100).toFixed(1)}%;background:#6e7787;"></div></div>
      <div class="pval">${p.toFixed(2)}</div>
    `;
    audiosetTopEl.appendChild(row);
  }
}

function updateWidgets(t) {
  if (!currentShow) return;
  // Pick which data source the widgets read from. In causal mode, the
  // panel must reflect only the live overlay (causal*), never the
  // precomputed/streamed track in currentShow.*. In precomputed mode,
  // it is the reverse. Fall back to causalNext-1 (most recent live
  // second) when the requested second has not been computed yet, so
  // the panel does not flicker to "—" every second.
  let displayT = t;
  let listening = false;
  let srcSegment, srcPost, srcAudio;
  // Source priority must match drawRibbon: causal overlay wins when active,
  // otherwise the audio-derived precomputed/streamed track.
  if (causalActive) {
    srcSegment = causalSegmentInt;
    srcPost = causalPosterior;
    srcAudio = causalAudiosetTop;
  } else {
    srcSegment = currentShow.segment_int || [];
    srcPost = currentShow.posterior || [];
    srcAudio = currentShow.audioset_top || [];
  }
  const hereIsValid = (v) => v !== undefined && v !== null && v >= 0;
  const here = srcSegment[t];
  if (!hereIsValid(here)) {
    // No prediction yet for this second — fall back to the most recent
    // valid second so the panel does not flicker between a real reading
    // and gray "—" every time the playhead crosses the inference
    // frontier (mid-stream uploads, lagging causal pump, etc.).
    let fb = -1;
    const upper = Math.min(t, srcSegment.length - 1);
    for (let i = upper; i >= 0; i--) {
      if (hereIsValid(srcSegment[i])) { fb = i; break; }
    }
    if (fb >= 0) displayT = fb;
    else if (causalActive) listening = true;
  } else if (causalActive && t < causalStart) {
    // Playhead behind the causal switch-on point — refuse to show the
    // overlay's older entry from the previous session.
    listening = causalNext === causalStart;
    if (!listening) {
      let fb = -1;
      const upper = Math.min(srcSegment.length - 1, causalNext - 1);
      for (let i = upper; i >= causalStart; i--) {
        if (hereIsValid(srcSegment[i])) { fb = i; break; }
      }
      if (fb >= 0) displayT = fb;
      else listening = true;
    }
  }
  // Current segment (6-class, includes heuristic Pre_Concert/Intermission)
  let segmentName = "—";
  if (listening) {
    segmentNow.textContent = "(listening…)";
    // Distinct from segment-ambient so the gray (listening) badge cannot
    // be confused with the gray Ambient segment.
    segmentNow.className = "segment-now segment-listening";
  } else {
    const idx = srcSegment[displayT];
    if (idx !== undefined && idx >= 0
        && currentShow.segment_classes && currentShow.segment_classes[idx]) {
      segmentName = currentShow.segment_classes[idx];
    }
    segmentNow.textContent = SEGMENT_DISPLAY[segmentName] || segmentName;
    segmentNow.className = "segment-now segment-" + (SEGMENT_CLASS_MAP[segmentName] || "ambient");
  }
  timeNow.textContent = `t = ${t} s`;

  // Pipeline diagram: light up all stages while we have data for t,
  // dim them when we are past the inference frontier.
  setPipelineActive(!listening && srcSegment[displayT] !== undefined);
  if (pipeOutDetail) pipeOutDetail.textContent =
    listening ? "(listening…)" : (SEGMENT_DISPLAY[segmentName] || segmentName);

  // Encoder raw output (top-7 AudioSet classes).
  if (listening && audiosetTopEl) {
    const _encT = (typeof ENCODER_META !== "undefined" && ENCODER_META[currentEncoder])
      ? ENCODER_META[currentEncoder].title : "the encoder";
    audiosetTopEl.innerHTML = '<div class="muted" style="font-size:12px">'
      + `(listening — waiting for the first ${_encT} prediction…)` + '</div>';
  } else if (srcAudio[displayT] && srcAudio[displayT].length) {
    drawAudiosetTop(srcAudio[displayT]);
  } else if (audiosetTopEl) {
    audiosetTopEl.innerHTML = '<div class="muted" style="font-size:12px">'
      + '(no AudioSet top-7 data for this second)'
      + '</div>';
  }

  // Per-row update: PANN posterior for the 4 audio classes; heuristic-
  // active highlight for Pre_Concert / Intermission when the final
  // 6-class segment_int picks one of them.
  const postClasses = currentShow.posterior_classes
    || ["Performance", "MC_Talk", "Applause", "Ambient"];
  const row_t = listening ? null : srcPost[displayT];
  const activeSegment = (!listening && srcSegment[displayT] !== undefined
                       && srcSegment[displayT] >= 0)
    ? currentShow.segment_classes[srcSegment[displayT]] : null;
  for (const row of posteriorList.children) {
    const name = row.dataset.segment;
    const fill = row.querySelector(".pfill");
    const val = row.querySelector(".pval");
    if (HEURISTIC_ONLY.has(name)) {
      const isOn = activeSegment === name;
      fill.style.width = isOn ? "100%" : "0%";
      val.textContent = isOn ? "active" : "—";
    } else {
      const i = postClasses.indexOf(name);
      const p = (row_t && i >= 0) ? row_t[i] : 0;
      fill.style.width = `${Math.min(100, Math.max(0, p * 100))}%`;
      val.textContent = p.toFixed(2);
    }
  }
}

// Debug scrubber — lets the user verify bars animate without needing
// to hit Play on the media element. Slider value is interpreted as a
// fraction of the show's duration.
const scrubber = document.getElementById("scrubber");
const scrubberReadout = document.getElementById("scrubber-readout");
if (scrubber) {
  scrubber.addEventListener("input", (e) => {
    if (!currentShow) return;
    const frac = parseFloat(e.target.value) / 100;
    // Single time base: scrub fraction × ribbonDuration() gives the
    // audio file's currentTime. segment_int is indexed by that same
    // second number. Using n_seconds here (instead of duration) was
    // causing a ~9 s offset on shows where the mp3 is longer than
    // n_seconds — audio seeked to 1553 s while the widget read
    // segment_int[1544], so the bars and badge tracked the wrong
    // moment of the recording.
    const total = ribbonDuration();
    const t = Math.floor(frac * total);
    if (mediaEl && isFinite(mediaEl.duration)) {
      try { mediaEl.currentTime = frac * mediaEl.duration; } catch (_) {}
    }
    lastReportedSec = -1;            // force a refresh
    updateWidgets(t);
    playhead.style.left = `${frac * 100}%`;
    scrubberReadout.textContent = `t = ${t} s`;
  });
}

// ===== Inference mode: precomputed track vs causal-from-the-playhead =====
// Precomputed: replay the offline segment track stored with the show.
// Causal: predict second by second as the playhead reaches each second —
// no precompute, no peeking ahead. Each second t is one on-demand call to
// /api/infer_causal_step over a trailing window floored at the switch-on
// point, with the HMM forward state (alpha) carried client-side. A seek
// resets that state, so a rehearsal-style jump starts genuinely fresh:
// the audio just before the jump is never seen, exactly like deploying
// the online system at that instant.
const modePre = document.getElementById("mode-pre");
const modeCausal = document.getElementById("mode-causal");
let inferMode = "precomputed";
let currentEncoder = "efficientat";   // AudioSet backbone for causal inference
let causalActive = false;     // ribbon clips to [causalStart, computed frontier]
let causalStart = 0;          // second the causal system was switched on
let causalNext = 0;           // next second to compute (computed = [causalStart, causalNext))
let causalAlpha = null;       // HMM forward state, carried across seconds; null = fresh
let causalBusy = false;       // a per-second request is in flight
let causalGen = 0;            // generation token — bumped on every seek to drop stale work
let causalAbort = null;       // AbortController for the in-flight fetch
// Causal output lives in its own overlay arrays keyed by absolute second t,
// completely separate from the precomputed / streamed track held in
// currentShow.segment_int. Switching mode is then a free toggle — there is
// no snapshot to take, no precomputed array to mutate, and a still-
// streaming upload keeps writing to currentShow unaffected by anything
// causal is doing in parallel.
let causalSegmentInt = [];
let causalPosterior = [];
let causalAudiosetTop = [];

// Reset every causal/mode-toggle bit of state to "precomputed, off".
// Loading a gallery show, starting live mic, and starting an upload all
// share the same need: previous-session causal state and a stale mode
// toggle must not leak into the new source. Without this, e.g., a user
// who left causal mode active during a live-mic session would later see
// their upload's panel stuck reading from an empty causal overlay.
function resetToPrecomputed() {
  inferMode = "precomputed";
  causalActive = false;
  causalBusy = false;
  causalGen++;
  if (causalAbort) { try { causalAbort.abort(); } catch (e) {} causalAbort = null; }
  causalStart = 0;
  causalNext = 0;
  causalAlpha = null;
  causalSegmentInt = [];
  causalPosterior = [];
  causalAudiosetTop = [];
  if (typeof modePre !== "undefined" && modePre) modePre.classList.add("active");
  if (typeof modeCausal !== "undefined" && modeCausal) modeCausal.classList.remove("active");
}
const CAUSAL_POST6 = {0: 1, 1: 2, 2: 3, 3: 5};
const CAUSAL_WIN = 10;        // trailing window length (s), matches the encoder

// (Re)start causal inference at startSec: fresh HMM, fresh window floor,
// and the predictions from that second onward are wiped so nothing stale
// from a precomputed track or an earlier jump shows through.
function startCausalFrom(startSec) {
  if (!currentShow || !currentShow.show_id) return;
  // Accept gallery shows AND uploads (show_id starts with "upload:");
  // anything else (e.g., live mic) is rejected — there is no file path
  // to re-decode windows from.
  const isGallery = catalog.some(c => c.show_id === currentShow.show_id);
  const isUpload = String(currentShow.show_id).startsWith("upload:");
  if (!isGallery && !isUpload) {
    uploadStatus.textContent =
      "causal mode applies to gallery shows and uploads (not live mic)";
    return;
  }
  causalGen++;                  // invalidate any in-flight request
  if (causalAbort) { try { causalAbort.abort(); } catch (e) {} causalAbort = null; }
  causalActive = true;
  causalStart = startSec;
  causalNext = startSec;
  causalAlpha = null;           // no memory of audio before the jump
  causalBusy = false;
  // Reset the causal overlay — switching to causal is a fresh start, no
  // memory of any previous causal session and no peek at the precomputed
  // track held in currentShow.*.
  causalSegmentInt = [];
  causalPosterior = [];
  causalAudiosetTop = [];
  if (mediaEl && isFinite(mediaEl.currentTime)) {
    const t = Math.floor(mediaEl.currentTime);
    if (t >= startSec) updateWidgets(t);
  }
  uploadStatus.textContent =
    `causal: live from t = ${startSec} s — listening, no future, no precompute`;
  try { drawRibbon(); } catch (e) {}
}

// Compute one second at a time, never ahead of the playhead. Calls chain
// sequentially because each step's HMM alpha feeds the next; if the
// playhead has run ahead (e.g. during the first call's audio decode) the
// backlog is drained one request after another until it catches up.
async function pumpCausal() {
  if (!causalActive || causalBusy || !currentShow) return;
  const head = (lastReportedSec >= 0) ? lastReportedSec : causalStart;
  if (causalNext > head) return;            // nothing reached by the playhead yet
  causalBusy = true;
  const gen = causalGen;
  const t = causalNext;
  causalAbort = new AbortController();
  const sig = causalAbort.signal;
  try {
    const alphaParam = (causalAlpha && causalAlpha.length)
      ? `&alpha=${encodeURIComponent(causalAlpha.join(","))}` : "";
    const resp = await fetch(
      `/api/infer_causal_step?show=${encodeURIComponent(currentShow.show_id)}`
      + `&t=${t}&start=${causalStart}&encoder=${currentEncoder}${alphaParam}`,
      {signal: sig});
    let row;
    try {
      row = await resp.json();
    } catch (e) {
      // A non-JSON body means the route answered with an error page — in
      // practice the Space swapped to a freshly deployed container. The new
      // container has an empty upload registry, so the session cannot resume.
      uploadStatus.textContent = "⚠️ causal: the server was redeployed mid-session "
        + "— please re-upload the file to continue";
      causalActive = false; causalBusy = false;
      return;
    }
    if (gen !== causalGen) { causalBusy = false; return; }   // a seek happened — drop it
    if (row.error) {
      if (/Executor shutdown|not registered|No such file/.test(row.error)) {
        uploadStatus.textContent = "⚠️ causal: the server was redeployed mid-session "
          + "— please re-upload the file to continue";
        causalActive = false;
      } else {
        uploadStatus.textContent = "causal: " + row.error;
      }
      causalBusy = false;
      return;
    }
    causalPosterior[t] = row.posterior;
    causalAudiosetTop[t] = row.audioset_top || [];
    const idx = (row.segment_idx_smoothed !== undefined)
      ? row.segment_idx_smoothed : row.segment_idx;
    causalSegmentInt[t] = CAUSAL_POST6[idx] ?? 5;
    causalAlpha = row.alpha || null;
    causalNext = t + 1;
    // Status stays quiet while the frontier keeps up (rewriting it every
    // second made the picker row shake), but a heavy encoder on a small
    // CPU can fall behind the playhead — then silence reads as "broken",
    // so surface the lag explicitly, throttled to every other second.
    if (mediaEl && isFinite(mediaEl.currentTime)) {
      const lag = Math.floor(mediaEl.currentTime) - causalNext;
      if (lag >= 3 && t % 2 === 0 && !uploadChooserOpen) {
        const encName = (ENCODER_META[currentEncoder] || {}).title || currentEncoder;
        uploadStatus.textContent =
          `🐢 causal (${encName}): labeled to ${causalNext} s, playhead at `
          + `${Math.floor(mediaEl.currentTime)} s — this encoder runs slower `
          + `than real time on this CPU. ⏸ Pause to let it catch up.`;
      }
    }
    if (mediaEl && t === Math.floor(mediaEl.currentTime || 0)) updateWidgets(t);
    else updateWidgets(Math.floor(mediaEl ? mediaEl.currentTime : t));
    try { drawRibbon(); } catch (e) {}
    triggerFlowPulse();          // visible "live" cue — pipeline dots pulse
                                 // every time a fresh causal prediction lands
  } catch (e) {
    if (e && e.name === "AbortError") {
      // seek/mode-change canceled this fetch — silently drop
    } else {
      uploadStatus.textContent = "causal error: " + e;
    }
  } finally {
    // Only release the busy flag if no fresher generation has started
    // its own request; otherwise an aborted old pump would clobber the
    // new pump's causalBusy=true and let a duplicate request fire.
    if (gen === causalGen) causalBusy = false;
  }
  // Keep draining while the playhead is still ahead of the frontier.
  if (causalActive && gen === causalGen) pumpCausal();
}

function setInferMode(m) {
  inferMode = m;
  if (modePre) modePre.classList.toggle("active", m === "precomputed");
  if (modeCausal) modeCausal.classList.toggle("active", m === "causal");
  if (m === "precomputed") {
    causalGen++;                // cancel any in-flight causal request
    if (causalAbort) { try { causalAbort.abort(); } catch (e) {} causalAbort = null; }
    causalActive = false;
    causalBusy = false;
    // The causal overlay is kept around for the lifetime of the show
    // in case the user toggles back; it does not affect what precomputed
    // mode displays, which reads only from currentShow.*.
    if (!uploadChooserOpen) uploadStatus.textContent = "";
    // Force the panel to refresh from the precomputed track immediately.
    if (mediaEl && isFinite(mediaEl.currentTime)) {
      updateWidgets(Math.floor(mediaEl.currentTime));
    }
    try { drawRibbon(); } catch (e) {}
  } else {
    const t = (mediaEl && isFinite(mediaEl.currentTime))
      ? Math.floor(mediaEl.currentTime) : 0;
    startCausalFrom(t);
    pumpCausal();
  }
  refreshScrubberVisibility();    // causalActive may have flipped either way
}
if (modePre) modePre.addEventListener("click", () => setInferMode("precomputed"));
if (modeCausal) modeCausal.addEventListener("click", () => setInferMode("causal"));

function onCausalSeek() {
  if (inferMode !== "causal") return;
  // No debounce: each seek must restart causal immediately so the user
  // sees the panels wipe and the new prediction land within ~one frame.
  // Rapid drag-style seeks coalesce naturally because startCausalFrom
  // bumps causalGen and aborts any in-flight request.
  const t = (mediaEl && isFinite(mediaEl.currentTime))
    ? Math.floor(mediaEl.currentTime) : 0;
  startCausalFrom(t);
  pumpCausal();
}
// "seeking" fires the instant currentTime changes (before the audio
// element actually buffers the new position), so we wipe the panels
// immediately on click; "seeked" runs again when buffering is done in
// case the user landed slightly off the click target.
audioEl.addEventListener("seeking", onCausalSeek);
audioEl.addEventListener("seeked", onCausalSeek);
videoEl.addEventListener("seeking", onCausalSeek);
videoEl.addEventListener("seeked", onCausalSeek);

// ===== Encoder switcher: the EfficientAT MN10 pipeline box is clickable =====
// All four encoders emit 527-class AudioSet posteriors into the same group
// mapping, so swapping the backbone is a drop-in change. Switching while
// causal mode is live restarts the prediction from the playhead with the
// new encoder; the precomputed gallery tracks are unaffected (they were
// computed offline with EfficientAT).
const ENCODER_META = {
  efficientat: {title: "EfficientAT MN10", detail: "5 M params · 527-class AudioSet",
                window: "10 s @ 32 kHz mono"},
  htsat:       {title: "HTS-AT",           detail: "31 M params · 527-class AudioSet",
                window: "10 s @ 32 kHz mono"},
  pann:        {title: "PANN CNN14",       detail: "81 M params · 527-class AudioSet",
                window: "2 s @ 16 kHz mono"},
  ast:         {title: "AST",              detail: "87 M params · 527-class AudioSet",
                window: "10 s @ 16 kHz mono"},
};
const pipeSwitch = document.getElementById("pipe-pann");
const encoderMenu = document.getElementById("encoder-menu");
const encTitleEl = document.getElementById("pipe-enc-title");
const encDetailEl = document.getElementById("pipe-enc-detail");
const audioDetailEl = document.getElementById("pipe-audio-detail");

let ENCODER_SCORES = {};   // {efficientat: {pooled_wf1_pct: 93.9, ...}, ...}

async function loadEncoderScores() {
  try {
    const r = await fetch("data/encoder_scores.json");
    if (r.ok) ENCODER_SCORES = await r.json();
  } catch (e) {}
  renderEncoder();
}

function _scoreSuffix(name) {
  const s = ENCODER_SCORES[name];
  return s && s.pooled_wf1_pct != null ? ` · wF1 ${s.pooled_wf1_pct}%` : "";
}

function renderEncoder() {
  const m = ENCODER_META[currentEncoder] || ENCODER_META.efficientat;
  if (encTitleEl) encTitleEl.innerHTML =
    `${m.title}<span class="pipe-caret">&#9662;</span>`;
  if (encDetailEl) encDetailEl.textContent = m.detail + _scoreSuffix(currentEncoder);
  if (audioDetailEl) audioDetailEl.textContent = m.window;
  if (encoderMenu) {
    for (const b of encoderMenu.querySelectorAll(".encoder-opt")) {
      b.classList.toggle("active", b.dataset.enc === currentEncoder);
      // Append wF1 to the span detail line if not already there.
      const span = b.querySelector("span");
      if (span && !span.dataset.baseText) span.dataset.baseText = span.textContent;
      if (span) span.textContent = span.dataset.baseText + _scoreSuffix(b.dataset.enc);
    }
  }
  _renderShowWf1();
}

function closeEncoderMenu() {
  if (encoderMenu) encoderMenu.hidden = true;
}

function setEncoder(name) {
  if (!ENCODER_META[name] || name === currentEncoder) { closeEncoderMenu(); return; }
  currentEncoder = name;
  renderEncoder();
  closeEncoderMenu();
  // If causal mode is live, recompute from the playhead so the switch is
  // visible immediately; otherwise it just takes effect on the next run.
  if (causalActive) {
    const t = (mediaEl && isFinite(mediaEl.currentTime))
      ? Math.floor(mediaEl.currentTime) : 0;
    startCausalFrom(t);
    pumpCausal();
  } else if (currentShow
             && String(currentShow.show_id).startsWith("upload:")) {
    // Same choice as at upload time: recompute this upload with the new
    // encoder either causally from the playhead (interactive, per-second)
    // or as a fresh offline pass over the whole file (needs the original
    // file, which we keep in memory from the upload).
    uploadChooserOpen = true;
    uploadStatus.innerHTML = "";
    const note = document.createElement("span");
    note.textContent = `recompute with ${ENCODER_META[name].title}: `;
    uploadStatus.appendChild(note);
    const mkBtn = (label, title, fn, primary) => {
      const b = document.createElement("button");
      b.type = "button"; b.className = "hdr-btn"; b.style.margin = "0 6px";
      if (primary) b.style.color = "var(--accent)";
      b.textContent = label; b.title = title;
      b.addEventListener("click", fn);
      return b;
    };
    uploadStatus.appendChild(mkBtn("Causal from playhead",
      "Per-second trailing windows from the current position — interactive, no look-ahead.",
      () => { uploadChooserOpen = false; uploadStatus.textContent = ""; setInferMode("causal"); }, true));
    if (lastUploadedFile) {
      uploadStatus.appendChild(mkBtn("Offline (whole file)",
        "One batched pass over the full file with the new encoder — fastest, sees ahead.",
        async () => {
          uploadChooserOpen = false;
          uploadCausalChoice = "0";
          uploadStatus.textContent = "";
          await runUpload(lastUploadedFile, "audio");
        }, false));
    }
  } else if (currentShow && currentShow.show_id
             && catalog.some((e) => e.show_id === currentShow.show_id)) {
    // Membership in the loaded catalog is the gallery test. Upload sessions
    // carry upload_<ts> or upload:<id> ids and live sessions carry "live";
    // none of those are in the catalog, so they fall through to the
    // applies-on-next-run message instead of a doomed JSON fetch.
    // Gallery mode: reload the show with the encoder-specific JSON so the ribbon
    // and per-second card reflect the new encoder's output. Upload and live
    // sessions have no per-encoder JSON on disk, so reloading would 404 and
    // wipe the UI; for those the new encoder simply applies to the next run.
    const t = (mediaEl && isFinite(mediaEl.currentTime))
      ? mediaEl.currentTime : 0;
    loadShow(currentShow.show_id).then(() => {
      if (mediaEl) mediaEl.currentTime = t;
    });
  } else {
    uploadStatus.textContent =
      `encoder set to ${ENCODER_META[name].title} — applies to gallery and uploads (live mic always runs EfficientAT MN10)`;
  }
}

if (pipeSwitch && encoderMenu) {
  pipeSwitch.addEventListener("click", (e) => {
    if (e.target.closest(".encoder-opt")) return;   // handled below
    encoderMenu.hidden = !encoderMenu.hidden;
  });
  for (const b of encoderMenu.querySelectorAll(".encoder-opt")) {
    b.addEventListener("click", (e) => {
      e.stopPropagation();
      setEncoder(b.dataset.enc);
    });
  }
  document.addEventListener("click", (e) => {
    if (!encoderMenu.hidden && !pipeSwitch.contains(e.target)) closeEncoderMenu();
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") closeEncoderMenu();
  });
  renderEncoder();
  loadEncoderScores();
}

showSelect.addEventListener("change", (e) => {
  // Navigating away abandons whatever was in progress: an open chooser,
  // and a live recording (otherwise the mic keeps streaming and its rows
  // land inside the newly loaded show's arrays).
  dismissUploadChooser();
  if (liveOn) stopLive();
  loadShow(e.target.value);
});

const MAX_UPLOAD_MB = 5000;
const MAX_INFER_SEC = 300;   // matches server-side MAX_INFER_SEC

// Remember the last uploaded file so the inline mode-picker can re-process
// it without forcing the user to re-pick from disk.
let lastUploadedFile = null;
let streamDrawPending = false;
let uploadTicker = null;        // the "(Ns elapsed)" interval — one at a time
function stopUploadTicker() {
  if (uploadTicker) { clearInterval(uploadTicker); uploadTicker = null; }
}
let currentUploadXHR = null;

uploadInput.addEventListener("change", async (e) => {
  dismissUploadChooser();
  if (liveOn) stopLive();
  const file = e.target.files[0];
  if (!file) return;
  if (file.size > MAX_UPLOAD_MB * 1024 * 1024) {
    uploadStatus.textContent = `File too large: ${(file.size / 1024 / 1024).toFixed(1)} MB exceeds ${MAX_UPLOAD_MB} MB cap.`;
    return;
  }
  const isVideo = file.type.startsWith("video/");
  const uploadMode = "audio";
  console.log("[CueSheet] upload start",
              {file: file.name, isVideo, mode: uploadMode,
               sizeMB: (file.size / 1024 / 1024).toFixed(1)});
  lastUploadedFile = file;
  _askUploadProcessing(file, uploadMode);
  e.target.value = "";
});

// Upload-time processing chooser: the same audio can be labeled two ways,
// so make the user pick instead of silently defaulting. Causal mirrors a
// live deployment (trailing windows, no look-ahead, 10 s head cold-start);
// Offline labels each second from the window starting there (sees ahead,
// fastest, the alignment the stored gallery tracks use).
let uploadCausalChoice = "1";
let uploadChooserOpen = false;   // guards housekeeping wipes of the status row
function dismissUploadChooser() {
  // Explicit user navigation (selecting a show, starting live capture,
  // picking another file) closes an open chooser; only background fetch
  // completions are barred from wiping it.
  stopUploadTicker();
  if (uploadChooserOpen) {
    uploadChooserOpen = false;
    uploadStatus.textContent = "";
  }
}
function _askUploadProcessing(file, uploadMode) {
  uploadChooserOpen = true;
  uploadStatus.innerHTML = "";
  const note = document.createElement("span");
  note.textContent = `process "${file.name}" as: `;
  const mk = (label, title, flag, primary) => {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "hdr-btn";
    b.style.margin = "0 6px";
    if (primary) b.style.color = "var(--accent)";
    b.textContent = label;
    b.title = title;
    b.addEventListener("click", async () => {
      uploadChooserOpen = false;
      uploadCausalChoice = flag;
      uploadStatus.textContent = "";
      await runUpload(file, uploadMode);
    });
    return b;
  };
  uploadStatus.appendChild(note);
  uploadStatus.appendChild(mk("Causal (live pace)",
    "Press play and the pipeline computes one second at a time as the audio arrives, exactly like a live deployment: trailing windows, no future, decode on demand. Pausing pauses computation.",
    "live", true));
  uploadStatus.appendChild(mk("Offline (full track now)",
    "One pass over the whole file up front. Each second is labeled from the 10 s window starting there — uses future audio, same alignment as the stored gallery tracks.",
    "0", false));
}

async function runUpload(file, uploadMode) {
  const isVideo = file.type.startsWith("video/");
  // Abort any in-flight audio stream before restarting.
  if (currentUploadXHR) { try { currentUploadXHR.abort(); } catch (e) {} currentUploadXHR = null; }

  // Clear the previously-selected show immediately. Without this the
  // playhead on the newly-dropped media element keeps indexing into the
  // previous show's segment_int / posterior arrays, so the old gallery
  // show's predictions appear to "follow" the new video.
  currentShow = null;
  lastReportedSec = -1;
  ribbon.innerHTML = "";
  posteriorList.innerHTML = "";
  segmentNow.textContent = "(running inference…)";
  segmentNow.className = "segment-now segment-ambient";
  timeNow.textContent = "t = 0 s";
  genreBadge.textContent = isVideo ? "video upload" : "audio upload";

  // Clear the show dropdown selection so it's visually obvious we're
  // no longer on a preloaded gallery entry.
  showSelect.value = "";

  uploadStatus.textContent = `uploading ${file.name} (${(file.size / 1024 / 1024).toFixed(1)} MB) — running the pipeline (CPU encoder, ~real-time, please wait)…`;

  // Display the file locally so the user can see it while the backend
  // grinds, but pause playback — playing now would scroll an empty
  // playhead over an empty ribbon, which is just confusing.
  const blobUrl = URL.createObjectURL(file);
  if (isVideo) {
    useMediaElement("video");
    videoEl.src = blobUrl;
    videoEl.load();
    videoEl.pause();
  } else {
    useMediaElement("audio");
    audioEl.src = blobUrl;
    audioEl.load();
    audioEl.pause();
  }

  // === Streaming upload via XHR (fetch does not expose upload progress
  // for the file body). XHR's readyState===3 gives us partial response
  // as the server emits NDJSON lines, which we parse incrementally.
  const t0 = performance.now();

  // Reset any leftover causal / mode state from a previous session
  // (e.g., a live-mic run that left causalActive=true would otherwise
  // bind the panel to an empty causal overlay for this upload too).
  resetToPrecomputed();
  // Seed an empty show payload that streaming rows will append into.
  hideExcerptUI();
  if (modePre) {
    modePre.textContent = (uploadCausalChoice === "0")
      ? "Replay · offline track" : "Precomputed";
  }
  currentShow = {
    show_id: `upload_${Date.now()}`,
    label: file.name,
    genre: isVideo ? "video upload" : "audio upload",
    n_seconds: 0,
    segment_int: [],
    segment_classes: ["Pre_Concert", "Performance", "MC_Talk", "Applause",
                    "Intermission", "Ambient"],
    posterior: [],
    posterior_classes: ["Performance", "MC_Talk", "Applause", "Ambient"],
    audioset_top: [],
  };
  drawPosteriorRows();

  const POST_TO_SEGMENT_IDX = {0: 1, 1: 2, 2: 3, 3: 5};
  // Mirror of cuesheet.bootstrap_labels.apply_pre_concert_heuristic:
  // once we have observed ≥ MIN_ACTIVE_RUN consecutive Performance windows,
  // back-fill everything before the run's start as Pre_Concert.
  const MIN_ACTIVE_RUN = 10;
  let runLen = 0;
  let showStart = null;

  const xhr = new XMLHttpRequest();
  xhr.open("POST", "/api/infer_stream", true);
  xhr.responseType = "";  // text
  const sizeMB = (file.size / 1024 / 1024).toFixed(1);
  let uploadDone = false;
  let lastIndex = 0;
  let nRows = 0;

  function parseChunk() {
    const text = xhr.responseText;
    if (text.length <= lastIndex) return;
    const slice = text.slice(lastIndex);
    lastIndex = text.length;
    const lines = slice.split("\n");
    // Last segment may be partial; only process up to len-1, then re-include trailing fragment.
    for (let i = 0; i < lines.length - 1; i++) {
      const line = lines[i].trim();
      if (!line) continue;
      let evt;
      try { evt = JSON.parse(line); }
      catch (e) { console.warn("[CueSheet] bad NDJSON:", line); continue; }
      handleEvent(evt);
    }
    // Pull the dangling partial back so next chunk completes it.
    if (lines.length > 0) {
      const last = lines[lines.length - 1];
      if (last !== "") lastIndex -= last.length;
    }
  }

  function handleEvent(evt) {
    if (evt.meta) {
      console.log("[CueSheet] stream meta", evt.meta);
      uploadStatus.textContent = `upload complete · decoding audio + running the encoder…`;
      // The server stamps an upload_id into the meta event so the
      // upload's audio file stays addressable for /api/infer_causal_step.
      // Rebrand the show_id under the "upload:" namespace and pre-warm
      // the causal cache so the user can immediately switch to causal-
      // from-playhead with the same low latency as a gallery show.
      if (currentShow && evt.meta.upload_id) {
        currentShow.show_id = `upload:${evt.meta.upload_id}`;
        fetch(`/api/causal_warm?show=${encodeURIComponent(currentShow.show_id)}`,
              {method: "POST"}).catch(() => {});
      }
      // Light up the pipeline while inference is in flight so the flowing
      // dots make 'something is happening' obvious even before the user
      // hits Play.
      if (pipelineEl) pipelineEl.classList.add("flowing");
      for (const el of pipeStages) if (el) el.classList.add("active");
      return;
    }
    if (evt.progress) {
      const elapsed = ((performance.now() - t0) / 1000).toFixed(0);
      uploadStatus.textContent = `[${elapsed}s] ${evt.message || evt.progress}`;
      return;
    }
    if (evt.error) {
      segmentNow.textContent = "(decode error)";
      uploadStatus.textContent = `server: ${evt.error}`;
      console.error("[CueSheet] stream error event:", evt.error);
      return;
    }
    if (evt.done) {
      stopUploadTicker();
      const dt = (performance.now() - t0) / 1000;
      if (evt.registered) {
        // Live-pace causal upload: the file is registered server-side and
        // nothing is precomputed. Enter causal-from-playhead mode; pressing
        // play simulates the audio arriving from that moment.
        setInferMode("causal");
        segmentNow.textContent = "—";
        uploadStatus.textContent =
          "▶️ causal (live pace): press Play — each second is computed as the audio arrives";
        return;
      }
      if (!evt.n_seconds || evt.n_seconds === 0) {
        segmentNow.textContent = "(no audio detected)";
        uploadStatus.textContent = `inference finished in ${dt.toFixed(1)} s but the file produced 0 seconds of audio — check that the file contains a decodable audio track.`;
      } else {
        const rtf = (evt.timing_sec / evt.n_seconds).toFixed(3);
        uploadStatus.textContent = `streamed ${evt.n_seconds} s in ${dt.toFixed(1)} s wall (~${rtf} s per second of audio · press Play)`;
        segmentNow.textContent = "—";
      }
      console.log("[CueSheet] stream done", evt);
      try { drawRibbon(); }
      catch (e) { console.error("[CueSheet] final drawRibbon failed:", e); }
      return;
    }
    if (typeof evt.t === "number" && Array.isArray(evt.posterior)) {
      currentShow.posterior.push(evt.posterior);
      currentShow.audioset_top.push(evt.audioset_top || []);
      // Prefer HMM-smoothed labels for the ribbon + final 6-class segment so
      // segments do not flicker frame-by-frame; the raw posterior bars
      // still display the unsmoothed PANN/EfficientAT softmax above.
      const labelIdx = (evt.segment_idx_smoothed !== undefined)
        ? evt.segment_idx_smoothed : evt.segment_idx;
      const segmentIdx6 = POST_TO_SEGMENT_IDX[labelIdx] ?? -1;
      currentShow.segment_int.push(segmentIdx6);
      currentShow.n_seconds = currentShow.posterior.length;
      // Pre_Concert streaming heuristic: track consecutive Performance runs
      // and once we cross MIN_ACTIVE_RUN, back-fill the prefix as Pre_Concert.
      // segment index 1 == Performance in the 6-class scheme.
      if (showStart === null) {
        if (segmentIdx6 === 1) {
          runLen++;
          if (runLen >= MIN_ACTIVE_RUN) {
            showStart = currentShow.segment_int.length - runLen;
            for (let k = 0; k < showStart; k++) currentShow.segment_int[k] = 0;
          }
        } else {
          runLen = 0;
        }
      }
      nRows++;
      // Redraw at most once per animation frame. The heavy encoders' offline
      // path streams every row in one burst, so a per-N-rows redraw rebuilt
      // the full ribbon DOM hundreds of times inside a single progress event
      // and froze the page ("Page unresponsive"). A single-flight rAF keeps
      // data ingestion cheap and pays for exactly one redraw per frame.
      if (!streamDrawPending) {
        streamDrawPending = true;
        requestAnimationFrame(() => {
          streamDrawPending = false;
          try { drawRibbon(); }
          catch (e) { console.error("[CueSheet] drawRibbon failed:", e); }
          const dur = mediaEl && isFinite(mediaEl.duration) && mediaEl.duration > 0
            ? mediaEl.duration : null;
          const pct = dur ? ` (${(nRows / dur * 100).toFixed(0)}% of clip)` : "";
          uploadStatus.textContent = `streaming inference… ${nRows} s inferred${pct}`;
        });
      }
    }
  }

  xhr.upload.onprogress = (e) => {
    if (e.lengthComputable && !uploadDone) {
      const pct = (e.loaded / e.total * 100).toFixed(0);
      const sentMB = (e.loaded / 1024 / 1024).toFixed(1);
      uploadStatus.textContent = `uploading ${file.name} … ${pct}% (${sentMB}/${sizeMB} MB)`;
    }
  };
  xhr.upload.onload = () => {
    uploadDone = true;
    const t1 = performance.now();
    uploadStatus.textContent = `upload complete (${sizeMB} MB) · server picking it up…`;
    // Belt-and-suspenders elapsed-time ticker: if the server is in a
    // slow state that doesn't yield a progress event for a while
    // (large videos can sit in ffmpeg for tens of seconds), at least
    // show the elapsed second count so the user knows the page is not
    // frozen. Cleared as soon as the first progress / row / done event
    // arrives.
    stopUploadTicker();
    uploadTicker = setInterval(() => {
      if (nRows > 0) { stopUploadTicker(); return; }
      // Rewriting textContent would destroy the chooser's buttons (child
      // elements), so never touch the row while a chooser is open.
      if (uploadChooserOpen) return;
      const sec = ((performance.now() - t1) / 1000).toFixed(0);
      const txt = uploadStatus.textContent || "";
      if (!txt.match(/\(\d+s elapsed\)/)) {
        uploadStatus.textContent = `${txt} (${sec}s elapsed)`;
      } else {
        uploadStatus.textContent = txt.replace(/\(\d+s elapsed\)/, `(${sec}s elapsed)`);
      }
    }, 1000);
  };
  xhr.onreadystatechange = () => {
    if (xhr.readyState === 3 || xhr.readyState === 4) parseChunk();
  };
  xhr.onerror = () => {
    stopUploadTicker();
    segmentNow.textContent = "(network error)";
    uploadStatus.textContent = `network error during upload — check the server is up.`;
    console.error("[CueSheet] xhr.onerror");
  };
  xhr.onabort = () => {
    stopUploadTicker();
    segmentNow.textContent = "(aborted)";
    uploadStatus.textContent = `upload aborted.`;
  };
  xhr.onload = () => {
    if (xhr.status >= 400) {
      let detail = "";
      try { detail = JSON.parse(xhr.responseText).detail || ""; } catch (_) {}
      segmentNow.textContent = `(error ${xhr.status})`;
      uploadStatus.textContent = `server returned ${xhr.status}: ${detail}`;
      console.error("[CueSheet] xhr error", xhr.status, detail);
    }
  };
  const fd = new FormData();
  fd.append("file", file);
  fd.append("causal", uploadCausalChoice === "live" ? "1" : uploadCausalChoice);
  fd.append("encoder", currentEncoder);
  fd.append("register_only", uploadCausalChoice === "live" ? "1" : "0");
  currentUploadXHR = xhr;
  xhr.send(fd);
}

// Expose live state for DevTools-console debugging. If the user reports
// "still broken", these let us see what the page actually has in memory.
window.__cuesheet = {
  get currentShow() { return currentShow; },
  get lastReportedSec() { return lastReportedSec; },
  get mediaEl() { return mediaEl; },
  get audioEl() { return audioEl; },
  get videoEl() { return videoEl; },
};

// ===== Real-time microphone input =========================================
// Capture the mic with WebAudio, stream raw float32 PCM over a WebSocket to
// /api/infer_live, and render each per-second prediction through the same
// widgets the gallery and upload paths use. The pipeline is online and
// causal, so the live segment is a true real-time read rather than a replay.
const LIVE_POST_TO_SEGMENT6 = {0: 1, 1: 2, 2: 3, 3: 5};  // 4-class -> 6-class
let liveCtx = null, liveStream = null, liveWS = null, liveProc = null;
let liveAnalyser = null, liveDrawRAF = 0;
let liveChunks = [];          // Float32Array per onaudioprocess for WAV playback
let liveChunkSamples = 0;
let liveInputSr = 0;
let liveOn = false;
const LIVE_MAX_SAMPLES = 48_000 * 600;   // 10 min @ 48 kHz, soft cap on memory

function drawLiveWaveform() {
  if (!liveAnalyser || !liveWaveformEl) return;
  const dpr = window.devicePixelRatio || 1;
  const w = liveWaveformEl.clientWidth;
  const h = liveWaveformEl.clientHeight;
  if (liveWaveformEl.width !== w * dpr) liveWaveformEl.width = w * dpr;
  if (liveWaveformEl.height !== h * dpr) liveWaveformEl.height = h * dpr;
  const ctx = liveWaveformEl.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  // zero line
  ctx.strokeStyle = "#2c3142"; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(0, h / 2); ctx.lineTo(w, h / 2); ctx.stroke();
  const N = liveAnalyser.fftSize;
  const buf = new Float32Array(N);
  liveAnalyser.getFloatTimeDomainData(buf);
  ctx.strokeStyle = "#ff7d3b"; ctx.lineWidth = 1.5;
  ctx.beginPath();
  for (let i = 0; i < w; i++) {
    const v = buf[Math.floor(i / w * N)] || 0;   // clamp [-1,1]
    const y = h / 2 - Math.max(-1, Math.min(1, v)) * (h / 2 - 2);
    if (i === 0) ctx.moveTo(i, y); else ctx.lineTo(i, y);
  }
  ctx.stroke();
  liveDrawRAF = requestAnimationFrame(drawLiveWaveform);
}

// Encode mono Float32 PCM to a 16-bit PCM WAV Blob in-browser.
function float32ToWavBlob(samples, sampleRate) {
  const n = samples.length;
  const buf = new ArrayBuffer(44 + n * 2);
  const v = new DataView(buf);
  const ws = (off, s) => { for (let i = 0; i < s.length; i++) v.setUint8(off + i, s.charCodeAt(i)); };
  ws(0, "RIFF"); v.setUint32(4, 36 + n * 2, true); ws(8, "WAVE");
  ws(12, "fmt "); v.setUint32(16, 16, true);
  v.setUint16(20, 1, true);          // PCM
  v.setUint16(22, 1, true);          // mono
  v.setUint32(24, sampleRate, true);
  v.setUint32(28, sampleRate * 2, true);
  v.setUint16(32, 2, true); v.setUint16(34, 16, true);
  ws(36, "data"); v.setUint32(40, n * 2, true);
  for (let i = 0; i < n; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    v.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return new Blob([buf], {type: "audio/wav"});
}


function hideExcerptUI() {
  const hintEl = document.getElementById("audible-hint");
  if (hintEl) hintEl.hidden = true;
  const noteEl = document.getElementById("excerpt-note");
  if (noteEl) noteEl.hidden = true;
  // The per-show wF1 readout describes a gallery show measured against its
  // hand labels; uploads and live capture have no ground truth, so clear it.
  const wf1El = document.getElementById("show-wf1");
  if (wf1El) wf1El.textContent = "";
}

function ingestLiveRow(row) {
  if (!currentShow) return;
  currentShow.posterior.push(row.posterior);
  currentShow.audioset_top.push(row.audioset_top || []);
  const labelIdx = (row.segment_idx_smoothed !== undefined)
    ? row.segment_idx_smoothed : row.segment_idx;
  currentShow.segment_int.push(LIVE_POST_TO_SEGMENT6[labelIdx] ?? 5);
  currentShow.n_seconds = currentShow.posterior.length;
  updateWidgets(currentShow.n_seconds - 1);
  try { drawRibbon(); }
  catch (e) { console.error("[CueSheet] live drawRibbon failed:", e); }
  triggerFlowPulse();
  playhead.style.left = "100%";
}

// Recording-time readout state. liveStartMs is wall-clock ms when the
// current capture began; liveTimerId is the setInterval that paints the
// record button's "■ Stop · m:ss" label; liveActiveBtn is the stop control.
let liveStartMs = 0;
let liveTimerId = 0;
let liveActiveBtn = null;

// Idle label for the live record button. In MM mode it reads as a generic
// record control ("● Record" / 録音 / 녹음).
function idleLiveBtnLabel(b) {
  if (APP_MODE === "mm") return t("live.record");
  return "● " + t("pick.live.audio");
}

// Debug scrubber is a replay-only affordance: hide it whenever the source
// is being treated as a stream (live capture OR causal-from-playhead),
// because neither has a "scrub through this clip" semantic — both are
// strictly per-second forward inference from the current playhead.
function refreshScrubberVisibility() {
  const scrub = document.querySelector(".scrubber-wrap");
  if (!scrub) return;
  scrub.hidden = !!(liveOn || causalActive);
}

async function startLive() {
  dismissUploadChooser();
  hideExcerptUI();
  // The Precomputed/Causal toggle controls gallery and upload replay; a live
  // session is intrinsically causal and has no stored track to replay, so
  // showing an active "Precomputed" there would be wrong. Hide it while live.
  const modepick = document.querySelector(".modepick");
  if (modepick) modepick.hidden = true;
  // Mic/camera capture needs a secure context — HTTPS, or http://localhost.
  // Plain HTTP from a LAN IP is blocked by browsers — they silently block getUserMedia
  // there. Give them the actionable diagnosis up front instead of a
  // generic "permission denied".
  if (!window.isSecureContext) {
    uploadStatus.textContent =
      "live capture needs HTTPS or http://localhost. "
      + "Try SSH-tunneling (ssh -L 8001:localhost:8001 <host>) and "
      + "opening http://localhost:8001/cuesheet, or enable the current origin "
      + "in chrome://flags/#unsafely-treat-insecure-origin-as-secure.";
    return;
  }
  try {
    liveStream = await navigator.mediaDevices.getUserMedia({audio: true});
  } catch (e) {
    uploadStatus.textContent = `live capture blocked: ${e && e.name ? e.name : "unknown"}`
      + " — check the mic permissions for this site in the browser address bar.";
    return;
  }
  liveOn = true;
  refreshScrubberVisibility();
  // Mark the record button as recording with a live elapsed-time readout
  // ("■ Stop · 0:08") so the user always sees how long they've been capturing.
  liveStartMs = Date.now();
  if (liveBtn) {
    liveActiveBtn = liveBtn;
    liveBtn.textContent = t("live.stop") + " · 0:00";
    liveBtn.classList.add("recording");
  }
  if (liveTimerId) clearInterval(liveTimerId);
  liveTimerId = setInterval(() => {
    if (!liveActiveBtn || !liveOn) return;
    const sec = Math.floor((Date.now() - liveStartMs) / 1000);
    const mm = Math.floor(sec / 60);
    const ss = String(sec % 60).padStart(2, "0");
    liveActiveBtn.textContent = t("live.stop") + " · " + mm + ":" + ss;
  }, 500);

  // Reset any leftover causal / mode state — live capture doesn't use
  // causal mode at all, and stale causalActive from a previous gallery
  // session would otherwise hide the live panel readouts.
  resetToPrecomputed();
  // Fresh growing payload — same shape the upload path seeds.
  currentShow = {
    show_id: "live", label: "live microphone", genre: "live input",
    n_seconds: 0, segment_int: [],
    segment_classes: ["Pre_Concert", "Performance", "MC_Talk", "Applause",
                    "Intermission", "Ambient"],
    posterior: [],
    posterior_classes: ["Performance", "MC_Talk", "Applause", "Ambient"],
    audioset_top: [],
  };
  showSelect.value = "";
  genreBadge.textContent = "live input";
  ribbon.innerHTML = "";
  drawPosteriorRows();
  segmentNow.textContent = "(listening…)";
  segmentNow.className = "segment-now segment-listening";
  timeNow.textContent = "t = 0 s";
  // No media element drives the playhead in live mode: select the audio
  // element and clear it so tick() sees no duration and stays idle.
  useMediaElement("audio");
  audioEl.pause();
  audioEl.removeAttribute("src");
  audioEl.load();
  // Swap the audio element for the live waveform / hide any prior
  // recording playback element.
  if (audioEl) audioEl.style.display = "none";
  if (livePlaybackEl) {
    livePlaybackEl.hidden = true;
    if (livePlaybackEl.src) {
      try { URL.revokeObjectURL(livePlaybackEl.src); } catch (e) {}
      livePlaybackEl.removeAttribute("src");
    }
  }
  if (liveWaveformEl) liveWaveformEl.hidden = false;
  playhead.style.left = "100%";

  liveChunks = []; liveChunkSamples = 0;

  liveCtx = new (window.AudioContext || window.webkitAudioContext)();
  liveInputSr = liveCtx.sampleRate;
  const src = liveCtx.createMediaStreamSource(liveStream);
  liveAnalyser = liveCtx.createAnalyser();
  liveAnalyser.fftSize = 2048;
  liveProc = liveCtx.createScriptProcessor(4096, 1, 1);
  // ScriptProcessor only fires onaudioprocess while it reaches the graph
  // destination; route through a muted gain so the mic is not echoed.
  const sink = liveCtx.createGain();
  sink.gain.value = 0;

  const proto = location.protocol === "https:" ? "wss" : "ws";
  liveWS = new WebSocket(`${proto}://${location.host}/api/infer_live`);
  liveWS.binaryType = "arraybuffer";
  liveWS.onopen = () => liveWS.send(JSON.stringify({sr: liveCtx.sampleRate}));
  liveWS.onmessage = (ev) => {
    let m;
    try { m = JSON.parse(ev.data); } catch (e) { return; }
    if (m.meta) {
      uploadStatus.textContent = "live: listening to the microphone…";
      return;
    }
    if (m.error) { uploadStatus.textContent = "live error: " + m.error; return; }
    if (typeof m.t === "number" && Array.isArray(m.posterior)) ingestLiveRow(m);
  };
  liveWS.onclose = () => { if (liveOn) stopLive(); };
  liveProc.onaudioprocess = (e) => {
    // copy: the input buffer is reused by the audio thread
    const frame = new Float32Array(e.inputBuffer.getChannelData(0));
    if (liveWS && liveWS.readyState === WebSocket.OPEN) liveWS.send(frame.buffer);
    if (liveChunkSamples < LIVE_MAX_SAMPLES) {     // bounded buffer for playback
      liveChunks.push(frame);
      liveChunkSamples += frame.length;
    }
  };
  src.connect(liveAnalyser);
  liveAnalyser.connect(liveProc);
  liveProc.connect(sink);
  sink.connect(liveCtx.destination);
  liveDrawRAF = requestAnimationFrame(drawLiveWaveform);
}

function stopLive() {
  liveOn = false;
  const modepick = document.querySelector(".modepick");
  if (modepick) modepick.hidden = false;
  // Back to a replay-style layout: debug scrubber visible again.
  refreshScrubberVisibility();
  if (liveTimerId) { clearInterval(liveTimerId); liveTimerId = 0; }
  liveActiveBtn = null;
  if (liveBtn) {
    liveBtn.textContent = idleLiveBtnLabel(liveBtn);
    liveBtn.classList.remove("recording");
    liveBtn.disabled = false;
  }
  if (liveDrawRAF) { cancelAnimationFrame(liveDrawRAF); liveDrawRAF = 0; }
  if (liveProc) {
    liveProc.onaudioprocess = null;
    try { liveProc.disconnect(); } catch (e) { /* already gone */ }
  }
  if (liveAnalyser) { try { liveAnalyser.disconnect(); } catch (e) {} }
  if (liveStream) liveStream.getTracks().forEach(t => t.stop());
  if (liveCtx && liveCtx.state !== "closed") liveCtx.close();
  if (liveWS && liveWS.readyState <= 1) liveWS.close();
  liveProc = liveStream = liveCtx = liveWS = liveAnalyser = null;
  // Bind the captured audio to the playback element so the user can hear
  // back what they just recorded.
  if (liveWaveformEl) liveWaveformEl.hidden = true;
  // Leave the main gallery audio element hidden — it has no src after a
  // live session and an empty player below the recorded clip is just
  // visual noise. loadShow() will reveal it again when a gallery show
  // (or upload) is selected.
  if (liveChunks.length && livePlaybackEl) {
    const total = liveChunkSamples;
    const merged = new Float32Array(total);
    let off = 0;
    for (const c of liveChunks) { merged.set(c, off); off += c.length; }
    const blob = float32ToWavBlob(merged, liveInputSr || 48_000);
    const url = URL.createObjectURL(blob);
    livePlaybackEl.src = url;
    livePlaybackEl.hidden = false;
    const sec = (total / (liveInputSr || 48_000)).toFixed(1);
    const cap = liveChunkSamples >= LIVE_MAX_SAMPLES ? " (capped at 10 min)" : "";
    uploadStatus.textContent =
      `live session ended — ${sec} s captured${cap}. Press play below to hear it back.`;
  } else {
    uploadStatus.textContent = "live session ended";
  }
  liveChunks = []; liveChunkSamples = 0;
}

if (liveBtn) {
  liveBtn.addEventListener("click", () => {
    if (liveOn) stopLive();
    else startLive();
  });
}

// ===== i18n (English / 日本語 / 한국어) ====================================
// Scope: UI chrome and the "How it works" explainer only. Encoder names
// (EfficientAT, PANN CNN14, AST, HTS-AT), AudioSet class
// names, spec numbers, segment class labels (Performance / MC talk / …) and
// author/affiliation stay in English in every language so the technical
// vocabulary stays canonical.
const LANG_KEY = "cuesheet-lang";
const LANG_CYCLE = ["en", "ja", "ko"];
const LANG_LABEL = { en: "EN", ja: "日本語", ko: "한국어" };
const langBtn = document.getElementById("lang-btn");

const I18N = {
  en: {
    "hdr.tagline": "Multi-genre concert segment detection · live demo",
    "btn.theme.toLight": "☀ Light",
    "btn.theme.toDark":  "☽ Dark",
    "btn.theme.titleToLight": "Switch to light theme",
    "btn.theme.titleToDark":  "Switch to dark theme",
    "btn.fit.width":  "↔ Width",
    "btn.fit.screen": "▣ Screen",
    "btn.fit.title":  "Fit mode: Width fills the window (vertical scroll OK) / Screen caps height so everything fits one screen (no scroll). Applies whether or not you are fullscreen.",
    "btn.fs.enter":   "⛶ Fullscreen",
    "btn.fs.exit":    "⛶ Exit",
    "btn.fs.title":   "Fullscreen (press F)",
    "btn.lang.title": "Switch language (English / 日本語 / 한국어)",
    "pick.show":          "Show",
    "pick.mode.pre":      "Precomputed",
    "pick.mode.causal":   "Causal from playhead",
    "pick.mode.title":    "Precomputed: replay the offline segment track. Causal: recompute from the playhead, as the online pipeline would.",
    "pick.live.title":    "Live capture: record from the microphone and run the pipeline in real time.",
    "pick.live.audio":    "Audio",
    "live.record":        "● Record",
    "live.stop":          "■ Stop",
    "pick.upmode.title":  "Pick how the next upload is processed. Click a different mode after upload to re-run on the same file.",
    "pick.upload.title":  "Upload an audio file to run the pipeline on",
    "pick.upload.label":  "Upload",
    "pipe.audio.title":     "Audio window",
    "pipe.enc.menuHead":    "AudioSet encoder",
    "pipe.enc.switchTitle": "Click to switch the AudioSet encoder",
    "pipe.group.title":     "Group mapping",
    "pipe.group.detail":    "4 concert classes",
    "pipe.heur.title":      "HMM smoothing + heuristic",
    "pipe.heur.detail":     "Causal forward + heuristics",
    "pipe.out.title":       "6-class segment",
    "how.summary": "How does the live inference actually work?",
    "how.window": "<strong>Per-second sliding window.</strong> Each second <em>t</em>, the audio encoder consumes the trailing 10&nbsp;s of audio (window <em>[t-9,&nbsp;t+1]</em>) and emits one prediction. There is no peek into the future — only what the system has actually heard.",
    "how.p2": "<strong>Cold start.</strong> At the very first second after a jump or session start, only ~1&nbsp;s of real audio is available; the encoder window is left-padded with silence. Expect the first ~5-10&nbsp;s of predictions to be biased toward whatever the encoder reports for mostly-silent input (often Ambient) before the trailing buffer fills.",
    "how.p3": "<strong>HMM memory.</strong> The 4-class encoder posterior is smoothed by an online causal HMM forward pass (stay&nbsp;=&nbsp;0.95). The HMM carries one floating-point state vector (<em>alpha</em>) across seconds — this is the only thing the live system \"remembers\". On a jump or mode switch, <em>alpha</em> is reset to a uniform prior so no memory leaks across sessions.",
    "how.p4": "<strong>Heuristics.</strong> Two of the six classes — Pre-concert and Intermission — are not encoder outputs; they're temporal rules applied on top of the smoothed segment track (first sustained run of Performance ends the pre-concert lead-in; long low-activity gaps between Performance segments are intermissions).",
    "card.now.label":      "Current segment",
    "card.post.label":     "Per-second posterior",
    "card.post.sub":       "(6 segment classes: 4 encoder posteriors + 2 heuristic-only)",
    "card.audioset.label": "Encoder raw output",
    "card.audioset.sub":   "(top-7 AudioSet classes activated this second)",
    "card.mapping.label":  "Group mapping",
    "card.mapping.sub":    "(AudioSet \u2192 concert classes)",
    "scrub.label":         "Scrub time (debug — drag to advance):",
    "foot.lastupdate":     "last update",
    "foot.lastupdate.title": "Demo asset version — bumped on every code push",
  },
  ja: {
    "hdr.tagline": "マルチジャンル・コンサートのフェーズ検出 · ライブデモ",
    "btn.theme.toLight": "☀ ライト",
    "btn.theme.toDark":  "☽ ダーク",
    "btn.theme.titleToLight": "ライトテーマに切り替え",
    "btn.theme.titleToDark":  "ダークテーマに切り替え",
    "btn.fit.width":  "↔ 幅優先",
    "btn.fit.screen": "▣ 画面に収める",
    "btn.fit.title":  "表示モード：「幅優先」はウィンドウ幅いっぱいに広げます（縦スクロール可）。「画面に収める」は高さも制限して一画面に収めます（スクロールなし）。全画面表示かどうかに関わらず適用されます。",
    "btn.fs.enter":   "⛶ 全画面",
    "btn.fs.exit":    "⛶ 終了",
    "btn.fs.title":   "全画面表示（F キー）",
    "btn.lang.title": "言語切替（English / 日本語 / 한국어）",
    "pick.show":          "公演",
    "pick.mode.pre":      "事前計算",
    "pick.mode.causal":   "再生位置から因果的に再計算",
    "pick.mode.title":    "事前計算：オフラインのフェーズトラックを再生。因果的：オンラインパイプラインと同じように、再生ヘッドから再計算します。",
    "pick.live.title":    "ライブキャプチャ：マイクから録音し、リアルタイムでパイプラインを実行します。",
    "pick.live.audio":    "音声",
    "live.record":        "● 録音",
    "live.stop":          "■ 停止",
    "pick.upmode.title":  "次回アップロードの処理方法を選択。アップロード後に別のモードを押すと、同じファイルで再実行します。",
    "pick.upload.title":  "パイプラインを実行する音声ファイルをアップロード",
    "pick.upload.label":  "アップロード",
    "pipe.audio.title":     "音声ウィンドウ",
    "pipe.enc.menuHead":    "AudioSet エンコーダ",
    "pipe.enc.switchTitle": "クリックして AudioSet エンコーダを切り替え",
    "pipe.group.title":     "グループマッピング",
    "pipe.group.detail":    "4 つのコンサートクラス",
    "pipe.heur.title":      "HMM 平滑化 + ヒューリスティック",
    "pipe.heur.detail":     "因果的フォワードパス + ヒューリスティック",
    "pipe.out.title":       "6 クラスのフェーズ",
    "how.summary": "ライブ推論は実際どのように動いているのか？",
    "how.window": "<strong>1 秒ごとのスライディングウィンドウ。</strong> 各秒 <em>t</em> で、音声エンコーダは直前 10&nbsp;秒の音声（ウィンドウ <em>[t-9,&nbsp;t+1]</em>）を受け取り、1 つの予測を出力します。未来を覗くことはなく、システムが実際に聞いた分だけを使います。",
    "how.p2": "<strong>コールドスタート。</strong> ジャンプやセッション開始直後の最初の秒では、実音声は約 1&nbsp;秒分しかなく、エンコーダウィンドウは無音で左パディングされます。最初の 5～10&nbsp;秒の予測は、ほぼ無音入力に対するエンコーダの出力（多くの場合 Ambient）に偏ります — トレーリングバッファが埋まるまでの挙動です。",
    "how.p3": "<strong>HMM のメモリ。</strong> 4 クラスのエンコーダ事後確率は、オンライン因果的 HMM のフォワードパス（stay&nbsp;=&nbsp;0.95）で平滑化されます。HMM が秒をまたいで保持するのは 1 つの浮動小数状態ベクトル <em>alpha</em> のみで、これがライブシステムの唯一の「記憶」です。ジャンプやモード切替時には <em>alpha</em> を一様事前分布にリセットし、セッション間で記憶が漏れないようにしています。",
    "how.p4": "<strong>ヒューリスティック。</strong> 6 クラスのうち 2 つ — Pre-concert と Intermission — はエンコーダ出力ではなく、平滑化後のフェーズトラックに対する時間ルールで決めています（Performance が最初に持続的に続いた時点で Pre-concert を終了、Performance 区間の間にある長い低活動区間を Intermission とみなす）。",
    "card.now.label":      "現在のフェーズ",
    "card.post.label":     "毎秒の事後確率",
    "card.post.sub":       "（6 フェーズクラス：エンコーダ事後確率 4 + ヒューリスティック専用 2）",
    "card.audioset.label": "エンコーダ生出力",
    "card.audioset.sub":   "（この 1 秒で活性化した AudioSet クラス Top-7）",
    "card.mapping.label":  "グループマッピング",
    "card.mapping.sub":    "（AudioSet \u2192 コンサートクラス）",
    "scrub.label":         "再生位置（デバッグ — ドラッグで進める）：",
    "foot.lastupdate":     "最終更新",
    "foot.lastupdate.title": "デモアセットのバージョン — コード push のたびに更新",
  },
  ko: {
    "hdr.tagline": "다중 장르 콘서트 단계 검출 · 라이브 데모",
    "btn.theme.toLight": "☀ 라이트",
    "btn.theme.toDark":  "☽ 다크",
    "btn.theme.titleToLight": "라이트 테마로 전환",
    "btn.theme.titleToDark":  "다크 테마로 전환",
    "btn.fit.width":  "↔ 너비 맞춤",
    "btn.fit.screen": "▣ 화면 맞춤",
    "btn.fit.title":  "맞춤 모드: 「너비 맞춤」은 창 너비에 맞춰 채웁니다(세로 스크롤 가능). 「화면 맞춤」은 높이도 제한해 한 화면에 모두 들어가게 합니다(스크롤 없음). 전체화면 여부와 관계없이 적용됩니다.",
    "btn.fs.enter":   "⛶ 전체화면",
    "btn.fs.exit":    "⛶ 나가기",
    "btn.fs.title":   "전체화면(F 키)",
    "btn.lang.title": "언어 전환(English / 日本語 / 한국어)",
    "pick.show":          "공연",
    "pick.mode.pre":      "사전 계산",
    "pick.mode.causal":   "재생 위치부터 인과적 재계산",
    "pick.mode.title":    "사전 계산: 오프라인으로 만든 단계 트랙을 재생합니다. 인과적: 온라인 파이프라인과 동일하게 재생 헤드부터 다시 계산합니다.",
    "pick.live.title":    "라이브 캡처: 마이크로 녹음하며 파이프라인을 실시간으로 실행합니다.",
    "pick.live.audio":    "오디오",
    "live.record":        "● 녹음",
    "live.stop":          "■ 중지",
    "pick.upmode.title":  "다음 업로드의 처리 방식을 선택합니다. 업로드 후 다른 모드를 누르면 같은 파일로 재실행합니다.",
    "pick.upload.title":  "파이프라인을 실행할 오디오 파일 업로드",
    "pick.upload.label":  "업로드",
    "pipe.audio.title":     "오디오 윈도우",
    "pipe.enc.menuHead":    "AudioSet 인코더",
    "pipe.enc.switchTitle": "클릭해서 AudioSet 인코더 전환",
    "pipe.group.title":     "그룹 매핑",
    "pipe.group.detail":    "콘서트 클래스 4개",
    "pipe.heur.title":      "HMM 평활화 + 휴리스틱",
    "pipe.heur.detail":     "인과적 정방향 패스 + 휴리스틱",
    "pipe.out.title":       "6 클래스 단계",
    "how.summary": "라이브 추론은 실제로 어떻게 동작하나요?",
    "how.window": "<strong>초당 슬라이딩 윈도우.</strong> 매 초 <em>t</em>에서 오디오 인코더는 직전 10&nbsp;초 오디오(윈도우 <em>[t-9,&nbsp;t+1]</em>)를 받아 예측 하나를 출력합니다. 미래는 들여다보지 않으며, 시스템이 실제로 들은 것만 사용합니다.",
    "how.p2": "<strong>콜드 스타트.</strong> 점프나 세션 시작 직후 첫 초에는 실제 오디오가 약 1&nbsp;초뿐이라 인코더 윈도우의 앞부분은 무음으로 패딩됩니다. 따라서 첫 5~10&nbsp;초의 예측은 거의 무음 입력에 대해 인코더가 내는 값(보통 Ambient)으로 치우치는 경향이 있습니다 — 트레일링 버퍼가 채워지기 전까지입니다.",
    "how.p3": "<strong>HMM 메모리.</strong> 4 클래스 인코더 사후확률은 온라인 인과적 HMM forward 패스(stay&nbsp;=&nbsp;0.95)로 평활화됩니다. HMM이 초 사이에 이어가는 것은 부동소수 상태 벡터 <em>alpha</em> 하나뿐이며, 이것이 라이브 시스템이 \"기억\"하는 유일한 정보입니다. 점프나 모드 전환 시 <em>alpha</em>는 균일 사전분포로 리셋되어 세션 간에 메모리가 새지 않도록 합니다.",
    "how.p4": "<strong>휴리스틱.</strong> 6 클래스 중 2개(Pre-concert, Intermission)는 인코더 출력이 아니라 평활화된 단계 트랙 위에 적용되는 시간 규칙입니다(Performance가 처음으로 일정 시간 지속되면 Pre-concert 도입부가 끝나는 것으로 처리하고, Performance 구간 사이의 긴 저활동 구간은 Intermission으로 간주).",
    "card.now.label":      "현재 단계",
    "card.post.label":     "초당 사후확률",
    "card.post.sub":       "(6 단계 클래스: 인코더 사후확률 4 + 휴리스틱 전용 2)",
    "card.audioset.label": "인코더 원시 출력",
    "card.audioset.sub":   "(이 1초에 활성화된 AudioSet 클래스 상위 7개)",
    "card.mapping.label":  "그룹 매핑",
    "card.mapping.sub":    "(AudioSet \u2192 콘서트 클래스)",
    "scrub.label":         "재생 위치(디버그 — 드래그로 이동):",
    "foot.lastupdate":     "마지막 업데이트",
    "foot.lastupdate.title": "데모 에셋 버전 — 코드 push마다 갱신",
  },
};

function getLang() {
  let v = "en";
  try { v = localStorage.getItem(LANG_KEY) || "en"; } catch (e) {}
  return LANG_CYCLE.includes(v) ? v : "en";
}
function t(key) {
  const lang = getLang();
  return (I18N[lang] && I18N[lang][key]) || (I18N.en[key]) || key;
}
function applyLang(lang) {
  if (!LANG_CYCLE.includes(lang)) lang = "en";
  try { localStorage.setItem(LANG_KEY, lang); } catch (e) {}
  document.documentElement.setAttribute("lang", lang);
  document.querySelectorAll("[data-i18n]").forEach((el) => {
    el.textContent = t(el.getAttribute("data-i18n"));
  });
  document.querySelectorAll("[data-i18n-html]").forEach((el) => {
    el.innerHTML = t(el.getAttribute("data-i18n-html"));
  });
  document.querySelectorAll("[data-i18n-title]").forEach((el) => {
    el.setAttribute("title", t(el.getAttribute("data-i18n-title")));
  });
  // Re-apply dynamic button labels (theme, fit, fullscreen, lang)
  if (typeof applyTheme === "function") {
    applyTheme(document.documentElement.getAttribute("data-theme") || "dark");
  }
  if (typeof setFitMode === "function") setFitMode(getFitMode());
  if (fsBtn) {
    fsBtn.textContent = document.fullscreenElement
      ? t("btn.fs.exit") : t("btn.fs.enter");
  }
  if (langBtn) {
    // Show the NEXT language in the cycle so the label reads "click to
    // switch to this", matching the theme / fit / fullscreen buttons.
    const next = LANG_CYCLE[(LANG_CYCLE.indexOf(lang) + 1) % LANG_CYCLE.length];
    langBtn.textContent = LANG_LABEL[next];
  }
  // Re-paint the live-button label: while idle, applyLang must rewrite it
  // because its text is JS-driven (not data-i18n). While recording the timer
  // interval will catch up within 500 ms on its own.
  if (typeof liveOn !== "undefined" && !liveOn &&
      typeof idleLiveBtnLabel === "function" && liveBtn) {
    liveBtn.textContent = idleLiveBtnLabel(liveBtn);
  }
}
if (langBtn) {
  langBtn.addEventListener("click", () => {
    const cur = getLang();
    const next = LANG_CYCLE[(LANG_CYCLE.indexOf(cur) + 1) % LANG_CYCLE.length];
    applyLang(next);
  });
}
// NOTE: applyLang(getLang()) is called *after* the theme and fit-mode
// blocks below, so the `const THEME_KEY`/`const FIT_KEY` they declare
// are out of TDZ before applyLang -> applyTheme/setFitMode reach them.
// Calling it here would throw a ReferenceError and halt the script.

// ===== Light / dark theme toggle (persisted in localStorage) ============
const THEME_KEY = "cuesheet-theme";
function applyTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  if (themeBtn) {
    themeBtn.textContent = theme === "dark"
      ? t("btn.theme.toLight") : t("btn.theme.toDark");
    themeBtn.title = t(theme === "dark" ? "btn.theme.titleToLight" : "btn.theme.titleToDark");
  }
  try { localStorage.setItem(THEME_KEY, theme); } catch (e) {}
}
(function initTheme() {
  let saved = "dark";
  try { saved = localStorage.getItem(THEME_KEY) || "dark"; } catch (e) {}
  applyTheme(saved === "light" ? "light" : "dark");
})();
if (themeBtn) {
  themeBtn.addEventListener("click", () => {
    const now = document.documentElement.getAttribute("data-theme") || "dark";
    applyTheme(now === "dark" ? "light" : "dark");
  });
}

// ===== Fit-mode toggle (Width vs Screen, persisted in localStorage) =======
const FIT_KEY = "cuesheet-fit-mode";
const fitBtn = document.getElementById("fit-btn");
function getFitMode() {
  let v = "width";
  try { v = localStorage.getItem(FIT_KEY) || "width"; } catch (e) {}
  return v === "screen" ? "screen" : "width";
}
function setFitMode(mode) {
  const m = mode === "screen" ? "screen" : "width";
  try { localStorage.setItem(FIT_KEY, m); } catch (e) {}
  if (fitBtn) {
    // Show the OTHER mode so the label reads "click to switch to this" —
    // matches the theme button ("☀ Light" when currently dark) and the
    // fullscreen button ("⛶ Fullscreen" when not yet fullscreen).
    fitBtn.textContent = t(m === "screen" ? "btn.fit.width" : "btn.fit.screen");
  }
  fitStage();
}
setFitMode(getFitMode());
if (fitBtn) {
  fitBtn.addEventListener("click", () => {
    setFitMode(getFitMode() === "screen" ? "width" : "screen");
  });
}

// Now that THEME_KEY and FIT_KEY are initialized, apply the saved
// language: walks every data-i18n element and re-syncs the theme / fit /
// fullscreen / language button labels in one pass.
applyLang(getLang());

loadCatalog().then(() => {
  // Allow URL ?t=<seconds> to jump straight to a specific moment so
  // the user can verify bar animation without scrubbing manually.
  const params = new URLSearchParams(window.location.search);
  const tWanted = parseInt(params.get("t") || "", 10);
  if (Number.isFinite(tWanted) && currentShow) {
    // Use ribbonDuration so the URL ?t=<sec> seek lands on the same
    // audio-file second that the playhead's left% encodes — n_seconds
    // would put the playhead ~0.5 % earlier than the audio for shows
    // whose mp3 is longer than the precomputed track.
    const frac = Math.max(0, Math.min(1, tWanted / ribbonDuration()));
    if (scrubber) scrubber.value = String(frac * 100);
    if (scrubberReadout) scrubberReadout.textContent = `t = ${tWanted} s (from URL)`;
    updateWidgets(tWanted);
    playhead.style.left = `${frac * 100}%`;
    lastReportedSec = tWanted;  // prevent tick() from immediately overwriting
  }
  requestAnimationFrame(tick);
  // ?autolive=1 starts the live-mic session on load (used for headless
  // verification and for a hands-free demo-table kiosk mode).
  if (params.get("autolive") === "1") startLive();
});

// ===== Timestamps export ==================================================
// HF Space (audio-only) build of the Timestamps feature. This build
// serves the audio cascade only (no modality mode toggle), and keeps the
// Model / Hand-labeled GT source toggle since every gallery show
// includes GT.
(function setupTimestamps() {
  const SEGMENT_NAMES = ["Pre-concert", "Performance", "MC talk",
                       "Applause", "Intermission", "Ambient"];
  const MIN_SEG_SEC = 10;
  const PIPELINE_TAG = "Audio cascade (EfficientAT MN10 → AudioSet group-map → online HMM + temporal heuristics)";
  function pipelineTag() {
    if (!causalActive) return PIPELINE_TAG;
    const enc = (typeof ENCODER_META !== "undefined" && ENCODER_META[currentEncoder])
      ? ENCODER_META[currentEncoder].title : "EfficientAT MN10";
    return `Audio cascade, causal from t=${causalStart}s (${enc} → AudioSet group-map → online HMM)`;
  }

  function buildSeq(source) {
    if (!currentShow) return {seq: [], source: "none", hasGt: false};
    // In causal mode the user is looking at the causal overlay, not the
    // stored track — export what is on screen (the computed region only).
    const segment = causalActive
      ? Array.from(causalSegmentInt.slice(0, causalNext))
      : (currentShow.segment_int || []);
    const T = segment.length;
    if (T === 0) return {seq: [], source: "none", hasGt: false};
    const hasGt = Array.isArray(currentShow.gt_labels)
      && currentShow.gt_labels.length > 0;
    source = (source === "gt" && hasGt) ? "gt" : "model";
    const out = new Int16Array(T);
    if (source === "gt") {
      const gt = currentShow.gt_labels;
      const n = Math.min(gt.length, T);
      for (let t = 0; t < n; t++) {
        const g = gt[t];
        out[t] = (g != null && g >= 0) ? g : -1;
      }
    } else {
      for (let t = 0; t < T; t++) {
        const a = segment[t];
        out[t] = (a != null && a >= 0) ? a : -1;
      }
    }
    return {seq: out, source, hasGt};
  }

  function toSegments(seq) {
    const segs = []; let i = 0;
    while (i < seq.length) {
      const lab = seq[i]; let j = i + 1;
      while (j < seq.length && seq[j] === lab) j++;
      if (lab >= 0) segs.push({start: i, end: j, label: lab});
      i = j;
    }
    return segs;
  }
  function mergeShort(segs) {
    if (segs.length === 0) return [];
    const out = [];
    for (const s of segs) {
      if ((s.end - s.start) < MIN_SEG_SEC && out.length > 0) {
        out[out.length - 1].end = s.end;
      } else out.push({...s});
    }
    const collapsed = [];
    for (const s of out) {
      if (collapsed.length > 0 && collapsed[collapsed.length - 1].label === s.label) {
        collapsed[collapsed.length - 1].end = s.end;
      } else collapsed.push(s);
    }
    if (collapsed.length > 0 && collapsed[0].start !== 0) collapsed[0].start = 0;
    return collapsed;
  }
  function fmtHMS(sec) {
    sec = Math.max(0, Math.floor(sec));
    const h = Math.floor(sec / 3600);
    const m = Math.floor((sec % 3600) / 60);
    const s = sec % 60;
    if (h > 0) return `${h}:${m.toString().padStart(2,"0")}:${s.toString().padStart(2,"0")}`;
    return `${m}:${s.toString().padStart(2,"0")}`;
  }
  function fmtVtt(sec) {
    sec = Math.max(0, Math.floor(sec));
    const h = Math.floor(sec / 3600);
    const m = Math.floor((sec % 3600) / 60);
    const s = sec % 60;
    return `${h.toString().padStart(2,"0")}:${m.toString().padStart(2,"0")}:${s.toString().padStart(2,"0")}.000`;
  }
  function asYouTube(segs, info) {
    const showLabel = (currentShow && currentShow.label) || "(show)";
    const isGt = info.source === "gt";
    const lines = [];
    lines.push(`# Chapter timestamps for: ${showLabel}`);
    if (isGt) {
      lines.push(`# Source: Hand-labeled GROUND TRUTH (per-second annotator labels) — not a model prediction.`);
    } else {
      lines.push(`# Pipeline used: Audio mode — ${pipelineTag()}.`);
    }
    lines.push(`# CueSheet: Live Demo of Multi-Genre Concert Segment Detection`);
    lines.push(`# Kim, Yamamoto, Kikuchi, Lerch, Kondo — CueSheet`);
    lines.push(`# Yamaha Corporation × Georgia Institute of Technology`);
    lines.push("");
    for (const s of segs) {
      const ts = fmtHMS(s.start);
      const name = SEGMENT_NAMES[s.label] || `class_${s.label}`;
      lines.push(`${ts} ${name}`);
    }
    return lines.join("\n");
  }
  function asWebVTT(segs, info) {
    const showLabel = (currentShow && currentShow.label) || "(show)";
    const isGt = info.source === "gt";
    const out = ["WEBVTT", `NOTE Chapter timestamps for: ${showLabel}`];
    if (isGt) {
      out.push("NOTE Source: Hand-labeled GROUND TRUTH (per-second annotator labels) — not a model prediction.");
    } else {
      out.push(`NOTE Pipeline used: Audio mode — ${pipelineTag()}.`);
    }
    out.push(
      `NOTE CueSheet — Multi-Genre Concert Segment Detection`,
      `NOTE Kim, Yamamoto, Kikuchi, Lerch, Kondo — CueSheet`,
      `NOTE Yamaha Corporation × Georgia Institute of Technology`,
      "",
    );
    for (const s of segs) {
      out.push(`${fmtVtt(s.start)} --> ${fmtVtt(s.end)}`);
      out.push(SEGMENT_NAMES[s.label] || `class_${s.label}`);
      out.push("");
    }
    return out.join("\n");
  }
  function copyToClipboard(text, btn) {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).then(() => {
        if (btn) {
          const orig = btn.textContent;
          btn.textContent = "✓ Copied";
          setTimeout(() => { btn.textContent = orig; }, 1500);
        }
      });
    }
  }
  function download(text, filename) {
    const blob = new Blob([text], {type: "text/plain;charset=utf-8"});
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url; a.download = filename;
    document.body.appendChild(a); a.click();
    document.body.removeChild(a);
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  function openPopup() {
    let activeSource = "model";
    const rebuild = () => {
      const info = buildSeq(activeSource);
      const seq = info.seq;
      if (!seq.length) return {info, segs: [], yt: "", vtt: ""};
      const segs = mergeShort(toSegments(seq));
      return {info, segs, yt: asYouTube(segs, info), vtt: asWebVTT(segs, info)};
    };
    let built = rebuild();
    if (!built.info.seq || !built.info.seq.length) {
      alert("No predictions available yet — pick a show first.");
      return;
    }
    if (!built.segs.length) {
      alert("No timestamps to export.");
      return;
    }
    const hasGt = built.info.hasGt;
    document.querySelectorAll(".timestamps-popup").forEach(p => p.remove());
    const overlay = document.createElement("div");
    overlay.className = "timestamps-popup-overlay";
    overlay.addEventListener("click", e => { if (e.target === overlay) overlay.remove(); });
    const popup = document.createElement("div");
    popup.className = "timestamps-popup";
    const title = document.createElement("div");
    title.className = "timestamps-popup-title";
    title.textContent = `Chapter timestamps — ${(currentShow && currentShow.label) || "show"}`;
    const sub = document.createElement("div");
    sub.className = "timestamps-popup-sub muted";
    const setSubText = (b) => {
      if (b.info.source === "gt") {
        sub.textContent = `Source: Hand-labeled ground truth (annotator labels per second). ${b.segs.length} chapters, min ${MIN_SEG_SEC} s each.`;
      } else {
        sub.textContent = `Pipeline: Audio — ${pipelineTag()}. ${b.segs.length} chapters, min ${MIN_SEG_SEC} s each. Paste the YouTube block directly under a video description.`;
      }
    };
    setSubText(built);
    let srcToggle = null;
    if (hasGt) {
      srcToggle = document.createElement("div");
      srcToggle.className = "timestamps-popup-tabs";
      const lbl = document.createElement("span");
      lbl.className = "muted";
      lbl.style.cssText = "font-size:11px; margin-right:8px; align-self:center;";
      lbl.textContent = "Source:";
      const bM = document.createElement("button");
      bM.className = "timestamps-popup-tab active"; bM.type = "button";
      bM.textContent = "Model predictions";
      const bG = document.createElement("button");
      bG.className = "timestamps-popup-tab"; bG.type = "button";
      bG.textContent = "Hand-labeled GT";
      srcToggle.appendChild(lbl);
      srcToggle.appendChild(bM);
      srcToggle.appendChild(bG);
      bM.addEventListener("click", () => {
        if (activeSource === "model") return;
        activeSource = "model";
        bM.classList.add("active"); bG.classList.remove("active");
        built = rebuild(); setSubText(built);
        pre.value = tabYt.classList.contains("active") ? built.yt : built.vtt;
      });
      bG.addEventListener("click", () => {
        if (activeSource === "gt") return;
        activeSource = "gt";
        bG.classList.add("active"); bM.classList.remove("active");
        built = rebuild(); setSubText(built);
        pre.value = tabYt.classList.contains("active") ? built.yt : built.vtt;
      });
    }
    const tabs = document.createElement("div");
    tabs.className = "timestamps-popup-tabs";
    const tabYt = document.createElement("button");
    tabYt.className = "timestamps-popup-tab active"; tabYt.type = "button";
    tabYt.textContent = "YouTube chapters";
    const tabVtt = document.createElement("button");
    tabVtt.className = "timestamps-popup-tab"; tabVtt.type = "button";
    tabVtt.textContent = "WebVTT";
    tabs.appendChild(tabYt); tabs.appendChild(tabVtt);
    const pre = document.createElement("textarea");
    pre.className = "timestamps-popup-pre";
    pre.readOnly = true;
    pre.value = built.yt;
    tabYt.addEventListener("click", () => {
      tabYt.classList.add("active"); tabVtt.classList.remove("active");
      pre.value = built.yt;
    });
    tabVtt.addEventListener("click", () => {
      tabVtt.classList.add("active"); tabYt.classList.remove("active");
      pre.value = built.vtt;
    });
    const actions = document.createElement("div");
    actions.className = "timestamps-popup-actions";
    const copyBtn = document.createElement("button");
    copyBtn.type = "button"; copyBtn.className = "timestamps-popup-btn";
    copyBtn.textContent = "Copy to clipboard";
    copyBtn.addEventListener("click", () => copyToClipboard(pre.value, copyBtn));
    const dlBtn = document.createElement("button");
    dlBtn.type = "button"; dlBtn.className = "timestamps-popup-btn";
    dlBtn.textContent = "Download";
    dlBtn.addEventListener("click", () => {
      const base = (currentShow && currentShow.show_id) || "cuesheet";
      const suffix = activeSource === "gt" ? "_gt" : "";
      const ext = tabYt.classList.contains("active") ? "txt" : "vtt";
      download(pre.value, `${base}_chapters${suffix}.${ext}`);
    });
    const closeBtn = document.createElement("button");
    closeBtn.type = "button"; closeBtn.className = "timestamps-popup-btn timestamps-popup-close";
    closeBtn.textContent = "Close";
    closeBtn.addEventListener("click", () => overlay.remove());
    actions.appendChild(copyBtn); actions.appendChild(dlBtn); actions.appendChild(closeBtn);
    // Community ask: please credit CueSheet when publishing
    // generated timestamps. Not a legal requirement under the
    // Apache-2.0 code license or CC BY 4.0 dataset license --
    // the user's own audio + the timestamps it generates are
    // not covered as derived work.
    const creditNote = document.createElement("div");
    creditNote.className = "timestamps-popup-credit muted";
    creditNote.style.cssText = "font-size: 11px; opacity: 0.7; margin-top: 6px; text-align: center;";
    creditNote.textContent = "If you publish these timestamps, please credit CueSheet (for example, \"Chapters by CueSheet\").";
    popup.appendChild(title);
    popup.appendChild(sub);
    if (srcToggle) popup.appendChild(srcToggle);
    popup.appendChild(tabs);
    popup.appendChild(pre);
    popup.appendChild(actions);
    popup.appendChild(creditNote);
    overlay.appendChild(popup);
    document.body.appendChild(overlay);
    const onKey = e => { if (e.key === "Escape") { overlay.remove(); document.removeEventListener("keydown", onKey); } };
    document.addEventListener("keydown", onKey);
  }
  const btn = document.getElementById("timestamps-btn");
  if (btn) btn.addEventListener("click", openPopup);
})();


// Group-mapping ⓘ popup: click the icon to reveal the full AudioSet class
// list for that row in a fixed-position card; click outside, Esc, or any
// scroll dismisses it.
(function setupGmapPopups() {
  let openPopup = null;
  let dismissBound = false;
  const _dismiss = () => {
    if (openPopup && openPopup.parentNode) openPopup.parentNode.removeChild(openPopup);
    openPopup = null;
  };
  const _onDocClick = (e) => {
    if (!openPopup) return;
    if (openPopup.contains(e.target)) return;
    if (e.target.classList && e.target.classList.contains("gmap-info")) return;
    _dismiss();
  };
  const _onKey = (e) => { if (e.key === "Escape") _dismiss(); };
  document.addEventListener("click", function (ev) {
    const btn = ev.target.closest && ev.target.closest(".gmap-info");
    if (!btn) return;
    ev.stopPropagation();
    const row = btn.closest(".gmap-row");
    if (!row) return;
    const text = row.getAttribute("data-gmap-full") || row.textContent;
    _dismiss();
    const popup = document.createElement("div");
    popup.className = "gmap-popup";
    const head = document.createElement("div");
    head.className = "gmap-popup-title";
    const label = (row.querySelector(".gmap-text")
      && row.querySelector(".gmap-text").textContent || "Group")
      .split("\u2190")[0].trim();
    head.textContent = "AudioSet classes \u2192 " + label;
    const body = document.createElement("div");
    body.className = "gmap-popup-body";
    body.textContent = text;
    const close = document.createElement("button");
    close.className = "gmap-popup-close";
    close.type = "button";
    close.textContent = "\u00d7";
    close.addEventListener("click", _dismiss);
    popup.appendChild(close);
    popup.appendChild(head);
    popup.appendChild(body);
    document.body.appendChild(popup);
    const r = btn.getBoundingClientRect();
    const pw = Math.min(420, window.innerWidth - 32);
    popup.style.maxWidth = pw + "px";
    const left = Math.min(window.innerWidth - pw - 16, Math.max(16, r.left - pw + r.width));
    popup.style.left = left + "px";
    const pH = popup.getBoundingClientRect().height;
    const top = (r.bottom + 8 + pH <= window.innerHeight)
      ? r.bottom + 8
      : Math.max(16, r.top - 8 - pH);
    popup.style.top = top + "px";
    openPopup = popup;
    if (!dismissBound) {
      document.addEventListener("click", _onDocClick);
      document.addEventListener("keydown", _onKey);
      window.addEventListener("scroll", _dismiss, true);
      dismissBound = true;
    }
  });
})();


// Footer build stamp: derive from the script tag's cache-buster so the page
// always reports the bundle that is actually loaded (a mismatch with the
// latest deploy means a stale browser cache).
(function fillBuildStamp() {
  const el = document.getElementById("last-update-value");
  if (!el) return;
  const tag = document.querySelector('script[src*="cuesheet.js?v="]');
  const m = tag && tag.src.match(/[?&]v=([^&]+)/);
  if (!m) { el.textContent = "unknown build"; return; }
  // v format: YYYY-MM-DD-HHMMJST-mmNN  ->  "YYYY-MM-DD HH:MM JST · build mmNN"
  const parts = m[1].match(/^(\d{4}-\d{2}-\d{2})-(\d{2})(\d{2})JST-(.+)$/);
  el.textContent = parts
    ? `${parts[1]} ${parts[2]}:${parts[3]} JST · build ${parts[4]}`
    : m[1];
})();
