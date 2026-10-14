/* Speech to Decisions: the page.

   Captures the microphone at 16 kHz, posts it to the Space in half-second
   chunks, and redraws from the state every reply carries: the transcript,
   what the speaker is doing, the safety check and the yes/no answers. The
   yes/no questions can be edited until the microphone is on.

   A classic script on purpose (no modules): on a private Space the page sits
   in a cross-site iframe, where module fetches aren't authenticated. The
   capture worklet is loaded from a blob: URL for the same reason. */
(function () {
  "use strict";

  function $(id) { return document.getElementById(id); }

  var panel = $("record-panel");
  var cluster = $("voice-cluster");
  var micBtn = $("record-trigger");
  var stopBtn = $("rec-stop");
  var recorderTitle = $("recorder-title");
  var recordTitle = $("record-title");
  var recordHint = $("record-hint");
  var elapsedEl = $("rec-elapsed");
  var deviceEl = $("rec-device");
  var waveBars = $("rec-wave-bars");
  var transcriptEl = $("transcript");
  var noticeEl = $("notice");
  var doingEl = $("doing");
  var safetyEl = $("safety");
  var safetyList = $("safety-list");
  var verdictEl = $("verdict");
  var noulList = $("noul-list");
  var lockedTag = $("tag-locked");
  var infoPanel = $("info-panel");
  var infoBtn = $("open-info");

  var IDLE_HINT = "Your words are judged as you say them";
  var config = null;
  var sid = null;
  var questions = [];        // the yes/no questions, as the page last sent them
  var customQuestions = false;
  var answers = {};          // question -> its latest noul
  var hasTranscript = false;
  var localError = "";       // a problem on this side: the microphone or the network

  /* ---- Talking to the Space ------------------------------------------------ */

  function api(path, body) {
    return fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body || {})
    }).then(function (response) {
      if (!response.ok) throw new Error("The Space answered " + response.status + ".");
      return response.json();
    });
  }

  function sendQuestions() {
    return api("/api/session", { sid: sid, questions: questions }).then(apply).catch(netError);
  }

  function netError(error) {
    localError = "Lost contact with the Space. " + (error && error.message ? error.message : "");
    showNotice([]);
  }

  /* ---- Drawing ------------------------------------------------------------- */

  // One row: a label, a bar as wide as the noul, and the number. An editable
  // row has a text field for its label.
  function noulRow(label, editable) {
    var row = document.createElement("div");
    row.className = "noul";
    var text;
    if (editable) {
      text = document.createElement("input");
      text.type = "text";
      text.className = "noul__input";
      text.value = label;
      text.maxLength = 300;
      text.spellcheck = false;
      text.placeholder = "Add a yes/no question…";
      text.setAttribute("aria-label", label ? "Yes/no question" : "Add a yes/no question");
      row.classList.add("is-editable");
      if (!label) row.classList.add("is-new");
    } else {
      text = document.createElement("div");
      text.className = "noul__label";
      text.textContent = label;
    }
    var bar = document.createElement("div");
    bar.className = "noul__bar";
    var fill = document.createElement("div");
    fill.className = "noul__fill";
    bar.appendChild(fill);
    var value = document.createElement("div");
    value.className = "noul__value";
    value.textContent = "—";
    row.appendChild(text);
    row.appendChild(bar);
    row.appendChild(value);
    if (editable) {
      var remove = document.createElement("button");
      remove.type = "button";
      remove.className = "noul__remove";
      remove.setAttribute("aria-label", "Remove this question");
      // Carbon: close
      remove.innerHTML = '<svg viewBox="0 0 32 32" fill="currentColor" aria-hidden="true"><polygon points="17.4141 16 24 9.4141 22.5859 8 16 14.5859 9.4143 8 8 9.4141 14.5859 16 8 22.5859 9.4143 24 16 17.4141 22.5859 24 24 22.5859 17.4141 16"/></svg>';
      row.appendChild(remove);
    }
    return row;
  }

  function setRow(row, noul, threshold) {
    var known = typeof noul === "number";
    row.classList.toggle("is-yes", known && noul > threshold);
    row.querySelector(".noul__fill").style.width = known ? Math.round(noul * 100) + "%" : "0";
    row.querySelector(".noul__value").textContent = known ? noul.toFixed(2) : "—";
  }

  // Rebuild a list only when its labels change, so the bars animate between answers.
  function drawList(container, labels, values, threshold) {
    var key = labels.join("\n");
    if (container.dataset.key !== key) {
      container.textContent = "";
      labels.forEach(function (label) { container.appendChild(noulRow(label, false)); });
      container.dataset.key = key;
    }
    for (var i = 0; i < labels.length; i++) setRow(container.children[i], values ? values[i] : null, threshold);
  }

  function drawTranscript(state) {
    hasTranscript = !!(state.committed || state.live);
    if (!hasTranscript) {
      transcriptEl.dataset.placeholder = "true";
      transcriptEl.textContent = "Your words will appear here as you speak…";
      return;
    }
    transcriptEl.dataset.placeholder = "false";
    transcriptEl.textContent = state.committed ? state.committed + " " : "";
    if (state.live) {
      var live = document.createElement("span");
      live.className = "transcript-interim";
      live.textContent = state.live;
      transcriptEl.appendChild(live);
    }
    transcriptEl.scrollTop = transcriptEl.scrollHeight;
  }

  function pill(id, text) {
    var el = $(id);
    el.hidden = !text;
    if (text) el.textContent = text;
  }

  function drawTelemetry(state) {
    var t = state.timings;
    pill("tel-stt", t.stt_s ? "Parakeet " + t.stt_s.toFixed(2) + " s" : "");
    pill("tel-jev", t.jev_s ? "Jev " + t.jev_s.toFixed(2) + " s" : "");
    pill("tel-describe", t.describe_s ? shortName(config.models.describe) + " " + t.describe_s.toFixed(2) + " s" : "");
    pill("tel-cost", state.cost ? "$" + state.cost.toFixed(4) : "");
  }

  function showNotice(errors) {
    var lines = (errors || []).slice();
    if (localError) lines.unshift(localError);
    noticeEl.hidden = !lines.length;
    noticeEl.textContent = lines.join(" ");
  }

  function drawBoard(state) {
    // After a long silence the Space wipes the judgments, so everything here
    // draws blank until the speaker starts again.
    $("tag-quiet").hidden = !(state.quiet && listening);
    doingEl.dataset.placeholder = state.description ? "false" : "true";
    doingEl.textContent = state.description || "—";

    var threshold = config.safety_threshold;
    var labels = config.safety.map(function (c) { return c.label; });
    var values = state.safety && config.safety.map(function (c) { return state.safety[c.key]; });
    drawList(safetyList, labels, values, threshold);
    var flagged = values ? labels.filter(function (_, i) { return values[i] > threshold; }) : [];
    safetyEl.classList.toggle("is-flagged", flagged.length > 0);
    verdictEl.hidden = !values;
    verdictEl.textContent = flagged.length ? "Flagged" : "Nothing harmful";

    answers = {};
    (state.nouls || []).forEach(function (n) { answers[n.question] = n.noul; });
    paintQuestions();
  }

  /* ---- The yes/no questions -------------------------------------------------
     Each row is a question, its bar and its noul. While the microphone is off
     the question is a text field: change it, remove it with its x button (or
     by clearing it), or type in the empty last row to add one. While the microphone is on the rows are
     plain text and can't be changed. */

  function rowText(row) {
    var input = row.querySelector(".noul__input");
    return (input ? input.value : row.querySelector(".noul__label").textContent).trim();
  }

  // Rebuild the rows from `questions`.
  function buildQuestions() {
    var editable = !listening && !stopping;
    noulList.textContent = "";
    questions.forEach(function (q) { noulList.appendChild(noulRow(q, editable)); });
    if (editable && questions.length < config.max_questions) noulList.appendChild(noulRow("", true));
    lockedTag.hidden = editable;
    paintQuestions();
  }

  // Fill in each row's bar and number from the latest answers.
  function paintQuestions() {
    for (var i = 0; i < noulList.children.length; i++) {
      var row = noulList.children[i];
      setRow(row, answers[rowText(row)], 0.5);
    }
  }

  // The questions as the rows now read: no blanks, no repeats.
  function readQuestions() {
    var seen = {}, list = [];
    for (var i = 0; i < noulList.children.length; i++) {
      var q = rowText(noulList.children[i]);
      if (q && !seen[q.toLowerCase()]) { seen[q.toLowerCase()] = true; list.push(q); }
    }
    return list.slice(0, config.max_questions);
  }

  var editTimer = null;
  // Send the list if it changed. `tidy` also rebuilds the rows, which drops
  // cleared ones and keeps one empty row at the end; it waits until typing is over.
  function questionsEdited(tidy) {
    clearTimeout(editTimer);
    if (listening || stopping) return;
    var next = readQuestions();
    var changed = next.join("\n") !== questions.join("\n");
    if (changed) {
      questions = next;
      customQuestions = true;
      sendQuestions();
    }
    if (tidy) buildQuestions();
  }

  noulList.addEventListener("input", function (event) {
    var row = event.target.closest(".noul");
    if (row) row.classList.toggle("is-new", !event.target.value.trim() && row === noulList.lastElementChild);
    clearTimeout(editTimer);
    editTimer = setTimeout(function () { questionsEdited(false); }, 700);
  });
  // Leaving the list, or pressing Enter, tidies it.
  noulList.addEventListener("focusout", function (event) {
    if (!noulList.contains(event.relatedTarget)) questionsEdited(true);
    else questionsEdited(false);
  });
  noulList.addEventListener("click", function (event) {
    var remove = event.target.closest(".noul__remove");
    if (!remove) return;
    remove.closest(".noul").remove();
    questionsEdited(true);
  });
  noulList.addEventListener("keydown", function (event) {
    if (event.key !== "Enter" || !event.target.classList.contains("noul__input")) return;
    event.preventDefault();
    var index = Array.prototype.indexOf.call(noulList.children, event.target.closest(".noul"));
    questionsEdited(true);
    // Move to the next row, which after tidying may be the empty one at the end.
    var next = noulList.children[Math.min(index + 1, noulList.children.length - 1)];
    var input = next && next.querySelector(".noul__input");
    if (input) input.focus();
  });

  // Every reply from the Space is the whole state: draw it.
  function apply(state) {
    if (state.sid !== sid) {
      // A new session (the first reply, Start over, or the Space restarted):
      // it has the default questions, so send ours again if they differ.
      sid = state.sid;
      if (customQuestions && state.questions.join("\n") !== questions.join("\n")) sendQuestions();
    }
    if (!customQuestions) questions = state.questions.slice();
    // Rebuild the rows when they don't match the list, unless someone is typing in them.
    if (!noulList.contains(document.activeElement) && readQuestions().join("\n") !== questions.join("\n")) buildQuestions();
    if (state.ready === false) disable("This Space has no OPENROUTER_API_KEY, so it can't listen.");
    drawTranscript(state);
    drawBoard(state);
    drawTelemetry(state);
    showNotice(state.errors);
    if (!listening && !stopping) idleText();
    return state;
  }

  function idleText() {
    recordTitle.textContent = hasTranscript ? "Tap to keep talking" : "Tap to start talking";
    recordHint.textContent = IDLE_HINT;
  }

  function disable(message) {
    panel.classList.add("is-disabled");
    recordTitle.textContent = message;
    recordHint.textContent = "";
  }

  function shortName(model) { return String(model).split("/").pop(); }

  /* ---- Microphone ---------------------------------------------------------- */

  // Runs on the audio thread: hands the page 100 ms blocks of the first channel.
  var WORKLET =
    "class Capture extends AudioWorkletProcessor {" +
    "  constructor() { super(); this.size = Math.round(sampleRate / 10); this.block = new Float32Array(this.size); this.n = 0; }" +
    "  process(inputs) {" +
    "    var channel = inputs[0] && inputs[0][0];" +
    "    if (channel) for (var i = 0; i < channel.length; i++) {" +
    "      this.block[this.n++] = channel[i];" +
    "      if (this.n === this.size) { this.port.postMessage(this.block.slice(0)); this.n = 0; }" +
    "    }" +
    "    return true;" +
    "  }" +
    "}" +
    "registerProcessor('capture', Capture);";

  var listening = false;
  var stopping = false;
  var stream = null;
  var context = null;
  var nodes = [];
  var blocks = [];           // audio not yet sent
  var sending = null;        // the chunk request in flight, if any
  var chunkTimer = null;
  var clockTimer = null;
  var startedAt = 0;
  var level = 0;             // smoothed microphone level, 0 to 1

  var WAVE_COLUMNS = 48;
  var WAVE_DOTS = 7;
  var waveLevels = [];

  function buildWave() {
    waveBars.textContent = "";
    waveLevels = [];
    for (var c = 0; c < WAVE_COLUMNS; c++) {
      var col = document.createElement("div");
      col.className = "rec-wave__col";
      for (var d = 0; d < WAVE_DOTS; d++) {
        var dot = document.createElement("i");
        dot.className = "rec-wave__dot";
        col.appendChild(dot);
      }
      waveBars.appendChild(col);
      waveLevels.push(0);
    }
  }

  // Scroll the meter one column to the left and light the newest from the middle out.
  function pushWave(value) {
    waveLevels.push(value);
    waveLevels.shift();
    var middle = (WAVE_DOTS - 1) / 2;
    for (var c = 0; c < WAVE_COLUMNS; c++) {
      var lit = Math.max(1, Math.round(waveLevels[c] * WAVE_DOTS));
      var dots = waveBars.children[c].children;
      for (var d = 0; d < WAVE_DOTS; d++) dots[d].classList.toggle("is-lit", Math.abs(d - middle) < lit / 2);
    }
  }

  function onBlock(block) {
    if (!listening) return;
    blocks.push(block);
    var sum = 0;
    for (var i = 0; i < block.length; i++) sum += block[i] * block[i];
    // Speech sits around 0.02 to 0.2 RMS; map that onto 0 to 1.
    var now = Math.min(1, Math.sqrt(sum / block.length) * 6);
    level = now > level ? now : level * 0.7 + now * 0.3;
    cluster.style.setProperty("--vscale", (0.85 + level * 0.5).toFixed(3));
    pushWave(level);
  }

  // Post the audio gathered so far. One request at a time, so chunks arrive in order.
  function sendChunk() {
    if (sending || !blocks.length) return sending || Promise.resolve();
    var length = 0;
    blocks.forEach(function (b) { length += b.length; });
    var pcm = new Int16Array(length);
    var at = 0;
    blocks.forEach(function (b) {
      for (var i = 0; i < b.length; i++) {
        var v = Math.max(-1, Math.min(1, b[i]));
        pcm[at++] = v < 0 ? v * 32768 : v * 32767;
      }
    });
    blocks = [];
    var url = "/api/chunk?rate=" + Math.round(context.sampleRate) + (sid ? "&sid=" + encodeURIComponent(sid) : "");
    sending = fetch(url, { method: "POST", headers: { "Content-Type": "application/octet-stream" }, body: pcm.buffer })
      .then(function (response) {
        if (!response.ok) throw new Error("The Space answered " + response.status + ".");
        return response.json();
      })
      .then(function (state) { localError = ""; apply(state); })
      .catch(netError)
      .then(function () { sending = null; });
    return sending;
  }

  function tick() {
    var s = Math.floor((Date.now() - startedAt) / 1000);
    elapsedEl.textContent = ("0" + Math.floor(s / 60)).slice(-2) + ":" + ("0" + (s % 60)).slice(-2);
  }

  function connect() {
    var source = context.createMediaStreamSource(stream);
    nodes = [source];
    if (context.audioWorklet && window.AudioWorkletNode) {
      var url = URL.createObjectURL(new Blob([WORKLET], { type: "application/javascript" }));
      return context.audioWorklet.addModule(url).then(function () {
        var node = new AudioWorkletNode(context, "capture");
        node.port.onmessage = function (event) { onBlock(event.data); };
        source.connect(node);
        nodes.push(node);
      });
    }
    // Older browsers: the main-thread processor does the same job.
    var processor = context.createScriptProcessor(4096, 1, 1);
    processor.onaudioprocess = function (event) { onBlock(new Float32Array(event.inputBuffer.getChannelData(0))); };
    source.connect(processor);
    processor.connect(context.destination);
    nodes.push(processor);
    return Promise.resolve();
  }

  function start() {
    if (listening || stopping || panel.classList.contains("is-disabled")) return;
    // Take in a question that's still being typed, so it's sent before the audio.
    questionsEdited(false);
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      localError = "This browser can't reach a microphone here. The page needs HTTPS.";
      showNotice([]);
      return;
    }
    localError = "";
    showNotice([]);
    panel.classList.add("is-busy");
    recordTitle.textContent = "Waiting for the microphone…";
    navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true } })
      .then(function (media) {
        stream = media;
        var AudioContextClass = window.AudioContext || window.webkitAudioContext;
        // Ask for 16 kHz so the browser resamples; the Space handles any other rate.
        try { context = new AudioContextClass({ sampleRate: config.sample_rate }); }
        catch (e) { context = new AudioContextClass(); }
        return connect();
      })
      .then(function () {
        listening = true;
        blocks = [];
        level = 0;
        buildWave();
        startedAt = Date.now();
        tick();
        var track = stream.getAudioTracks()[0];
        deviceEl.textContent = (track && track.label) || "Microphone";
        recorderTitle.textContent = "Listening…";
        buildQuestions();  // lock the questions
        panel.classList.remove("is-busy");
        panel.classList.add("is-listening");
        micBtn.setAttribute("aria-pressed", "true");
        micBtn.setAttribute("aria-label", "Stop");
        closeInfo();
        chunkTimer = setInterval(sendChunk, config.chunk_s * 1000);
        clockTimer = setInterval(tick, 500);
      })
      .catch(function (error) {
        release();
        panel.classList.remove("is-busy");
        localError = error && error.name === "NotAllowedError"
          ? "Microphone access was blocked. Allow it in the browser and try again."
          : "Couldn't open the microphone. " + (error && error.message ? error.message : "");
        showNotice([]);
        idleText();
      });
  }

  function release() {
    nodes.forEach(function (node) { try { node.disconnect(); } catch (e) { /* already gone */ } });
    nodes = [];
    if (stream) stream.getTracks().forEach(function (track) { track.stop(); });
    stream = null;
  }

  function stop() {
    if (!listening) return;
    listening = false;
    stopping = true;
    clearInterval(chunkTimer);
    clearInterval(clockTimer);
    release();
    panel.classList.remove("is-listening");
    panel.classList.add("is-busy");
    cluster.style.removeProperty("--vscale");
    micBtn.setAttribute("aria-pressed", "false");
    micBtn.setAttribute("aria-label", "Start talking");
    recordTitle.textContent = "Finishing the last words…";
    recordHint.textContent = "";
    // Let the chunk in flight land, send what's left, then ask for the final answers.
    (sending || Promise.resolve())
      .then(sendChunk)
      .then(function () { return api("/api/stop", { sid: sid }).then(apply).catch(netError); })
      .then(function () {
        if (context) context.close();
        context = null;
        stopping = false;
        panel.classList.remove("is-busy");
        $("tag-quiet").hidden = true;
        idleText();
        buildQuestions();  // unlock the questions
      });
  }

  /* ---- Controls ------------------------------------------------------------ */

  function closeInfo() {
    infoPanel.hidden = true;
    infoBtn.setAttribute("aria-pressed", "false");
  }

  infoBtn.addEventListener("click", function () {
    infoPanel.hidden = !infoPanel.hidden;
    infoBtn.setAttribute("aria-pressed", String(!infoPanel.hidden));
  });
  $("close-info").addEventListener("click", closeInfo);

  micBtn.addEventListener("click", function () { if (listening) stop(); else start(); });
  stopBtn.addEventListener("click", stop);

  $("start-over").addEventListener("click", function () {
    var wasListening = listening;
    // Stop without waiting for the last answers: they're about to be thrown away.
    if (wasListening) {
      listening = false;
      clearInterval(chunkTimer);
      clearInterval(clockTimer);
      release();
      if (context) context.close();
      context = null;
      blocks = [];
      panel.classList.remove("is-listening");
      cluster.style.removeProperty("--vscale");
      micBtn.setAttribute("aria-pressed", "false");
    }
    localError = "";
    api("/api/clear", { sid: sid, questions: customQuestions ? questions : null }).then(apply).catch(netError)
      .then(buildQuestions);
  });

  /* ---- Start-up ------------------------------------------------------------ */

  fetch("/api/config")
    .then(function (response) { return response.json(); })
    .then(function (loaded) {
      config = loaded;
      questions = config.questions.slice();
      $("tag-scope").textContent = "Last " + config.recent_words + " words";
      $("info-words").textContent = config.recent_words;
      $("info-quiet").textContent = config.quiet_after_s;
      $("tag-stt").textContent = shortName(config.models.stt);
      $("tag-describe").textContent = shortName(config.models.describe);
      $("tag-jev").textContent = shortName(config.models.jev);
      $("tag-jev-safety").textContent = shortName(config.models.jev);
      var tags = $("info-tags");
      tags.textContent = "";
      [config.models.stt, config.models.jev, config.models.describe, "OpenRouter"].forEach(function (name) {
        var tag = document.createElement("span");
        tag.className = "tag";
        tag.textContent = shortName(name);
        tags.appendChild(tag);
      });
      buildWave();
      buildQuestions();
      return api("/api/session", {});
    })
    .then(apply)
    .catch(function () { disable("Couldn't reach the Space. Reload the page to try again."); });
})();
