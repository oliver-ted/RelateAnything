/* Client-side helpers for the Aether demo UI (injected into <head>). */
(function () {
  "use strict";

  /* ---------- Export: build CSV / JSON from the results table ---------- */

  function tableRows(df) {
    if (!df) return { headers: [], rows: [] };
    if (Array.isArray(df)) return { headers: [], rows: df };
    return { headers: df.headers || [], rows: df.data || [] };
  }

  function stamp() {
    var d = new Date();
    var p = function (n) { return String(n).padStart(2, "0"); };
    return d.getFullYear() + p(d.getMonth() + 1) + p(d.getDate()) + "-" +
      p(d.getHours()) + p(d.getMinutes()) + p(d.getSeconds());
  }

  function download(text, mime, name) {
    var blob = new Blob([text], { type: mime });
    var url = URL.createObjectURL(blob);
    var a = document.createElement("a");
    a.href = url;
    a.download = name;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(function () { URL.revokeObjectURL(url); }, 1000);
  }

  function csvCell(v) {
    var s = v === null || v === undefined ? "" : String(v);
    return /[",\r\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s;
  }

  var DEFAULT_HEADERS = ["Subject", "Relationship", "Object", "Confidence"];

  window.aeExport = function (kind, df) {
    var t = tableRows(df);
    var headers = t.headers.length ? t.headers : DEFAULT_HEADERS;
    var rows = t.rows.filter(function (r) {
      return r && r.some(function (c) { return c !== "" && c !== null; });
    });
    var name = "relationships-" + stamp();
    if (kind === "json") {
      var keys = headers.map(function (h) {
        return String(h).toLowerCase().replace(/\s+/g, "_");
      });
      var records = rows.map(function (r) {
        var o = {};
        keys.forEach(function (k, i) { o[k] = r[i]; });
        return o;
      });
      download(JSON.stringify(records, null, 2), "application/json",
        name + ".json");
    } else {
      var lines = [headers].concat(rows).map(function (r) {
        return r.map(csvCell).join(",");
      });
      download(lines.join("\r\n") + "\r\n", "text/csv", name + ".csv");
    }
    return [];
  };

  /* ---------- Camera: friendly messages instead of raw errors ---------- */

  function cameraMessage(err) {
    var name = err && err.name;
    if (name === "NotAllowedError" || name === "SecurityError")
      return "Camera access is blocked. Allow camera access for this site " +
        "in your browser settings, then try again. You can keep using " +
        "image upload in the meantime.";
    if (name === "NotFoundError" || name === "OverconstrainedError")
      return "No camera was found. Connect a camera, or switch to Image " +
        "to analyze a photo instead.";
    if (name === "NotReadableError" || name === "AbortError")
      return "The camera is in use by another application. Close it and " +
        "try again.";
    return "The camera could not be started. Try again, or switch to " +
      "Image to analyze a photo instead.";
  }

  function setNotice(msg) {
    var host = document.querySelector("#ae-cam-notice .ae-notice");
    if (!host) return;
    var text = host.querySelector(".ae-notice__text");
    if (text) text.textContent = msg || "";
    host.classList.toggle("is-visible", !!msg);
  }

  var md = navigator.mediaDevices;
  var cameraUnavailable = !md || !md.getUserMedia;
  if (!cameraUnavailable) {
    var original = md.getUserMedia.bind(md);
    md.getUserMedia = function (constraints) {
      return original(constraints).then(function (stream) {
        if (constraints && constraints.video) setNotice("");
        return stream;
      }, function (err) {
        if (constraints && constraints.video) setNotice(cameraMessage(err));
        throw err;
      });
    };
  }

  /* Called when the input source switches; returns the value unchanged. */
  window.aeSourceChanged = function (source) {
    if (source === "Camera" && cameraUnavailable) {
      setNotice("The camera needs a secure connection (HTTPS or localhost). " +
        "Open this page over HTTPS, or switch to Image to analyze a photo.");
    } else if (source !== "Camera") {
      setNotice("");
    }
    return source;
  };
})();
