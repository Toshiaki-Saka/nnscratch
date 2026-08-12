// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Toshiaki Saka
#ifndef NNSCRATCH_REPORT_HPP
#define NNSCRATCH_REPORT_HPP

#include <cmath>
#include <cstddef>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "nnscratch/tensor.hpp"

/// Writes a self-contained HTML report: charts, weight images and tables in one
/// file that opens in a browser with no server, no toolchain and no network.
///
/// Kept header-only and dependency-free for the same reason as pgm.hpp -- the
/// demos should be able to emit something you can actually look at without the
/// library growing a plotting dependency. Everything (CSS, JS, data) is inlined
/// into the document, so the result survives being copied around or attached to
/// a mail. Charts are drawn on a canvas by about a hundred lines of vanilla JS;
/// there is no chart library involved.
namespace nn::report {

// --- one line on a chart ---------------------------------------------------
struct Series {
    std::string name;
    std::vector<double> y;
};

/// A line chart. Two measures of different scale belong in two charts, never on
/// two y-axes, so there is deliberately no secondary-axis option here.
struct Chart {
    std::string title;
    std::string subtitle;
    std::string x_label;
    std::string y_label;
    std::vector<double> x;
    std::vector<Series> series;
    bool log_y = false;    ///< offer a log scale toggle, and start on it
    bool percent = false;  ///< values are fractions; render them as percentages
};

/// A grid of equally-sized square images -- learned weights, conv filters.
/// Each cell is min-max normalised independently, as in pgm.hpp, so structure
/// is visible regardless of scale. Brightness is therefore comparable within a
/// cell and not between cells; the caption should say so.
struct ImageGrid {
    std::string title;
    std::string caption;
    std::vector<std::vector<double>> cells;
    std::size_t cell = 8;  ///< edge of one cell, in values (not pixels)
    std::size_t cols = 8;
};

struct Table {
    std::string title;
    std::vector<std::string> headers;
    std::vector<std::vector<std::string>> rows;
};

/// A headline number. Use when the story is one figure, not a shape.
struct Stat {
    std::string label;
    std::string value;
    std::string note;
};

namespace detail {

inline std::string html_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        switch (c) {
            case '&':
                out += "&amp;";
                break;
            case '<':
                out += "&lt;";
                break;
            case '>':
                out += "&gt;";
                break;
            case '"':
                out += "&quot;";
                break;
            default:
                out += c;
        }
    }
    return out;
}

inline std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 2);
    for (char c : s) {
        switch (c) {
            case '"':
                out += "\\\"";
                break;
            case '\\':
                out += "\\\\";
                break;
            case '\n':
                out += "\\n";
                break;
            case '\r':
                out += "\\r";
                break;
            case '\t':
                out += "\\t";
                break;
            default:
                // Keep it ASCII-safe; anything below 0x20 would break the literal.
                if (static_cast<unsigned char>(c) < 0x20) {
                    out += ' ';
                } else {
                    out += c;
                }
        }
    }
    return out;
}

/// JSON has no inf/nan, so they become null and the chart skips those points.
inline std::string num(double v) {
    if (!std::isfinite(v)) return "null";
    std::ostringstream os;
    os.precision(9);
    os << v;
    return os.str();
}

inline std::string array(const std::vector<double>& v) {
    std::string out = "[";
    for (std::size_t i = 0; i < v.size(); ++i) {
        if (i) out += ',';
        out += num(v[i]);
    }
    out += ']';
    return out;
}

}  // namespace detail

/// Accumulates sections, then writes the document.
class Report {
public:
    explicit Report(std::string title, std::string subtitle = {})
        : title_(std::move(title)), subtitle_(std::move(subtitle)) {}

    Report& heading(const std::string& text) {
        body_ += "<h2>" + detail::html_escape(text) + "</h2>\n";
        return *this;
    }

    Report& note(const std::string& text) {
        body_ += "<p class=\"note\">" + detail::html_escape(text) + "</p>\n";
        return *this;
    }

    Report& add(const std::vector<Stat>& stats) {
        body_ += "<div class=\"stats\">\n";
        for (const Stat& s : stats) {
            body_ += "  <div class=\"stat\"><div class=\"stat-l\">" +
                     detail::html_escape(s.label) + "</div><div class=\"stat-v\">" +
                     detail::html_escape(s.value) + "</div>";
            if (!s.note.empty()) {
                body_ += "<div class=\"stat-n\">" + detail::html_escape(s.note) + "</div>";
            }
            body_ += "</div>\n";
        }
        body_ += "</div>\n";
        return *this;
    }

    Report& add(const Chart& c) {
        const std::string id = "c" + std::to_string(++counter_);
        body_ += "<figure class=\"card\" id=\"" + id + "\">\n";
        body_ += "  <figcaption><h3>" + detail::html_escape(c.title) + "</h3>";
        if (!c.subtitle.empty()) {
            body_ += "<p class=\"sub\">" + detail::html_escape(c.subtitle) + "</p>";
        }
        body_ += "</figcaption>\n";
        // A legend restates the title when there is only one series, so it is
        // emitted only from two upwards.
        body_ += "  <div class=\"legend\"></div>\n";
        body_ += "  <div class=\"plot\"><canvas></canvas><div class=\"tip\"></div></div>\n";
        body_ +=
            "  <div class=\"tools\"><button class=\"tbl-btn\" type=\"button\">"
            "Show data table</button>";
        if (c.log_y) {
            body_ += "<button class=\"log-btn\" type=\"button\">Linear scale</button>";
        }
        body_ += "</div>\n  <div class=\"tbl\" hidden></div>\n</figure>\n";

        std::string js = "{id:\"" + id + "\",kind:\"chart\",xLabel:\"" +
                         detail::json_escape(c.x_label) + "\",yLabel:\"" +
                         detail::json_escape(c.y_label) +
                         "\",logY:" + (c.log_y ? "true" : "false") +
                         ",percent:" + (c.percent ? "true" : "false") +
                         ",x:" + detail::array(c.x) + ",series:[";
        for (std::size_t i = 0; i < c.series.size(); ++i) {
            if (i) js += ',';
            js += "{name:\"" + detail::json_escape(c.series[i].name) +
                  "\",y:" + detail::array(c.series[i].y) + "}";
        }
        js += "]}";
        push(js);
        return *this;
    }

    Report& add(const ImageGrid& g) {
        const std::string id = "g" + std::to_string(++counter_);
        body_ += "<figure class=\"card\" id=\"" + id + "\">\n";
        body_ += "  <figcaption><h3>" + detail::html_escape(g.title) + "</h3></figcaption>\n";
        body_ += "  <div class=\"grid\"><canvas></canvas></div>\n";
        if (!g.caption.empty()) {
            body_ += "  <p class=\"note\">" + detail::html_escape(g.caption) + "</p>\n";
        }
        body_ += "</figure>\n";

        std::string js = "{id:\"" + id + "\",kind:\"grid\",cell:" + std::to_string(g.cell) +
                         ",cols:" + std::to_string(g.cols) + ",cells:[";
        for (std::size_t i = 0; i < g.cells.size(); ++i) {
            if (i) js += ',';
            js += detail::array(g.cells[i]);
        }
        js += "]}";
        push(js);
        return *this;
    }

    Report& add(const Table& t) {
        body_ += "<figure class=\"card\">\n";
        if (!t.title.empty()) {
            body_ +=
                "  <figcaption><h3>" + detail::html_escape(t.title) + "</h3></figcaption>\n";
        }
        body_ += "  <table><thead><tr>";
        for (const std::string& h : t.headers) {
            body_ += "<th>" + detail::html_escape(h) + "</th>";
        }
        body_ += "</tr></thead><tbody>\n";
        for (const std::vector<std::string>& row : t.rows) {
            body_ += "    <tr>";
            for (const std::string& cellv : row) {
                body_ += "<td>" + detail::html_escape(cellv) + "</td>";
            }
            body_ += "</tr>\n";
        }
        body_ += "  </tbody></table>\n</figure>\n";
        return *this;
    }

    void write(const std::string& path) const;

private:
    void push(const std::string& js) {
        if (!data_.empty()) data_ += ",\n";
        data_ += js;
    }

    std::string title_;
    std::string subtitle_;
    std::string body_;
    std::string data_;
    int counter_ = 0;
};

namespace detail {

// Palette, chrome and layout. Dark mode is a selected set of steps for the dark
// surface, not an automatic inversion, and is declared under both the OS media
// query and an explicit [data-theme] scope so the toggle wins either way.
inline const char* css() {
    return R"CSS(
:root{color-scheme:light;
--surface:#fcfcfb;--plane:#f9f9f7;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;
--grid:#e1e0d9;--axis:#c3c2b7;--border:rgba(11,11,11,.10);
--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s4:#eda100}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;
--surface:#1a1a19;--plane:#0d0d0d;--ink:#fff;--ink2:#c3c2b7;--muted:#898781;
--grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.10);
--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500}}
:root[data-theme="dark"]{color-scheme:dark;
--surface:#1a1a19;--plane:#0d0d0d;--ink:#fff;--ink2:#c3c2b7;--muted:#898781;
--grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.10);
--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500}
*{box-sizing:border-box}
body{margin:0;background:var(--plane);color:var(--ink);
font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
.wrap{max-width:1080px;margin:0 auto;padding:32px 20px 64px}
header.top{display:flex;align-items:flex-start;gap:16px;margin-bottom:8px}
header.top h1{font-size:24px;margin:0;letter-spacing:-.01em}
header.top .sub{color:var(--ink2);margin:4px 0 0}
#theme{margin-left:auto;flex:none;background:var(--surface);color:var(--ink2);
border:1px solid var(--border);border-radius:8px;padding:6px 12px;cursor:pointer;font:inherit;
font-size:13px}
#theme:hover{color:var(--ink)}
h2{font-size:14px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);
margin:36px 0 12px;font-weight:600}
.card{background:var(--surface);border:1px solid var(--border);border-radius:12px;
margin:0 0 16px;padding:18px 18px 14px}
.card h3{margin:0;font-size:16px}
.card .sub{margin:3px 0 0;color:var(--ink2);font-size:13px}
figcaption{margin-bottom:10px}
.note{color:var(--ink2);font-size:13px;margin:10px 0 0;max-width:74ch}
.plot{position:relative;height:280px}
.plot canvas{width:100%;height:100%;display:block}
.grid{overflow-x:auto}
.grid canvas{image-rendering:pixelated;display:block;max-width:100%}
.legend{display:flex;flex-wrap:wrap;gap:6px 16px;margin:0 0 10px}
.legend button{display:inline-flex;align-items:center;gap:7px;background:none;border:0;
padding:2px 0;cursor:pointer;color:var(--ink2);font:inherit;font-size:13px}
.legend button[aria-pressed="false"]{opacity:.4}
.legend .key{width:14px;height:3px;border-radius:2px;flex:none}
.tools{display:flex;gap:8px;margin-top:10px}
.tools button{background:none;border:1px solid var(--border);border-radius:7px;
color:var(--ink2);padding:4px 10px;font:inherit;font-size:12px;cursor:pointer}
.tools button:hover{color:var(--ink)}
table{border-collapse:collapse;width:100%;font-size:13px;
font-variant-numeric:tabular-nums}
th,td{text-align:right;padding:5px 10px;border-bottom:1px solid var(--border)}
th:first-child,td:first-child{text-align:left}
thead th{color:var(--muted);font-weight:600;font-size:12px;text-transform:uppercase;
letter-spacing:.05em}
.tbl{margin-top:12px;max-height:300px;overflow:auto}
.tip{position:absolute;pointer-events:none;opacity:0;transition:opacity .1s;
background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:8px 10px;
font-size:12px;box-shadow:0 4px 14px rgba(0,0,0,.13);min-width:110px;
font-variant-numeric:tabular-nums}
.tip b{display:block;color:var(--muted);font-weight:600;margin-bottom:4px;font-size:11px;
text-transform:uppercase;letter-spacing:.05em}
.tip .row{display:flex;align-items:center;gap:7px;color:var(--ink)}
.tip .row i{width:9px;height:9px;border-radius:50%;flex:none}
.tip .row span{margin-left:auto;padding-left:12px}
.stats{display:flex;flex-wrap:wrap;gap:12px;margin-bottom:16px}
.stat{background:var(--surface);border:1px solid var(--border);border-radius:12px;
padding:14px 18px;min-width:150px;flex:1}
.stat-l{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.06em}
.stat-v{font-size:26px;letter-spacing:-.02em;margin-top:2px}
.stat-n{color:var(--ink2);font-size:12px;margin-top:2px}
footer{color:var(--muted);font-size:12px;margin-top:36px;border-top:1px solid var(--border);
padding-top:14px}
)CSS";
}

inline const char* js() {
    return R"JS(
const CSS = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const SLOTS = ["--s1","--s2","--s3","--s4"];
const fmt = (v, pct) => v === null ? "--" :
  pct ? (v*100).toFixed(1) + "%" :
  (Math.abs(v) >= 1000 || (Math.abs(v) < 0.01 && v !== 0) ? v.toExponential(2)
                                                          : v.toFixed(4));

// Clean tick values: 1/2/5 x 10^n covering the range.
function ticks(lo, hi, n){
  const span = hi - lo; if (!(span > 0)) return [lo];
  const raw = span / n, mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1,2,5,10].find(m => m*mag >= raw) * mag;
  const out = []; for (let t = Math.ceil(lo/step)*step; t <= hi + 1e-9; t += step) out.push(t);
  return out;
}
function logTicks(lo, hi){
  const out = [];
  for (let e = Math.floor(Math.log10(lo)); e <= Math.ceil(Math.log10(hi)); e++){
    const v = Math.pow(10, e); if (v >= lo*0.999 && v <= hi*1.001) out.push(v);
  }
  return out.length >= 2 ? out : ticks(lo, hi, 4);
}

function drawChart(c){
  const cv = c.el.querySelector("canvas"), ctx = cv.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const w = cv.clientWidth, h = cv.clientHeight;
  cv.width = Math.round(w*dpr); cv.height = Math.round(h*dpr);
  ctx.setTransform(dpr,0,0,dpr,0,0); ctx.clearRect(0,0,w,h);

  const surface = CSS("--surface"), grid = CSS("--grid"), axis = CSS("--axis"),
        muted = CSS("--muted"), colors = SLOTS.map(CSS);
  const on = c.series.filter((_,i) => c.visible[i]);
  const pad = {l:58, r:14, t:8, b:30};
  const pw = w - pad.l - pad.r, ph = h - pad.t - pad.b;
  if (pw <= 0 || ph <= 0) return;

  const vals = on.flatMap(s => s.y).filter(v => v !== null && isFinite(v));
  const pos = vals.filter(v => v > 0);
  const useLog = c.logOn && pos.length > 0;
  let lo = useLog ? Math.min(...pos) : Math.min(...vals, 0);
  let hi = Math.max(...vals, useLog ? Math.max(...pos) : 0);
  if (!(hi > lo)) { hi = lo + 1; }
  if (useLog){ lo = Math.pow(10, Math.floor(Math.log10(lo)));
               hi = Math.pow(10, Math.ceil(Math.log10(hi))); }
  else { const m = (hi-lo)*0.06; hi += m; if (lo < 0) lo -= m; }

  const X = i => pad.l + (c.x.length < 2 ? pw/2 : pw * i/(c.x.length-1));
  const Y = v => { const t = useLog ? (Math.log10(v)-Math.log10(lo))/(Math.log10(hi)-Math.log10(lo))
                                    : (v-lo)/(hi-lo);
                   return pad.t + ph - t*ph; };
  c._X = X; c._pad = pad; c._pw = pw; c._ph = ph;

  // Gridlines: hairline, solid, recessive.
  const ys = useLog ? logTicks(lo,hi) : ticks(lo,hi,5);
  ctx.lineWidth = 1; ctx.font = "11px system-ui,-apple-system,'Segoe UI',sans-serif";
  ctx.textAlign = "right"; ctx.textBaseline = "middle"; ctx.fillStyle = muted;
  for (const t of ys){
    const y = Math.round(Y(t)) + .5;
    ctx.strokeStyle = grid; ctx.beginPath();
    ctx.moveTo(pad.l, y); ctx.lineTo(pad.l+pw, y); ctx.stroke();
    ctx.fillText(c.percent ? (t*100).toFixed(0)+"%" :
                 (useLog || Math.abs(t) >= 1000 ? t.toExponential(0) : String(+t.toFixed(4))),
                 pad.l-8, y);
  }
  ctx.strokeStyle = axis; ctx.beginPath();
  ctx.moveTo(pad.l+.5, pad.t); ctx.lineTo(pad.l+.5, pad.t+ph); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(pad.l, pad.t+ph+.5); ctx.lineTo(pad.l+pw, pad.t+ph+.5); ctx.stroke();

  ctx.textAlign = "center"; ctx.textBaseline = "top";
  const step = Math.max(1, Math.round(c.x.length/8));
  for (let i = 0; i < c.x.length; i += step)
    ctx.fillText(String(c.x[i]), X(i), pad.t+ph+8);
  if (c.xLabel){ ctx.fillText(c.xLabel, pad.l+pw/2, pad.t+ph+20); }

  // Lines: 2px, round join/cap.
  ctx.lineWidth = 2; ctx.lineJoin = "round"; ctx.lineCap = "round";
  c.series.forEach((s,si) => {
    if (!c.visible[si]) return;
    ctx.strokeStyle = colors[si % colors.length];
    ctx.beginPath(); let pen = false;
    s.y.forEach((v,i) => {
      if (v === null || !isFinite(v) || (useLog && v <= 0)) { pen = false; return; }
      const x = X(i), y = Y(v);
      pen ? ctx.lineTo(x,y) : ctx.moveTo(x,y); pen = true;
    });
    ctx.stroke();
    // End marker, >=8px, with a 2px surface ring so overlaps stay legible.
    for (let i = s.y.length-1; i >= 0; i--){
      const v = s.y[i];
      if (v === null || !isFinite(v) || (useLog && v <= 0)) continue;
      ctx.beginPath(); ctx.arc(X(i), Y(v), 4, 0, 6.284);
      ctx.fillStyle = colors[si % colors.length]; ctx.fill();
      ctx.lineWidth = 2; ctx.strokeStyle = surface; ctx.stroke(); ctx.lineWidth = 2;
      break;
    }
  });
}

function hover(c, ev){
  const box = c.el.querySelector(".plot").getBoundingClientRect();
  const tip = c.el.querySelector(".tip");
  const x = ev.clientX - box.left;
  if (!c._X || x < c._pad.l - 12 || x > c._pad.l + c._pw + 12){ tip.style.opacity = 0; return; }
  const i = Math.max(0, Math.min(c.x.length-1,
            Math.round((x - c._pad.l) / (c._pw || 1) * (c.x.length-1))));
  const colors = SLOTS.map(CSS);
  let html = "<b>" + (c.xLabel || "x") + " " + c.x[i] + "</b>";
  c.series.forEach((s,si) => {
    if (!c.visible[si]) return;
    html += '<div class="row"><i style="background:' + colors[si%colors.length] + '"></i>' +
            s.name + "<span>" + fmt(s.y[i] ?? null, c.percent) + "</span></div>";
  });
  tip.innerHTML = html; tip.style.opacity = 1;
  const tw = tip.offsetWidth, px = c._X(i);
  tip.style.left = Math.min(Math.max(px - tw/2, 0), box.width - tw) + "px";
  tip.style.top = "6px";
  // Crosshair on top of the redrawn plot.
  drawChart(c);
  const ctx = c.el.querySelector("canvas").getContext("2d");
  ctx.save(); ctx.strokeStyle = CSS("--axis"); ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(Math.round(px)+.5, c._pad.t);
  ctx.lineTo(Math.round(px)+.5, c._pad.t + c._ph); ctx.stroke(); ctx.restore();
}

function buildTable(c){
  let h = "<table><thead><tr><th>" + (c.xLabel || "x") + "</th>";
  c.series.forEach(s => h += "<th>" + s.name + "</th>");
  h += "</tr></thead><tbody>";
  c.x.forEach((xv,i) => {
    h += "<tr><td>" + xv + "</td>";
    c.series.forEach(s => h += "<td>" + fmt(s.y[i] ?? null, c.percent) + "</td>");
    h += "</tr>";
  });
  return h + "</tbody></table>";
}

function drawGrid(g){
  const cv = g.el.querySelector("canvas"), ctx = cv.getContext("2d");
  const n = g.cells.length, cols = g.cols, rows = Math.ceil(n/cols), pad = 1;
  // Scale so one cell lands around 44px whatever its native size -- a 3x3 conv
  // kernel is unreadable at the zoom that suits a 8x8 weight tile.
  const zoom = Math.max(4, Math.ceil(44/g.cell));
  const W = cols*g.cell + (cols+1)*pad, H = rows*g.cell + (rows+1)*pad;
  cv.width = W*zoom; cv.height = H*zoom;
  cv.style.width = (W*zoom) + "px";
  ctx.imageSmoothingEnabled = false;
  const img = ctx.createImageData(W, H);
  for (let i = 0; i < W*H; i++){ img.data[i*4] = img.data[i*4+1] = img.data[i*4+2] = 128;
                                 img.data[i*4+3] = 255; }
  g.cells.forEach((src,idx) => {
    const cr = Math.floor(idx/cols), cc = idx % cols;
    let lo = src[0], hi = src[0];
    for (const v of src){ if (v < lo) lo = v; if (v > hi) hi = v; }
    const range = (hi - lo) > 1e-12 ? (hi - lo) : 1;
    for (let y = 0; y < g.cell; y++) for (let x = 0; x < g.cell; x++){
      const q = Math.round((src[y*g.cell + x] - lo)/range * 255);
      const p = ((pad + cr*(g.cell+pad) + y)*W + (pad + cc*(g.cell+pad) + x))*4;
      img.data[p] = img.data[p+1] = img.data[p+2] = q;
    }
  });
  const off = document.createElement("canvas");
  off.width = W; off.height = H; off.getContext("2d").putImageData(img, 0, 0);
  ctx.drawImage(off, 0, 0, W*zoom, H*zoom);
}

const items = [];
function init(){
  for (const d of DATA){
    const el = document.getElementById(d.id);
    const it = Object.assign({el}, d);
    items.push(it);
    if (d.kind === "grid"){ drawGrid(it); continue; }

    it.visible = d.series.map(() => true);
    it.logOn = !!d.logY;
    const colors = SLOTS.map(CSS);
    const legend = el.querySelector(".legend");
    // One series needs no legend: the title already names what is plotted.
    if (d.series.length >= 2){
      d.series.forEach((s,i) => {
        const b = document.createElement("button");
        b.type = "button"; b.setAttribute("aria-pressed","true");
        b.innerHTML = '<i class="key" style="background:' + colors[i%colors.length] +
                      '"></i>' + s.name;
        b.onclick = () => {
          if (it.visible.filter(Boolean).length === 1 && it.visible[i]) return;
          it.visible[i] = !it.visible[i];
          b.setAttribute("aria-pressed", String(it.visible[i]));
          drawChart(it);
        };
        legend.appendChild(b);
      });
    } else { legend.remove(); }

    const tbl = el.querySelector(".tbl"), tb = el.querySelector(".tbl-btn");
    tb.onclick = () => {
      const show = tbl.hasAttribute("hidden");
      if (show){ if (!tbl.innerHTML) tbl.innerHTML = buildTable(it); tbl.removeAttribute("hidden"); }
      else tbl.setAttribute("hidden","");
      tb.textContent = show ? "Hide data table" : "Show data table";
    };
    const lb = el.querySelector(".log-btn");
    if (lb) lb.onclick = () => { it.logOn = !it.logOn;
      lb.textContent = it.logOn ? "Linear scale" : "Log scale"; drawChart(it); };

    const plot = el.querySelector(".plot");
    plot.addEventListener("mousemove", e => hover(it, e));
    plot.addEventListener("mouseleave", () => {
      el.querySelector(".tip").style.opacity = 0; drawChart(it); });
    drawChart(it);
  }
}
const redraw = () => items.forEach(i => i.kind === "grid" ? drawGrid(i) : drawChart(i));
window.addEventListener("resize", redraw);
document.getElementById("theme").onclick = () => {
  const dark = document.documentElement.getAttribute("data-theme") === "dark" ||
    (!document.documentElement.hasAttribute("data-theme") &&
     matchMedia("(prefers-color-scheme: dark)").matches);
  document.documentElement.setAttribute("data-theme", dark ? "light" : "dark");
  redraw();
};
init();
)JS";
}

}  // namespace detail

inline void Report::write(const std::string& path) const {
    std::ofstream out(path);
    if (!out) return;
    out << "<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
        << "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">\n"
        << "<title>" << detail::html_escape(title_) << "</title>\n<style>" << detail::css()
        << "</style>\n</head>\n<body>\n<div class=\"wrap\">\n"
        << "<header class=\"top\"><div><h1>" << detail::html_escape(title_) << "</h1>";
    if (!subtitle_.empty()) {
        out << "<p class=\"sub\">" << detail::html_escape(subtitle_) << "</p>";
    }
    out << "</div><button id=\"theme\" type=\"button\">Theme</button></header>\n"
        << body_
        << "<footer>Generated by nnscratch &mdash; a dependency-free neural network in "
           "C++20. This file is self-contained: no scripts, styles or data are loaded "
           "from anywhere.</footer>\n</div>\n<script>\nconst DATA=[\n"
        << data_ << "\n];\n"
        << detail::js() << "</script>\n</body>\n</html>\n";
}

/// Convenience: the columns of a Dense weight matrix as one image per unit.
inline std::vector<std::vector<double>> weight_cells(const Tensor& w) {
    std::vector<std::vector<double>> cells;
    cells.reserve(w.cols());
    for (std::size_t unit = 0; unit < w.cols(); ++unit) {
        std::vector<double> cell(w.rows());
        for (std::size_t in = 0; in < w.rows(); ++in) cell[in] = w(in, unit);
        cells.push_back(std::move(cell));
    }
    return cells;
}

/// Convenience: a rank-4 (out_c, in_c, k, k) kernel as one image per filter.
inline std::vector<std::vector<double>> filter_cells(const Tensor& w) {
    const std::size_t out_c = w.dim(0);
    const std::size_t per = w.size() / out_c;
    std::vector<std::vector<double>> cells;
    cells.reserve(out_c);
    for (std::size_t oc = 0; oc < out_c; ++oc) {
        cells.emplace_back(w.data().begin() + static_cast<std::ptrdiff_t>(oc * per),
                           w.data().begin() + static_cast<std::ptrdiff_t>((oc + 1) * per));
    }
    return cells;
}

}  // namespace nn::report

#endif  // NNSCRATCH_REPORT_HPP
