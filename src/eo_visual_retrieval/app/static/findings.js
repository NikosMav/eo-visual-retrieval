/* Charts for the findings page.

   Every value drawn here arrives in one JSON payload the server built from the
   committed evidence files. Nothing is computed in the browser beyond scaling
   to pixels, so a figure on this page cannot disagree with the recorded report.
*/
(function () {
  "use strict";

  var node = document.getElementById("findings-data");
  if (!node) {
    return;
  }
  var D = JSON.parse(node.textContent);
  var SVGNS = "http://www.w3.org/2000/svg";
  var COLORS = ["var(--s1)", "var(--s2)", "var(--s3)", "var(--s4)", "var(--s5)"];

  function el(name, attrs, text) {
    var n = document.createElementNS(SVGNS, name);
    for (var k in attrs) {
      n.setAttribute(k, attrs[k]);
    }
    if (text !== undefined) {
      n.textContent = text;
    }
    return n;
  }
  function colorOf(i) {
    return COLORS[i % COLORS.length];
  }

  var tip = document.getElementById("tip");
  function bindTip(target, text) {
    target.setAttribute("data-tip", "");
    target.addEventListener("pointerenter", function (e) {
      tip.textContent = text;
      tip.style.opacity = "1";
      tip.style.left = e.clientX + "px";
      tip.style.top = e.clientY + "px";
    });
    target.addEventListener("pointermove", function (e) {
      tip.style.left = e.clientX + "px";
      tip.style.top = e.clientY + "px";
    });
    target.addEventListener("pointerleave", function () {
      tip.style.opacity = "0";
    });
  }

  function overlapBetween(a, b) {
    var v = D.overlap[a + " vs " + b];
    return v === undefined ? D.overlap[b + " vs " + a] : v;
  }

  /* 1 — pairwise overlap matrix */
  (function () {
    var g = document.getElementById("heat");
    if (!g) {
      return;
    }
    var names = D.series.map(function (s) {
      return s.key;
    });
    var left = 156;
    var top = 44;
    var cw = 100;
    var ch = 38;
    var ramp = ["var(--r1)", "var(--r2)", "var(--r3)", "var(--r4)", "var(--r5)"];
    var top_ = D.overlap_max > 0 ? D.overlap_max : 1;

    names.forEach(function (n, c) {
      g.appendChild(el("text", {
        x: left + c * cw + cw / 2, y: top - 13, "text-anchor": "middle", class: "dlab-s"
      }, n));
    });
    names.forEach(function (rn, r) {
      g.appendChild(el("text", {
        x: left - 11, y: top + r * ch + ch / 2 + 4, "text-anchor": "end", class: "dlab-s"
      }, rn));
      names.forEach(function (cn, c) {
        var x = left + c * cw;
        var y = top + r * ch;
        if (rn === cn) {
          g.appendChild(el("rect", {
            x: x + 1, y: y + 1, width: cw - 2, height: ch - 2, fill: "var(--paper)"
          }));
          g.appendChild(el("text", {
            x: x + cw / 2, y: y + ch / 2 + 4, "text-anchor": "middle", class: "tick"
          }, "—"));
          return;
        }
        var v = overlapBetween(rn, cn);
        var band = Math.min(4, Math.floor((v / top_) * 5 - 1e-9));
        var rect = el("rect", {
          x: x + 1, y: y + 1, width: cw - 2, height: ch - 2,
          fill: ramp[band < 0 ? 0 : band]
        });
        bindTip(rect, rn + "  vs  " + cn + "\n" + v.toFixed(3) + " of " + D.k + " results shared");
        g.appendChild(rect);
        g.appendChild(el("text", {
          x: x + cw / 2, y: y + ch / 2 + 4, "text-anchor": "middle", class: "dlab",
          fill: band >= 3 ? "#ffffff" : "var(--ink)"
        }, v.toFixed(2)));
      });
    });
    g.appendChild(el("text", {
      x: left, y: top + names.length * ch + 28, class: "axlab"
    }, "mean overlap@" + D.k + "  ·  0.00 to " + top_.toFixed(2)));
  })();

  /* 2 — quality against consistency */
  (function () {
    var g = document.getElementById("scatter");
    if (!g) {
      return;
    }
    var L = 86, R = 620, T = 24, B = 300;
    var sds = D.series.map(function (s) { return s.sd; });
    var maps = D.series.map(function (s) { return s.map; });
    var xMin = Math.min.apply(null, sds) - 0.015;
    var xMax = Math.max.apply(null, sds) + 0.015;
    var yMin = Math.max(0, Math.min.apply(null, maps) - 0.08);
    var yMax = Math.min(1, Math.max.apply(null, maps) + 0.08);
    function X(v) { return L + (v - xMin) / (xMax - xMin) * (R - L); }
    function Y(v) { return B - (v - yMin) / (yMax - yMin) * (B - T); }

    [0.2, 0.4, 0.6, 0.8].forEach(function (v) {
      if (v < yMin || v > yMax) { return; }
      g.appendChild(el("line", { x1: L, x2: R, y1: Y(v), y2: Y(v), class: "gridline" }));
      g.appendChild(el("text", { x: L - 9, y: Y(v) + 4, "text-anchor": "end", class: "tick" }, v.toFixed(1)));
    });
    g.appendChild(el("line", { x1: L, x2: R, y1: B, y2: B, class: "axis" }));
    g.appendChild(el("line", { x1: L, x2: L, y1: T, y2: B, class: "axis" }));
    sds.forEach(function (v) {
      g.appendChild(el("text", { x: X(v), y: B + 18, "text-anchor": "middle", class: "tick" }, v.toFixed(3)));
    });
    g.appendChild(el("text", { x: L, y: B + 40, class: "axlab" },
      "more consistent  ←   spread of per-cell score (SD)   →  more erratic"));
    g.appendChild(el("text", {
      x: 0, y: 0, class: "axlab", "text-anchor": "middle",
      transform: "translate(" + (L - 46) + "," + ((T + B) / 2) + ") rotate(-90)"
    }, "mAP@" + D.k));

    D.series.forEach(function (s, i) {
      var cx = X(s.sd), cy = Y(s.map);
      g.appendChild(el("circle", { cx: cx, cy: cy, r: 9, fill: "var(--card)" }));
      var dot = el("circle", { cx: cx, cy: cy, r: 6, fill: colorOf(i) });
      bindTip(dot, s.key + "\nmAP@" + D.k + "  " + s.map.toFixed(4) +
        "\nspread   " + s.sd.toFixed(4) + "\nworst cell " + s.min.toFixed(3));
      g.appendChild(dot);
      var right = cx < (L + R) / 2;
      g.appendChild(el("text", {
        x: cx + (right ? 14 : -14), y: cy - 8,
        "text-anchor": right ? "start" : "end", class: "dlab"
      }, s.key));
      g.appendChild(el("text", {
        x: cx + (right ? 14 : -14), y: cy + 6,
        "text-anchor": right ? "start" : "end", class: "tick"
      }, s.map.toFixed(3)));
    });
  })();

  /* 3 — quality by distance to the nearest same-class image */
  (function () {
    var g = document.getElementById("slope");
    if (!g) {
      return;
    }
    var L = 68, R = 520, T = 22, B = 286;
    var all = D.series.reduce(function (acc, s) { return acc.concat(s.q); }, []);
    var yMin = Math.max(0, Math.min.apply(null, all) - 0.06);
    var yMax = Math.min(1, Math.max.apply(null, all) + 0.06);
    function X(i) { return L + i * (R - L) / 3; }
    function Y(v) { return B - (v - yMin) / (yMax - yMin) * (B - T); }

    var e = D.quartile_edges_km;
    var qlab = [
      "Q1  " + e[0] + "–" + e[1] + " km",
      "Q2  " + e[1] + "–" + e[2] + " km",
      "Q3  " + e[2] + "–" + e[3] + " km",
      "Q4  " + e[3] + "–" + e[4] + " km"
    ];
    [0.2, 0.4, 0.6, 0.8].forEach(function (v) {
      if (v < yMin || v > yMax) { return; }
      g.appendChild(el("line", { x1: L, x2: R, y1: Y(v), y2: Y(v), class: "gridline" }));
      g.appendChild(el("text", { x: L - 9, y: Y(v) + 4, "text-anchor": "end", class: "tick" }, v.toFixed(1)));
    });
    qlab.forEach(function (t, i) {
      g.appendChild(el("line", { x1: X(i), x2: X(i), y1: T, y2: B, class: "gridline" }));
      g.appendChild(el("text", { x: X(i), y: B + 20, "text-anchor": "middle", class: "tick" }, t));
    });
    g.appendChild(el("line", { x1: L, x2: R, y1: B, y2: B, class: "axis" }));
    g.appendChild(el("text", { x: L, y: B + 40, class: "axlab" },
      "distance from query to the nearest index image of the same class"));

    D.series.forEach(function (s, i) {
      var d = s.q.map(function (v, j) { return (j ? "L" : "M") + X(j) + " " + Y(v); }).join(" ");
      g.appendChild(el("path", {
        d: d, fill: "none", stroke: colorOf(i), "stroke-width": 2, "stroke-linejoin": "round"
      }));
      s.q.forEach(function (v, j) {
        g.appendChild(el("circle", { cx: X(j), cy: Y(v), r: 5.4, fill: "var(--card)" }));
        var dot = el("circle", { cx: X(j), cy: Y(v), r: 3.4, fill: colorOf(i) });
        bindTip(dot, s.key + "\n" + qlab[j] + "\nmAP@" + D.k + "  " + v.toFixed(4));
        g.appendChild(dot);
      });
    });

    /* Two series can end within a pixel of each other, so end labels are placed
       in order with a minimum gap and joined back to their line by a leader. */
    var placed = D.series.map(function (s, i) {
      return { s: s, i: i, at: Y(s.q[3]), y: Y(s.q[3]) };
    }).sort(function (a, b) { return a.y - b.y; });
    for (var n = 1; n < placed.length; n++) {
      if (placed[n].y - placed[n - 1].y < 32) {
        placed[n].y = placed[n - 1].y + 32;
      }
    }
    placed.forEach(function (p) {
      if (Math.abs(p.y - p.at) > 2) {
        g.appendChild(el("path", {
          d: "M" + (R + 4) + " " + p.at + " L" + (R + 10) + " " + p.y,
          fill: "none", stroke: "var(--line)", "stroke-width": 1
        }));
      }
      g.appendChild(el("text", { x: R + 14, y: p.y + 4, class: "dlab" }, p.s.key));
      var peak = Math.max.apply(null, p.s.q);
      var drop = (peak - p.s.q[3]) / peak * 100;
      g.appendChild(el("text", { x: R + 14, y: p.y + 17, class: "tick" },
        "−" + drop.toFixed(0) + "% from peak"));
    });

    var lg = document.getElementById("legend-slope");
    if (lg) {
      D.series.forEach(function (s, i) {
        var li = document.createElement("li");
        var sw = document.createElement("span");
        sw.className = "swatch";
        sw.style.background = colorOf(i);
        li.appendChild(sw);
        li.appendChild(document.createTextNode(s.key));
        lg.appendChild(li);
      });
    }
  })();

  /* 4 — clear-sky days by place */
  (function () {
    var g = document.getElementById("bars");
    if (!g) {
      return;
    }
    var L = 116, T = 16, bh = 24, gap = 6, W = 440;
    var days = D.places.map(function (p) { return p.days; });
    var maxV = Math.max.apply(null, days);
    var minV = Math.min.apply(null, days);
    function w(v) { return v / maxV * W; }

    var ticks = [0, Math.round(maxV / 3), Math.round(2 * maxV / 3), maxV];
    ticks.forEach(function (v) {
      g.appendChild(el("line", {
        x1: L + w(v), x2: L + w(v), y1: T - 5,
        y2: T + D.places.length * (bh + gap) - gap + 3, class: "gridline"
      }));
      g.appendChild(el("text", { x: L + w(v), y: T - 11, "text-anchor": "middle", class: "tick" }, v));
    });
    D.places.forEach(function (p, i) {
      var y = T + i * (bh + gap);
      var extreme = p.days === maxV || p.days === minV;
      g.appendChild(el("text", {
        x: L - 11, y: y + bh / 2 + 4, "text-anchor": "end", class: "dlab-s"
      }, p.label));
      var bar = el("rect", {
        x: L, y: y, width: Math.max(2, w(p.days)), height: bh, rx: 4, ry: 4,
        fill: extreme ? "var(--s1)" : "var(--r3)"
      });
      bindTip(bar, p.place_id + "\n" + p.latitude.toFixed(2) + "° N\n" +
        p.days + " clear days");
      g.appendChild(bar);
      g.appendChild(el("text", { x: L + w(p.days) + 8, y: y + bh / 2 + 4, class: "dlab" }, p.days));
      g.appendChild(el("text", {
        x: L + w(p.days) + 30, y: y + bh / 2 + 4, class: "tick"
      }, p.latitude.toFixed(1) + "° N"));
    });
    g.appendChild(el("text", {
      x: L, y: T + D.places.length * (bh + gap) + 14, class: "axlab"
    }, "clear-sky acquisition days  ·  ordered south to north"));
  })();

  /* table views, required because several series sit below 3:1 on this surface */
  (function () {
    var rows = document.getElementById("series-rows");
    if (rows) {
      D.series.forEach(function (s) {
        var tr = document.createElement("tr");
        [s.key, s.map.toFixed(4), s.sd.toFixed(4), s.min.toFixed(3), s.med.toFixed(3)]
          .forEach(function (v, i) {
            var td = document.createElement("td");
            if (i) { td.className = "num"; }
            td.textContent = v;
            tr.appendChild(td);
          });
        rows.appendChild(tr);
      });
    }
    var pr = document.getElementById("place-rows");
    if (pr) {
      D.places.forEach(function (p) {
        var tr = document.createElement("tr");
        [p.place_id, p.latitude.toFixed(2) + "° N", String(p.days)]
          .forEach(function (v, i) {
            var td = document.createElement("td");
            if (i) { td.className = "num"; }
            td.textContent = v;
            tr.appendChild(td);
          });
        pr.appendChild(tr);
      });
    }
  })();
})();
