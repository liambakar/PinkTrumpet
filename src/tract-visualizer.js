const TAU = Math.PI * 2;
const lerp = (a, b, t) => a + (b - a) * t;
const clamp = (value, min, max) => Math.min(max, Math.max(min, value));

export class TractVisualizer {
  constructor(canvas, onChange) {
    this.canvas = canvas;
    this.context = canvas.getContext("2d");
    this.onChange = onChange;
    this.parameters = null;
    this.dragMode = null;
    this.frame = null;
    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(canvas);
    canvas.addEventListener("pointerdown", (event) => this.pointerDown(event));
    canvas.addEventListener("pointermove", (event) => this.pointerMove(event));
    canvas.addEventListener("pointerup", () => this.pointerUp());
    canvas.addEventListener("pointercancel", () => this.pointerUp());
  }

  setParameters(parameters) {
    this.parameters = parameters;
    this.draw();
  }

  resize() {
    const rect = this.canvas.getBoundingClientRect();
    const dpr = Math.min(2, devicePixelRatio || 1);
    this.canvas.width = Math.round(rect.width * dpr);
    this.canvas.height = Math.round(rect.height * dpr);
    this.context.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.draw();
  }

  geometry() {
    const { width, height } = this.canvas.getBoundingClientRect();
    const scale = Math.min(width / 730, height / 510);
    const origin = { x: width * 0.53, y: height * 0.72 };
    const inner = 114 * scale;
    const outer = 278 * scale;
    return { width, height, scale, origin, inner, outer, startAngle: Math.PI * 1.09, endAngle: Math.PI * 1.88 };
  }

  tractPoint(index, diameter, geometry = this.geometry()) {
    const t = clamp(index / 44, 0, 1);
    const angle = lerp(geometry.startAngle, geometry.endAngle, t);
    const radius = lerp(geometry.outer, geometry.inner, clamp(diameter / 4, 0, 1));
    return {
      x: geometry.origin.x + Math.cos(angle) * radius,
      y: geometry.origin.y + Math.sin(angle) * radius,
    };
  }

  diameterAt(index) {
    const p = this.parameters;
    let diameter = index < 7 ? 0.6 : index < 12 ? 1.1 : 1.5;
    if (index >= 10 && index < 39) {
      const t = 1.1 * Math.PI * (p.tongueIndex - index) / 22;
      diameter = 1.5 - (1.5 - (2 + (p.tongueDiameter - 2) / 1.5)) * Math.cos(t);
    }
    const distance = Math.abs(index - p.constrictionIndex);
    const width = p.constrictionIndex >= 32 ? 5 : 8;
    if (distance < width) {
      const blend = 0.5 * (1 + Math.cos(Math.PI * distance / width));
      diameter = lerp(diameter, p.constrictionDiameter, blend);
    }
    return clamp(diameter, 0.05, 3.7);
  }

  draw() {
    if (!this.parameters || !this.canvas.width) return;
    cancelAnimationFrame(this.frame);
    this.frame = requestAnimationFrame(() => this.paint());
  }

  paint() {
    const ctx = this.context;
    const g = this.geometry();
    ctx.clearRect(0, 0, g.width, g.height);

    const guide = (diameter, alpha, dash = []) => {
      ctx.beginPath();
      for (let i = 0; i <= 44; i += 1) {
        const point = this.tractPoint(i, diameter, g);
        if (!i) ctx.moveTo(point.x, point.y); else ctx.lineTo(point.x, point.y);
      }
      ctx.setLineDash(dash);
      ctx.strokeStyle = `rgba(240, 108, 155, ${alpha})`;
      ctx.lineWidth = 1;
      ctx.stroke();
      ctx.setLineDash([]);
    };
    guide(0, 0.12);
    guide(1.5, 0.09, [3, 7]);
    guide(3.6, 0.13);

    const outerPoints = [];
    const innerPoints = [];
    for (let i = 0; i <= 44; i += 1) {
      outerPoints.push(this.tractPoint(i, 0, g));
      innerPoints.push(this.tractPoint(i, this.diameterAt(i), g));
    }

    const gradient = ctx.createRadialGradient(g.origin.x, g.origin.y, g.inner, g.origin.x, g.origin.y, g.outer);
    gradient.addColorStop(0, "rgba(247, 220, 215, 0.34)");
    gradient.addColorStop(1, "rgba(240, 108, 155, 0.1)");
    ctx.beginPath();
    outerPoints.forEach((p, i) => i ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y));
    [...innerPoints].reverse().forEach((p) => ctx.lineTo(p.x, p.y));
    ctx.closePath();
    ctx.fillStyle = gradient;
    ctx.fill();

    ctx.beginPath();
    innerPoints.forEach((p, i) => i ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y));
    ctx.strokeStyle = "rgba(247, 220, 215, 0.82)";
    ctx.lineWidth = 2.5;
    ctx.shadowColor = "rgba(240,108,155,.22)";
    ctx.shadowBlur = 12;
    ctx.stroke();
    ctx.shadowBlur = 0;

    // Nasal branch and velum gate.
    const nasalBase = this.tractPoint(17, this.diameterAt(17), g);
    const nasalEnd = { x: g.width * 0.72, y: g.height * 0.10 };
    ctx.beginPath();
    ctx.moveTo(nasalBase.x, nasalBase.y);
    ctx.quadraticCurveTo(g.width * 0.58, g.height * 0.17, nasalEnd.x, nasalEnd.y);
    ctx.strokeStyle = `rgba(240,108,155,${0.12 + this.parameters.velum * 1.4})`;
    ctx.lineWidth = 14 * g.scale + this.parameters.velum * 24;
    ctx.lineCap = "round";
    ctx.stroke();

    // Tongue control target.
    const tongue = this.tractPoint(this.parameters.tongueIndex, this.parameters.tongueDiameter, g);
    ctx.beginPath();
    ctx.arc(tongue.x, tongue.y, 14 * g.scale + 5, 0, TAU);
    ctx.fillStyle = "rgba(240,108,155,.15)";
    ctx.strokeStyle = "#f06c9b";
    ctx.lineWidth = 2;
    ctx.fill();
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(tongue.x, tongue.y, 3.5, 0, TAU);
    ctx.fillStyle = "#ffd1de";
    ctx.fill();

    // Constriction control target.
    const constriction = this.tractPoint(this.parameters.constrictionIndex, this.parameters.constrictionDiameter, g);
    ctx.beginPath();
    ctx.arc(constriction.x, constriction.y, 7, 0, TAU);
    ctx.fillStyle = this.parameters.fricativeIntensity > 0.1 ? "#ffbdcf" : "#f06c9b";
    ctx.fill();

    ctx.fillStyle = "rgba(239,225,231,.55)";
    ctx.font = `${Math.max(10, 11 * g.scale)}px ui-sans-serif, system-ui`;
    ctx.textAlign = "center";
    ctx.fillText("NASAL CAVITY", g.width * .67, g.height * .055);
    ctx.fillText("LIPS", g.width * .87, g.height * .47);
    ctx.textAlign = "left";
    ctx.fillText("GLOTTIS", g.width * .12, g.height * .9);
  }

  localPointer(event) {
    const rect = this.canvas.getBoundingClientRect();
    return { x: event.clientX - rect.left, y: event.clientY - rect.top };
  }

  inversePoint(point) {
    const g = this.geometry();
    const dx = point.x - g.origin.x;
    const dy = point.y - g.origin.y;
    let angle = Math.atan2(dy, dx);
    if (angle < g.startAngle) angle += TAU;
    const index = clamp((angle - g.startAngle) / (g.endAngle - g.startAngle) * 44, 2, 42);
    const radius = Math.hypot(dx, dy);
    const diameter = clamp((g.outer - radius) / (g.outer - g.inner) * 4, 0, 3.5);
    return { index, diameter };
  }

  pointerDown(event) {
    this.canvas.setPointerCapture(event.pointerId);
    const pointer = this.localPointer(event);
    const tongue = this.tractPoint(this.parameters.tongueIndex, this.parameters.tongueDiameter);
    this.dragMode = Math.hypot(pointer.x - tongue.x, pointer.y - tongue.y) < 42 ? "tongue" : "constriction";
    this.updateFromPointer(pointer);
  }

  pointerMove(event) {
    if (!this.dragMode) return;
    this.updateFromPointer(this.localPointer(event));
  }

  pointerUp() { this.dragMode = null; }

  updateFromPointer(pointer) {
    const value = this.inversePoint(pointer);
    if (this.dragMode === "tongue") {
      this.onChange({ tongueIndex: clamp(value.index, 12, 29), tongueDiameter: clamp(value.diameter, 2.05, 3.5) });
    } else {
      this.onChange({ constrictionIndex: value.index, constrictionDiameter: value.diameter });
    }
  }
}
