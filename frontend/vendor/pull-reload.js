/* Pull to reload for touch phones and tablets, shared by the chat and the ontology page.
 *
 * Both pages keep the page itself fixed and scroll inside it (or scroll a 3D view), so the
 * browser's own pull-to-refresh can't do its job. Instead, a round badge drops in from the
 * top of the screen, over the header, with a muscular arm inside it (vendor/curl-arm) that
 * curls its dumbbell as you pull: hanging at the start, at the top of the curl when letting go
 * would reload, straining harder on the way. A ring around the badge fills with the pull and
 * turns the accent colour once it closes; there the badge stops, however much further the finger
 * goes. Letting go then reloads, the arm carrying on with its reps in that same spot while the
 * page goes; letting go early lowers it and the badge slides back up. The page doesn't move.
 *
 *   pullToReload({
 *     anchor,            // the header: at full pull the badge sits just over its bottom edge
 *     zone(target),      // may a pull start from this touch target?
 *     allowed(),         // the page's own conditions (e.g. not while dictating)
 *     beforeReload(),    // optional, e.g. keep an unsent message
 *   });
 *
 * The badge takes its colours from CSS custom properties (--ptr-*) with the chat's dark
 * defaults; a page sets its own on :root. The arm keeps its own colours on every page.
 * Only runs on touch screens up to 1024px wide, and only those download the arm (about 3 MB
 * with its three.js), once the page has loaded. Until it is in, the badge shows just the ring.
 */
(function () {
  'use strict';
  const READY = 70, D = 92;
  const TOUCH = matchMedia('(hover: none) and (pointer: coarse)'), COMPACT = matchMedia('(max-width: 1024px)');
  const STILL = matchMedia('(prefers-reduced-motion: reduce)');
  const ARM_MODULE = new URL('curl-arm/curl-arm.js', document.currentScript.src).href;

  const CSS = `
  .ptr-badge { position: fixed; left: 50%; top: 0; width: ${D}px; height: ${D}px; margin-left: -${D / 2}px;
    border-radius: 50%; background: var(--ptr-bg, #1e293b); box-shadow: var(--ptr-shadow, 0 6px 18px rgba(2, 6, 23, .55));
    z-index: 60; pointer-events: none; visibility: hidden; will-change: transform; }
  .ptr-ringbox { position: absolute; inset: 0; overflow: visible; transform: rotate(-90deg); z-index: 1; }
  .ptr-track, .ptr-ring { fill: none; stroke-width: 2.5; }
  .ptr-track { stroke: var(--ptr-border, #334155); }
  .ptr-ring { stroke: var(--ptr-ring, #94a3b8); stroke-linecap: round; transition: stroke .15s; }
  .ptr-badge.ready .ptr-ring, .ptr-badge.reloading .ptr-ring { stroke: var(--ptr-ring-on, #3b82f6); }
  .ptr-clip { position: absolute; inset: 0; border-radius: 50%; overflow: hidden; }
  .ptr-arm { position: absolute; inset: 0; }
  .ptr-arm canvas { display: block; width: 100%; height: 100%; }`;

  window.pullToReload = function ({anchor, zone, allowed = () => true, beforeReload = () => {}}) {
    const style = document.createElement('style');
    style.textContent = CSS;
    document.head.appendChild(style);
    const badge = document.createElement('div');
    badge.className = 'ptr-badge';
    badge.setAttribute('aria-hidden', 'true');
    badge.innerHTML = `<svg class="ptr-ringbox" viewBox="0 0 ${D} ${D}">
      <circle class="ptr-track" cx="${D / 2}" cy="${D / 2}" r="${D / 2 - 1.5}"/>
      <circle class="ptr-ring" cx="${D / 2}" cy="${D / 2}" r="${D / 2 - 1.5}" pathLength="100" stroke-dasharray="0 100"/>
      </svg><div class="ptr-clip"><div class="ptr-arm"></div></div>`;
    document.body.appendChild(badge);
    const ring = badge.querySelector('.ptr-ring'), armBox = badge.querySelector('.ptr-arm');
    let startY = null, startX = 0, pulling = false, ready = false, reloading = false;
    let shown = 0, stop = 0, tween = 0;

    // The arm arrives after the page has loaded, on the devices that can pull. While the badge
    // is out it is re-posed every frame, so the strain shake keeps going even with the finger still.
    let arm = null, armFrame = 0, armLast = 0;
    if (TOUCH.matches && COMPACT.matches) {
      const load = () => import(ARM_MODULE)
        .then(m => m.createCurlArm(armBox, {padding: 0.09}))
        .then(a => { arm = a; if (reloading) { arm.play(); armFrame = requestAnimationFrame(armLoop); } })
        .catch(() => {});                          // no WebGL or a failed download: the ring still works
      if (document.readyState === 'complete') setTimeout(load, 0); else addEventListener('load', load, {once: true});
    }
    // The arm is framed for the whole rep, so once curled it would sit in the top half of the
    // badge; it is lowered as it curls to stay centred, and raised again as it hangs, so the
    // hand and dumbbell stay inside the circle at the bottom of each rep.
    const centre = curl => { armBox.style.transform = `translateY(${(.2 * D * curl).toFixed(1)}px)`; };
    function armLoop(now) {
      armFrame = 0;
      if (!arm || shown <= 0) return;
      if (reloading) centre(arm.curl);
      else {
        const p = shown / READY;
        centre(p);
        arm.lift(p, STILL.matches ? 0 : Math.min(.1, (now - armLast) / 1000));
      }
      armLast = now;
      armFrame = requestAnimationFrame(armLoop);
    }

    // px is the damped pull (0-READY). The badge starts out of sight above the screen and at
    // READY its middle sits just below the header's bottom edge, so it travels over the header.
    function show(px) {
      shown = px;
      badge.style.visibility = px > 0 ? 'visible' : 'hidden';
      const p = px / READY;
      badge.style.transform = `translateY(${-D - 8 + (stop + D + 8) * p}px)`;
      ring.setAttribute('stroke-dasharray', `${(p * 100).toFixed(1)} 100`);
      if (px > 0 && !armFrame) { armLast = performance.now(); armFrame = requestAnimationFrame(armLoop); }
      if ((px >= READY) !== ready) {
        ready = px >= READY;
        badge.classList.toggle('ready', ready);
        if (ready && navigator.vibrate && navigator.userActivation?.hasBeenActive) navigator.vibrate(8);
      }
    }
    function settleTo(px) {
      cancelAnimationFrame(tween);
      if (STILL.matches) return show(px);
      const from = shown, t0 = performance.now();
      const step = now => {
        const t = Math.min(1, (now - t0) / 250), e = 1 - (1 - t) ** 3;
        show(from + (px - from) * e);
        if (t < 1) tween = requestAnimationFrame(step);
      };
      tween = requestAnimationFrame(step);
    }
    function reload() {
      reloading = true;
      cancelAnimationFrame(tween);
      show(READY);
      badge.classList.remove('ready');
      badge.classList.add('reloading');
      try { beforeReload(); } catch {}
      if (arm) arm.play();                         // the squeeze at the top, then on into the rep
      setTimeout(() => location.reload(), 800);
    }

    document.addEventListener('touchstart', e => {
      startY = null;
      if (!TOUCH.matches || !COMPACT.matches || reloading || e.touches.length !== 1) return;
      if (!allowed() || !zone(e.target)) return;
      cancelAnimationFrame(tween);
      stop = Math.max(0, anchor.getBoundingClientRect().bottom) - D / 2 + 6;
      startY = e.touches[0].clientY;
      startX = e.touches[0].clientX;
      pulling = false;
    }, {passive: true});
    document.addEventListener('touchmove', e => {
      if (startY === null) return;
      const dy = e.touches[0].clientY - startY, dx = e.touches[0].clientX - startX;
      if (!pulling) {
        if (dy < 0 || Math.abs(dx) > Math.abs(dy)) { startY = null; return; }   // scrolling or a sideways swipe
        if (dy < 6) return;
        pulling = true;
      }
      e.preventDefault();                          // the pull, not the browser's scroll or bounce
      show(Math.min(READY, dy * 0.55));          // stops where the ring closes
    }, {passive: false});
    const release = cancelled => {
      if (startY === null) return;
      startY = null;
      if (!pulling) return;
      pulling = false;
      if (ready && !cancelled) reload(); else settleTo(0);
    };
    document.addEventListener('touchend', () => release(false));
    document.addEventListener('touchcancel', () => release(true));
  };
})();
