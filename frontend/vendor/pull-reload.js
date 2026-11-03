/* Pull to reload for touch phones and tablets, shared by the chat and the ontology page.
 *
 * Both pages keep the page itself fixed and scroll inside it (or scroll a 3D view), so the
 * browser's own pull-to-refresh can't do its job. Instead, a little lifter in a small card
 * slides out from under the page's header: the further the pull, the higher he curls the
 * dumbbell and the harder his face strains (data-effort 0-3). Letting go at the top of the
 * curl reloads the page; letting go early lowers the weight and tucks the card away.
 *
 *   pullToReload({
 *     anchor,            // the header the card slides out from under
 *     zone(target),      // may a pull start from this touch target?
 *     allowed(),         // the page's own conditions (e.g. not while dictating)
 *     shift,             // optional element moved down with the pull (the chat's messages)
 *     beforeReload(),    // optional, e.g. keep an unsent message
 *   });
 *
 * Colours come from CSS custom properties (--ptr-*) with the chat's dark defaults; a page sets
 * its own on :root. Only runs on touch screens up to 1024px wide.
 */
(function () {
  'use strict';
  const READY = 70, MAX = 100;
  const TOUCH = matchMedia('(hover: none) and (pointer: coarse)'), COMPACT = matchMedia('(max-width: 1024px)');

  const CSS = `
  .ptr-clip { position: fixed; left: 0; right: 0; height: 0; overflow: hidden; pointer-events: none; z-index: 60; }
  .ptr-card { position: absolute; left: 50%; bottom: 6px; transform: translateX(-50%);
    display: flex; flex-direction: column; align-items: center; gap: 3px; padding: 6px 10px 6px;
    background: var(--ptr-bg, #1e293b); border: 1px solid var(--ptr-border, #334155); border-radius: 14px;
    box-shadow: var(--ptr-shadow, 0 6px 18px rgba(2, 6, 23, .55)); }
  .ptr-clip.settling { transition: height .25s ease; }
  .ptr-card svg { overflow: visible; display: block; }
  .ptr-card .body path, .ptr-card .forearm path { fill: none; stroke: var(--ptr-stroke, #94a3b8); stroke-width: 1.8;
    stroke-linecap: round; stroke-linejoin: round; }
  .ptr-card .head { fill: var(--ptr-bg, #1e293b); stroke: var(--ptr-stroke, #94a3b8); stroke-width: 1.6; }
  .ptr-card .forearm { transform-box: view-box; transform-origin: 26px 24px; }
  .ptr-clip.settling .forearm { transition: transform .25s ease; }
  .ptr-card .forearm .bell { stroke: var(--ptr-bell, #cbd5e1); stroke-width: 1.4; }
  .ptr-card .plate { fill: var(--ptr-plate, #3b82f6); }
  .ptr-card .face path { fill: none; stroke: var(--ptr-face, #e2e8f0); stroke-width: 1; stroke-linecap: round; stroke-linejoin: round; }
  .ptr-card .face .calm circle, .ptr-card .m-grit rect { fill: var(--ptr-face, #e2e8f0); stroke: none; }
  .ptr-card .m-grit path { stroke: var(--ptr-bg, #1e293b); stroke-width: .6; }
  .ptr-card .cheek { fill: #f87171; opacity: .55; }
  .ptr-card .face .sweat { fill: #60a5fa; stroke: none; }
  .ptr-card .squint, .ptr-card .brows, .ptr-card .m-flat, .ptr-card .m-grit, .ptr-card .strain { display: none; }
  .ptr-card [data-effort="1"] .m-smile, .ptr-card [data-effort="2"] .m-smile, .ptr-card [data-effort="3"] .m-smile { display: none; }
  .ptr-card [data-effort="1"] .m-flat, .ptr-card [data-effort="1"] .brows { display: inline; }
  .ptr-card [data-effort="2"] .calm, .ptr-card [data-effort="3"] .calm { display: none; }
  .ptr-card [data-effort="2"] .squint, .ptr-card [data-effort="3"] .squint,
  .ptr-card [data-effort="2"] .m-grit, .ptr-card [data-effort="3"] .m-grit, .ptr-card [data-effort="3"] .strain { display: inline; }
  .ptr-label { font: 10px/1 system-ui, sans-serif; color: var(--ptr-label, #94a3b8); white-space: nowrap; }
  .ptr-card.ready .ptr-label, .ptr-card.reloading .ptr-label { color: var(--ptr-label-on, #e2e8f0); }
  .ptr-card.ready svg { animation: ptr-tremble .12s linear infinite; }
  .ptr-card.reloading .forearm { animation: ptr-curl .7s ease-in-out infinite; }
  @keyframes ptr-tremble { 25% { transform: translateX(-.5px); } 75% { transform: translateX(.5px); } }
  @keyframes ptr-curl { 0%, 100% { transform: rotate(-20deg); } 50% { transform: rotate(-150deg); } }
  @media (prefers-reduced-motion: reduce) {
    .ptr-card.ready svg, .ptr-card.reloading .forearm { animation: none; }
    .ptr-clip.settling, .ptr-clip.settling .forearm { transition: none; }
  }`;

  // A stick lifter facing the viewer; the curling forearm turns about the elbow at (26, 24).
  const SVG = `
  <svg viewBox="0 0 40 46" width="22" height="26" data-effort="0" aria-hidden="true">
    <g class="body"><circle class="head" cx="20" cy="8" r="6"/>
      <path d="M20 14v15M20 29l-5 14M20 29l5 14M20 17l-6 6-1 7M20 17l6 7"/></g>
    <g class="forearm"><path d="M26 24v9"/><path class="bell" d="M22.5 33.5h7"/>
      <rect class="plate" x="20.5" y="31" width="2.4" height="5" rx=".8"/>
      <rect class="plate" x="29.1" y="31" width="2.4" height="5" rx=".8"/></g>
    <g class="face">
      <g class="calm"><circle cx="17.8" cy="7.4" r=".9"/><circle cx="22.2" cy="7.4" r=".9"/></g>
      <g class="squint"><path d="M16.8 6.6l1.8.9-1.8.9M23.2 6.6l-1.8.9 1.8.9"/></g>
      <path class="brows" d="M16.6 5.3l2.2.6M23.4 5.3l-2.2.6"/>
      <path class="m-smile" d="M17.6 10.2q2.4 1.8 4.8 0"/>
      <path class="m-flat" d="M17.8 10.8h4.4"/>
      <g class="m-grit"><rect x="17.3" y="9.8" width="5.4" height="2.2" rx=".6"/><path d="M17.5 10.9h5"/></g>
      <g class="strain"><circle class="cheek" cx="15.9" cy="9.6" r="1.1"/><circle class="cheek" cx="24.1" cy="9.6" r="1.1"/>
        <path class="sweat" d="M28.4 2.4q1.5 2.3 0 3.2q-1.5-.9 0-3.2z"/></g>
    </g>
  </svg>`;

  window.pullToReload = function ({anchor, zone, allowed = () => true, shift = null, beforeReload = () => {}}) {
    const style = document.createElement('style');
    style.textContent = CSS;
    document.head.appendChild(style);
    const clip = document.createElement('div');
    clip.className = 'ptr-clip';
    clip.setAttribute('aria-hidden', 'true');
    clip.innerHTML = `<div class="ptr-card">${SVG}<span class="ptr-label">Pull to reload</span></div>`;
    document.body.appendChild(clip);
    const card = clip.firstElementChild, lifter = card.querySelector('svg');
    const forearm = card.querySelector('.forearm'), label = card.querySelector('.ptr-label');
    let startY = null, startX = 0, pulling = false, ready = false, reloading = false;

    function show(px) {
      clip.style.height = px + 'px';
      const moved = shift && shift();
      if (moved) moved.style.transform = px ? `translateY(${px}px)` : '';
      // The curl follows the pull: arm hanging at rest, the dumbbell beside his head
      // (150 degrees) when letting go would reload. His face works harder with it.
      const p = Math.min(1, px / READY);
      forearm.style.transform = `rotate(${-Math.round(150 * p)}deg)`;
      lifter.dataset.effort = p >= 1 ? 3 : p >= 0.66 ? 2 : p >= 0.33 ? 1 : 0;
      if ((px >= READY) !== ready) {
        ready = px >= READY;
        card.classList.toggle('ready', ready);
        label.textContent = ready ? 'Release to reload' : 'Pull to reload';
        if (ready && navigator.vibrate && navigator.userActivation?.hasBeenActive) navigator.vibrate(8);
      }
    }
    function settleTo(px) {
      clip.classList.add('settling');
      const moved = shift && shift();
      if (moved) moved.style.transition = 'transform .25s ease';
      show(px);
      setTimeout(() => { clip.classList.remove('settling'); if (moved) moved.style.transition = ''; }, 260);
    }
    function reload() {
      reloading = true;
      settleTo(READY - 8);
      card.classList.add('reloading');
      label.textContent = 'Reloading…';
      try { beforeReload(); } catch {}
      setTimeout(() => location.reload(), 650);   // one rep before the page goes
    }

    document.addEventListener('touchstart', e => {
      startY = null;
      if (!TOUCH.matches || !COMPACT.matches || reloading || e.touches.length !== 1) return;
      if (!allowed() || !zone(e.target)) return;
      clip.style.top = Math.max(0, anchor.getBoundingClientRect().bottom) + 'px';
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
      show(Math.min(MAX, dy * 0.55));
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
