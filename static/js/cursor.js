(function () {
  // Only activate for mouse/pointer devices, skip touch screens
  if (window.matchMedia && window.matchMedia('(pointer: coarse)').matches) return;

  var orb = document.createElement('div');
  orb.id = 'gf-cursor-orb';
  var aura = document.createElement('div');
  aura.id = 'gf-cursor-aura';

  function initCursor() {
    if (!document.body) return;
    if (!document.getElementById('gf-cursor-aura')) {
      document.body.appendChild(aura);
    }
    if (!document.getElementById('gf-cursor-orb')) {
      document.body.appendChild(orb);
    }
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initCursor);
  } else {
    initCursor();
  }

  var mouseX = -200, mouseY = -200;
  var auraX = -200, auraY = -200;
  var isVisible = false;

  window.addEventListener('mousemove', function (e) {
    mouseX = e.clientX;
    mouseY = e.clientY;

    if (!isVisible) {
      isVisible = true;
      orb.style.opacity = '1';
      aura.style.opacity = '1';
      auraX = mouseX;
      auraY = mouseY;
    }

    orb.style.transform = 'translate3d(' + mouseX + 'px, ' + mouseY + 'px, 0)';
  }, { passive: true });

  document.addEventListener('mouseleave', function () {
    orb.style.opacity = '0';
    aura.style.opacity = '0';
    isVisible = false;
  });

  document.addEventListener('mouseenter', function () {
    orb.style.opacity = '1';
    aura.style.opacity = '1';
    isVisible = true;
  });

  document.addEventListener('mousedown', function () {
    orb.classList.add('is-active');
    aura.classList.add('is-active');
  });

  document.addEventListener('mouseup', function () {
    orb.classList.remove('is-active');
    aura.classList.remove('is-active');
  });

  var interactiveSelector = 'a, button, input, label, select, textarea, [role="button"], [tabindex]:not([tabindex="-1"]), .cursor-pointer';

  document.addEventListener('mouseover', function (e) {
    if (e.target && e.target.closest && e.target.closest(interactiveSelector)) {
      orb.classList.add('is-hover');
      aura.classList.add('is-hover');
    }
  }, { passive: true });

  document.addEventListener('mouseout', function (e) {
    if (e.target && e.target.closest && e.target.closest(interactiveSelector)) {
      orb.classList.remove('is-hover');
      aura.classList.remove('is-hover');
    }
  }, { passive: true });

  function renderAura() {
    if (isVisible) {
      auraX += (mouseX - auraX) * 0.18;
      auraY += (mouseY - auraY) * 0.18;
      aura.style.transform = 'translate3d(' + auraX + 'px, ' + auraY + 'px, 0)';
    }
    requestAnimationFrame(renderAura);
  }
  requestAnimationFrame(renderAura);
})();
