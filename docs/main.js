// User-initiated playback only. Pause offscreen media and competing players.
const videos = [...document.querySelectorAll('video')];
const playerIcons = {
  play: '<path d="m8 5 11 7-11 7Z" fill="currentColor" stroke="none"/>',
  pause: '<path d="M8 5v14M16 5v14" stroke-width="3"/>',
  muted: '<path d="M11 5 6 9H3v6h3l5 4Z"/><path d="m16 9 6 6m0-6-6 6"/>',
  sound: '<path d="M11 5 6 9H3v6h3l5 4Z"/><path d="M15 8a6 6 0 0 1 0 8M18 5a10 10 0 0 1 0 14"/>',
  expand: '<path d="M9 3H3v6m12-6h6v6M3 15v6h6m12-6v6h-6"/>',
  collapse: '<path d="M3 9h6V3m6 0v6h6M9 21v-6H3m12 6v-6h6"/>'
};
function playerButton(label, icon) {
  const button = document.createElement('button');
  button.type = 'button';
  setPlayerButton(button, label, icon);
  return button;
}
function setPlayerButton(button, label, icon) {
  button.setAttribute('aria-label', label);
  button.title = label;
  button.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${playerIcons[icon]}</svg>`;
}
function mediaTime(seconds) {
  const value = Number.isFinite(seconds) ? Math.max(0, Math.floor(seconds)) : 0;
  return `${Math.floor(value / 60)}:${String(value % 60).padStart(2, '0')}`;
}
// Enhance native controls only when JavaScript is available. Keep controls out
// of the picture in normal and fullscreen layouts alike.
videos.forEach(video => {
  const player = document.createElement('div');
  player.className = 'video-player';
  const frame = document.createElement('div');
  frame.className = 'video-frame';
  const controls = document.createElement('div');
  controls.className = 'video-controls';
  controls.setAttribute('role', 'group');
  controls.setAttribute('aria-label', `${video.getAttribute('aria-label') || 'Video'} controls`);
  const play = playerButton('Play', 'play');
  const mute = playerButton('Unmute', 'muted');
  const seek = document.createElement('input');
  seek.type = 'range';
  seek.className = 'video-seek';
  seek.min = '0';
  seek.max = '0';
  seek.step = '0.1';
  seek.value = '0';
  seek.disabled = true;
  seek.setAttribute('aria-label', 'Playback position');
  const time = document.createElement('span');
  time.className = 'video-time';
  const status = document.createElement('div');
  status.className = 'video-status';
  status.setAttribute('role', 'status');
  status.hidden = true;
  const showError = () => {
    status.replaceChildren(document.createTextNode('Unable to play here. '));
    const link = document.createElement('a');
    link.textContent = 'Open video';
    link.href = video.currentSrc || video.querySelector('source').src;
    link.target = '_blank';
    link.rel = 'noopener noreferrer';
    status.append(link);
    status.hidden = false;
  };
  function updateProgress() {
    const duration = Number.isFinite(video.duration) ? video.duration : 0;
    seek.max = String(duration);
    seek.disabled = duration <= 0;
    seek.value = String(video.currentTime);
    seek.style.setProperty('--played', `${duration ? video.currentTime / duration * 100 : 0}%`);
    seek.setAttribute('aria-valuetext', `${mediaTime(video.currentTime)} of ${mediaTime(duration)}`);
    time.textContent = `${mediaTime(video.currentTime)} / ${mediaTime(duration)}`;
  }
  play.addEventListener('click', async () => {
    if (!video.paused) { video.pause(); return; }
    status.hidden = true;
    try { await video.play(); }
    catch (error) { if (error.name !== 'AbortError') showError(); }
  });
  mute.addEventListener('click', () => { video.muted = !video.muted; });
  seek.addEventListener('input', () => {
    video.currentTime = Number(seek.value);
    updateProgress();
  });
  const updatePlayback = () => setPlayerButton(play, video.paused ? 'Play' : 'Pause', video.paused ? 'play' : 'pause');
  const updateVolume = () => setPlayerButton(mute, video.muted ? 'Unmute' : 'Mute', video.muted ? 'muted' : 'sound');
  ['play', 'pause', 'ended'].forEach(event => video.addEventListener(event, updatePlayback));
  ['loadedmetadata', 'durationchange', 'timeupdate', 'seeked'].forEach(event => video.addEventListener(event, updateProgress));
  video.addEventListener('volumechange', updateVolume);
  video.addEventListener('error', showError);
  controls.append(play, seek, time, mute);
  if (document.fullscreenEnabled && player.requestFullscreen) {
    const fullscreen = playerButton('Enter fullscreen', 'expand');
    fullscreen.addEventListener('click', async () => {
      try {
        if (document.fullscreenElement === player) await document.exitFullscreen();
        else await player.requestFullscreen();
      } catch { /* Fullscreen may be blocked by the embedding browser. */ }
    });
    document.addEventListener('fullscreenchange', () => {
      const active = document.fullscreenElement === player;
      setPlayerButton(fullscreen, active ? 'Exit fullscreen' : 'Enter fullscreen', active ? 'collapse' : 'expand');
    });
    controls.append(fullscreen);
  }
  video.before(player);
  frame.append(video);
  player.append(frame, controls, status);
  video.controls = false;
  updateProgress();
  updateVolume();
});
// Fetch durations near the viewport, without preloading every chapter's video.
if ('IntersectionObserver' in window) {
  const metadataObserver = new IntersectionObserver(entries => {
    entries.forEach(({target, isIntersecting}) => {
      if (isIntersecting) {
        if (target.readyState === 0) { target.preload = 'metadata'; target.load(); }
        metadataObserver.unobserve(target);
      }
    });
  }, {rootMargin: '200px'});
  videos.forEach(video => metadataObserver.observe(video));
}
videos.forEach(video => video.addEventListener('play', () => {
  videos.forEach(other => { if (other !== video) other.pause(); });
}));
if ('IntersectionObserver' in window) {
  const observer = new IntersectionObserver(entries => {
    entries.forEach(({target, isIntersecting}) => { if (!isIntersecting) target.pause(); });
  }, {threshold: 0});
  videos.forEach(video => observer.observe(video));
}
document.addEventListener('visibilitychange', () => {
  if (document.hidden) videos.forEach(video => video.pause());
});

// Section navigation follows document position, including long video chapters.
const chapters = [...document.querySelectorAll('main > section')];
const chapterLabels = ['Overview', 'Human–Humanoid Gap', 'Method', 'Collection',
  'Scale Alignment', 'Demonstrations', 'Summary', 'Citation'];
const chapterNav = document.createElement('nav');
chapterNav.className = 'chapter-dots';
chapterNav.setAttribute('aria-label', 'Chapter navigation');
const chapterLinks = chapters.map((section, index) => {
  if (!section.id) section.id = index === 0 ? 'overview' : 'summary';
  const link = document.createElement('a');
  const label = chapterLabels[index] || `Chapter ${index + 1}`;
  link.href = `#${section.id}`;
  link.setAttribute('aria-label', label);
  const tooltip = document.createElement('span');
  tooltip.className = 'chapter-dot-label';
  tooltip.textContent = label;
  link.append(tooltip);
  chapterNav.append(link);
  return link;
});
if (chapters.length) document.body.append(chapterNav);
function updateChapter() {
  const anchor = window.innerHeight * 0.5;
  let active = 0;
  chapters.forEach((section, index) => {
    if (section.getBoundingClientRect().top <= anchor) active = index;
  });
  if (window.scrollY + window.innerHeight >= document.documentElement.scrollHeight - 2) {
    active = chapters.length - 1;
  }
  chapterLinks.forEach((link, index) => {
    if (index === active) link.setAttribute('aria-current', 'location');
    else link.removeAttribute('aria-current');
  });
}
let chapterUpdatePending = false;
function scheduleChapterUpdate() {
  if (chapterUpdatePending) return;
  chapterUpdatePending = true;
  requestAnimationFrame(() => { updateChapter(); chapterUpdatePending = false; });
}
window.addEventListener('scroll', scheduleChapterUpdate, {passive: true});
window.addEventListener('resize', scheduleChapterUpdate);
window.addEventListener('load', scheduleChapterUpdate);
updateChapter();
