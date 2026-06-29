const API = {
  BASE:     'http://127.0.0.1:8000',
  REGISTER: '/register',
  LOGIN:    '/login',
  SET_LANG: '/language',
  PREDICT:  '/predict',
  HISTORY:  '/history',
};

/* ═══ TEXT STRINGS (EN / HI) ══════════════════════ */
const TXT = {
  en: {
    h:        'Analyse Your Crop',
    sub:      'Upload a clear photo of the affected crop leaves or stem',
    uzt:      'Drop your crop image here',
    uzh:      'or click to browse · JPG, PNG, WebP up to 10 MB',
    an:       '🔬 Analyse Crop',
    clear:    'Clear image',
    scanning: 'Analysing...',
    conf:     'Confidence',
    lTreat:   'Treatment',
    lSev:     'Severity',
    lNote:    'Additional Notes',
    tagDis:   '⚠ Disease Detected',
    tagOk:    '✓ Healthy Crop',
    cropPfx:  'Crop: ',
    histTitle:'Scan History',
    histEmpty:'No scans yet.',
    histBtn:  'History',
    healthy:  'No treatment needed — crop is healthy.',
    noTreat:  'No treatment data available for this class.',
    back:     'Back',
    viewTreat:'View treatment',
    fPest:    'Pesticide',
    fTrade:   'Trade name',
    fDose:    'Dosage',
    fMethod:  'Method',
    fFreq:    'Frequency',
  },
  hi: {
    h:        'अपनी फसल की जाँच करें',
    sub:      'प्रभावित फसल की पत्तियों या तने की स्पष्ट फ़ोटो अपलोड करें',
    uzt:      'यहाँ अपनी फसल की तस्वीर डालें',
    uzh:      'या ब्राउज़ करने के लिए क्लिक करें · JPG, PNG, WebP (10 MB तक)',
    an:       '🔬 फसल की जाँच करें',
    clear:    'तस्वीर हटाएं',
    scanning: 'जाँच हो रही है...',
    conf:     'सटीकता',
    lTreat:   'उपचार',
    lSev:     'गंभीरता',
    lNote:    'अतिरिक्त जानकारी',
    tagDis:   '⚠ रोग पाया गया',
    tagOk:    '✓ स्वस्थ फसल',
    cropPfx:  'फसल: ',
    histTitle:'स्कैन इतिहास',
    histEmpty:'अभी तक कोई स्कैन नहीं।',
    histBtn:  'इतिहास',
    healthy:  'कोई उपचार की आवश्यकता नहीं — फसल स्वस्थ है।',
    noTreat:  'इस वर्ग के लिए उपचार डेटा उपलब्ध नहीं है।',
    back:     'वापस',
    viewTreat:'उपचार देखें',
    fPest:    'कीटनाशक',
    fTrade:   'व्यापारिक नाम',
    fDose:    'मात्रा',
    fMethod:  'छिड़काव विधि',
    fFreq:    'आवृत्ति',
  }
};

/* ═══ STATE ════════════════════════════════════════ */
const S = {
  get token()    { return localStorage.getItem('ka_token'); },
  set token(v)   { v ? localStorage.setItem('ka_token', v) : localStorage.removeItem('ka_token'); },
  get username() { return localStorage.getItem('ka_user') || 'User'; },
  set username(v){ localStorage.setItem('ka_user', v); },
  get lang()     { return localStorage.getItem('ka_lang') || 'en'; },
  set lang(v)    { localStorage.setItem('ka_lang', v); },
  get langSet()  { return !!localStorage.getItem('ka_lang_set'); },
  set langSet(v) { v ? localStorage.setItem('ka_lang_set', '1') : localStorage.removeItem('ka_lang_set'); },
  selectedFile: null,
  chosenLang:   null,
  lastResult:   null,
};

/* ═══ ROUTING ══════════════════════════════════════ */
function show(id) {
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.getElementById('page-' + id).classList.add('active');
}

function boot() {
  if (!S.token)   { show('auth');     return; }
  if (!S.langSet) { show('language'); return; }
  initPredict();
  show('predict');
}

/* ═══ UTILS ════════════════════════════════════════ */
function togglePw(id, el) {
  const inp = document.getElementById(id);
  inp.type = inp.type === 'password' ? 'text' : 'password';
  el.textContent = inp.type === 'password' ? '👁' : '🙈';
}

function setAlert(id, msg, type) {
  const el = document.getElementById(id);
  if (!el) return;
  el.textContent = msg || '';
  el.className   = msg ? `alert show alert-${type}` : 'alert';
}

function setBtnLoading(id, loading, defaultLabel) {
  const btn    = document.getElementById(id);
  btn.disabled = loading;
  btn.innerHTML = loading
    ? '<div class="spinner"></div>Please wait…'
    : defaultLabel;
}

/* ═══ AUTH — tab switch ════════════════════════════ */
function switchTab(tab) {
  ['login', 'register'].forEach(t => {
    document.getElementById('tab-' + t).classList.toggle('active', t === tab);
    document.getElementById('frm-' + t).classList.toggle('active', t === tab);
  });
  setAlert('login-err', '', 'error');
  setAlert('reg-err',   '', 'error');
  setAlert('reg-ok',    '', 'success');
}

/* ═══ PASSWORD STRENGTH ════════════════════════════ */
function onPwInput(v) {
  const checks = {
    'rc-len': v.length >= 8,
    'rc-up':  /[A-Z]/.test(v),
    'rc-lo':  /[a-z]/.test(v),
    'rc-num': /[0-9]/.test(v),
    'rc-sp':  /[!@#$%^&*]/.test(v),
  };
  let met = 0;
  for (const [id, ok] of Object.entries(checks)) {
    document.getElementById(id).classList.toggle('ok', ok);
    if (ok) met++;
  }
  const colors = ['#E74C3C', '#E67E22', '#F1C40F', '#52B788', '#1B4332'];
  const widths = ['15%', '33%', '56%', '80%', '100%'];
  const fill   = document.getElementById('pw-fill');
  fill.style.width      = met ? widths[met - 1] : '0%';
  fill.style.background = met ? colors[met - 1] : 'transparent';
}

/* ═══ LOGIN ════════════════════════════════════════ */
async function handleLogin() {
  setAlert('login-err', '', 'error');
  const user = document.getElementById('l-user').value.trim();
  const pass = document.getElementById('l-pass').value;
  if (!user || !pass) { setAlert('login-err', 'Please fill in all fields.', 'error'); return; }

  setBtnLoading('login-btn', true, 'Sign In');
  try {
    const r = await fetch(API.BASE + API.LOGIN, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ username: user, password: pass }),
    });
    const d = await r.json();
    if (!r.ok) throw new Error(d.detail || 'Incorrect username or password.');

    S.token    = d.access_token;
    S.username = user;
    S.langSet  = d.is_onboarded;
    S.lang     = d.language || 'en';

    if (!S.langSet) { show('language'); }
    else            { initPredict(); show('predict'); }
  } catch (e) {
    setAlert('login-err', e.message || 'Login failed. Please try again.', 'error');
  } finally {
    setBtnLoading('login-btn', false, 'Sign In');
  }
}

/* ═══ REGISTER ═════════════════════════════════════ */
async function handleRegister() {
  setAlert('reg-err', '', 'error');
  setAlert('reg-ok',  '', 'success');

  const username         = document.getElementById('r-user').value.trim();
  const email            = document.getElementById('r-email').value.trim();
  const password         = document.getElementById('r-pass').value;
  const confirm_password = document.getElementById('r-conf').value;

  if (!username || !email || !password || !confirm_password) {
    setAlert('reg-err', 'Please fill in all fields.', 'error'); return;
  }
  if (password !== confirm_password) {
    setAlert('reg-err', 'Passwords do not match.', 'error'); return;
  }

  setBtnLoading('reg-btn', true, 'Create Account');
  try {
    const r = await fetch(API.BASE + API.REGISTER, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ username, email, password, confirm_password }),
    });
    const d = await r.json();
    if (!r.ok) {
      let msg = d.detail;
      if (Array.isArray(msg)) msg = msg.map(x => x.msg || JSON.stringify(x)).join(' · ');
      throw new Error(msg || 'Registration failed.');
    }
    S.token    = d.access_token;
    S.username = username;
    S.langSet  = d.is_onboarded;
    if (!S.langSet) { show('language'); }
    else            { initPredict(); show('predict'); }
  } catch (e) {
    setAlert('reg-err', e.message || 'Registration failed. Please try again.', 'error');
  } finally {
    setBtnLoading('reg-btn', false, 'Create Account');
  }
}

/* ═══ LANGUAGE SELECTION ═══════════════════════════ */
function pickLang(lang) {
  S.chosenLang = lang;
  document.getElementById('lc-en').classList.toggle('selected', lang === 'en');
  document.getElementById('lc-hi').classList.toggle('selected', lang === 'hi');
  document.getElementById('lang-continue-btn').classList.add('show');
}

async function confirmLang() {
  if (!S.chosenLang) return;
  S.lang    = S.chosenLang;
  S.langSet = true;
  try {
    await fetch(API.BASE + API.SET_LANG, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json', 'Authorization': 'Bearer ' + S.token },
      body:    JSON.stringify({ language: S.lang }),
    });
  } catch (_) { /* non-critical */ }
  initPredict();
  show('predict');
}

/* ═══ PREDICT PAGE — setup ═════════════════════════ */
function initPredict() {
  applyLang();
  const n = S.username;
  document.getElementById('u-name').textContent = n;
  document.getElementById('u-avi').textContent  = n[0].toUpperCase();
  setupDnD();
}

function applyLang() {
  const t = TXT[S.lang] || TXT.en;
  const map = {
    'txt-h':         t.h,
    'txt-sub':       t.sub,
    'txt-uz-t':      t.uzt,
    'txt-uz-h':      t.uzh,
    'txt-an-btn':    t.an,
    'txt-clear':     t.clear,
    'txt-scan':      t.scanning,
    'txt-conf':      t.conf,
    'txt-lbl-treat': t.lTreat,
    'txt-lbl-sev':   t.lSev,
    'txt-lbl-note':  t.lNote,
    'txt-hist-btn':  t.histBtn,
  };
  for (const [id, val] of Object.entries(map)) {
    const el = document.getElementById(id);
    if (el) el.textContent = val;
  }
  document.getElementById('p-en').classList.toggle('active', S.lang === 'en');
  document.getElementById('p-hi').classList.toggle('active', S.lang === 'hi');
  if (S.lastResult) renderResults(S.lastResult);
}

function switchLang(lang) {
  S.lang = lang;
  applyLang();
}

/* ═══ DRAG & DROP ══════════════════════════════════ */
function setupDnD() {
  const zone = document.getElementById('drop-zone');
  zone.addEventListener('dragover',  e => { e.preventDefault(); zone.classList.add('drag-over'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'));
  zone.addEventListener('drop', e => {
    e.preventDefault();
    zone.classList.remove('drag-over');
    const f = e.dataTransfer.files[0];
    if (f && f.type.startsWith('image/')) loadFile(f);
  });
}

function onFileChosen(e) {
  const f = e.target.files[0];
  if (f) loadFile(f);
}

function loadFile(file) {
  if (file.size > 10 * 1024 * 1024) { alert('File is too large. Max 10 MB.'); return; }
  S.selectedFile = file;
  const reader   = new FileReader();
  reader.onload  = ev => {
    document.getElementById('prev-img').src = ev.target.result;
    document.getElementById('drop-zone').classList.add('hidden');
    document.getElementById('prev-wrap').classList.add('show');
    document.getElementById('action-btns').classList.add('show');
    document.getElementById('results').classList.remove('show');
    S.lastResult = null;
  };
  reader.readAsDataURL(file);
}

function clearAll() {
  S.selectedFile = null;
  S.lastResult   = null;
  document.getElementById('drop-zone').classList.remove('hidden');
  document.getElementById('prev-wrap').classList.remove('show');
  document.getElementById('action-btns').classList.remove('show');
  document.getElementById('results').classList.remove('show');
  document.getElementById('scan-ov').classList.remove('show');
  document.getElementById('file-inp').value = '';
}

/* ═══ ANALYSE ══════════════════════════════════════ */
async function runAnalysis() {
  if (!S.selectedFile) return;
  const t   = TXT[S.lang] || TXT.en;
  const btn = document.getElementById('an-btn');
  btn.disabled  = true;
  btn.innerHTML = `<div class="spinner"></div><span>${t.scanning}</span>`;

  document.getElementById('scan-ov').classList.add('show');
  document.getElementById('results').classList.remove('show');

  try {
    const fd = new FormData();
    fd.append('file', S.selectedFile);

    const r = await fetch(API.BASE + API.PREDICT, {
      method:  'POST',
      headers: { 'Authorization': 'Bearer ' + S.token },
      body:    fd,
    });

    if (r.status === 401) { handleLogout(); return; }
    if (!r.ok) {
      const d = await r.json().catch(() => ({}));
      throw new Error(d.detail || 'Analysis failed. Please try again.');
    }

    const data   = await r.json();
    S.lastResult = data;
    renderResults(data);
  } catch (e) {
    alert(e.message || 'Something went wrong. Please try again.');
  } finally {
    btn.disabled  = false;
    btn.innerHTML = `<span id="txt-an-btn">${(TXT[S.lang] || TXT.en).an}</span>`;
    document.getElementById('scan-ov').classList.remove('show');
  }
}

/* ═══ SHARED RENDER HELPERS ════════════════════════ */
// "Tomato___Bacterial_spot" → { crop:"Tomato", disease:"Bacterial spot" }
function parseClass(predicted_class) {
  const parts   = (predicted_class || '').split('___');
  const crop    = parts[0]?.replace(/_/g, ' ').replace(/\(.*\)/, '').trim() || '—';
  const disease = parts[1]?.replace(/_/g, ' ') || (predicted_class || '').replace(/_/g, ' ');
  return { crop, disease };
}

// Severity is curated client-side (no LLM) so it's bilingual + consistent with history,
// which only stores the level string ("High"/"Medium"/"Low").
const SEV = {
  High:   { en: { label: '🔴 Severe',      advice: 'Immediate treatment required. Disease is well-established.' },
            hi: { label: '🔴 गंभीर',        advice: 'तुरंत उपचार आवश्यक है। रोग पूरी तरह फैल चुका है।' } },
  Medium: { en: { label: '🟡 Moderate',    advice: 'Treat within 3-5 days. Disease is progressing.' },
            hi: { label: '🟡 मध्यम',         advice: '3-5 दिनों के भीतर उपचार करें। रोग बढ़ रहा है।' } },
  Low:    { en: { label: '🟢 Early Stage', advice: 'Monitor closely. Treat as a precaution.' },
            hi: { label: '🟢 प्रारंभिक अवस्था', advice: 'ध्यान से निगरानी करें। एहतियात के तौर पर उपचार करें।' } },
};
function severityDisplay(level) {
  const e = SEV[level];
  if (!e) return { label: level || '—', advice: '' };
  return e[S.lang] || e.en;
}

// Pick the Hindi variant of a prose field when in Hindi mode, else the English one.
// Pesticide/trade/dosage have no _hi and stay English by design.
function trField(tr, base) {
  return (S.lang === 'hi' && tr[base + '_hi']) ? tr[base + '_hi'] : tr[base];
}

// Build the treatment block HTML from a treatment object (shared by results + history detail)
function buildTreatmentHtml(tr, healthy, t) {
  if (healthy) return `<span style="color:var(--green-leaf)">${t.healthy}</span>`;
  if (!tr)     return t.noTreat;
  const rows = [
    tr.pesticide_common_name && `<b>${t.fPest}:</b> ${tr.pesticide_common_name}`,
    tr.trade_name            && `<b>${t.fTrade}:</b> ${tr.trade_name}`,
    tr.dosage                && `<b>${t.fDose}:</b> ${tr.dosage}`,
    tr.application_method    && `<b>${t.fMethod}:</b> ${trField(tr, 'application_method')}`,
    tr.frequency             && `<b>${t.fFreq}:</b> ${trField(tr, 'frequency')}`,
  ].filter(Boolean);
  return rows.length ? rows.join('<br>') : t.noTreat;
}

/* ═══ RENDER RESULTS ═══════════════════════════════ */
function renderResults(data) {
  const t       = TXT[S.lang] || TXT.en;
  const healthy = data.is_healthy;
  const pct     = Math.round((data.confidence || 0) * 100);
  const fillClr = pct >= 80 ? '#40916C' : pct >= 55 ? '#D4A017' : '#C0392B';
  const { crop, disease } = parseClass(data.predicted_class);

  document.getElementById('r-badge').textContent = healthy ? t.tagOk : t.tagDis;
  document.getElementById('r-badge').className   = 'r-badge ' + (healthy ? 'ok' : 'dis');
  document.getElementById('r-disease').textContent = healthy ? 'No disease found' : disease;
  document.getElementById('r-crop').textContent    = t.cropPfx + crop;
  document.getElementById('conf-pct').textContent  = pct + '%';

  const fill = document.getElementById('conf-fill');
  fill.style.width      = pct + '%';
  fill.style.background = fillClr;

  // severity (curated bilingual, keyed by level)
  const sev = severityDisplay(data.severity && data.severity.level);
  document.getElementById('r-sev-label').textContent  = sev.label;
  document.getElementById('r-sev-advice').textContent = sev.advice || '—';

  // treatment block
  const tr = data.treatment;
  document.getElementById('r-treat').innerHTML = buildTreatmentHtml(tr, healthy, t);

  // additional notes (Hindi when available)
  document.getElementById('r-note').textContent   = (tr && trField(tr, 'additional_note')) || '—';
  document.getElementById('r-source').textContent = tr?.source ? `Source: ${tr.source}` : '';

  // refresh bilingual labels
  document.getElementById('txt-conf').textContent      = t.conf;
  document.getElementById('txt-lbl-treat').textContent = t.lTreat;
  document.getElementById('txt-lbl-sev').textContent   = t.lSev;
  document.getElementById('txt-lbl-note').textContent  = t.lNote;

  document.getElementById('results').classList.add('show');
  document.getElementById('results').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

/* ═══ HISTORY PANEL ════════════════════════════════ */
let _histScans = {};   // id → scan object, so detail view needs no extra fetch

function showHistView(which) {
  document.getElementById('hist-list-view').style.display   = which === 'list'   ? 'block' : 'none';
  document.getElementById('hist-detail-view').style.display = which === 'detail' ? 'flex'  : 'none';
}

async function openHistory() {
  const t = TXT[S.lang] || TXT.en;
  document.getElementById('hist-title').textContent     = t.histTitle;
  document.getElementById('txt-hist-back').textContent  = t.back;
  document.getElementById('hist-overlay').classList.add('show');
  showHistView('list');
  document.getElementById('hist-list').innerHTML =
    '<div style="text-align:center;padding:32px;color:var(--text-light)">Loading…</div>';

  try {
    const r = await fetch(API.BASE + API.HISTORY, {
      headers: { 'Authorization': 'Bearer ' + S.token },
    });
    if (r.status === 401) { handleLogout(); return; }
    const scans = await r.json();

    _histScans = {};
    if (!scans.length) {
      document.getElementById('hist-list').innerHTML =
        `<div style="text-align:center;padding:32px;color:var(--text-light)">${t.histEmpty}</div>`;
      return;
    }

    document.getElementById('hist-list').innerHTML = scans.map(s => {
      _histScans[s.id] = s;
      const { crop, disease } = parseClass(s.predicted_class);
      const date  = new Date(s.scanned_at).toLocaleString();
      const pct   = Math.round((s.confidence || 0) * 100);
      const badge = s.is_healthy
        ? `<span class="r-badge ok" style="font-size:11px;padding:2px 8px">✓ Healthy</span>`
        : `<span class="r-badge dis" style="font-size:11px;padding:2px 8px">⚠ Disease</span>`;
      return `
        <div class="hist-item" onclick="showScanDetail(${s.id})">
          <div class="hist-item-top">
            ${badge}
            <span class="hist-date">${date}</span>
          </div>
          <div class="hist-disease">${s.is_healthy ? 'Healthy' : disease}</div>
          <div class="hist-crop">${t.cropPfx}${crop} &nbsp;·&nbsp; ${pct}% confidence</div>
          <div class="hist-chevron">${t.viewTreat} →</div>
        </div>`;
    }).join('');
  } catch (e) {
    document.getElementById('hist-list').innerHTML =
      `<div style="text-align:center;padding:32px;color:var(--red)">Failed to load history.</div>`;
  }
}

/* Clicking a history item shows its full treatment, reusing the already-fetched data */
function showScanDetail(id) {
  const t = TXT[S.lang] || TXT.en;
  const s = _histScans[id];
  if (!s) return;

  const { crop, disease } = parseClass(s.predicted_class);
  const pct     = Math.round((s.confidence || 0) * 100);
  const fillClr = pct >= 80 ? '#40916C' : pct >= 55 ? '#D4A017' : '#C0392B';
  const date    = new Date(s.scanned_at).toLocaleString();
  const badge   = s.is_healthy
    ? `<span class="r-badge ok">${t.tagOk}</span>`
    : `<span class="r-badge dis">${t.tagDis}</span>`;
  const tr      = s.treatment;

  document.getElementById('hist-detail').innerHTML = `
    <div class="hist-date" style="margin-bottom:10px">${date}</div>
    ${badge}
    <div class="r-title" style="margin-top:10px">${s.is_healthy ? 'No disease found' : disease}</div>
    <div class="r-sub">${t.cropPfx}${crop}</div>

    <div class="conf-row" style="margin:16px 0">
      <span class="conf-label">${t.conf}</span>
      <div class="conf-track"><div class="conf-fill" style="width:${pct}%;background:${fillClr}"></div></div>
      <span class="conf-pct">${pct}%</span>
    </div>

    <div class="r-section-label">${t.lSev}</div>
    <div class="r-section-body" style="margin-bottom:16px">
      <div style="font-weight:600">${severityDisplay(s.severity_level).label}</div>
      <div style="color:var(--text-mid);font-size:0.9rem">${severityDisplay(s.severity_level).advice}</div>
    </div>

    <div class="r-section-label">${t.lTreat}</div>
    <div class="r-section-body" style="margin-bottom:16px">${buildTreatmentHtml(tr, s.is_healthy, t)}</div>

    <div class="r-section-label">${t.lNote}</div>
    <div class="r-section-body">${(tr && trField(tr, 'additional_note')) || '—'}</div>
    ${tr && tr.source ? `<div style="margin-top:8px;font-size:0.78rem;color:var(--text-light)">Source: ${tr.source}</div>` : ''}
  `;
  showHistView('detail');
}

function backToHistory() {
  showHistView('list');
}

function closeHistory() {
  document.getElementById('hist-overlay').classList.remove('show');
}

/* ═══ LOGOUT ═══════════════════════════════════════ */
function handleLogout() {
  localStorage.clear();
  clearAll();
  resetAuthForms();
  show('auth');
}

function resetAuthForms() {
  ['l-user', 'l-pass', 'r-user', 'r-email', 'r-pass', 'r-conf'].forEach(id => {
    document.getElementById(id).value = '';
  });
  ['rc-len', 'rc-up', 'rc-lo', 'rc-num', 'rc-sp'].forEach(id => {
    document.getElementById(id).classList.remove('ok');
  });
  document.getElementById('pw-fill').style.width = '0%';
  document.getElementById('pw-fill').style.background = 'transparent';
  switchTab('login');
  document.getElementById('lc-en').classList.remove('selected');
  document.getElementById('lc-hi').classList.remove('selected');
  document.getElementById('lang-continue-btn').classList.remove('show');
}

/* ═══ BOOT ═════════════════════════════════════════ */
boot();
