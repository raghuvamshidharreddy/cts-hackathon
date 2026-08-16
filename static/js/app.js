/**
 * CAR SENCE — SPA Frontend v2 (Lasso Regression Edition)
 * Fields: brand, model, year, engine_size(L), mileage(miles),
 *         fuel_type, transmission, doors
 * Output: USD pricing
 */

// Brand → Models from Flask template
const BRAND_MODELS_DATA = window.BRAND_MODELS_DATA || {
    "Audi":       ["A3", "A4", "Q5"],
    "BMW":        ["5 Series", "X5"],
    "Chevrolet":  ["Equinox", "Impala", "Malibu"],
    "Ford":       ["Explorer", "Fiesta", "Focus"],
    "Honda":      ["Accord", "CR-V", "Civic"],
    "Hyundai":    ["Elantra", "Sonata", "Tucson"],
    "Kia":        ["Optima", "Rio", "Sportage"],
    "Mercedes":   ["C-Class", "E-Class", "GLA"],
    "Toyota":     ["Camry", "Corolla", "RAV4"],
    "Volkswagen": ["Golf", "Passat", "Tiguan"],
};

// Global App State
const AppState = {
    isCompareMode: false,
};

// ── Init ──────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    initBrandDropdowns();
    initCompareToggle();
    initForms();
    initModals();
    initMobileMenu();
});

// ── Brand → Model Dynamic Dropdowns ───────────────────────────────────
function initBrandDropdowns() {
    linkBrandModel('primary_brand', 'primary_model');
    linkBrandModel('compare_brand', 'compare_model');
}

function linkBrandModel(brandId, modelId) {
    const brandEl = document.getElementById(brandId);
    const modelEl = document.getElementById(modelId);
    if (!brandEl || !modelEl) return;

    brandEl.addEventListener('change', () => {
        const brand = brandEl.value;
        const models = BRAND_MODELS_DATA[brand] || [];
        modelEl.innerHTML = '<option value="" disabled selected>Select Model</option>';
        models.forEach(m => {
            const opt = document.createElement('option');
            opt.value = m;
            opt.textContent = m;
            modelEl.appendChild(opt);
        });
        modelEl.disabled = models.length === 0;
    });
}

// ── Compare Toggle ────────────────────────────────────────────────────
function initCompareToggle() {
    const toggleBtn  = document.getElementById('toggleCompareBtn');
    const closeBtn   = document.getElementById('closeCompareBtn');
    const cardBtn    = document.getElementById('triggerCompareFromCard');

    if (toggleBtn) toggleBtn.addEventListener('click', () => setCompareMode(!AppState.isCompareMode));
    if (closeBtn)  closeBtn.addEventListener('click',  () => setCompareMode(false));
    if (cardBtn)   cardBtn.addEventListener('click',   () => setCompareMode(true));
}

function setCompareMode(active) {
    AppState.isCompareMode = active;
    const toggleBtn  = document.getElementById('toggleCompareBtn');
    const panel      = document.getElementById('comparePanel');
    const illus      = document.getElementById('illustrationCard');

    if (active) {
        if (panel)     panel.classList.add('active');
        if (illus)     illus.style.display = 'none';
        if (toggleBtn) {
            toggleBtn.classList.add('active');
            toggleBtn.innerHTML = '<span>❌</span> Close Compare';
        }
    } else {
        if (panel)     panel.classList.remove('active');
        if (illus)     illus.style.display = 'flex';
        if (toggleBtn) {
            toggleBtn.classList.remove('active');
            toggleBtn.innerHTML = '<span>⚖️</span> Compare';
        }
    }
}

// ── Form Submit Handlers ──────────────────────────────────────────────
function initForms() {
    const primaryForm  = document.getElementById('primaryCarForm');
    const compareForm  = document.getElementById('compareCarForm');
    const recalcBtn    = document.getElementById('recalculateBtn');

    if (primaryForm) primaryForm.addEventListener('submit', e => {
        e.preventDefault(); handlePrediction();
    });
    if (compareForm) compareForm.addEventListener('submit', e => {
        e.preventDefault(); handlePrediction();
    });
    if (recalcBtn) recalcBtn.addEventListener('click', handlePrediction);
}

// ── Main Prediction Handler ───────────────────────────────────────────
async function handlePrediction() {
    const d1 = collectFormData('primary');
    const v1 = validateData(d1);
    if (!v1.valid) {
        showAlert('Fix Vehicle 1 errors:\n• ' + v1.errors.join('\n• '));
        return;
    }

    let d2 = null;
    if (AppState.isCompareMode) {
        d2 = collectFormData('compare');
        const v2 = validateData(d2);
        if (!v2.valid) {
            showAlert('Fix Vehicle 2 errors:\n• ' + v2.errors.join('\n• '));
            return;
        }
    }

    showLoading(AppState.isCompareMode);

    try {
        if (AppState.isCompareMode && d2) {
            const [r1, r2] = await Promise.all([callApi(d1), callApi(d2)]);
            if (r1.success && r2.success) {
                renderComparison(d1, r1, d2, r2);
            } else {
                showError(r1.error || r2.error || 'Prediction failed for one or both vehicles.');
            }
        } else {
            const r1 = await callApi(d1);
            if (r1.success) {
                renderSingle(d1, r1);
            } else {
                showError(r1.error || 'Could not estimate price. Please check your inputs.');
            }
        }
    } catch (err) {
        console.error('API error:', err);
        showError('Network error. Make sure the Flask server is running on port 5000.');
    }
}

// ── API Call ──────────────────────────────────────────────────────────
async function callApi(payload) {
    const res = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
    });
    return res.json();
}

// ── Collect Form Data ─────────────────────────────────────────────────
function collectFormData(prefix) {
    return {
        brand:        val(`${prefix}_brand`),
        model:        val(`${prefix}_model`),
        year:         intVal(`${prefix}_year`),
        engine_size:  floatVal(`${prefix}_engine_size`),
        mileage:      floatVal(`${prefix}_mileage`),
        fuel_type:    val(`${prefix}_fuel_type`),
        transmission: val(`${prefix}_transmission`),
        doors:        intVal(`${prefix}_doors`),
    };
}

function val(id)      { return document.getElementById(id)?.value?.trim() || ''; }
function intVal(id)   { return parseInt(document.getElementById(id)?.value || '0'); }
function floatVal(id) { return parseFloat(document.getElementById(id)?.value || '0'); }

// ── Client-Side Validation ────────────────────────────────────────────
function validateData(d) {
    const errors = [];
    if (!d.brand)
        errors.push('Select a Brand.');
    if (!d.year || d.year < 2000 || d.year > 2023)
        errors.push('Select a valid Year (2000–2023).');
    if (isNaN(d.engine_size) || d.engine_size < 1.0 || d.engine_size > 5.0)
        errors.push('Enter Engine Size between 1.0 and 5.0 litres (e.g. 2.0).');
    if (isNaN(d.mileage) || d.mileage < 0 || d.mileage > 300000)
        errors.push('Enter Mileage between 0 and 300,000 miles.');
    if (!d.fuel_type)
        errors.push('Select Fuel Type.');
    if (!d.transmission)
        errors.push('Select Transmission.');
    if (!d.doors || d.doors < 2 || d.doors > 5)
        errors.push('Select number of Doors (2–5).');
    return { valid: errors.length === 0, errors };
}

// ── UI State Helpers ──────────────────────────────────────────────────
function getResultsSection() {
    return {
        section:  document.getElementById('resultsSection'),
        loading:  document.getElementById('resultsLoadingState'),
        content:  document.getElementById('resultsContentState'),
        loadText: document.getElementById('loadingTextMain'),
    };
}

function showLoading(isCompare) {
    const { section, loading, content, loadText } = getResultsSection();
    if (section) { section.classList.add('active'); section.scrollIntoView({ behavior: 'smooth', block: 'nearest' }); }
    if (loading) loading.style.display = 'flex';
    if (content) content.style.display = 'none';
    if (loadText) loadText.textContent = isCompare
        ? 'Comparing both vehicles...'
        : 'Estimating car value...';
}

function showError(msg) {
    const { loading, content } = getResultsSection();
    if (loading) loading.style.display = 'none';
    if (content) {
        content.style.display = 'block';
        content.innerHTML = `
            <div style="text-align:center;padding:2.5rem 1rem;">
                <div style="font-size:2.2rem;margin-bottom:0.6rem;">⚠️</div>
                <div style="font-size:1.05rem;font-weight:700;color:#FFC727;margin-bottom:0.35rem;">Prediction Error</div>
                <div style="font-size:0.85rem;color:#78909C;">${escHtml(msg)}</div>
            </div>`;
    }
}

function showAlert(msg) {
    // Minimal styled alert fallback
    alert(msg);
}

// ── Single Car Result ─────────────────────────────────────────────────
function renderSingle(specs, result) {
    const { loading, content } = getResultsSection();
    if (loading) loading.style.display = 'none';
    if (!content) return;
    content.style.display = 'block';

    const year      = parseInt(specs.year);
    const age       = 2025 - year;
    const mileage   = parseFloat(specs.mileage);
    const price     = result.predicted_price;

    // Derived metrics
    const depRate   = Math.min(22, Math.max(4, (5.5 + age * 0.6))).toFixed(1);
    const valueScore = Math.min(97, Math.max(55, Math.round(85 - age * 2.2 + (mileage < 30000 ? 8 : mileage > 80000 ? -8 : 0))));
    const relScore   = Math.min(98, Math.max(60, Math.round(92 - age * 1.5)));
    const marketPos  = price > 25000 ? 'Premium' : price > 12000 ? 'Mid-Range' : 'Value';

    content.innerHTML = `
        <div class="results-display-grid">
            <div class="price-metric-main">
                <div class="price-label-text">Estimated Market Value</div>
                <div class="price-val-usd">${result.formatted_usd}</div>
                <div class="price-val-range">Range: ${result.price_low} – ${result.price_high}</div>
            </div>

            <div class="metrics-strip">
                <div class="metric-box">
                    <div class="metric-icon-circle">📊</div>
                    <div class="metric-data">
                        <span class="metric-name">Market</span>
                        <span class="metric-val">${marketPos}</span>
                        <span class="metric-sub">Segment</span>
                    </div>
                </div>
                <div class="metric-box">
                    <div class="metric-icon-circle">📈</div>
                    <div class="metric-data">
                        <span class="metric-name">Value Score</span>
                        <span class="metric-val">${valueScore}/100</span>
                        <span class="metric-sub">Higher = Better</span>
                    </div>
                </div>
                <div class="metric-box">
                    <div class="metric-icon-circle">📉</div>
                    <div class="metric-data">
                        <span class="metric-name">Depreciation</span>
                        <span class="metric-val">${depRate}%/yr</span>
                        <span class="metric-sub">Estimated Rate</span>
                    </div>
                </div>
                <div class="metric-box">
                    <div class="metric-icon-circle">🛡️</div>
                    <div class="metric-data">
                        <span class="metric-name">Reliability</span>
                        <span class="metric-val">${relScore}/100</span>
                        <span class="metric-sub">By Age</span>
                    </div>
                </div>
            </div>

            <div style="
                background:#111;
                border:1px solid rgba(69,90,100,0.3);
                border-radius:12px;
                padding:1rem 1.25rem;
                display:grid;
                grid-template-columns:repeat(3,1fr);
                gap:0.75rem;
                font-size:0.82rem;">
                ${specRow('Brand', specs.brand)}
                ${specRow('Model', specs.model || '—')}
                ${specRow('Year', specs.year)}
                ${specRow('Engine', specs.engine_size + 'L')}
                ${specRow('Mileage', fmtNum(mileage) + ' mi')}
                ${specRow('Fuel', specs.fuel_type)}
                ${specRow('Gearbox', specs.transmission)}
                ${specRow('Doors', specs.doors)}
            </div>
        </div>

        <div class="results-tip-banner">
            <span>💡</span>
            <span class="tip-banner-text">
                <strong>Tip:</strong> Adjust any spec above and click <strong>Get Price Estimate</strong> again — your inputs are preserved!
            </span>
        </div>`;
}

// ── Comparison Result ─────────────────────────────────────────────────
function renderComparison(d1, r1, d2, r2) {
    const { loading, content } = getResultsSection();
    if (loading) loading.style.display = 'none';
    if (!content) return;
    content.style.display = 'block';

    const p1 = r1.predicted_price;
    const p2 = r2.predicted_price;
    const diff      = Math.abs(p1 - p2);
    const pctDiff   = Math.max(p1, p2) > 0 ? ((diff / Math.max(p1, p2)) * 100).toFixed(1) : '0.0';
    const cheaperIs1 = p1 <= p2;
    const higherMileIs1 = parseFloat(d1.mileage) <= parseFloat(d2.mileage); // lower mileage = better

    content.innerHTML = `
        <div style="text-align:center;margin-bottom:1.25rem;">
            <div style="font-size:0.72rem;text-transform:uppercase;letter-spacing:0.09em;color:#78909C;font-weight:700;">Price Difference</div>
            <div style="font-family:'Space Grotesk',sans-serif;font-size:1.7rem;font-weight:900;color:#FFC727;margin-top:0.15rem;">
                $${fmtNum(Math.round(diff))} &nbsp;<span style="font-size:1rem;font-weight:600;">(${pctDiff}%)</span>
            </div>
            <div style="font-size:0.82rem;color:rgba(255,255,255,0.7);margin-top:0.2rem;">
                ${cheaperIs1
                    ? `${d1.brand} ${d1.model||''} is the better value by $${fmtNum(Math.round(diff))}`
                    : `${d2.brand} ${d2.model||''} is the better value by $${fmtNum(Math.round(diff))}`}
            </div>
        </div>

        <div class="comparison-results-grid">
            ${compCard(d1, r1, cheaperIs1, higherMileIs1, 'Vehicle 1')}
            ${compCard(d2, r2, !cheaperIs1, !higherMileIs1, 'Vehicle 2')}
        </div>

        <div class="results-tip-banner">
            <span>💡</span>
            <span class="tip-banner-text">
                <strong>Tip:</strong> Edit any field above and click <strong>Compare Both</strong> to refresh the live comparison.
            </span>
        </div>`;
}

function compCard(specs, result, isCheaper, isLowerMileage, label) {
    const age     = 2025 - parseInt(specs.year);
    const mileage = parseFloat(specs.mileage);
    return `
        <div class="car-compare-card ${isCheaper ? 'winner' : ''}">
            <span class="car-compare-badge">
                ${label} · ${specs.brand} ${isCheaper ? '✅ Better Value' : ''}
            </span>
            <div class="car-compare-title">${specs.brand} ${specs.model || ''} (${specs.year})</div>
            <div class="car-compare-price">${result.formatted_usd}</div>
            <div style="font-size:0.78rem;color:#78909C;margin-bottom:0.75rem;">
                ${result.formatted_k} · ${specs.fuel_type} · ${specs.transmission}
            </div>
            <div style="font-size:0.8rem;border-top:1px solid rgba(69,90,100,0.25);padding-top:0.65rem;display:flex;flex-direction:column;gap:0.32rem;">
                <div style="display:flex;justify-content:space-between;">
                    <span style="color:#78909C;">Mileage</span>
                    <span style="font-weight:700;color:${isLowerMileage ? '#FFC727' : '#fff'};">
                        ${fmtNum(mileage)} mi ${isLowerMileage ? '⚡' : ''}
                    </span>
                </div>
                <div style="display:flex;justify-content:space-between;">
                    <span style="color:#78909C;">Engine</span>
                    <span style="font-weight:600;color:#fff;">${specs.engine_size}L</span>
                </div>
                <div style="display:flex;justify-content:space-between;">
                    <span style="color:#78909C;">Doors</span>
                    <span style="font-weight:600;color:#fff;">${specs.doors}</span>
                </div>
                <div style="display:flex;justify-content:space-between;">
                    <span style="color:#78909C;">Age</span>
                    <span style="font-weight:600;color:#fff;">${age} yr${age !== 1 ? 's' : ''}</span>
                </div>
                <div style="display:flex;justify-content:space-between;">
                    <span style="color:#78909C;">Price Range</span>
                    <span style="font-weight:600;color:#fff;font-size:0.75rem;">${result.price_low} – ${result.price_high}</span>
                </div>
            </div>
        </div>`;
}

// ── Spec Row Helper ───────────────────────────────────────────────────
function specRow(label, value) {
    return `
        <div style="display:flex;flex-direction:column;gap:0.15rem;">
            <span style="font-size:0.65rem;text-transform:uppercase;letter-spacing:0.07em;color:#78909C;font-weight:700;">${label}</span>
            <span style="font-weight:700;color:#fff;font-size:0.84rem;">${value}</span>
        </div>`;
}

// ── Modals ────────────────────────────────────────────────────────────
function initModals() {
    bindModal('tipsBtn',   'tipsModal',  'closeTipsModal');
    bindModal('resetBtn',  'resetModal', 'cancelResetCross');
    const cancelReset  = document.getElementById('cancelResetBtn');
    const confirmReset = document.getElementById('confirmResetBtn');
    if (cancelReset)  cancelReset.addEventListener('click',  () => closeModal('resetModal'));
    if (confirmReset) confirmReset.addEventListener('click', () => { doReset(); closeModal('resetModal'); });
}

function bindModal(openId, modalId, closeId) {
    const openBtn  = document.getElementById(openId);
    const closeBtn = document.getElementById(closeId);
    if (openBtn)  openBtn.addEventListener('click',  () => openModal(modalId));
    if (closeBtn) closeBtn.addEventListener('click', () => closeModal(modalId));
}

function openModal(id)  { document.getElementById(id)?.classList.add('active');    }
function closeModal(id) { document.getElementById(id)?.classList.remove('active'); }

function doReset() {
    document.getElementById('primaryCarForm')?.reset();
    document.getElementById('compareCarForm')?.reset();
    setCompareMode(false);
    const resultsSection = document.getElementById('resultsSection');
    if (resultsSection) resultsSection.classList.remove('active');
    // Reset model dropdowns
    ['primary_model', 'compare_model'].forEach(id => {
        const el = document.getElementById(id);
        if (el) { el.innerHTML = '<option value="" disabled selected>Select Brand First</option>'; el.disabled = true; }
    });
}

// ── Mobile Menu ───────────────────────────────────────────────────────
function initMobileMenu() {
    const toggle = document.getElementById('mobileMenuToggle');
    const nav    = document.getElementById('headerNav');
    if (toggle && nav) toggle.addEventListener('click', () => nav.classList.toggle('active'));
}

// ── Utilities ─────────────────────────────────────────────────────────
function fmtNum(n) { return Number(n).toLocaleString('en-US'); }
function escHtml(s) {
    return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}
