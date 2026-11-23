/**
 * Client-side form handling and prediction rendering.
 */

let optionsMap = {};

document.addEventListener("DOMContentLoaded", async () => {
  initThemeToggle();
  await loadOptions();
  setupEventListeners();
});

function initThemeToggle() {
  const toggleBtn = document.getElementById("theme-toggle");
  if (!toggleBtn) return;

  function updateButton(theme) {
    const icon = toggleBtn.querySelector(".theme-icon");
    const text = toggleBtn.querySelector(".theme-text");
    if (theme === "dark") {
      if (icon) icon.textContent = "☀️";
      if (text) text.textContent = "Day Mode";
      toggleBtn.setAttribute("title", "Switch to Day Mode");
    } else {
      if (icon) icon.textContent = "🌙";
      if (text) text.textContent = "Night Mode";
      toggleBtn.setAttribute("title", "Switch to Night Mode");
    }
  }

  const currentTheme = document.documentElement.getAttribute("data-theme") || "light";
  updateButton(currentTheme);

  toggleBtn.addEventListener("click", () => {
    const activeTheme = document.documentElement.getAttribute("data-theme") || "light";
    const newTheme = activeTheme === "dark" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", newTheme);
    localStorage.setItem("theme", newTheme);
    updateButton(newTheme);
  });
}

async function loadOptions() {
  try {
    const res = await fetch("/api/options");
    if (!res.ok) return;
    optionsMap = await res.json();
    populateSelects();
  } catch (err) {
    console.error("Failed to fetch /api/options:", err);
  }
}

const FIELD_ID_MAP = {
  "Course": "field-course",
  "Application mode": "field-app-mode",
  "Marital Status": "field-marital-status",
  "Gender": "field-gender",
  "Daytime/evening attendance": "field-attendance",
  "Displaced": "field-displaced",
  "Educational special needs": "field-special-needs",
  "Debtor": "field-debtor",
  "Tuition fees up to date": "field-tuition",
  "Scholarship holder": "field-scholarship",
  "International": "field-international",
  "Nacionality": "field-nacionality",
  "Previous qualification": "field-previous-qualification",
  "Mother's qualification": "field-mother-s-qualification",
  "Father's qualification": "field-father-s-qualification",
  "Mother's occupation": "field-mother-s-occupation",
  "Father's occupation": "field-father-s-occupation",
};

function getFieldId(name) {
  return FIELD_ID_MAP[name] || `field-${name.toLowerCase().replace(/[^a-z0-9]/g, "-")}`;
}

function populateSelects() {
  const fields = [
    "Course",
    "Application mode",
    "Marital Status",
    "Gender",
    "Daytime/evening attendance",
    "Displaced",
    "Educational special needs",
    "Debtor",
    "Tuition fees up to date",
    "Scholarship holder",
    "International",
    "Nacionality",
    "Previous qualification",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
  ];

  fields.forEach(field => {
    const select = document.getElementById(getFieldId(field));
    if (select && optionsMap[field]) {
      select.innerHTML = "";
      optionsMap[field].forEach(opt => {
        const optionEl = document.createElement("option");
        optionEl.value = opt.code;
        optionEl.textContent = opt.label;
        select.appendChild(optionEl);
      });
    }
  });

  // Set sensible initial defaults for demonstration
  setBaselineDefaults();
}

function setBaselineDefaults() {
  setVal("field-course", 9254); // Tourism
  setVal("field-app-mode", 1); // 1st phase
  setVal("field-app-order", 1);
  setVal("field-admission-grade", 125.0);
  setVal("field-prev-grade", 130.0);
  setVal("field-age", 19);
  setVal("field-gender", 1);
  setVal("field-tuition", 1);
  setVal("field-debtor", 0);
  setVal("field-scholarship", 0);
  setVal("field-attendance", 1);
  setVal("field-marital-status", 1);
  setVal("field-displaced", 1);
  setVal("field-special-needs", 0);
  setVal("field-international", 0);
  setVal("field-nacionality", 1);
  setVal("field-previous-qualification", 1);
  setVal("field-mother-s-qualification", 1);
  setVal("field-father-s-qualification", 1);
  setVal("field-mother-s-occupation", 5);
  setVal("field-father-s-occupation", 9);
  setVal("field-unemployment-rate", 10.8);
  setVal("field-inflation-rate", 1.4);
  setVal("field-gdp", 1.74);

  // Sem 1 & 2
  setVal("field-s1-enrolled", 6);
  setVal("field-s1-approved", 5);
  setVal("field-s1-evaluations", 7);
  setVal("field-s1-grade", 12.5);
  setVal("field-s1-credited", 0);
  setVal("field-s1-noeval", 0);

  setVal("field-s2-enrolled", 6);
  setVal("field-s2-approved", 5);
  setVal("field-s2-evaluations", 7);
  setVal("field-s2-grade", 13.0);
  setVal("field-s2-credited", 0);
  setVal("field-s2-noeval", 0);

  const s1Check = document.getElementById("toggle-s1");
  const s2Check = document.getElementById("toggle-s2");
  if (s1Check) s1Check.checked = false;
  if (s2Check) s2Check.checked = false;
  const s1Fields = document.getElementById("s1-fields");
  const s2Fields = document.getElementById("s2-fields");
  if (s1Fields) s1Fields.classList.add("hidden");
  if (s2Fields) s2Fields.classList.add("hidden");
  updateStageBadge();
}

function setVal(id, val) {
  const el = document.getElementById(id);
  if (el) el.value = val;
}

function setupEventListeners() {
  const form = document.getElementById("prediction-form");
  if (!form) return;

  const s1Check = document.getElementById("toggle-s1");
  const s2Check = document.getElementById("toggle-s2");
  const s1Fields = document.getElementById("s1-fields");
  const s2Fields = document.getElementById("s2-fields");
  const advancedToggle = document.getElementById("toggle-advanced");
  const advancedFields = document.getElementById("advanced-fields");

  if (s1Check) {
    s1Check.addEventListener("change", () => {
      if (s1Fields) s1Fields.classList.toggle("hidden", !s1Check.checked);
      if (!s1Check.checked && s2Check && s2Check.checked) {
        s2Check.checked = false;
        if (s2Fields) s2Fields.classList.add("hidden");
      }
      updateStageBadge();
    });
  }

  if (s2Check) {
    s2Check.addEventListener("change", () => {
      if (s2Fields) s2Fields.classList.toggle("hidden", !s2Check.checked);
      if (s2Check.checked && s1Check && !s1Check.checked) {
        s1Check.checked = true;
        if (s1Fields) s1Fields.classList.remove("hidden");
      }
      updateStageBadge();
    });
  }

  if (advancedToggle && advancedFields) {
    advancedToggle.addEventListener("click", () => {
      const isHidden = advancedFields.classList.toggle("hidden");
      advancedToggle.textContent = isHidden ? "+ Show Advanced Admission Fields" : "- Hide Advanced Fields";
    });
  }

  // Presets
  const btnHigh = document.getElementById("preset-high-risk");
  const btnMed = document.getElementById("preset-medium-risk");
  const btnLow = document.getElementById("preset-low-risk");
  const btnReset = document.getElementById("preset-reset");

  if (btnHigh) {
    btnHigh.addEventListener("click", () => {
      setVal("field-course", 9003); // Agronomy
      setVal("field-age", 24);
      setVal("field-admission-grade", 108.0);
      setVal("field-prev-grade", 112.0);
      setVal("field-tuition", 0); // Unpaid fees!
      setVal("field-debtor", 1); // Debtor!
      setVal("field-scholarship", 0);
      setVal("field-attendance", 0); // Evening
      // Unlock Stage 2 to show zero approvals
      s1Check.checked = true;
      s1Fields.classList.remove("hidden");
      s2Check.checked = false;
      s2Fields.classList.add("hidden");
      setVal("field-s1-enrolled", 6);
      setVal("field-s1-approved", 0); // 0 approvals!
      setVal("field-s1-evaluations", 2);
      setVal("field-s1-grade", 0.0);
      updateStageBadge();
    });
  }

  if (btnMed) {
    btnMed.addEventListener("click", () => {
      setVal("field-course", 9254); // Tourism
      setVal("field-age", 20);
      setVal("field-admission-grade", 122.0);
      setVal("field-prev-grade", 126.0);
      setVal("field-tuition", 1);
      setVal("field-debtor", 0);
      setVal("field-scholarship", 0);
      setVal("field-attendance", 1);
      s1Check.checked = true;
      s1Fields.classList.remove("hidden");
      s2Check.checked = false;
      s2Fields.classList.add("hidden");
      setVal("field-s1-enrolled", 6);
      setVal("field-s1-approved", 3); // 50% approval
      setVal("field-s1-evaluations", 6);
      setVal("field-s1-grade", 11.0);
      updateStageBadge();
    });
  }

  if (btnLow) {
    btnLow.addEventListener("click", () => {
      setVal("field-course", 9254); // Tourism
      setVal("field-age", 18);
      setVal("field-admission-grade", 160.0);
      setVal("field-prev-grade", 155.0);
      setVal("field-tuition", 1);
      setVal("field-debtor", 0);
      setVal("field-scholarship", 1); // Scholarship holder
      setVal("field-attendance", 1);
      s1Check.checked = true;
      s1Fields.classList.remove("hidden");
      s2Check.checked = true;
      s2Fields.classList.remove("hidden");
      setVal("field-s1-enrolled", 6);
      setVal("field-s1-approved", 6);
      setVal("field-s1-evaluations", 7);
      setVal("field-s1-grade", 16.0);
      setVal("field-s2-enrolled", 6);
      setVal("field-s2-approved", 6);
      setVal("field-s2-evaluations", 7);
      setVal("field-s2-grade", 16.5);
      updateStageBadge();
    });
  }

  if (btnReset) {
    btnReset.addEventListener("click", () => {
      setBaselineDefaults();
    });
  }

  form.addEventListener("submit", handleSubmit);
}

function updateStageBadge() {
  const toggleS1 = document.getElementById("toggle-s1");
  const toggleS2 = document.getElementById("toggle-s2");
  if (!toggleS1 || !toggleS2) return;

  const s1 = toggleS1.checked;
  const s2 = toggleS2.checked;
  const badge = document.getElementById("active-stage-badge");
  const badgeS1 = document.getElementById("badge-s1");
  const badgeS2 = document.getElementById("badge-s2");

  const completionPct = document.getElementById("completion-percentage");
  const completionBar = document.getElementById("completion-progress-bar");
  const checkS2 = document.getElementById("check-s2");
  const checkS3 = document.getElementById("check-s3");
  const subnavS1 = document.getElementById("subnav-s1");
  const subnavS2 = document.getElementById("subnav-s2");
  const subnavS3 = document.getElementById("subnav-s3");

  if (s2) {
    if (badge) badge.textContent = "Stage 3 (After Semester 2)";
    if (completionPct) completionPct.textContent = "100%";
    if (completionBar) completionBar.style.width = "100%";
    if (checkS2) checkS2.classList.add("checked");
    if (checkS3) checkS3.classList.add("checked");
    if (subnavS3) subnavS3.classList.add("active");
    if (subnavS2) subnavS2.classList.remove("active");
    if (subnavS1) subnavS1.classList.remove("active");
  } else if (s1) {
    if (badge) badge.textContent = "Stage 2 (After Semester 1)";
    if (completionPct) completionPct.textContent = "66%";
    if (completionBar) completionBar.style.width = "66%";
    if (checkS2) checkS2.classList.add("checked");
    if (checkS3) checkS3.classList.remove("checked");
    if (subnavS2) subnavS2.classList.add("active");
    if (subnavS3) subnavS3.classList.remove("active");
    if (subnavS1) subnavS1.classList.remove("active");
  } else {
    if (badge) badge.textContent = "Stage 1 (Admission)";
    if (completionPct) completionPct.textContent = "33%";
    if (completionBar) completionBar.style.width = "33%";
    if (checkS2) checkS2.classList.remove("checked");
    if (checkS3) checkS3.classList.remove("checked");
    if (subnavS1) subnavS1.classList.add("active");
    if (subnavS2) subnavS2.classList.remove("active");
    if (subnavS3) subnavS3.classList.remove("active");
  }

  if (badgeS1) {
    if (s1) {
      badgeS1.className = "badge badge-active";
      badgeS1.textContent = "✓ Stage 2 Active";
    } else {
      badgeS1.className = "badge badge-unlock";
      badgeS1.textContent = "Unlocks Stage 2";
    }
  }

  if (badgeS2) {
    if (s2) {
      badgeS2.className = "badge badge-active";
      badgeS2.textContent = "✓ Stage 3 Active";
    } else {
      badgeS2.className = "badge badge-unlock";
      badgeS2.textContent = "Unlocks Stage 3";
    }
  }
}

async function handleSubmit(e) {
  e.preventDefault();

  const errorBanner = document.getElementById("error-banner");
  errorBanner.classList.add("hidden");
  errorBanner.innerHTML = "";

  const submitBtn = document.getElementById("submit-btn");
  const btnText = document.getElementById("btn-text");
  const btnSpinner = document.getElementById("btn-spinner");

  btnText.textContent = "Analyzing Student Profile...";
  btnSpinner.classList.remove("hidden");
  submitBtn.disabled = true;

  const payload = buildPayload();

  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const data = await res.json();

    if (!res.ok) {
      showError(data);
    } else {
      renderResults(data);
    }
  } catch (err) {
    errorBanner.textContent = "Server communication error. Please check your network or server logs.";
    errorBanner.classList.remove("hidden");
  } finally {
    btnText.textContent = "Run Risk Assessment";
    btnSpinner.classList.add("hidden");
    submitBtn.disabled = false;
  }
}

function buildPayload() {
  const s1Checked = document.getElementById("toggle-s1").checked;
  const s2Checked = document.getElementById("toggle-s2").checked;

  const student = {
    "Course": parseInt(getVal("field-course")),
    "Application mode": parseInt(getVal("field-app-mode")),
    "Application order": parseInt(getVal("field-app-order")),
    "Admission grade": parseFloat(getVal("field-admission-grade")),
    "Previous qualification (grade)": parseFloat(getVal("field-prev-grade")),
    "Age at enrollment": parseInt(getVal("field-age")),
    "Gender": parseInt(getVal("field-gender")),
    "Marital Status": parseInt(getVal("field-marital-status") || 1),
    "Tuition fees up to date": parseInt(getVal("field-tuition")),
    "Debtor": parseInt(getVal("field-debtor")),
    "Scholarship holder": parseInt(getVal("field-scholarship")),
    "Daytime/evening attendance": parseInt(getVal("field-attendance")),
    "Displaced": parseInt(getVal("field-displaced")),
    "Educational special needs": parseInt(getVal("field-special-needs")),
    "International": parseInt(getVal("field-international")),
    "Nacionality": parseInt(getVal("field-nacionality")),
    "Previous qualification": parseInt(getVal("field-previous-qualification")),
    "Mother's qualification": parseInt(getVal("field-mother-s-qualification")),
    "Father's qualification": parseInt(getVal("field-father-s-qualification")),
    "Mother's occupation": parseInt(getVal("field-mother-s-occupation")),
    "Father's occupation": parseInt(getVal("field-father-s-occupation")),
    "Unemployment rate": parseFloat(getVal("field-unemployment-rate") || 10.8),
    "Inflation rate": parseFloat(getVal("field-inflation-rate") || 1.4),
    "GDP": parseFloat(getVal("field-gdp") || 1.74),
  };

  if (s1Checked) {
    student["Curricular units 1st sem (enrolled)"] = parseInt(getVal("field-s1-enrolled"));
    student["Curricular units 1st sem (approved)"] = parseInt(getVal("field-s1-approved"));
    student["Curricular units 1st sem (evaluations)"] = parseInt(getVal("field-s1-evaluations"));
    student["Curricular units 1st sem (grade)"] = parseFloat(getVal("field-s1-grade"));
    student["Curricular units 1st sem (credited)"] = parseInt(getVal("field-s1-credited") || 0);
    student["Curricular units 1st sem (without evaluations)"] = parseInt(getVal("field-s1-noeval") || 0);
  }

  if (s2Checked) {
    student["Curricular units 2nd sem (enrolled)"] = parseInt(getVal("field-s2-enrolled"));
    student["Curricular units 2nd sem (approved)"] = parseInt(getVal("field-s2-approved"));
    student["Curricular units 2nd sem (evaluations)"] = parseInt(getVal("field-s2-evaluations"));
    student["Curricular units 2nd sem (grade)"] = parseFloat(getVal("field-s2-grade"));
    student["Curricular units 2nd sem (credited)"] = parseInt(getVal("field-s2-credited") || 0);
    student["Curricular units 2nd sem (without evaluations)"] = parseInt(getVal("field-s2-noeval") || 0);
  }

  return { student };
}

function getVal(id) {
  const el = document.getElementById(id);
  return el ? el.value : "";
}

function showError(data) {
  const errorBanner = document.getElementById("error-banner");
  errorBanner.classList.remove("hidden");

  let msg = `<strong>${data.message || "Error"}</strong>`;
  if (data.details && Array.isArray(data.details)) {
    msg += "<ul>";
    data.details.forEach(d => {
      msg += `<li><strong>${d.field}:</strong> ${d.problem}</li>`;
    });
    msg += "</ul>";
  }
  errorBanner.innerHTML = msg;
}

function renderResults(data) {
  document.getElementById("results-placeholder").classList.add("hidden");
  const panel = document.getElementById("results-content");
  panel.classList.remove("hidden");

  // Outcome label & risk pill
  const predEl = document.getElementById("res-prediction");
  predEl.textContent = data.prediction;
  predEl.className = "prediction-badge pred-" + data.prediction;
  predEl.removeAttribute("style");

  const riskEl = document.getElementById("res-risk-level");
  riskEl.textContent = `${data.risk_level} Risk (${(data.dropout_probability * 100).toFixed(1)}% Dropout Probability)`;
  riskEl.className = `risk-pill risk-${data.risk_level}`;

  document.getElementById("res-stage-used").textContent = `Evaluated at Stage ${data.stage_used}`;

  // Probabilities
  const pDrop = (data.probabilities.Dropout * 100).toFixed(1);
  const pEnr = (data.probabilities.Enrolled * 100).toFixed(1);
  const pGrad = (data.probabilities.Graduate * 100).toFixed(1);

  document.getElementById("val-dropout").textContent = `${pDrop}%`;
  document.getElementById("bar-dropout").style.width = `${pDrop}%`;

  document.getElementById("val-enrolled").textContent = `${pEnr}%`;
  document.getElementById("bar-enrolled").style.width = `${pEnr}%`;

  document.getElementById("val-graduate").textContent = `${pGrad}%`;
  document.getElementById("bar-graduate").style.width = `${pGrad}%`;

  // Update KPI metric cards
  const kpiDropout = document.getElementById("val-dropout-kpi");
  const kpiEnrolled = document.getElementById("val-enrolled-kpi");
  const kpiGraduate = document.getElementById("val-graduate-kpi");
  const kpiTrend = document.getElementById("val-dropout-trend");
  const kpiGradTrend = document.getElementById("val-graduate-trend");

  if (kpiDropout) kpiDropout.textContent = `${pDrop}%`;
  if (kpiEnrolled) kpiEnrolled.textContent = `${pEnr}%`;
  if (kpiGraduate) kpiGraduate.textContent = `${pGrad}%`;

  if (kpiTrend) {
    if (parseFloat(pDrop) >= 66) {
      kpiTrend.textContent = "↑ Alert";
      kpiTrend.className = "trend-pill trend-up";
    } else if (parseFloat(pDrop) >= 33) {
      kpiTrend.textContent = "↑ Moderate";
      kpiTrend.className = "trend-pill trend-up";
    } else {
      kpiTrend.textContent = "↓ Safe Tier";
      kpiTrend.className = "trend-pill trend-down";
    }
  }

  if (kpiGradTrend) {
    kpiGradTrend.textContent = parseFloat(pGrad) >= 50 ? "✓ Strong" : "—";
    kpiGradTrend.className = parseFloat(pGrad) >= 50 ? "trend-pill trend-down" : "trend-pill";
  }

  // Top Factors
  const factorsUl = document.getElementById("res-factors");
  factorsUl.innerHTML = "";
  if (data.top_factors && data.top_factors.length > 0) {
    data.top_factors.forEach(f => {
      const li = document.createElement("li");
      li.className = "factor-item";
      const isRisk = f.impact === "increases_risk";
      li.innerHTML = `
        <span class="${isRisk ? 'factor-icon-up' : 'factor-icon-down'}">${isRisk ? '▲' : '▼'}</span>
        <div>
          <strong>${f.feature}</strong>: ${f.value !== null ? f.value : ''} 
          <span style="color: ${isRisk ? '#dc2626' : '#16a34a'}; font-size: 0.8rem;">
            (${isRisk ? 'Increases dropout risk' : 'Pushes toward graduation'})
          </span>
        </div>
      `;
      factorsUl.appendChild(li);
    });
  }

  // Actions
  const actionsUl = document.getElementById("res-actions");
  actionsUl.innerHTML = "";
  data.suggested_actions.forEach(a => {
    const li = document.createElement("li");
    li.textContent = a;
    actionsUl.appendChild(li);
  });
}

