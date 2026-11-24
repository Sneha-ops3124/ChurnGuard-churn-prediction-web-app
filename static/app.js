const form = document.getElementById("predict-form");
const result = document.getElementById("result");
const submitBtn = document.getElementById("submit-btn");
const batchFile = document.getElementById("batch-file");
const fileLabel = document.getElementById("file-label");

const SAMPLES = {
  high: {
    gender: "Female",
    SeniorCitizen: "0",
    Partner: "No",
    Dependents: "No",
    PhoneService: "Yes",
    MultipleLines: "No",
    InternetService: "Fiber optic",
    OnlineSecurity: "No",
    OnlineBackup: "No",
    DeviceProtection: "No",
    TechSupport: "No",
    StreamingTV: "Yes",
    StreamingMovies: "Yes",
    Contract: "Month-to-month",
    PaperlessBilling: "Yes",
    PaymentMethod: "Electronic check",
    tenure: 2,
    MonthlyCharges: 89.5,
    TotalCharges: 179.0,
  },
  low: {
    gender: "Male",
    SeniorCitizen: "0",
    Partner: "Yes",
    Dependents: "Yes",
    PhoneService: "Yes",
    MultipleLines: "Yes",
    InternetService: "DSL",
    OnlineSecurity: "Yes",
    OnlineBackup: "Yes",
    DeviceProtection: "Yes",
    TechSupport: "Yes",
    StreamingTV: "No",
    StreamingMovies: "No",
    Contract: "Two year",
    PaperlessBilling: "No",
    PaymentMethod: "Credit card (automatic)",
    tenure: 64,
    MonthlyCharges: 68.0,
    TotalCharges: 4352.0,
  },
};

function applySample(sample) {
  Object.entries(sample).forEach(([key, val]) => {
    const el = form.elements[key];
    if (el) el.value = val;
  });
  form.dispatchEvent(new Event("submit"));
}

document.getElementById("fill-high-risk")?.addEventListener("click", () => applySample(SAMPLES.high));
document.getElementById("fill-low-risk")?.addEventListener("click", () => applySample(SAMPLES.low));
document.getElementById("reset-form")?.addEventListener("click", () => {
  form.reset();
  result.innerHTML = `
    <div class="empty-state">
      <div class="empty-icon">📊</div>
      <h4>Ready to score</h4>
      <p>Fill out the customer profile or load a sample profile to see churn probability, recommended action, and top drivers.</p>
    </div>
  `;
});

batchFile?.addEventListener("change", () => {
  if (batchFile.files.length > 0) {
    fileLabel.textContent = `Selected: ${batchFile.files[0].name}`;
  }
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  submitBtn.disabled = true;
  result.innerHTML = `
    <div class="loading-box">
      <div class="spinner"></div>
      <p>Computing churn risk &amp; SHAP explanations...</p>
    </div>
  `;

  const data = Object.fromEntries(new FormData(form).entries());

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });
    if (!response.ok) throw new Error(`Server returned ${response.status}`);
    const r = await response.json();

    const pct = (r.probability * 100).toFixed(1);
    const threshPct = (r.threshold * 100).toFixed(0);
    const riskClass = r.risk.toLowerCase();

    const actionTitle = r.flagged
      ? "Priority Action: Send Retention Offer"
      : "Status: No Immediate Intervention Needed";
    const actionDesc = r.flagged
      ? `Probability exceeds the ${threshPct}% profit-optimal threshold. Contact this customer this week.`
      : `Probability is below the ${threshPct}% action threshold. Continue standard engagement.`;

    const maxImpact = Math.max(...r.drivers.map((d) => Math.abs(d.impact)), 0.01);
    const driversHtml = r.drivers
      .map((d) => {
        const up = d.impact > 0;
        const barWidth = Math.min(100, Math.round((Math.abs(d.impact) / maxImpact) * 100));
        return `
          <li class="driver-item">
            <div class="driver-top">
              <span class="driver-name">${d.feature}</span>
              <span class="driver-tag ${up ? "tag-up" : "tag-down"}">
                ${up ? "+" : "-"}${Math.abs(d.impact).toFixed(2)} SHAP
              </span>
            </div>
            <div class="driver-track">
              <div class="driver-fill ${up ? "fill-up" : "fill-down"}" style="width: ${barWidth}%"></div>
            </div>
          </li>
        `;
      })
      .join("");

    result.innerHTML = `
      <div class="score-summary fade-in">
        <div class="score-header">
          <div>
            <span class="score-kicker">Predicted Churn Probability</span>
            <div class="score-number">${pct}%</div>
          </div>
          <span class="badge badge-${riskClass}">${r.risk} Risk</span>
        </div>

        <div class="meter-wrap">
          <div class="meter-track">
            <div class="meter-fill meter-${riskClass}" style="width: ${Math.min(100, r.probability * 100)}%"></div>
            <div class="meter-marker" style="left: ${Math.min(100, r.threshold * 100)}%" title="Threshold ${threshPct}%"></div>
          </div>
          <div class="meter-labels">
            <span>0%</span>
            <span>Action Threshold (${threshPct}%)</span>
            <span>100%</span>
          </div>
        </div>

        <div class="action-box ${r.flagged ? "action-alert" : "action-ok"}">
          <strong>${actionTitle}</strong>
          <p>${actionDesc}</p>
        </div>

        <div class="drivers-section">
          <h4>Top 3 Drivers For This Customer</h4>
          <ul class="driver-list">${driversHtml}</ul>
        </div>
      </div>
    `;
  } catch (err) {
    result.innerHTML = `<div class="error-box">Something went wrong: ${err.message}</div>`;
  } finally {
    submitBtn.disabled = false;
  }
});
