document.getElementById('riskForm').addEventListener('submit', function (e) {
  e.preventDefault();

  const absenceDays = parseInt(document.getElementById('absence_days').value);
  const weeklyHours = parseInt(document.getElementById('weekly_self_study_hours').value);
  const extra = document.getElementById('extracurricular_activities').value;
  const career = document.getElementById('career_aspiration').value.trim().toLowerCase();

  const scores = [
    parseFloat(document.getElementById('math_score').value),
    parseFloat(document.getElementById('history_score').value),
    parseFloat(document.getElementById('physics_score').value),
    parseFloat(document.getElementById('chemistry_score').value),
    parseFloat(document.getElementById('biology_score').value),
    parseFloat(document.getElementById('english_score').value),
    parseFloat(document.getElementById('geography_score').value)
  ];

  const totalScore = scores.reduce((a, b) => a + b, 0) / scores.length;
  const isAtRisk = totalScore < 60 || absenceDays > 5;

  let resultText = `<h2>Analysis Result</h2>`;
  resultText += `<p><strong>Total Average Score:</strong> ${totalScore.toFixed(2)}</p>`;
  resultText += `<p><strong>Predicted Risk Status:</strong> ${isAtRisk ? '⚠️ At Risk' : '✅ Not At Risk'}</p>`;

  if (isAtRisk) {
    resultText += `<p><strong>Recommended Interventions:</strong></p><ul>`;
    if (totalScore < 60) resultText += `<li>Seek help for low-performing subjects.</li>`;
    if (absenceDays > 5) resultText += `<li>Reduce absences and improve attendance.</li>`;
    if (weeklyHours < 5) resultText += `<li>Increase self-study with a daily plan.</li>`;
    if (extra === "no") resultText += `<li>Join extracurriculars to improve time management.</li>`;
    if (career === "unknown" || career === "") resultText += `<li>Get career guidance for setting goals.</li>`;
    resultText += `</ul>`;
  }

  document.getElementById('riskResult').innerHTML = resultText;
});