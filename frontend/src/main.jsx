import React from 'react';
import ReactDOM from 'react-dom/client';
import './styles.css';

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000';
const fields = [
  { name: 'ph', label: 'pH', unit: 'pH', min: 0, max: 14, step: 0.1 },
  { name: 'ec', label: 'Electrical conductivity', unit: 'mS/cm', min: 0, step: 0.1 },
  { name: 'water_temp', label: 'Water temperature', unit: '°C', min: 0, step: 0.1 },
  { name: 'humidity', label: 'Relative humidity', unit: '%', min: 0, max: 100, step: 1 },
  { name: 'air_temp', label: 'Air temperature', unit: '°C', min: -50, max: 80, step: 0.1 },
];

async function requestJson(path) {
  const response = await fetch(`${API_BASE}${path}`);
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.detail || 'The request could not be completed.');
  return payload;
}

function formatDate(value) {
  if (!value) return 'Time unavailable';
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString();
}

function App() {
  const [form, setForm] = React.useState({
    ph: '',
    ec: '',
    water_temp: '',
    humidity: '',
    air_temp: '',
  });
  const [image, setImage] = React.useState(null);
  const [preview, setPreview] = React.useState('');
  const [result, setResult] = React.useState(null);
  const [history, setHistory] = React.useState([]);
  const [latest, setLatest] = React.useState(null);
  const [health, setHealth] = React.useState(null);
  const [loading, setLoading] = React.useState(false);
  const [refreshing, setRefreshing] = React.useState(true);
  const [error, setError] = React.useState('');

  React.useEffect(() => {
    if (!image) {
      setPreview('');
      return undefined;
    }
    const objectUrl = URL.createObjectURL(image);
    setPreview(objectUrl);
    return () => URL.revokeObjectURL(objectUrl);
  }, [image]);

  const refreshData = async () => {
    setRefreshing(true);
    try {
      const [healthData, historyData, latestData] = await Promise.all([
        requestJson('/api/health'),
        requestJson('/api/history?limit=12'),
        requestJson('/api/sensor/latest'),
      ]);
      setHealth(healthData);
      setHistory(historyData.items || []);
      setLatest(latestData.status === 'no_data' ? null : latestData);
      if (historyData.items?.length) {
        const saved = historyData.items[0];
        setResult({
          observation_id: saved.observation_id,
          sensor: {
            prediction: saved.sensor_prediction || 'unknown',
            confidence: saved.sensor_confidence || 0,
            probabilities: saved.sensor_probabilities || {},
            readings: {
              ph: saved.ph,
              ec: saved.ec,
              water_temp: saved.water_temp,
              humidity: saved.humidity,
              air_temp: saved.air_temp,
            },
          },
          image: {
            predicted_class: saved.image_prediction || 'unknown',
            confidence: saved.image_confidence || 0,
            status: saved.image_prediction && saved.image_prediction !== 'unknown' ? 'ready' : 'pending_cnn_model',
          },
          fusion: { status: saved.fusion_status || 'insufficient_confidence' },
          recommendation: saved.recommendation || {
            summary: saved.final_diagnosis || 'No recommendation was saved for this observation.',
            actions: [],
            warnings: [],
          },
        });
      }
    } catch (requestError) {
      setError(requestError.message || 'Unable to connect to the API.');
    } finally {
      setRefreshing(false);
    }
  };

  React.useEffect(() => {
    refreshData();
  }, []);

  const handleChange = (event) => {
    setForm((current) => ({ ...current, [event.target.name]: event.target.value }));
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    setLoading(true);
    setError('');
    const body = new FormData();
    Object.entries(form).forEach(([key, value]) => body.append(key, value));
    if (image) body.append('image', image);

    try {
      const response = await fetch(`${API_BASE}/api/predict`, { method: 'POST', body });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || 'Analysis failed.');
      setResult(payload);
      await refreshData();
    } catch (requestError) {
      setError(requestError.message || 'Unable to reach the backend API.');
    } finally {
      setLoading(false);
    }
  };

  const probabilities = Object.entries(result?.sensor?.probabilities || {})
    .sort(([, first], [, second]) => second - first);

  return (
    <div className="app-shell">
      <header className="masthead">
        <a className="wordmark" href="#overview" aria-label="Hydroponic Nutrition AI home">
          <span className="wordmark-mark" aria-hidden="true">H</span>
          <span>Hydroponic<br />Nutrition AI</span>
        </a>
        <div className="masthead-meta">
          <span className={`connection-dot ${health?.sensor_model_ready ? 'is-live' : ''}`} />
          <span>{health?.sensor_model_ready ? 'Sensor model online' : 'API unavailable'}</span>
          <span className="meta-divider" />
          <span className="cnn-state">CNN pending</span>
        </div>
      </header>

      <main id="overview" className="page-content">
        <section className="page-heading">
          <div>
            <p className="eyebrow">Grow room / Overview</p>
            <h1>Crop health, in context.</h1>
            <p className="heading-copy">Review sensor classifications and keep each observation tied to its source.</p>
          </div>
          <button className="quiet-button" type="button" onClick={refreshData} disabled={refreshing}>
            <span aria-hidden="true">↻</span> {refreshing ? 'Refreshing' : 'Refresh data'}
          </button>
        </section>

        {error && <div className="notice notice-error" role="alert">{error}</div>}

        <section className="status-strip" aria-label="System status">
          <div className="status-item">
            <span className="status-label">Sensor classifier</span>
            <strong>{health?.sensor_model_ready ? 'Ready' : 'Offline'}</strong>
            <span className="status-detail">Saved model · inference only</span>
          </div>
          <div className="status-item">
            <span className="status-label">Visual classifier</span>
            <strong className="text-amber">Awaiting model</strong>
            <span className="status-detail">Images can be attached; no visual prediction is made</span>
          </div>
          <div className="status-item">
            <span className="status-label">Latest observation</span>
            <strong>{latest ? formatDate(latest.timestamp) : 'No readings yet'}</strong>
            <span className="status-detail">{latest ? `Source: ${latest.source}` : 'Manual and ESP32 readings appear here'}</span>
          </div>
        </section>

        <div className="workspace-grid">
          <section className="entry-section" aria-labelledby="entry-heading">
            <div className="section-heading">
              <div>
                <p className="eyebrow">New observation</p>
                <h2 id="entry-heading">Sensor readings</h2>
              </div>
              <span className="step-index">01</span>
            </div>
            <form onSubmit={handleSubmit} className="sensor-form">
              <div className="field-grid">
                {fields.map((field) => (
                  <label className="field" key={field.name}>
                    <span className="field-label">{field.label}</span>
                    <span className="input-wrap">
                      <input
                        type="number"
                        name={field.name}
                        value={form[field.name]}
                        min={field.min}
                        max={field.max}
                        step={field.step}
                        onChange={handleChange}
                        required
                      />
                      <span className="unit">{field.unit}</span>
                    </span>
                  </label>
                ))}
              </div>

              <label className="upload-field">
                <input
                  type="file"
                  accept="image/jpeg,image/png,image/webp"
                  onChange={(event) => setImage(event.target.files?.[0] || null)}
                />
                {preview ? (
                  <span className="upload-preview">
                    <img src={preview} alt="Selected lettuce sample" />
                    <span><strong>{image.name}</strong><small>Saved with this observation · visual analysis pending</small></span>
                    <button className="remove-image" type="button" onClick={(event) => { event.preventDefault(); event.stopPropagation(); setImage(null); }}>Remove</button>
                  </span>
                ) : (
                  <span className="upload-prompt">
                    <span className="upload-icon" aria-hidden="true">＋</span>
                    <span><strong>Attach a crop image</strong><small>JPEG, PNG or WebP · up to 10 MB</small></span>
                    <span className="upload-note">Optional</span>
                  </span>
                )}
              </label>

              <div className="form-footer">
                <p>Sensor predictions are classifications, not nutrient-dose instructions.</p>
                <button className="primary-button" type="submit" disabled={loading || !health?.sensor_model_ready}>
                  {loading ? 'Analyzing…' : 'Run sensor analysis'}
                  <span aria-hidden="true">→</span>
                </button>
              </div>
            </form>
          </section>

          <section className="analysis-section" aria-labelledby="analysis-heading">
            <div className="section-heading">
              <div>
                <p className="eyebrow">Most recent result</p>
                <h2 id="analysis-heading">Analysis</h2>
              </div>
              <span className="step-index">02</span>
            </div>
            {!result ? (
              <div className="analysis-empty">
                <div className="empty-mark" aria-hidden="true">H₂O</div>
                <p>Your next observation will appear here.</p>
                <span>Enter a complete sensor reading to begin.</span>
              </div>
            ) : (
              <div className="analysis-result" aria-live="polite">
                <div className="result-banner">
                  <div>
                    <span className="status-label">Sensor classification</span>
                    <h3>{result.sensor.prediction.replaceAll('_', ' ')}</h3>
                  </div>
                  <div className="confidence-value">
                    <strong>{Math.round(result.sensor.confidence * 100)}%</strong>
                    <span>confidence</span>
                  </div>
                </div>

                <div className="result-meta">
                  <span>Observation #{result.observation_id}</span>
                  <span className="result-state">{result.fusion.status.replaceAll('_', ' ')}</span>
                </div>

                <div className="probability-list">
                  <div className="subsection-title"><span>Class probabilities</span><span>Sensor model</span></div>
                  {probabilities.map(([label, value]) => (
                    <div className="probability-row" key={label}>
                      <span>{label.replaceAll('_', ' ')}</span>
                      <span className="probability-track"><span style={{ width: `${value * 100}%` }} /></span>
                      <strong>{Math.round(value * 100)}%</strong>
                    </div>
                  ))}
                </div>

                <div className="recommendation">
                  <span className="recommendation-kicker">Review note</span>
                  <p>{result.recommendation.summary}</p>
                  <ul>{result.recommendation.actions.map((action) => <li key={action}>{action}</li>)}</ul>
                  {result.recommendation.warnings.map((warning) => <p className="warning-text" key={warning}>{warning}</p>)}
                </div>

                <p className="visual-pending"><span aria-hidden="true">◌</span> Visual classifier pending. No image-based finding was generated.</p>
              </div>
            )}
          </section>
        </div>

        <section className="history-section" aria-labelledby="history-heading">
          <div className="section-heading history-heading">
            <div>
              <p className="eyebrow">Observation log</p>
              <h2 id="history-heading">Recent readings</h2>
            </div>
            <span className="history-count">{history.length} shown</span>
          </div>
          {history.length === 0 ? (
            <p className="history-empty">Stored manual and ESP32 observations will appear here.</p>
          ) : (
            <div className="history-table-wrap">
              <table>
                <thead><tr><th>Time / source</th><th>Sensor class</th><th>Confidence</th><th>pH</th><th>EC</th><th>Visual model</th></tr></thead>
                <tbody>
                  {history.map((item) => (
                    <tr key={item.observation_id}>
                      <td><strong>{formatDate(item.observed_at)}</strong><small>{item.source}</small></td>
                      <td>{item.sensor_prediction || 'Not predicted'}</td>
                      <td>{item.sensor_confidence == null ? '—' : `${Math.round(item.sensor_confidence * 100)}%`}</td>
                      <td>{item.ph ?? '—'}</td>
                      <td>{item.ec ?? '—'}</td>
                      <td>{item.image_prediction && item.image_prediction !== 'unknown' ? item.image_prediction : 'Pending CNN'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>
      </main>
      <footer className="page-footer"><span>Hydroponic Nutrition AI</span><span>Sensor and image evidence remain separate until validated together.</span></footer>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode><App /></React.StrictMode>
);
