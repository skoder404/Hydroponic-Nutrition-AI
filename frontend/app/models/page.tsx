export default function ModelsPage() {
  const models = [
    {
      name: "Custom CNN & EfficientNet-B0",
      target: "Foliar Deficiencies",
      framework: "PyTorch / torchvision",
      input: "Isolated per-leaf RGB crops",
      metric: "Top-1 Accuracy & F1-score",
    },
    {
      name: "Classical ML Classifiers",
      target: "Lettuce / Spinach OpenCV + GLCM Features",
      framework: "scikit-learn / scikit-image",
      input: "Extracted color & texture feature vectors",
      metric: "Macro F1 & Calibration Error",
    },
    {
      name: "SAM 2.1 Segmenter",
      target: "Leaf Mask Extraction",
      framework: "Meta Segment Anything 2.1",
      input: "Full plant photography",
      metric: "IoU & Mask Boundary Stability",
    },
    {
      name: "Sensor Condition Classifiers",
      target: "Water EC, pH, Temp Telemetry",
      framework: "Decision Tree / Random Forest / SVM",
      input: "Validated tabular sensor observations",
      metric: "Per-class Precision & Recall",
    },
  ];

  return (
    <div className="space-y-6">
      <div className="border-b border-stone-200 pb-4">
        <h1 className="text-2xl sm:text-3xl font-bold text-stone-900 tracking-tight">
          Models & Architectural Limitations
        </h1>
        <p className="text-sm text-stone-600 mt-1">
          Registered machine learning models, active checkpoints, and explicit agronomic boundary limitations.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {models.map((m) => (
          <div key={m.name} className="rounded-xl border border-stone-200 bg-white p-5 shadow-sm space-y-3">
            <h2 className="text-base font-bold text-stone-900">{m.name}</h2>
            <div className="space-y-1.5 text-xs text-stone-600">
              <p><strong className="text-stone-800">Target:</strong> {m.target}</p>
              <p><strong className="text-stone-800">Framework:</strong> {m.framework}</p>
              <p><strong className="text-stone-800">Input:</strong> {m.input}</p>
              <p><strong className="text-stone-800">Evaluation:</strong> {m.metric}</p>
            </div>
          </div>
        ))}
      </div>

      <div className="rounded-xl border border-amber-200 bg-amber-50/70 p-6 space-y-2 text-xs text-amber-900">
        <h3 className="font-bold text-sm text-amber-950">Strict Limitations & Boundary Rules</h3>
        <ul className="list-disc list-inside space-y-1 text-amber-800">
          <li><strong>No Fusion Model:</strong> Image and sensor data are collected independently; multimodal fusion without paired data is mathematically prohibited (TECH_RULES §15, Overview §19).</li>
          <li><strong>Non-Diagnostic:</strong> Results represent probabilistic statistical correlations, not certified veterinary/pathological diagnoses.</li>
          <li><strong>Rule Fallback:</strong> If Gemini or external API services fail, deterministic agronomic bounds are enforced as default.</li>
        </ul>
      </div>
    </div>
  );
}
