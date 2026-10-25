export default function AboutPage() {
  return (
    <div className="space-y-6 max-w-4xl">
      <div className="border-b border-stone-200 pb-4">
        <h1 className="text-2xl sm:text-3xl font-bold text-stone-900 tracking-tight">
          About Hydroponic Nutrition AI
        </h1>
        <p className="text-sm text-stone-600 mt-1">
          A machine learning decision-support system built for sustainable closed-loop hydroponic cultivation.
        </p>
      </div>

      <div className="space-y-4 text-sm text-stone-700 leading-relaxed">
        <p>
          Hydroponic cultivation requires continuous, precision monitoring of nutrient solutions (electrical conductivity, pH, water temperature, dissolved oxygen) and foliar health. Unbalanced nutrient formulations can cause irreversible chlorosis, tipburn, or yield reduction within hours.
        </p>
        <p>
          This system provides dual-pipeline intelligence:
        </p>
        <ol className="list-decimal list-inside space-y-2 pl-2">
          <li><strong>Computer Vision Pipeline:</strong> Segment individual plant leaves from photographs using Meta SAM 2.1, validate geometric mask integrity, extract GLCM texture and color features, and classify foliar deficiency states with custom CNNs and EfficientNet-B0.</li>
          <li><strong>Sensor Telemetry Pipeline:</strong> Validate multi-parameter water sensor streams against strictly defined agronomic bands for lettuce, spinach, and tomato crops using supervised classical classifiers.</li>
          <li><strong>Advisory Synthesis:</strong> Synthesize conservative recommendations via Google Gemini with automatic, deterministic local rule fallback.</li>
        </ol>
      </div>

      <div className="rounded-xl border border-stone-200 bg-white p-6 shadow-sm space-y-3">
        <h2 className="text-base font-bold text-stone-900">Project Standards & Provenance</h2>
        <p className="text-xs text-stone-600">
          Developed in pairing by Yashwant and Sushil under strict reproducibility, security (ADR 0005: no unauthenticated execution hazards), and educational mastery guidelines. Datasets sourced from Mendeley Data, Kaggle open research, and USDA Ag Data Commons.
        </p>
      </div>
    </div>
  );
}
