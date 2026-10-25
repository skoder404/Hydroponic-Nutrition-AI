export default function AnalyzePage() {
  return (
    <div className="space-y-6">
      <div className="border-b border-stone-200 pb-4">
        <h1 className="text-2xl sm:text-3xl font-bold text-stone-900 tracking-tight">
          Crop Analysis & Diagnostics
        </h1>
        <p className="text-sm text-stone-600 mt-1">
          Provide plant leaf imagery, reservoir sensor measurements, or stream live data from the campus adapter.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="rounded-xl border border-stone-200 bg-white p-6 shadow-sm space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-bold text-stone-900">1. Image Analysis</h2>
            <span className="text-xs bg-stone-100 text-stone-600 px-2 py-0.5 rounded font-medium">
              SAM 2.1 + Vision
            </span>
          </div>
          <div className="border-2 border-dashed border-stone-300 rounded-lg p-8 text-center bg-stone-50/50">
            <span className="text-3xl block mb-2">📷</span>
            <p className="text-sm font-semibold text-stone-700">
              Drag & drop plant image or click to upload
            </p>
            <p className="text-xs text-stone-500 mt-1">
              Supports JPEG / PNG up to 15 MB. Autorotated and segmented per leaf.
            </p>
          </div>
        </div>

        <div className="rounded-xl border border-stone-200 bg-white p-6 shadow-sm space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-bold text-stone-900">2. Sensor Telemetry</h2>
            <span className="text-xs bg-stone-100 text-stone-600 px-2 py-0.5 rounded font-medium">
              ML Classifier
            </span>
          </div>
          <div className="space-y-3 text-sm">
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="block text-xs font-semibold text-stone-600 mb-1">
                  pH Level
                </label>
                <input
                  type="number"
                  step="0.1"
                  placeholder="e.g. 6.2"
                  disabled
                  className="w-full rounded-md border border-stone-300 bg-stone-100 px-3 py-1.5 text-stone-600"
                />
              </div>
              <div>
                <label className="block text-xs font-semibold text-stone-600 mb-1">
                  EC (mS/cm)
                </label>
                <input
                  type="number"
                  step="0.05"
                  placeholder="e.g. 1.8"
                  disabled
                  className="w-full rounded-md border border-stone-300 bg-stone-100 px-3 py-1.5 text-stone-600"
                />
              </div>
            </div>
            <p className="text-xs text-stone-500 italic">
              Input form will be active once telemetry providers are connected in Phase 6.
            </p>
          </div>
        </div>
      </div>

      <div className="rounded-xl border border-dashed border-stone-300 bg-stone-50/80 p-6 text-center text-sm text-stone-600">
        <p className="font-semibold text-stone-700">Analysis Pipeline Standby</p>
        <p className="text-xs text-stone-500 mt-1">
          Predictions are computed separately for vision and sensors with strict evidence separation (no fusion).
        </p>
      </div>
    </div>
  );
}
