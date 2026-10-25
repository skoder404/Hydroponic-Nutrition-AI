import { Suspense } from "react";
import Link from "next/link";

export const instant = false;

export function generateStaticParams() {
  return [
    { id: "rec_spinach_001" },
    { id: "rec_lettuce_002" },
    { id: "rec_tomato_003" },
  ];
}

async function RecordDetailContent({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between border-b border-stone-200 pb-4">
        <div>
          <Link
            href="/history"
            className="text-xs font-semibold text-stone-500 hover:text-emerald-800 mb-1 inline-block"
          >
            ← Back to History
          </Link>
          <h1 className="text-2xl sm:text-3xl font-bold text-stone-900 tracking-tight">
            Record Detail: <span className="font-mono text-emerald-800">{id}</span>
          </h1>
        </div>
        <span className="text-xs bg-emerald-100 text-emerald-800 px-3 py-1 rounded-full font-medium">
          Saved Audit Record
        </span>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="rounded-xl border border-stone-200 bg-white p-6 shadow-sm space-y-4">
          <h2 className="text-base font-bold text-stone-900">Image Classification Evidence</h2>
          <div className="p-4 bg-stone-50 rounded-lg text-xs space-y-2 text-stone-600 font-mono">
            <p><strong>Status:</strong> Segmented with SAM 2.1</p>
            <p><strong>Leaves Detected:</strong> 4</p>
            <p><strong>Per-Leaf Predictions:</strong> Leaf #1: Optimal (96%), Leaf #2: Optimal (94%)</p>
          </div>
        </div>

        <div className="rounded-xl border border-stone-200 bg-white p-6 shadow-sm space-y-4">
          <h2 className="text-base font-bold text-stone-900">Sensor Telemetry Evidence</h2>
          <div className="p-4 bg-stone-50 rounded-lg text-xs space-y-2 text-stone-600 font-mono">
            <p><strong>Crop Target:</strong> Spinach</p>
            <p><strong>pH:</strong> 6.3 (Band: 5.8 - 6.5) [Normal]</p>
            <p><strong>EC:</strong> 1.7 mS/cm (Band: 1.5 - 2.2) [Normal]</p>
          </div>
        </div>
      </div>

      <div className="rounded-xl border border-stone-200 bg-white p-6 shadow-sm space-y-3">
        <h2 className="text-base font-bold text-stone-900">Advisory Recommendation</h2>
        <p className="text-sm text-stone-700 leading-relaxed">
          Nutrient levels and foliar development remain within optimal agronomic parameters. Continue standard recirculation cycle.
        </p>
        <div className="text-xs text-stone-500 pt-2 border-t border-stone-100 flex items-center justify-between">
          <span>Source: Gemini / Fallback RuleProvider</span>
          <span>Schema Verified ✓</span>
        </div>
      </div>
    </div>
  );
}

export default function RecordDetailPage(props: {
  params: Promise<{ id: string }>;
}) {
  return (
    <Suspense
      fallback={
        <div className="p-8 text-center text-sm text-stone-500">
          Loading record details...
        </div>
      }
    >
      <RecordDetailContent params={props.params} />
    </Suspense>
  );
}
