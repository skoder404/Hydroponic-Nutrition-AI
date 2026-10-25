import Link from "next/link";

export default function HistoryPage() {
  const mockRecords = [
    {
      id: "rec_spinach_001",
      date: "2026-10-25 10:15 AM",
      crop: "Spinach",
      source: "Manual Telemetry",
      prediction: "Optimal",
      confidence: "94.2%",
    },
    {
      id: "rec_lettuce_002",
      date: "2026-10-24 04:30 PM",
      crop: "Lettuce",
      source: "Camera Upload",
      prediction: "Nitrogen Deficiency",
      confidence: "88.6%",
    },
    {
      id: "rec_tomato_003",
      date: "2026-10-23 09:00 AM",
      crop: "Tomato",
      source: "Campus Adapter",
      prediction: "Low EC Warning",
      confidence: "91.0%",
    },
  ];

  return (
    <div className="space-y-6">
      <div className="border-b border-stone-200 pb-4">
        <h1 className="text-2xl sm:text-3xl font-bold text-stone-900 tracking-tight">
          Analysis History & Audit Log
        </h1>
        <p className="text-sm text-stone-600 mt-1">
          Historical analysis records stored in PostgreSQL with leaf images, feature vectors, and recommendations.
        </p>
      </div>

      <div className="rounded-xl border border-stone-200 bg-white overflow-hidden shadow-sm">
        <table className="w-full text-left text-sm">
          <thead className="bg-stone-50 border-b border-stone-200 text-xs font-semibold text-stone-600 uppercase tracking-wider">
            <tr>
              <th className="px-6 py-3">Record ID</th>
              <th className="px-6 py-3">Timestamp</th>
              <th className="px-6 py-3">Crop</th>
              <th className="px-6 py-3">Source</th>
              <th className="px-6 py-3">Result</th>
              <th className="px-6 py-3 text-right">Action</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-stone-100">
            {mockRecords.map((rec) => (
              <tr key={rec.id} className="hover:bg-stone-50/60 transition-colors">
                <td className="px-6 py-4 font-mono text-xs text-stone-700">{rec.id}</td>
                <td className="px-6 py-4 text-stone-600">{rec.date}</td>
                <td className="px-6 py-4 font-semibold text-stone-900">{rec.crop}</td>
                <td className="px-6 py-4 text-stone-500 text-xs">{rec.source}</td>
                <td className="px-6 py-4">
                  <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-medium bg-emerald-50 text-emerald-800 border border-emerald-200">
                    {rec.prediction} ({rec.confidence})
                  </span>
                </td>
                <td className="px-6 py-4 text-right">
                  <Link
                    href={`/history/${rec.id}`}
                    className="text-emerald-700 hover:text-emerald-900 font-semibold text-xs"
                  >
                    View Details →
                  </Link>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
