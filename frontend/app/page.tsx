import Link from "next/link";

export default function HomePage() {
  const crops = [
    {
      name: "Lettuce",
      latin: "Lactuca sativa",
      icon: "🥬",
      description: "Deficiency detection and environmental tracking.",
      status: "Ready",
    },
    {
      name: "Spinach",
      latin: "Spinacia oleracea",
      icon: "🍃",
      description: "USDA Ag Data Commons telemetry and leaf image analysis.",
      status: "Ready",
    },
    {
      name: "Tomato",
      latin: "Solanum lycopersicum",
      icon: "🍅",
      description: "Vegetative stage sensor bounds and foliar nutrition checks.",
      status: "Ready",
    },
  ];

  return (
    <div className="space-y-10">
      <section className="text-center max-w-3xl mx-auto space-y-4 pt-4 sm:pt-8">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-emerald-100 text-emerald-800 text-xs font-semibold tracking-wide uppercase">
          Advisory Intelligence
        </div>
        <h1 className="text-3xl sm:text-5xl font-extrabold text-stone-900 tracking-tight">
          Precision Hydroponic Nutrition Decision Support
        </h1>
        <p className="text-stone-600 text-base sm:text-lg leading-relaxed">
          Analyze leaf imagery and water nutrient telemetry with computer vision and machine learning models. Receive conservative, evidence-backed advisory recommendations.
        </p>
        <div className="pt-2">
          <Link
            href="/analyze"
            className="inline-flex items-center justify-center rounded-lg bg-emerald-800 px-6 py-3 text-base font-semibold text-white shadow-sm hover:bg-emerald-700 transition-colors"
          >
            Start Analysis →
          </Link>
        </div>
      </section>

      <section className="space-y-4">
        <h2 className="text-xl font-bold text-stone-800">Supported Crops</h2>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-6">
          {crops.map((crop) => (
            <div
              key={crop.name}
              className="rounded-xl border border-stone-200 bg-white p-6 shadow-sm hover:shadow-md transition-shadow flex flex-col justify-between"
            >
              <div>
                <span className="text-4xl block mb-3">{crop.icon}</span>
                <h3 className="text-lg font-bold text-stone-900">{crop.name}</h3>
                <p className="text-xs italic text-stone-500 mb-2">{crop.latin}</p>
                <p className="text-sm text-stone-600">{crop.description}</p>
              </div>
              <div className="mt-5 pt-3 border-t border-stone-100 flex items-center justify-between text-xs font-medium">
                <span className="text-emerald-700 bg-emerald-50 px-2 py-0.5 rounded">
                  {crop.status}
                </span>
                <Link
                  href="/analyze"
                  className="text-stone-600 hover:text-emerald-800 font-semibold"
                >
                  Select Crop →
                </Link>
              </div>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}
