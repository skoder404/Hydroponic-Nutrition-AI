import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";
import { MotionProvider } from "@/components/motion";

export const metadata: Metadata = {
  title: "Hydroponic Nutrition AI",
  description:
    "Decision-support system for hydroponic crop nutrition, leaf deficiency classification, and sensor anomaly detection.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="h-full antialiased">
      <body className="min-h-full flex flex-col bg-stone-50 text-stone-900 selection:bg-emerald-200">
        <header className="sticky top-0 z-50 border-b border-stone-200 bg-white/80 backdrop-blur-md">
          <div className="mx-auto flex max-w-7xl items-center justify-between px-4 py-3 sm:px-6 lg:px-8">
            <Link href="/" className="flex items-center gap-2.5 group">
              <span className="flex h-9 w-9 items-center justify-center rounded-lg bg-emerald-800 text-white font-semibold text-lg shadow-sm group-hover:bg-emerald-700 transition-colors">
                🌿
              </span>
              <div>
                <span className="text-base font-bold tracking-tight text-emerald-950 block leading-tight">
                  Hydroponic Nutrition AI
                </span>
                <span className="text-xs text-stone-500 font-medium block">
                  Decision Support System
                </span>
              </div>
            </Link>

            <nav className="flex items-center gap-1 sm:gap-2 text-sm font-medium text-stone-600">
              <Link
                href="/analyze"
                className="rounded-md px-3 py-1.5 hover:bg-stone-100 hover:text-emerald-900 transition-colors"
              >
                Analyze
              </Link>
              <Link
                href="/history"
                className="rounded-md px-3 py-1.5 hover:bg-stone-100 hover:text-emerald-900 transition-colors"
              >
                History
              </Link>
              <Link
                href="/models"
                className="rounded-md px-3 py-1.5 hover:bg-stone-100 hover:text-emerald-900 transition-colors"
              >
                Models & Limits
              </Link>
              <Link
                href="/about"
                className="rounded-md px-3 py-1.5 hover:bg-stone-100 hover:text-emerald-900 transition-colors"
              >
                About
              </Link>
              <Link
                href="/demo"
                className="rounded-md px-3 py-1.5 hover:bg-stone-100 hover:text-emerald-900 transition-colors text-emerald-800 font-semibold"
              >
                Design System
              </Link>
            </nav>
          </div>
        </header>

        <main className="flex-1 mx-auto w-full max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
          <MotionProvider>{children}</MotionProvider>
        </main>

        <footer className="border-t border-stone-200 bg-stone-100/70 text-xs text-stone-500 py-6 mt-auto">
          <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-3">
            <p>
              <strong className="text-stone-700">Notice:</strong> Predicted — not a diagnosis. Advisory decision-support tool for academic & lab research.
            </p>
            <p>Lettuce · Spinach · Tomato | v1.0.0-dev</p>
          </div>
        </footer>
      </body>
    </html>
  );
}
