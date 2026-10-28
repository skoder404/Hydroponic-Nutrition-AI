import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";
import { MotionProvider } from "@/components/motion";
import { HealthBanner } from "@/components/ui/HealthBanner";

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
      <body className="min-h-full flex flex-col bg-[var(--bg)] text-[var(--bark)] selection:bg-[var(--sand)]">
        {/* Accessible skip-link for keyboard navigation */}
        <a
          href="#main-content"
          className="sr-only focus:not-sr-only focus:fixed focus:top-3 focus:left-3 focus:z-50 focus:px-4 focus:py-2 focus:bg-[var(--surface)] focus:text-[var(--forest)] focus:border focus:border-[var(--forest)] focus:rounded-md focus:shadow-md focus:ring-2 focus:ring-[var(--ring-color)] font-medium"
        >
          Skip to main content
        </a>

        {/* Top-level health connectivity banner */}
        <HealthBanner />

        <header className="sticky top-0 z-40 border-b border-[var(--border)] bg-[var(--surface)]/90 backdrop-blur-md">
          <div className="mx-auto flex max-w-7xl items-center justify-between px-4 py-3 sm:px-6 lg:px-8">
            <Link
              href="/"
              className="flex items-center gap-2.5 group rounded-md p-1 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--ring-color)]"
            >
              <span className="flex h-9 w-9 items-center justify-center rounded-lg bg-[var(--forest)] text-white font-semibold text-lg shadow-sm group-hover:bg-[var(--moss)] transition-colors">
                🌿
              </span>
              <div>
                <span className="text-base font-bold tracking-tight text-[var(--forest)] block leading-tight font-serif">
                  Hydroponic Nutrition AI
                </span>
                <span className="text-xs text-[var(--stone)] font-medium block">
                  Decision Support System
                </span>
              </div>
            </Link>

            <nav
              aria-label="Main Navigation"
              className="flex items-center gap-1 sm:gap-2 text-sm font-medium text-[var(--stone)]"
            >
              <Link
                href="/analyze"
                className="rounded-md px-3 py-1.5 hover:bg-[var(--sand)] hover:text-[var(--forest)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--ring-color)] transition-colors"
              >
                Analyze
              </Link>
              <Link
                href="/history"
                className="rounded-md px-3 py-1.5 hover:bg-[var(--sand)] hover:text-[var(--forest)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--ring-color)] transition-colors"
              >
                History
              </Link>
              <Link
                href="/models"
                className="rounded-md px-3 py-1.5 hover:bg-[var(--sand)] hover:text-[var(--forest)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--ring-color)] transition-colors"
              >
                Models &amp; Limits
              </Link>
              <Link
                href="/about"
                className="rounded-md px-3 py-1.5 hover:bg-[var(--sand)] hover:text-[var(--forest)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--ring-color)] transition-colors"
              >
                About
              </Link>
              <Link
                href="/demo"
                className="rounded-md px-3 py-1.5 hover:bg-[var(--sand)] hover:text-[var(--forest)] text-[var(--moss)] font-semibold focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--ring-color)] transition-colors"
              >
                Design System
              </Link>
            </nav>
          </div>
        </header>

        <main
          id="main-content"
          tabIndex={-1}
          className="flex-1 mx-auto w-full max-w-7xl px-4 py-8 sm:px-6 lg:px-8 outline-none"
        >
          <MotionProvider>{children}</MotionProvider>
        </main>

        <footer className="border-t border-[var(--border)] bg-[var(--surface)] text-xs text-[var(--stone)] py-6 mt-auto">
          <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-3">
            <p>
              <strong className="text-[var(--forest)]">Notice:</strong> Advisory decision-support system. Model predictions are probabilistic estimates for greenhouse monitoring. Not certified agronomic prescription.
            </p>
            <p>Lettuce · Spinach · Tomato | v1.0.0-dev</p>
          </div>
        </footer>
      </body>
    </html>
  );
}
