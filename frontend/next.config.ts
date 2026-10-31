import type { NextConfig } from "next";

const backendBaseUrl =
  process.env.NEXT_PUBLIC_API_BASE_URL ||
  (process.env.PORT ? `http://127.0.0.1:${process.env.PORT}` : "http://127.0.0.1:8003");

const nextConfig: NextConfig = {
  /* config options here */
  cacheComponents: true,
  partialPrefetching: true,
  allowedDevOrigins: ["localhost:3000", "127.0.0.1:3000", "10.174.6.63:3000"],
  async rewrites() {
    return [
      {
        source: "/api/v1/:path*",
        destination: `${backendBaseUrl}/api/v1/:path*`,
      },
    ];
  },
  turbopack: {
    rules: {
      "*.css": {
        loaders: ["@tailwindcss/turbopack"],
        as: "*.css",
      },
    },
  },
};

export default nextConfig;
