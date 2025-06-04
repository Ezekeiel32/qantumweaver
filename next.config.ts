import type { NextConfig } from 'next';

const nextConfig: NextConfig = {
  typescript: {
    ignoreBuildErrors: true,
  },
  eslint: {
    ignoreDuringBuilds: true,
  },
  images: {
    remotePatterns: [
      {
        protocol: 'https',
        hostname: 'placehold.co',
        port: '',
        pathname: '/**',
      },
    ],
  },
  webpack: (config, { isServer }) => {
    if (!isServer) {
      // Helps with some packages that might try to use Node.js modules on client
      config.resolve.fallback = {
        ...config.resolve.fallback, 
        fs: false,
        path: false,
        process: false, // Example, if some lib needs process.env on client
      };
    }
    return config;
  },
};

export default nextConfig;
