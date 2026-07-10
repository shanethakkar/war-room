import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ["var(--font-sans)", "system-ui", "sans-serif"],
        mono: ["var(--font-mono)", "ui-monospace", "monospace"],
      },
      colors: {
        brand: {
          DEFAULT: "#8251EE",
          hover: "#9366F5",
          light: "#A37EF5",
          subtle: "rgba(130, 81, 238, 0.15)",
        },
        neutral: {
          bg1: "hsl(240, 6%, 9%)",
          bg2: "hsl(240, 5%, 12%)",
          bg3: "hsl(240, 5%, 14%)",
          bg4: "hsl(240, 4%, 18%)",
          bg5: "hsl(240, 4%, 22%)",
          bg6: "hsl(240, 4%, 26%)",
        },
        text: {
          primary: "#FAFAFA",
          secondary: "#A1A1AA",
          muted: "#71717A",
        },
        border: {
          subtle: "hsla(0, 0%, 100%, 0.06)",
          DEFAULT: "hsla(0, 0%, 100%, 0.10)",
          strong: "hsla(0, 0%, 100%, 0.18)",
        },
        pos: {
          qb: "#A78BFA",
          rb: "#34D399",
          wr: "#60A5FA",
          te: "#FBBF24",
        },
        good: "#34D399",
        bad: "#F87171",
      },
      borderRadius: { DEFAULT: "0.5rem", lg: "0.75rem", xl: "1rem" },
      boxShadow: {
        glow: "0 0 24px rgba(130, 81, 238, 0.25)",
      },
    },
  },
  plugins: [],
};

export default config;
