/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./app.js", "./src/**/*.{js,jsx,ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: "#e7edf7",
        inksoft: "#b8c5d9",
        panel: "#151c28",
        panel2: "#1a2332",
        line: "#33455e",
        brand: "#7ca8df",
        emerald: "#67b483",
      },
      boxShadow: {
        velvet: "0 22px 44px rgba(6, 10, 18, 0.42)",
        soft: "0 10px 22px rgba(8, 12, 20, 0.24)",
      },
      keyframes: {
        rise: {
          "0%": { opacity: "0", transform: "translateY(4px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
      },
      animation: {
        rise: "rise 220ms ease-out",
      },
    },
  },
  plugins: [],
};
