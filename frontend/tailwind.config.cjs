/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      fontFamily: {
        display: ["Fraunces", "serif"],
        body: ["Manrope", "sans-serif"]
      },
      colors: {
        ink: {
          50: "#f6f7f8",
          100: "#e3e7eb",
          200: "#c9d1da",
          300: "#a7b4c2",
          400: "#8795a5",
          500: "#6c7b8f",
          600: "#566477",
          700: "#424f60",
          800: "#2e3946",
          900: "#1f2733"
        },
        tide: {
          50: "#eef7f6",
          100: "#d6efea",
          200: "#a7dcd3",
          300: "#77c7bb",
          400: "#4db3a4",
          500: "#2f9a8c",
          600: "#217a70",
          700: "#195e56",
          800: "#0f3f3b",
          900: "#082523"
        },
        ember: {
          50: "#fff5ed",
          100: "#ffe6d1",
          200: "#ffc9a0",
          300: "#ffa565",
          400: "#ff7f33",
          500: "#f55f12",
          600: "#cd4507",
          700: "#a53508",
          800: "#7e2a0b",
          900: "#4b1905"
        }
      }
    }
  },
  plugins: []
};
