import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import { ThemeProvider } from "@/context/ThemeContext";
import { ToastProvider } from "@/context/ToastContext";
import AdminShortcut from "@/components/AdminShortcut";

const inter = Inter({
  variable: "--font-inter",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "CardioDetect - AI Heart Disease Detection",
  description: "Advanced AI-powered heart disease detection and 10-year cardiovascular risk prediction",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body className={`${inter.variable} font-sans antialiased`}>
        <ThemeProvider>
          <ToastProvider>
            <AdminShortcut />
            {children}
          </ToastProvider>
        </ThemeProvider>
      </body>
    </html>
  );
}
