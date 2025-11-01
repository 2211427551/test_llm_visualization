import type { Metadata } from "next";
import "./globals.css";
import { ThemeProvider } from "@/contexts/ThemeContext";
import { VisualizationProvider } from "@/contexts/VisualizationContext";
import { AnimationProvider } from "@/contexts/AnimationContext";
import { Toaster } from "@/components/ui/toaster";

export const metadata: Metadata = {
  title: "Transformer Visualization - Interactive Learning Platform",
  description: "Interactive visualization and exploration of Transformer model architecture and computations",
  keywords: ["transformer", "visualization", "machine learning", "attention", "neural networks"],
  authors: [{ name: "Transformer Visualization Team" }],
  openGraph: {
    title: "Transformer Visualization - Interactive Learning Platform",
    description: "Interactive visualization and exploration of Transformer model architecture and computations",
    type: "website",
    locale: "en_US",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=5" />
        <meta name="theme-color" content="#0ea5e9" />
      </head>
      <body className="font-sans antialiased bg-slate-50 dark:bg-slate-900 min-h-screen">
        <ThemeProvider>
          <VisualizationProvider>
            <AnimationProvider>
              <div className="relative min-h-screen">
                {children}
                <Toaster />
              </div>
            </AnimationProvider>
          </VisualizationProvider>
        </ThemeProvider>
      </body>
    </html>
  );
}