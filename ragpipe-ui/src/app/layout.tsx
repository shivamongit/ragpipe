import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import Script from "next/script";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "ragpipe — Production RAG Studio",
  description: "Multi-LLM RAG playground: OpenAI, Anthropic, Gemini, Groq, Cohere, Mistral, Ollama. Chat with your documents.",
  icons: { icon: "/favicon.ico" },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} dark h-full antialiased`}
    >
      <body className="h-full bg-background text-foreground">
        <Script id="suppress-ext-errors" strategy="beforeInteractive">
          {`
            // Silence noisy "Cannot redefine property: ethereum" from wallet
            // browser extensions (MetaMask, Coinbase, etc.) that try to inject
            // window.ethereum on every page load. Not relevant to ragpipe.
            try {
              const desc = Object.getOwnPropertyDescriptor(window, 'ethereum');
              if (!desc || desc.configurable !== false) {
                Object.defineProperty(window, 'ethereum', {
                  configurable: true,
                  writable: true,
                  value: window.ethereum
                });
              }
            } catch (_) {}
            window.addEventListener('error', (e) => {
              const m = String(e?.message || '');
              if (m.includes('Cannot redefine property: ethereum') ||
                  m.includes('chrome-extension://')) {
                e.stopImmediatePropagation();
                e.preventDefault();
              }
            }, true);
            window.addEventListener('unhandledrejection', (e) => {
              const m = String(e?.reason?.message || e?.reason || '');
              if (m.includes('Cannot redefine property: ethereum')) {
                e.preventDefault();
              }
            });
          `}
        </Script>
        {children}
      </body>
    </html>
  );
}
