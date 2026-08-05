import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import { headers } from "next/headers";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export async function generateMetadata(): Promise<Metadata> {
  const requestHeaders = await headers();
  const host = requestHeaders.get("host") ?? "localhost:3003";
  const protocol = host.includes("localhost") ? "http" : "https";
  const socialImage = `${protocol}://${host}/og.png`;

  return {
    title: {
      default: "Flux MMM",
      template: "%s · Flux MMM",
    },
    description:
      "An open-source marketing mix modeling workspace for validated data, calibrated ROI, and transparent measurement.",
    icons: {
      icon: "/favicon.svg",
      shortcut: "/favicon.svg",
    },
    openGraph: {
      title: "Flux MMM",
      description: "Open measurement. Grounded ROI.",
      type: "website",
      images: [{ url: socialImage, width: 1727, height: 911, alt: "Flux MMM analytical workspace" }],
    },
    twitter: {
      card: "summary_large_image",
      title: "Flux MMM",
      description: "Open measurement. Grounded ROI.",
      images: [socialImage],
    },
  };
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        {children}
      </body>
    </html>
  );
}
