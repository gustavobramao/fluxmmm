import type { Metadata } from "next";
import { MmmWorkbench } from "./workbench";

export const metadata: Metadata = {
  title: { absolute: "Flux MMM — Open measurement workspace" },
  description:
    "Validate marketing data, explore model readiness, and estimate calibrated channel ROI.",
};

export default async function Home({
  searchParams,
}: {
  searchParams: Promise<{ demo?: string }>;
}) {
  const { demo } = await searchParams;
  return <MmmWorkbench demoMode={demo === "1"} />;
}
