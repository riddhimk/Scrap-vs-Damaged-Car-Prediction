import { Link, createFileRoute, useNavigate } from "@tanstack/react-router";
import { useEffect } from "react";
import { ArrowRight, CheckCircle2 } from "lucide-react";
import { Brand } from "@/components/brand";
import { getAssessmentDraft } from "@/lib/assessment-state";

export const Route = createFileRoute("/results")({
  head: () => ({ meta: [
    { title: "Assessment Results — GreenFleet" },
    { name: "description", content: "Review the GreenFleet vehicle damage classification, confidence, and summary." },
    { property: "og:title", content: "Assessment Results — GreenFleet" },
    { property: "og:description", content: "Vehicle damage classification and confidence report." },
    { property: "og:type", content: "website" },
    { name: "twitter:card", content: "summary_large_image" },
  ] }),
  component: ResultsPage,
});

function ResultsPage() {
  const navigate = useNavigate();
  const draft = getAssessmentDraft();
  useEffect(() => {
    if (!draft || !draft.result) {
      navigate({ to: "/upload", replace: true });
    }
  }, [draft, navigate]);

  if (!draft || !draft.result) return null;
  const result = draft.result;

  return (
    <div className="min-h-screen bg-ink text-paper">
      <header className="border-b border-paper/15"><div className="mx-auto flex max-w-[1200px] items-center justify-between px-6 py-5 lg:px-10"><Brand /><Link to="/upload" className="text-sm font-bold text-lime">New assessment <ArrowRight className="ml-1 inline" size={16} /></Link></div></header>
      <main className="mx-auto max-w-[1200px] px-6 py-14 lg:px-10 lg:py-20">
        <div className="mb-10 flex flex-wrap items-end justify-between gap-4">
          <div><p className="mb-3 flex items-center gap-2 text-xs font-bold uppercase text-lime"><CheckCircle2 size={16} />Assessment complete</p><h1 className="text-4xl font-black uppercase leading-none sm:text-5xl">Vehicle condition report</h1></div>
          <span className="rounded-full bg-lime/15 px-4 py-2 text-sm font-bold text-lime">Analysis complete</span>
        </div>

        <div className="grid gap-8 lg:grid-cols-[1.1fr_0.9fr]">
          <figure className="rounded-[16px] bg-paper/[0.04] p-4 ring-1 ring-paper/10">
            <img src={draft.imageUrl} alt="Uploaded vehicle assessed by GreenFleet" className="aspect-[4/3] w-full rounded-[10px] bg-ink object-contain" />
            <figcaption className="mt-4 flex flex-wrap justify-between gap-2 px-1 text-sm text-paper/55"><span className="truncate">{draft.fileName}</span><span className="font-bold text-lime">Image analyzed</span></figcaption>
          </figure>

          <div className="space-y-4">
            <div className="rounded-[14px] bg-lime px-6 py-6 text-ink ring-1 ring-lime"><p className="text-xs font-bold uppercase">Classification</p><p className="mt-2 text-4xl font-black uppercase leading-none">{result.classification}</p></div>
            <div className="rounded-[14px] bg-paper/[0.05] px-6 py-6 ring-1 ring-paper/10"><p className="text-xs font-bold uppercase text-paper/55">Confidence score</p><p className="mt-2 text-4xl font-black tabular-nums text-lime">{result.confidenceScore}</p><div className="mt-4 h-2 overflow-hidden rounded-full bg-paper/10"><div className="h-full rounded-full bg-lime" style={{ width: `${Math.min(100, Math.max(0, result.confidencePercent))}%` }} /></div></div>
            <div className="grid grid-cols-3 gap-3">
              {result.breakdown.map(([label, score]) => {
                const isSelected = label.toLowerCase() === result.classification.toLowerCase();
                return (
                  <div key={label} className={isSelected ? "rounded-[12px] bg-lime/15 p-4 text-center ring-1 ring-lime/40" : "rounded-[12px] bg-paper/[0.05] p-4 text-center ring-1 ring-paper/10"}>
                    <p className={isSelected ? "text-xs font-bold uppercase text-lime" : "text-xs font-bold uppercase text-paper/45"}>{label}</p>
                    <p className={isSelected ? "mt-2 text-xl font-black text-lime" : "mt-2 text-xl font-black text-paper/55"}>{score}</p>
                  </div>
                );
              })}
            </div>
          </div>
        </div>

        <section className="mt-8 rounded-[16px] bg-paper/[0.04] p-7 ring-1 ring-paper/10">
          <p className="text-xs font-bold uppercase text-lime">Summary report</p><h2 className="mt-2 text-2xl font-black uppercase">Condition verdict</h2>
          <p className="mt-4 max-w-[70ch] text-lg leading-relaxed text-paper/70">{result.summary}</p>
          <div className="mt-7"><Link to="/upload" className="inline-flex rounded-[10px] bg-lime px-5 py-3 text-sm font-bold text-ink">Assess another car</Link></div>
        </section>
      </main>
    </div>
  );
}
