import { Link, createFileRoute } from "@tanstack/react-router";
import { Brand } from "@/components/brand";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "GreenFleet — AI Car Damage Assessment" },
      { name: "description", content: "Assess vehicle damage from a single photo with GreenFleet." },
      { property: "og:title", content: "GreenFleet — AI Car Damage Assessment" },
      { property: "og:description", content: "Upload a car photo and receive a clear vehicle condition assessment." },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
  component: Index,
});

function Index() {
  return (
    <div className="min-h-screen bg-paper text-ink">
      <header className="relative overflow-hidden bg-ink text-paper">
        <div className="pointer-events-none absolute -right-[10%] -top-24 h-[520px] w-[700px] rotate-[22deg] bg-lime/10 blur-3xl" />
        <div className="relative mx-auto max-w-[1200px] px-6 py-6 lg:px-10">
          <div className="flex items-center justify-between">
            <Brand />
            <nav className="hidden items-center gap-8 text-sm font-medium text-paper/70 md:flex" aria-label="Primary navigation">
              <Link to="/upload" className="hover:text-lime">Assess</Link>
              <a href="#about" className="hover:text-lime">About</a>
              <a href="#process" className="hover:text-lime">How it works</a>
            </nav>
            <Link to="/upload" className="rounded-[9px] bg-lime px-4 py-2 text-sm font-bold text-ink">Upload a car</Link>
          </div>

          <div className="pb-20 pt-16 lg:pb-28 lg:pt-24">
            <p className="mb-6 flex items-center gap-2 text-xs font-bold uppercase text-lime"><span className="h-px w-8 bg-lime" />Instant vehicle condition report</p>
            <h1 className="max-w-[16ch] text-5xl font-black uppercase leading-[0.9] sm:text-7xl lg:text-8xl">Point. Shoot.<br /><span className="text-lime">Know the damage.</span></h1>
            <p className="mt-8 max-w-[46ch] text-lg leading-relaxed text-paper/70 lg:text-xl">Upload one car photo. GreenFleet reads visible dents, scratches, and panel damage, then returns a clear classification, confidence score, and summary report.</p>
            <div className="mt-10 flex flex-wrap gap-4">
              <Link to="/upload" className="rounded-[10px] bg-lime px-6 py-3 font-bold text-ink">Start an assessment</Link>
              <a href="#about" className="rounded-[10px] border border-paper/25 px-6 py-3 font-semibold text-paper hover:border-lime hover:text-lime">About GreenFleet</a>
            </div>
          </div>

          <div id="process" className="grid gap-px overflow-hidden rounded-[12px] bg-paper/15 sm:grid-cols-3">
            {[["01", "Upload", "Add a clear JPEG or PNG."], ["02", "Analyze", "The model inspects visible damage."], ["03", "Review", "Get a verdict and confidence score."]].map(([n, title, copy]) => (
              <div key={n} className="bg-ink px-6 py-5"><p className="text-2xl font-black text-lime">{n}</p><p className="mt-2 font-bold">{title}</p><p className="mt-1 text-sm text-paper/60">{copy}</p></div>
            ))}
          </div>
        </div>
      </header>

      <section id="about" className="mx-auto grid max-w-[1200px] gap-12 px-6 py-20 lg:grid-cols-[0.85fr_1.15fr] lg:px-10 lg:py-28">
        <div><p className="mb-4 flex items-center gap-2 text-xs font-bold uppercase text-lime-deep"><span className="h-px w-6 bg-lime-deep" />About us</p><h2 className="max-w-[18ch] text-4xl font-black uppercase leading-[1.05] lg:text-5xl">Smarter inspection starts with one photo</h2></div>
        <div className="space-y-5 text-lg leading-relaxed text-ink/70">
          <p>GreenFleet is an AI-powered vehicle assessment platform designed to make damage screening fast, consistent, and easy to understand.</p>
          <p>Our vision is to give drivers, fleet teams, and automotive professionals a practical first look at vehicle condition—without the wait or uncertainty of a manual first pass.</p>
          <Link to="/upload" className="inline-flex rounded-[10px] bg-ink px-6 py-3 font-bold text-paper hover:text-lime">Assess your car →</Link>
        </div>
      </section>

      <footer className="bg-ink px-6 py-8 text-paper/55"><div className="mx-auto flex max-w-[1200px] flex-wrap justify-between gap-4 border-t border-paper/15 pt-8"><span className="font-black text-paper">GreenFleet</span><span className="text-sm">AI-powered vehicle condition assessment · © 2026</span></div></footer>
    </div>
  );
}
