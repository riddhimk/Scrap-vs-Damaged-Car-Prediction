import { Link } from "@tanstack/react-router";

export function Brand() {
  return (
    <Link to="/" className="flex items-center gap-3" aria-label="GreenFleet home">
      <span className="grid size-9 place-items-center rounded-[8px] bg-lime text-sm font-black text-ink">GF</span>
      <span className="text-lg font-black text-paper">GreenFleet</span>
    </Link>
  );
}

export function SiteHeader() {
  return (
    <header className="border-b border-paper/15 bg-ink text-paper">
      <div className="mx-auto flex max-w-[1200px] items-center justify-between px-6 py-5 lg:px-10">
        <Brand />
        <nav className="flex items-center gap-3 sm:gap-7" aria-label="Primary navigation">
          <Link to="/" className="hidden text-sm text-paper/70 hover:text-lime sm:block">Home</Link>
          <Link to="/upload" className="rounded-[9px] bg-lime px-4 py-2 text-sm font-bold text-ink">Upload a car</Link>
        </nav>
      </div>
    </header>
  );
}