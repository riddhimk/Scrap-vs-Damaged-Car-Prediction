import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { useEffect, useState, useRef } from "react";
import { getAssessmentDraft, setAssessmentResult } from "@/lib/assessment-state";
import mercedes from "@/assets/loading-mercedes.png";
import defender from "@/assets/loading-defender.png";
import golf from "@/assets/loading-golf.png";
import mini from "@/assets/loading-mini.png";

export const Route = createFileRoute("/analyzing")({
  head: () => ({ meta: [
    { title: "Analyzing Vehicle — GreenFleet" },
    { name: "description", content: "GreenFleet is analyzing the uploaded vehicle photo." },
    { property: "og:title", content: "Analyzing Vehicle — GreenFleet" },
    { property: "og:description", content: "Vehicle damage assessment in progress." },
    { property: "og:type", content: "website" },
    { name: "twitter:card", content: "summary_large_image" },
  ] }),
  component: AnalyzingPage,
});

const cars = [
  { src: mercedes, name: "Mercedes S-Class" },
  { src: defender, name: "Land Rover Defender" },
  { src: golf, name: "VW Golf GTI" },
  { src: mini, name: "Mini Cooper Convertible" },
];
const stages = ["Receiving image", "Mapping panels", "Scoring damage", "Writing report"];

function AnalyzingPage() {
  const navigate = useNavigate();
  const [progress, setProgress] = useState(0);
  const [carIndex] = useState(() => Math.floor(Math.random() * cars.length));
  const draft = getAssessmentDraft();
  const navigatedRef = useRef(false);

  useEffect(() => {
    if (!draft) {
      navigate({ to: "/upload", replace: true });
      return;
    }

    let isApiDone = false;

    const goToResults = () => {
      if (navigatedRef.current) return;
      navigatedRef.current = true;
      setProgress(100);
      window.setTimeout(() => {
        navigate({ to: "/results", replace: true });
      }, 300);
    };

    const doAnalysis = async () => {
      try {
        let uploadFile = draft.file;
        if (!uploadFile && draft.imageUrl && draft.imageUrl.startsWith("blob:")) {
          try {
            const blobRes = await fetch(draft.imageUrl);
            const blobData = await blobRes.blob();
            uploadFile = new File([blobData], draft.fileName || "car.jpg", { type: blobData.type || "image/jpeg" });
          } catch (e) {
            console.warn("Could not retrieve blob as file:", e);
          }
        }

        if (uploadFile) {
          const formData = new FormData();
          formData.append("file", uploadFile);

          const res = await fetch("/api/analyze", { method: "POST", body: formData });
          if (res.ok) {
            const data = await res.json();
            if (data && data.classification) {
              setAssessmentResult(data);
              isApiDone = true;
              goToResults();
              return;
            }
          } else {
            console.error("API error response:", await res.text());
          }
        }
      } catch (err) {
        console.error("API request failed:", err);
      }
    };

    doAnalysis();

    const started = Date.now();
    const duration = 4500;
    const timer = window.setInterval(() => {
      if (navigatedRef.current) {
        window.clearInterval(timer);
        return;
      }
      const elapsed = Date.now() - started;
      const target = isApiDone ? 100 : 92;
      const calculated = Math.min(target, Math.round((elapsed / duration) * target));
      setProgress(calculated);

      if (isApiDone && calculated >= 92) {
        window.clearInterval(timer);
        goToResults();
      }
    }, 60);

    return () => {
      window.clearInterval(timer);
    };
  }, [draft, navigate]);

  const car = cars[carIndex] ?? cars[0];
  if (!car) return null;
  const stageIndex = Math.min(3, Math.floor(progress / 25));

  return (
    <main className="grid min-h-screen place-items-center overflow-hidden bg-paper px-6 py-12 text-ink">
      <div className="w-full max-w-[1100px]">
        <div className="mb-10 flex flex-wrap items-end justify-between gap-4">
          <div><p className="mb-3 flex items-center gap-2 text-xs font-bold uppercase text-lime-deep"><span className="h-px w-6 bg-lime-deep" />Inspection lane</p><h1 className="text-4xl font-black uppercase leading-none sm:text-6xl">Analyzing your car</h1><p className="mt-4 text-ink/55">Please keep this window open while we finish every inspection step.</p></div>
          <span className="text-6xl font-black tabular-nums sm:text-8xl">{progress}<span className="text-lime-deep">%</span></span>
        </div>

        <div className="relative overflow-hidden rounded-[16px] bg-ink px-5 py-10 ring-1 ring-ink/10 sm:px-10">
          <div className="absolute inset-x-0 top-0 h-1 bg-lime" />
          <div className="relative h-36">
            <div className="absolute inset-x-0 bottom-8 h-20 rounded-[10px] bg-paper/5" />
            <div className="lane-run absolute inset-x-0 bottom-[66px] h-1" />
            <div className="absolute bottom-9 transition-[left] duration-100 ease-linear" style={{ left: `clamp(0px, calc(${progress}% - ${progress * 1.6}px), calc(100% - 160px))` }}>
              <img src={car.src} alt="Vehicle driving along inspection lane" width={1152} height={576} loading="lazy" className="car-bob h-auto w-40 object-contain" />
            </div>
          </div>
          <div className="flex justify-between text-xs font-bold uppercase text-paper/45"><span>Start</span><span className="text-lime">Your car is being analyzed</span><span>Report</span></div>
        </div>

        <div className="mt-6 grid gap-3 sm:grid-cols-4">
          {stages.map((stage, index) => <div key={stage} className={index <= stageIndex ? "rounded-[12px] bg-lime px-5 py-4 text-ink ring-1 ring-lime" : "rounded-[12px] bg-paper px-5 py-4 text-ink/35 ring-1 ring-ink/10"}><p className="text-xs font-bold uppercase">{index < stageIndex ? "Done" : index === stageIndex ? "Now" : "Next"}</p><p className="mt-1 font-black">{stage}</p></div>)}
        </div>
      </div>
    </main>
  );
}
