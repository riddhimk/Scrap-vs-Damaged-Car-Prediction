import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { useRef, useState, type ChangeEvent, type DragEvent } from "react";
import { ImagePlus, Upload, X } from "lucide-react";
import { SiteHeader } from "@/components/brand";
import { Button } from "@/components/ui/button";
import { setAssessmentDraft } from "@/lib/assessment-state";

export const Route = createFileRoute("/upload")({
  head: () => ({ meta: [
    { title: "Upload a Car Photo — GreenFleet" },
    { name: "description", content: "Upload a JPEG or PNG vehicle photo for GreenFleet damage assessment." },
    { property: "og:title", content: "Upload a Car Photo — GreenFleet" },
    { property: "og:description", content: "Start an AI-powered car damage assessment from one photo." },
    { property: "og:type", content: "website" },
    { name: "twitter:card", content: "summary_large_image" },
  ] }),
  component: UploadPage,
});

const allowed = ["image/jpeg", "image/png"];

function UploadPage() {
  const navigate = useNavigate();
  const inputRef = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File>();
  const [preview, setPreview] = useState("");
  const [error, setError] = useState("");

  const selectFile = (next?: File) => {
    if (!next) return;
    if (!allowed.includes(next.type)) { setError("Please choose a JPEG, JPG, or PNG image."); return; }
    if (next.size > 12 * 1024 * 1024) { setError("Please choose an image smaller than 12 MB."); return; }
    if (preview.startsWith("blob:")) URL.revokeObjectURL(preview);
    setFile(next); setPreview(URL.createObjectURL(next)); setError("");
  };

  const onChange = (event: ChangeEvent<HTMLInputElement>) => selectFile(event.target.files?.[0]);
  const onDrop = (event: DragEvent<HTMLDivElement>) => { event.preventDefault(); selectFile(event.dataTransfer.files[0]); };
  const analyze = () => {
    if (!file || !preview) return;
    setAssessmentDraft({ imageUrl: preview, fileName: file.name, file });
    navigate({ to: "/analyzing" });
  };

  return (
    <div className="min-h-screen bg-ink text-paper">
      <SiteHeader />
      <main className="mx-auto max-w-[1200px] px-6 py-14 lg:px-10 lg:py-20">
        <p className="mb-4 flex items-center gap-2 text-xs font-bold uppercase text-lime"><span className="h-px w-6 bg-lime" />New assessment</p>
        <h1 className="max-w-[18ch] text-4xl font-black uppercase leading-none sm:text-5xl">Upload your car photo</h1>
        <p className="mt-4 max-w-xl text-paper/60">Use a clear, well-lit image with the affected area visible. Supported formats: JPEG, JPG, and PNG.</p>

        <div className="mt-10 grid gap-8 lg:grid-cols-[1.3fr_0.9fr]">
          <div onDragOver={(event) => event.preventDefault()} onDrop={onDrop} className="grid min-h-[390px] place-items-center rounded-[16px] border-2 border-dashed border-paper/25 bg-paper/[0.03] p-6 text-center hover:border-lime">
            {preview ? (
              <div className="relative h-full w-full"><img src={preview} alt="Selected car" className="h-full max-h-[440px] w-full rounded-[10px] object-contain" /><button onClick={() => { setFile(undefined); setPreview(""); }} className="absolute right-3 top-3 grid size-10 place-items-center rounded-full bg-ink text-paper" aria-label="Remove selected image"><X size={18} /></button></div>
            ) : (
              <button type="button" onClick={() => inputRef.current?.click()} className="flex flex-col items-center">
                <span className="grid size-16 place-items-center rounded-full bg-lime text-ink"><ImagePlus size={28} /></span>
                <span className="mt-5 text-lg font-bold">Drag a photo here, or browse</span>
                <span className="mt-2 text-sm text-paper/55">JPEG, JPG or PNG · up to 12 MB</span>
              </button>
            )}
            <input ref={inputRef} type="file" accept=".jpeg,.jpg,.png,image/jpeg,image/png" onChange={onChange} className="sr-only" />
          </div>

          <aside className="self-start rounded-[16px] bg-paper/[0.04] p-6 ring-1 ring-paper/10">
            <div className="flex items-center justify-between"><span className="text-sm font-bold uppercase">File details</span><span className={file ? "rounded-full bg-lime/15 px-3 py-1 text-xs font-bold text-lime" : "text-xs text-paper/40"}>{file ? "Ready" : "Waiting"}</span></div>
            <div className="my-7 border-y border-paper/10 py-5 text-sm">
              <div className="flex justify-between gap-4"><span className="text-paper/50">File</span><span className="truncate text-right">{file?.name ?? "No image selected"}</span></div>
              <div className="mt-3 flex justify-between"><span className="text-paper/50">Size</span><span>{file ? `${(file.size / 1024 / 1024).toFixed(2)} MB` : "—"}</span></div>
            </div>
            {error && <p className="mb-4 text-sm text-destructive" role="alert">{error}</p>}
            {!file && <Button onClick={() => inputRef.current?.click()} variant="outline" className="w-full"><Upload size={18} className="mr-2" />Choose image</Button>}
            <Button onClick={analyze} disabled={!file} className="mt-3 w-full">Analyze this car</Button>
            <p className="mt-4 text-center text-xs leading-relaxed text-paper/40">Your photo is used only for this assessment in this frontend demo.</p>
          </aside>
        </div>
      </main>
    </div>
  );
}
