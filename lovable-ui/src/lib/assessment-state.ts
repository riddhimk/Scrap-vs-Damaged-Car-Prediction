export type AssessmentResult = {
  classification: string;
  confidenceScore: string;
  confidencePercent: number;
  breakdown: [string, string][];
  summary: string;
  filename?: string;
};

export type AssessmentDraft = {
  imageUrl: string;
  fileName: string;
  file?: File;
  result?: AssessmentResult;
};

let draft: AssessmentDraft | undefined;

export function setAssessmentDraft(next: AssessmentDraft) {
  if (draft?.imageUrl && draft.imageUrl.startsWith("blob:") && draft.imageUrl !== next.imageUrl) {
    URL.revokeObjectURL(draft.imageUrl);
  }
  draft = next;
}

export function setAssessmentResult(res: AssessmentResult) {
  if (draft) {
    draft.result = res;
  }
}

export function getAssessmentDraft() {
  return draft;
}
