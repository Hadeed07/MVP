export interface DetectedBook {
  spine_idx: string;
  ocr_text: string;
  matched_title: string;
  matched_authors: string;
  isbn13: string;
  score: number;
}

export type ScanResponse = DetectedBook[];