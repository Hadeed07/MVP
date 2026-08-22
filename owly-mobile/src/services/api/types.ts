export type Corner = [number, number];

export interface RecommendedBook {
  id: string;
  title: string;
  author: string;
  isbn13?: string | null;
  description?: string | null;
  thumbnail?: string | null;
  crop_idx: number;
  query_score?: number | null;
  corners: Corner[];
}

export interface ScanResponse {
  scan_id: string;
  image_width: number;
  image_height: number;
  recommendations: RecommendedBook[];
}