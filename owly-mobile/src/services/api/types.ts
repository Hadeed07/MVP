export type Corner = [number, number];

export interface SpineResult {
  id: string;
  text: string;
  corners: Corner[];
}

export interface ScanResponse {
  scan_id: string;
  spines: SpineResult[];
}