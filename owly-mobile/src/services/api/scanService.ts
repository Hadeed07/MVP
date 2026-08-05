import axios from 'axios';
import { apiClient } from './client';
import { API_ENDPOINTS } from './endpoints';
import { ScanResponse } from './types';

interface ScanOptions {
  signal?: AbortSignal;
}

export async function scanBookshelf(
  imageUri: string,
  options?: ScanOptions
): Promise<ScanResponse> {
  const formData = new FormData();

  formData.append('file', {
    uri: imageUri,
    name: 'bookshelf.jpg',
    type: 'image/jpeg',
  } as any);

  try {
    const response = await apiClient.post<ScanResponse>(
      API_ENDPOINTS.SCAN,
      formData,
      {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        signal: options?.signal,
      }
    );

    if (!Array.isArray(response.data)) {
      throw new Error('Malformed server response.');
    }

    return response.data;
  } catch (error) {
    if (axios.isAxiosError(error)) {
      throw new Error(
        error.response?.data?.detail ??
          error.message ??
          'Failed to scan bookshelf.'
      );
    }

    throw error;
  }
}