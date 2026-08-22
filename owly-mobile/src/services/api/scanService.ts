import axios from 'axios';
import { apiClient } from './client';
import { API_ENDPOINTS } from './endpoints';
import { ScanResponse } from './types';

interface ScanOptions {
  signal?: AbortSignal;
}

export async function scanBookshelf(
  imageUri: string,
  query: string,
  options?: ScanOptions
): Promise<ScanResponse> {
  const formData = new FormData();

  formData.append('file', {
    uri: imageUri,
    name: 'bookshelf.jpg',
    type: 'image/jpeg',
  } as any);

  formData.append('query', query);

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

    if (!response.data) {
      throw new Error('Malformed server response.');
    }

    return response.data;
  } catch (error) {
    if (axios.isAxiosError(error)) {
      console.error('Scan request failed:', {
        code: error.code,
        message: error.message,
        status: error.response?.status,
        response: error.response?.data,
      });

      throw new Error(
        error.response?.data?.detail ??
          error.message ??
          'Failed to scan bookshelf.'
      );
    }

    throw error;
  }
}