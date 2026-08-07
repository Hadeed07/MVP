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
    console.log('📤 Uploading image...');

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

    console.log('✅ Response received');
    console.log('Status:', response.status);
    console.log('Data:', response.data);

    if (!response.data || !Array.isArray(response.data.spines)) {
      throw new Error('Malformed server response.');
    }

    return response.data;
  } catch (error) {
    console.log('❌ FULL ERROR:', error);

    if (axios.isAxiosError(error)) {
      console.log('Axios Code:', error.code);
      console.log('Axios Message:', error.message);
      console.log('Axios Status:', error.response?.status);
      console.log('Axios Response:', error.response?.data);

      throw new Error(
        error.response?.data?.detail ??
          error.message ??
          'Failed to scan bookshelf.'
      );
    }

    throw error;
  }
}