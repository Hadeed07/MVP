import { useCallback, useEffect, useRef, useState } from 'react';

import { scanBookshelf } from '@/services/api/scanService';
import { ScanResponse } from '@/services/api/types';

interface UseScanBookshelfResult {
  books: ScanResponse;
  loading: boolean;
  error: string | null;
  scan: (imageUri: string) => Promise<ScanResponse>;
  cancel: () => void;
  reset: () => void;
}

export function useScanBookshelf(): UseScanBookshelfResult {
  const [books, setBooks] = useState<ScanResponse>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const controllerRef = useRef<AbortController | null>(null);

  const cancel = useCallback(() => {
    controllerRef.current?.abort();
    controllerRef.current = null;
    setLoading(false);
  }, []);

  const reset = useCallback(() => {
    setBooks([]);
    setError(null);
    setLoading(false);
  }, []);

  const scan = useCallback(async (imageUri: string) => {
    cancel();

    const controller = new AbortController();
    controllerRef.current = controller;

    setLoading(true);
    setError(null);

    try {
      const result = await scanBookshelf(imageUri, {
        signal: controller.signal,
      });

      setBooks(result);

      return result;
    } catch (err) {
      const message =
        err instanceof Error ? err.message : 'Unexpected error occurred.';

      if (!controller.signal.aborted) {
        setError(message);
      }

      throw err;
    } finally {
      if (!controller.signal.aborted) {
        setLoading(false);
      }
    }
  }, [cancel]);

  useEffect(() => {
    return () => {
      controllerRef.current?.abort();
    };
  }, []);

  return {
    books,
    loading,
    error,
    scan,
    cancel,
    reset,
  };
}