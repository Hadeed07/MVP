import React, {
  createContext,
  ReactNode,
  useContext,
  useEffect,
  useState,
} from "react";
import AsyncStorage from "@react-native-async-storage/async-storage";

import { apiClient } from "@/services/api/client";
import { API_ENDPOINTS } from "@/services/api/endpoints";

const STORAGE_KEY = "@owly/recommendation_query";

interface RecommendationQueryResponse {
  query: string;
}

interface RecommendationContextValue {
  recommendationQuery: string;
  setRecommendationQuery: (query: string) => Promise<void>;
  clearRecommendationQuery: () => Promise<void>;
}

const RecommendationContext = createContext<
  RecommendationContextValue | undefined
>(undefined);

interface RecommendationProviderProps {
  children: ReactNode;
}

export function RecommendationProvider({
  children,
}: RecommendationProviderProps) {
  const [recommendationQuery, setRecommendationQueryState] = useState("");

  useEffect(() => {
    const loadQuery = async () => {
      try {
        const response = await apiClient.get<RecommendationQueryResponse>(
          API_ENDPOINTS.RECOMMENDATION_QUERY,
        );

        const query = response.data.query?.trim() ?? "";

        if (query) {
          setRecommendationQueryState(query);
          await AsyncStorage.setItem(STORAGE_KEY, query);
        }
      } catch (error) {
        console.error(
          "Failed to load recommendation query from backend:",
          error,
        );

        // Fallback to locally cached query
        try {
          const storedQuery = await AsyncStorage.getItem(STORAGE_KEY);

          if (storedQuery) {
            setRecommendationQueryState(storedQuery);
          }
        } catch (storageError) {
          console.error(
            "Failed to load cached recommendation query:",
            storageError,
          );
        }
      }
    };

    loadQuery();
  }, []);

  const setRecommendationQuery = async (query: string) => {
    const trimmedQuery = query.trim();

    if (!trimmedQuery) {
      throw new Error("Recommendation query cannot be empty.");
    }

    try {
      await apiClient.put(API_ENDPOINTS.RECOMMENDATION_QUERY, {
        query: trimmedQuery,
      });

      setRecommendationQueryState(trimmedQuery);
      await AsyncStorage.setItem(STORAGE_KEY, trimmedQuery);
    } catch (error) {
      console.error("Failed to save recommendation query:", error);
      throw error;
    }
  };

  const clearRecommendationQuery = async () => {
    await AsyncStorage.removeItem(STORAGE_KEY);
    setRecommendationQueryState("");
  };

  return (
    <RecommendationContext.Provider
      value={{
        recommendationQuery,
        setRecommendationQuery,
        clearRecommendationQuery,
      }}
    >
      {children}
    </RecommendationContext.Provider>
  );
}

export function useRecommendation() {
  const context = useContext(RecommendationContext);

  if (!context) {
    throw new Error(
      "useRecommendation must be used inside RecommendationProvider",
    );
  }

  return context;
}
