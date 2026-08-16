import React, {
  createContext,
  ReactNode,
  useContext,
  useEffect,
  useState,
} from "react";
import AsyncStorage from "@react-native-async-storage/async-storage";

const STORAGE_KEY = "@owly/recommendation_query";

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
        const storedQuery = await AsyncStorage.getItem(STORAGE_KEY);

        if (storedQuery) {
          setRecommendationQueryState(storedQuery);
        }
      } catch (error) {
        console.error("Failed to load recommendation query:", error);
      }
    };

    loadQuery();
  }, []);

  const setRecommendationQuery = async (query: string) => {
    const trimmedQuery = query.trim();

    try {
      if (!trimmedQuery) {
        await AsyncStorage.removeItem(STORAGE_KEY);
        setRecommendationQueryState("");
        return;
      }

      await AsyncStorage.setItem(STORAGE_KEY, trimmedQuery);
      setRecommendationQueryState(trimmedQuery);
    } catch (error) {
      console.error("Failed to save recommendation query:", error);
      throw error;
    }
  };

  const clearRecommendationQuery = async () => {
    try {
      await AsyncStorage.removeItem(STORAGE_KEY);
      setRecommendationQueryState("");
    } catch (error) {
      console.error("Failed to clear recommendation query:", error);
      throw error;
    }
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
