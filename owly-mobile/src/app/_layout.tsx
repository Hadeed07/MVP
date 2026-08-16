import React from "react";
import { Stack } from "expo-router";

import { RecommendationProvider } from "@/context/RecommendationContext";

export default function RootLayout() {
  return (
    <RecommendationProvider>
      <Stack screenOptions={{ headerShown: false }} />
    </RecommendationProvider>
  );
}
