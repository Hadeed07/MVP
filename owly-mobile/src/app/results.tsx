import React, { useMemo } from "react";
import { FlatList, SafeAreaView, StyleSheet, Text, View } from "react-native";
import { useLocalSearchParams } from "expo-router";

import BookCard from "@/components/books/BookCard";
import { ScanResponse, SpineResult } from "@/services/api/types";
import { Colors } from "@/theme";

export default function ResultsScreen() {
  const { scanResult } = useLocalSearchParams<{
    scanResult?: string;
  }>();

  const result = useMemo<ScanResponse | null>(() => {
    if (!scanResult) {
      return null;
    }

    try {
      return JSON.parse(scanResult);
    } catch {
      return null;
    }
  }, [scanResult]);

  const spines: SpineResult[] = result?.spines ?? [];

  return (
    <SafeAreaView style={styles.container}>
      <Text style={styles.heading}>Detected Spines</Text>

      <Text style={styles.subHeading}>
        {spines.length} {spines.length === 1 ? "spine" : "spines"} found
      </Text>

      <FlatList
        data={spines}
        keyExtractor={(item) => item.id}
        renderItem={({ item }) => <BookCard book={item} />}
        contentContainerStyle={styles.list}
        showsVerticalScrollIndicator={false}
        ListEmptyComponent={
          <View style={styles.emptyContainer}>
            <Text style={styles.emptyText}>No spines detected.</Text>
          </View>
        }
      />
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: Colors.background,
  },

  heading: {
    fontSize: 28,
    fontWeight: "700",
    color: Colors.text,
    paddingHorizontal: 20,
    paddingTop: 24,
  },

  subHeading: {
    fontSize: 15,
    color: Colors.textSecondary,
    paddingHorizontal: 20,
    marginTop: 4,
    marginBottom: 16,
  },

  list: {
    paddingHorizontal: 20,
    paddingBottom: 24,
    gap: 16,
  },

  emptyContainer: {
    flex: 1,
    alignItems: "center",
    marginTop: 80,
  },

  emptyText: {
    fontSize: 16,
    color: Colors.textSecondary,
  },
});
