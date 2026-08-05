import React, { useMemo } from "react";
import { FlatList, SafeAreaView, StyleSheet, Text, View } from "react-native";
import { useLocalSearchParams } from "expo-router";

import BookCard from "@/components/books/BookCard";
import { DetectedBook } from "@/services/api/types";
import { Colors, Spacing, Typography } from '@/theme';

export default function ResultsScreen() {
  const { books } = useLocalSearchParams<{ books?: string }>();

  const detectedBooks = useMemo<DetectedBook[]>(() => {
    if (!books) {
      return [];
    }

    try {
      return JSON.parse(books);
    } catch {
      return [];
    }
  }, [books]);

  return (
    <SafeAreaView style={styles.container}>
      <Text style={styles.heading}>Detected Books</Text>

      <Text style={styles.subHeading}>{detectedBooks.length} books found</Text>

      <FlatList
        data={detectedBooks}
        keyExtractor={(item) => item.isbn13 || item.spine_idx.toString()}
        renderItem={({ item }) => <BookCard book={item} />}
        contentContainerStyle={styles.list}
        showsVerticalScrollIndicator={false}
        ListEmptyComponent={
          <View style={styles.emptyContainer}>
            <Text style={styles.emptyText}>No books detected.</Text>
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
