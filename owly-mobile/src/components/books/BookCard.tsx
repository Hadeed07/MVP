import React from "react";
import { StyleSheet, Text, View } from "react-native";

import { DetectedBook } from "@/services/api/types";
import { Colors, Spacing, Typography, Radius, Shadows } from "@/theme";

interface BookCardProps {
  book: DetectedBook;
}

export default function BookCard({ book }: BookCardProps) {
  return (
    <View style={styles.container}>
      <Text style={styles.title}>{book.matched_title || "Unknown Title"}</Text>

      <Text style={styles.author}>
        {book.matched_authors || "Unknown Author"}
      </Text>

      <Text style={styles.ocr}>OCR: {book.ocr_text}</Text>

      <View style={styles.footer}>
        <Text style={styles.score}>Score: {book.score.toFixed(1)}%</Text>

        <Text style={styles.spine}>#{book.spine_idx}</Text>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    backgroundColor: Colors.surface,
    borderRadius: Radius.lg,
    padding: Spacing.lg,

    ...Shadows.md,
  },

  title: {
    ...Typography.title,
  },

  author: {
    ...Typography.bodySecondary,
    color: Colors.primary,
    marginTop: Spacing.xs,
  },

  ocr: {
    ...Typography.caption,
    marginTop: Spacing.md,
  },

  footer: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",

    marginTop: Spacing.lg,
  },

  score: {
    ...Typography.caption,
    color: Colors.primary,
    fontWeight: "600",
  },

  spine: {
    ...Typography.caption,
    color: Colors.textMuted,
  },
});
