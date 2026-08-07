import React from "react";
import { StyleSheet, Text, View } from "react-native";

import { SpineResult } from "@/services/api/types";
import { Colors, Spacing, Typography, Radius, Shadows } from "@/theme";

interface BookCardProps {
  book: SpineResult;
}

export default function BookCard({ book }: BookCardProps) {
  return (
    <View style={styles.container}>
      <Text style={styles.title}>Spine #{book.id}</Text>

      <Text style={styles.ocr}>{book.text}</Text>

      <View style={styles.footer}>
        <Text style={styles.cornerCount}>{book.corners.length} corners</Text>

        <Text style={styles.spine}>ID: {book.id}</Text>
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

  ocr: {
    ...Typography.body,
    marginTop: Spacing.md,
    color: Colors.text,
  },

  footer: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",

    marginTop: Spacing.lg,
  },

  cornerCount: {
    ...Typography.caption,
    color: Colors.primary,
    fontWeight: "600",
  },

  spine: {
    ...Typography.caption,
    color: Colors.textMuted,
  },
});
