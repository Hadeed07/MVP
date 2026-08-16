import { TextStyle } from "react-native";

import { Colors } from "./colors";

export const Typography: Record<string, TextStyle> = {
  // Large page heading
  h1: {
    fontSize: 32,
    lineHeight: 36,
    fontWeight: "700",
    color: Colors.text,
  },

  // Section heading
  h2: {
    fontSize: 24,
    lineHeight: 29,
    fontWeight: "700",
    color: Colors.text,
  },

  // Smaller section heading
  h3: {
    fontSize: 18,
    lineHeight: 23,
    fontWeight: "700",
    color: Colors.text,
  },

  // Compact title
  title: {
    fontSize: 17,
    lineHeight: 22,
    fontWeight: "600",
    color: Colors.text,
  },

  // Main body text
  body: {
    fontSize: 15,
    lineHeight: 21,
    color: Colors.text,
  },

  // Supporting body text
  bodySecondary: {
    fontSize: 13,
    lineHeight: 18,
    color: Colors.textSecondary,
  },

  // Small supporting text
  caption: {
    fontSize: 11,
    lineHeight: 15,
    color: Colors.textSecondary,
  },

  // Small uppercase contextual labels
  eyebrow: {
    fontSize: 9,
    lineHeight: 12,
    fontWeight: "800",
    letterSpacing: 2,
    color: Colors.textSecondary,
  },

  // Buttons
  button: {
    fontSize: 14,
    lineHeight: 18,
    fontWeight: "700",
    color: Colors.surface,
  },

  // Small interactive labels / pills
  label: {
    fontSize: 11,
    lineHeight: 14,
    fontWeight: "600",
    color: Colors.text,
  },
};