import React, { useEffect, useMemo, useState } from "react";
import {
  Image,
  KeyboardAvoidingView,
  Platform,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  View,
} from "react-native";
import { MaterialCommunityIcons } from "@expo/vector-icons";

import { Colors, Radius, Shadows, Spacing, Typography } from "@/theme";

type Mood =
  | "Make Me Think"
  | "Look Inward"
  | "Give Me Perspective"
  | "Make Me Feel"
  | "Keep Me Hooked"
  | "Lift Me Up"
  | "Take Me Somewhere Dark"
  | "Mess With My Mind";

type Experience =
  | "Fast & Addictive"
  | "Slow Burn"
  | "Piece Things Together"
  | "Unravel a Mystery"
  | "Sit With an Idea"
  | "Escape Reality"
  | "Live Through Someone Else"
  | "Explore Something New";

interface RecommendationQueryProps {
  initialQuery?: string;
  onSubmit: (query: string) => void;
}

const MOODS: Mood[] = [
  "Make Me Think",
  "Look Inward",
  "Give Me Perspective",
  "Make Me Feel",
  "Keep Me Hooked",
  "Lift Me Up",
  "Take Me Somewhere Dark",
  "Mess With My Mind",
];

const EXPERIENCES: Experience[] = [
  "Fast & Addictive",
  "Slow Burn",
  "Piece Things Together",
  "Unravel a Mystery",
  "Sit With an Idea",
  "Escape Reality",
  "Live Through Someone Else",
  "Explore Something New",
];

const MOOD_ICONS: Record<Mood, keyof typeof MaterialCommunityIcons.glyphMap> = {
  "Make Me Think": "brain",
  "Look Inward": "account-heart-outline",
  "Give Me Perspective": "compass-outline",
  "Make Me Feel": "heart-outline",
  "Keep Me Hooked": "link-variant",
  "Lift Me Up": "weather-sunny",
  "Take Me Somewhere Dark": "moon-waning-crescent",
  "Mess With My Mind": "head-cog-outline",
};

const EXPERIENCE_ICONS: Record<
  Experience,
  keyof typeof MaterialCommunityIcons.glyphMap
> = {
  "Fast & Addictive": "lightning-bolt",
  "Slow Burn": "clock-outline",
  "Piece Things Together": "puzzle-outline",
  "Unravel a Mystery": "magnify",
  "Sit With an Idea": "thought-bubble-outline",
  "Escape Reality": "island",
  "Live Through Someone Else": "account-outline",
  "Explore Something New": "compass-outline",
};

export default function RecommendationQuery({
  initialQuery = "",
  onSubmit,
}: RecommendationQueryProps) {
  const [customQuery, setCustomQuery] = useState(initialQuery);

  const [selectedMood, setSelectedMood] = useState<Mood | null>(null);

  const [selectedExperience, setSelectedExperience] =
    useState<Experience | null>(null);

  useEffect(() => {
    setCustomQuery(initialQuery);
  }, [initialQuery]);

  const selectMood = (mood: Mood) => {
    setSelectedMood((current) => (current === mood ? null : mood));
  };

  const selectExperience = (experience: Experience) => {
    setSelectedExperience((current) =>
      current === experience ? null : experience,
    );
  };

  const query = useMemo(() => {
    const parts: string[] = [];

    const text = customQuery.trim();

    if (text) {
      parts.push(text);
    }

    if (selectedMood) {
      parts.push(
        `I want something that makes me ${selectedMood
          .toLowerCase()
          .replace("make me ", "")}.`,
      );
    }

    if (selectedExperience) {
      parts.push(
        `I want the experience to feel ${selectedExperience.toLowerCase()}.`,
      );
    }

    return parts.join(" ");
  }, [customQuery, selectedMood, selectedExperience]);

  const canSave = customQuery.trim().length > 0;

  return (
    <KeyboardAvoidingView
      style={styles.container}
      behavior={Platform.OS === "ios" ? "padding" : undefined}
    >
      <ScrollView
        showsVerticalScrollIndicator={false}
        keyboardShouldPersistTaps="handled"
        contentContainerStyle={styles.content}
      >
        {/* ───────────────── HEADER ───────────────── */}

        <View style={styles.header}>
          <Image
            source={require("../../../assets/images/owl-icon.png")}
            style={styles.owlIcon}
            resizeMode="contain"
          />

          <Text style={[Typography.eyebrow, styles.eyebrow]}>
            READING PREFERENCE
          </Text>

          <Text style={styles.title}>
            What are you in the <Text style={styles.titleAccent}>mood</Text>{" "}
            for?
          </Text>

          <Text style={styles.description}>
            Tell Owly what you're looking for.
            {"\n"}
            You can be specific—or just give me a feeling.
          </Text>
        </View>

        {/* ───────────────── FREE-FORM QUERY ───────────────── */}

        <View style={styles.queryBox}>
          <TextInput
            value={customQuery}
            onChangeText={setCustomQuery}
            placeholder={
              "Something thoughtful, immersive,\nand a little unsettling..."
            }
            placeholderTextColor={Colors.textMuted}
            multiline
            maxLength={300}
            textAlignVertical="top"
            style={styles.queryInput}
          />

          <View style={styles.queryFooter}>
            <Text style={styles.queryHint}>Describe what you want to read</Text>

            <Text style={styles.counter}>{customQuery.length}/300</Text>
          </View>
        </View>

        {/* ───────────────── MOOD ───────────────── */}

        <View style={styles.section}>
          <View style={styles.sectionHeader}>
            <Text style={styles.sectionTitle}>How should it feel?</Text>

            <Text style={styles.sectionHint}>Pick one</Text>
          </View>

          <View style={styles.options}>
            {MOODS.map((mood) => {
              const selected = selectedMood === mood;

              return (
                <Pressable
                  key={mood}
                  onPress={() => selectMood(mood)}
                  style={({ pressed }) => [
                    styles.option,
                    selected && styles.optionSelected,
                    pressed && styles.optionPressed,
                  ]}
                >
                  <MaterialCommunityIcons
                    name={MOOD_ICONS[mood]}
                    size={15}
                    color={selected ? Colors.surface : Colors.primary}
                  />

                  <Text
                    style={[
                      styles.optionText,
                      selected && styles.optionTextSelected,
                    ]}
                  >
                    {mood}
                  </Text>
                </Pressable>
              );
            })}
          </View>
        </View>

        {/* ───────────────── EXPERIENCE ───────────────── */}

        <View style={styles.section}>
          <View style={styles.sectionHeader}>
            <Text style={styles.sectionTitle}>What kind of experience?</Text>

            <Text style={styles.sectionHint}>Pick one</Text>
          </View>

          <View style={styles.options}>
            {EXPERIENCES.map((experience) => {
              const selected = selectedExperience === experience;

              return (
                <Pressable
                  key={experience}
                  onPress={() => selectExperience(experience)}
                  style={({ pressed }) => [
                    styles.option,
                    selected && styles.optionSelected,
                    pressed && styles.optionPressed,
                  ]}
                >
                  <MaterialCommunityIcons
                    name={EXPERIENCE_ICONS[experience]}
                    size={15}
                    color={selected ? Colors.surface : Colors.primary}
                  />

                  <Text
                    style={[
                      styles.optionText,
                      selected && styles.optionTextSelected,
                    ]}
                  >
                    {experience}
                  </Text>
                </Pressable>
              );
            })}
          </View>
        </View>

        {/* ───────────────── SAVE ───────────────── */}

        <Pressable
          disabled={!canSave}
          onPress={() => onSubmit(query)}
          style={({ pressed }) => [
            styles.saveButton,
            !canSave && styles.saveButtonDisabled,
            pressed && canSave && styles.saveButtonPressed,
          ]}
        >
          <Text style={styles.saveText}>Find My Next Read</Text>

          <MaterialCommunityIcons
            name="arrow-right"
            size={18}
            color={Colors.surface}
          />
        </Pressable>

      </ScrollView>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: Colors.background,
  },

  content: {
    paddingHorizontal: Spacing.lg,
    paddingTop: Spacing.sm,
    paddingBottom: Spacing.xl,
  },

  /* ───────── HEADER ───────── */

  header: {
    alignItems: "center",
    paddingHorizontal: Spacing.sm,
    marginBottom: Spacing.xl,
  },

  owlIcon: {
    width: 48,
    height: 48,
    marginBottom: Spacing.xs,
  },

  eyebrow: {
    marginBottom: Spacing.sm,
    color: Colors.textSecondary,
  },

  title: {
    fontSize: 32,
    lineHeight: 36,
    fontWeight: "700",
    letterSpacing: -0.7,
    textAlign: "center",
    color: Colors.text,
  },

  titleAccent: {
    color: Colors.accent,
  },

  description: {
    marginTop: Spacing.sm,
    fontSize: 13,
    lineHeight: 19,
    fontWeight: "500",
    textAlign: "center",
    color: Colors.textSecondary,
  },

  /* ───────── QUERY BOX ───────── */

  queryBox: {
    minHeight: 100,
    borderRadius: Radius.xl,
    backgroundColor: Colors.surface,
    borderWidth: 1,
    borderColor: Colors.border,
    marginBottom: Spacing.xl,
    overflow: "hidden",
    ...Shadows.sm,
  },

  queryInput: {
    flex: 1,
    minHeight: 72,
    paddingHorizontal: Spacing.md,
    paddingTop: Spacing.md,
    paddingBottom: Spacing.md,

    fontSize: 13,
    lineHeight: 19,
    color: Colors.text,
  },

  queryFooter: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",

    paddingHorizontal: Spacing.md,
    paddingBottom: Spacing.md,
  },

  queryHint: {
    fontSize: 9,
    color: Colors.textMuted,
  },

  counter: {
    fontSize: 9,
    color: Colors.textMuted,
  },

  /* ───────── SECTIONS ───────── */

  section: {
    marginBottom: Spacing.xl,
  },

  sectionHeader: {
    flexDirection: "row",
    alignItems: "baseline",
    justifyContent: "space-between",
    marginBottom: Spacing.md,
  },

  sectionTitle: {
    fontSize: 18,
    lineHeight: 22,
    fontWeight: "700",
    color: Colors.text,
  },

  sectionHint: {
    fontSize: 9,
    fontWeight: "600",
    color: Colors.textSecondary,
  },

  /* ───────── OPTIONS ───────── */

  options: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: Spacing.sm,
  },

  option: {
    minHeight: 40,
    paddingHorizontal: Spacing.md,
    borderRadius: Radius.full,

    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",

    gap: Spacing.xs,

    backgroundColor: Colors.surface,
    borderWidth: 1,
    borderColor: Colors.border,
  },

  optionSelected: {
    backgroundColor: Colors.primary,
    borderColor: Colors.primary,
    ...Shadows.sm,
  },

  optionPressed: {
    transform: [{ scale: 0.97 }],
  },

  optionText: {
    fontSize: 11,
    lineHeight: 14,
    fontWeight: "600",
    color: Colors.text,
  },

  optionTextSelected: {
    color: Colors.surface,
  },

  /* ───────── CTA ───────── */

  saveButton: {
    minHeight: 48,
    borderRadius: Radius.md,

    paddingHorizontal: Spacing.lg,

    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    gap: Spacing.sm,

    backgroundColor: Colors.primary,

    ...Shadows.md,
  },

  saveButtonDisabled: {
    opacity: 0.35,
  },

  saveButtonPressed: {
    transform: [{ scale: 0.985 }],
  },

  saveText: {
    ...Typography.button,
    fontSize: 13,
  },

  /* ───────── SKIP ───────── */

  skipButton: {
    alignItems: "center",
    justifyContent: "center",
    paddingVertical: Spacing.md,
  },

  skipText: {
    fontSize: 10,
    fontWeight: "600",
    color: Colors.textMuted,
  },
});
