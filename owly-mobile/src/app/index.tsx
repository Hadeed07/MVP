import React from "react";
import {Alert, Image, StyleSheet, Text, TouchableOpacity, View,} from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { useRouter } from "expo-router";

import { pickImage } from "@/utils/imagePicker";
import { useScanBookshelf } from "@/hooks/useScanBookshelf";
import { Colors, Spacing, Typography } from "@/theme";

export default function HomeScreen() {
  const router = useRouter();

  const { scan, loading } = useScanBookshelf();

  const handleUploadPhoto = async () => {
    try {
      const imageUri = await pickImage();

      if (!imageUri) {
        return;
      }

      const books = await scan(imageUri);

      router.push({
        pathname: "/results",
        params: {
          books: JSON.stringify(books),
        },
      });
    } catch (error) {
      Alert.alert(
        "Scan Failed",
        error instanceof Error ? error.message : "Something went wrong.",
      );
    }
  };

  return (
    <SafeAreaView style={styles.safeArea}>
      <View style={styles.hero}>
        <Image
          source={require("../../assets/images/owl-icon.png")}
          style={styles.logo}
          resizeMode="contain"
        />

        <Text style={styles.title}>Owly</Text>

        <Text style={styles.tagline}>Your Personal Bookshelf Companion</Text>

        <Text style={styles.description}>
          Scan your bookshelf to discover, organize and rediscover your books.
        </Text>
      </View>

      <View style={styles.actions}>
        <TouchableOpacity
          style={[styles.secondaryButton, loading && styles.disabledButton]}
          disabled={loading}
          onPress={() =>
            router.push({
              pathname: "/scan",
              params: {
                source: "gallery",
              },
            })
          }
        >
          <Text style={styles.secondaryButtonText}>
            {loading ? "Scanning..." : "Upload Photo"}
          </Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={styles.primaryButton}
          onPress={() => router.push("/scan")}
        >
          <Text style={styles.primaryButtonText}>Scan Bookshelf</Text>
        </TouchableOpacity>
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
    backgroundColor: Colors.background,
    justifyContent: "space-between",
    paddingHorizontal: Spacing["2xl"],
    paddingBottom: Spacing["2xl"],
  },

  hero: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
    gap: Spacing.md,
  },

  logo: {
    width: 140,
    height: 140,
  },

  title: {
    ...Typography.h1,
  },

  tagline: {
    ...Typography.body,
    textAlign: "center",
  },

  description: {
    ...Typography.caption,
    textAlign: "center",
    paddingHorizontal: Spacing.lg,
  },

  actions: {
    gap: Spacing.md,
  },

  secondaryButton: {
    borderWidth: 1.5,
    borderColor: Colors.primary,
    borderRadius: 12,
    paddingVertical: 14,
    alignItems: "center",
  },

  secondaryButtonText: {
    ...Typography.button,
    color: Colors.primary,
  },

  primaryButton: {
    backgroundColor: Colors.primary,
    borderRadius: 12,
    paddingVertical: 14,
    alignItems: "center",
  },

  primaryButtonText: {
    ...Typography.button,
  },

  disabledButton: {
    opacity: 0.6,
  },
});
