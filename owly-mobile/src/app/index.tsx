import React from "react";
import {
  Alert,
  Image,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { useRouter } from "expo-router";

import { pickImage } from "@/utils/imagePicker";
import { useScanBookshelf } from "@/hooks/useScanBookshelf";
import { Colors, Spacing, Typography, Radius, Shadows } from "@/theme";

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

        <View style={styles.divider}>
          <View style={styles.line} />
          <View style={styles.dot} />
          <View style={styles.line} />
        </View>

        <Text style={styles.tagline}>
          Your Personal Bookshelf Companion
        </Text>

        <Text style={styles.description}>
          Scan your bookshelf to discover, organize and rediscover your books.
        </Text>
      </View>

      <View style={styles.actions}>
        <TouchableOpacity
          style={styles.primaryButton}
          onPress={() => router.push("/scan")}
        >
          <Text style={styles.primaryButtonText}>Scan Bookshelf</Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={[styles.secondaryButton, loading && styles.disabledButton]}
          disabled={loading}
          onPress={handleUploadPhoto}
        >
          <Text style={styles.secondaryButtonText}>
            {loading ? "Scanning..." : "Upload Photo"}
          </Text>
        </TouchableOpacity>
      </View>

      <View style={styles.footer}>
        <Text style={styles.footerText}>Private • Secure • AI Powered</Text>
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
    backgroundColor: "#FFFFFF",
    paddingHorizontal: Spacing["2xl"],
    paddingTop: 40,
    paddingBottom: 28,
    justifyContent: "space-between",
  },

  hero: {
    alignItems: "center",
  },

  logo: {
    width: 200,
    height: 200,
    marginBottom: Spacing.lg,
  },

  title: {
    ...Typography.h1,
    fontSize: 52,
    color: Colors.primary,
    marginBottom: 14,
  },

  divider: {
    flexDirection: "row",
    alignItems: "center",
    marginBottom: 24,
  },

  line: {
    width: 50,
    height: 1,
    backgroundColor: "#D7C8B4",
  },

  dot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: "#D7C8B4",
    marginHorizontal: 10,
  },

  tagline: {
    ...Typography.body,
    fontSize: 30,
    textAlign: "center",
    color: Colors.primary,
    marginBottom: 24,
    lineHeight: 42,
  },

  description: {
    ...Typography.caption,
    textAlign: "center",
    lineHeight: 26,
    paddingHorizontal: 10,
  },

  actions: {
    gap: 18,
  },

  primaryButton: {
    backgroundColor: Colors.primary,
    borderRadius: Radius.lg,
    paddingVertical: 18,
    alignItems: "center",
    ...Shadows.md,
  },

  primaryButtonText: {
    ...Typography.button,
    fontSize: 20,
  },

  secondaryButton: {
    borderWidth: 1.5,
    borderColor: Colors.primary,
    borderRadius: Radius.lg,
    paddingVertical: 18,
    alignItems: "center",
  },

  secondaryButtonText: {
    ...Typography.button,
    fontSize: 20,
    color: Colors.primary,
  },

  disabledButton: {
    opacity: 0.6,
  },

  footer: {
    alignItems: "center",
    marginTop: 24,
  },

  footerText: {
    color: "#7C7C7C",
    fontSize: 16,
    fontWeight: "500",
  },
});