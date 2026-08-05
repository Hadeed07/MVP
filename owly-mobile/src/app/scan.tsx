import React, { useState } from "react";
import {
  ActivityIndicator,
  Alert,
  Image,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { useRouter } from "expo-router";

import { pickImage, takePhoto } from "@/utils/imagePicker";
import { useScanBookshelf } from "@/hooks/useScanBookshelf";
import { Colors, Radius, Shadows, Spacing, Typography } from "@/theme";

export default function ScanScreen() {
  const router = useRouter();

  const [imageUri, setImageUri] = useState<string | null>(null);

  const { scan, loading } = useScanBookshelf();

  const handlePickImage = async () => {
    const uri = await pickImage();

    if (uri) {
      setImageUri(uri);
    }
  };

  const handleTakePhoto = async () => {
    const uri = await takePhoto();

    if (uri) {
      setImageUri(uri);
    }
  };

  const handleScan = async () => {
    if (!imageUri) {
      Alert.alert("No Image", "Please select or capture an image first.");
      return;
    }

    try {
      const books = await scan(imageUri);

      router.replace({
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
    <SafeAreaView style={styles.container}>
      <Text style={styles.title}>Scan Bookshelf</Text>

      <View style={styles.previewContainer}>
        {imageUri ? (
          <Image source={{ uri: imageUri }} style={styles.image} />
        ) : (
          <Text style={styles.placeholder}>No image selected</Text>
        )}
      </View>

      <View style={styles.actions}>
        <TouchableOpacity
          style={styles.secondaryButton}
          onPress={handleTakePhoto}
          disabled={loading}
        >
          <Text style={styles.secondaryButtonText}>Take Photo</Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={styles.secondaryButton}
          onPress={handlePickImage}
          disabled={loading}
        >
          <Text style={styles.secondaryButtonText}>Choose Photo</Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={[
            styles.primaryButton,
            (!imageUri || loading) && styles.disabled,
          ]}
          disabled={!imageUri || loading}
          onPress={handleScan}
        >
          {loading ? (
            <ActivityIndicator color={Colors.surface} />
          ) : (
            <Text style={styles.primaryButtonText}>Scan Bookshelf</Text>
          )}
        </TouchableOpacity>
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: Colors.background,
    padding: Spacing.xl,
  },

  title: {
    ...Typography.h2,
    marginBottom: Spacing.xl,
  },

  previewContainer: {
    flex: 1,
    backgroundColor: Colors.surface,
    borderRadius: Radius.lg,
    justifyContent: "center",
    alignItems: "center",
    overflow: "hidden",

    ...Shadows.md,
  },

  image: {
    width: "100%",
    height: "100%",
  },

  placeholder: {
    ...Typography.bodySecondary,
  },

  actions: {
    gap: Spacing.md,
    marginTop: Spacing.xl,
  },

  secondaryButton: {
    borderWidth: 1,
    borderColor: Colors.primary,
    borderRadius: Radius.md,
    paddingVertical: 14,
    alignItems: "center",
  },

  secondaryButtonText: {
    ...Typography.button,
    color: Colors.primary,
  },

  primaryButton: {
    backgroundColor: Colors.primary,
    borderRadius: Radius.md,
    paddingVertical: 14,
    alignItems: "center",
  },

  primaryButtonText: {
    ...Typography.button,
  },

  disabled: {
    opacity: 0.6,
  },
});
