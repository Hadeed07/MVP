import React, { useEffect, useState } from "react";
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
import Svg, { Circle } from "react-native-svg";

import { pickImage, takePhoto } from "@/utils/imagePicker";
import { useScanBookshelf } from "@/hooks/useScanBookshelf";
import { useRecommendation } from "@/context/RecommendationContext";
import { Colors, Radius, Shadows, Spacing, Typography } from "@/theme";

const RING_SIZE = 170;
const STROKE_WIDTH = 8;
const RADIUS = (RING_SIZE - STROKE_WIDTH) / 2;
const CIRCUMFERENCE = 2 * Math.PI * RADIUS;

export default function ScanScreen() {
  const router = useRouter();

  const [imageUri, setImageUri] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);

  const { scan, loading } = useScanBookshelf();
  const { recommendationQuery } = useRecommendation();

  useEffect(() => {
    if (!loading) {
      setProgress(0);
      return;
    }

    setProgress(5);

    const interval = setInterval(() => {
      setProgress((current) => {
        if (current >= 90) return current;

        const increment = current < 30 ? 4 : current < 60 ? 2 : 1;

        return Math.min(current + increment, 90);
      });
    }, 300);

    return () => clearInterval(interval);
  }, [loading]);

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
      const result = await scan(imageUri, recommendationQuery);

      setProgress(100);

      router.replace({
        pathname: "/results",
        params: {
          scanResult: JSON.stringify(result),
          imageUri,
        },
      });
    } catch (error) {
      console.log("HANDLE SCAN ERROR:", error);

      Alert.alert(
        "Scan Failed",
        error instanceof Error ? error.message : "Something went wrong.",
      );
    }
  };

  const progressOffset = CIRCUMFERENCE - (progress / 100) * CIRCUMFERENCE;

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.header}>
        <Image
          source={require("@/assets/images/owly_2.png")}
          style={styles.headerImage}
          resizeMode="contain"
        />

        <Text style={styles.title}>Scan Bookshelf</Text>
      </View>

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
          <Text style={styles.primaryButtonText}>Scan Bookshelf</Text>
        </TouchableOpacity>
      </View>

      {loading && (
        <View style={styles.loadingOverlay}>
          <View style={styles.loadingContent}>
            <View style={styles.ringContainer}>
              <Svg
                width={RING_SIZE}
                height={RING_SIZE}
                viewBox={`0 0 ${RING_SIZE} ${RING_SIZE}`}
                style={styles.ring}
              >
                <Circle
                  cx={RING_SIZE / 2}
                  cy={RING_SIZE / 2}
                  r={RADIUS}
                  stroke={Colors.surface}
                  strokeWidth={STROKE_WIDTH}
                  fill="none"
                />

                <Circle
                  cx={RING_SIZE / 2}
                  cy={RING_SIZE / 2}
                  r={RADIUS}
                  stroke={Colors.primary}
                  strokeWidth={STROKE_WIDTH}
                  fill="none"
                  strokeLinecap="round"
                  strokeDasharray={`${CIRCUMFERENCE} ${CIRCUMFERENCE}`}
                  strokeDashoffset={progressOffset}
                  rotation="-90"
                  origin={`${RING_SIZE / 2}, ${RING_SIZE / 2}`}
                />
              </Svg>

              <Image
                source={require("@/assets/images/owly_2.png")}
                style={styles.loadingIcon}
                resizeMode="contain"
              />
            </View>

            <Text style={styles.progressText}>{progress}%</Text>

            <Text style={styles.loadingTitle}>Reading your bookshelf...</Text>

            <Text style={styles.loadingSubtitle}>
              Owly is discovering what’s waiting on your shelves.
            </Text>
          </View>
        </View>
      )}
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: Colors.background,
    padding: Spacing.xl,
  },

  header: {
    flexDirection: "row",
    alignItems: "center",
    marginBottom: Spacing.xl,
  },

  headerImage: {
    width: 52,
    height: 52,
    marginRight: Spacing.md,
  },

  title: {
    ...Typography.h2,
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

  loadingOverlay: {
    position: "absolute",
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: Colors.background,
    justifyContent: "center",
    alignItems: "center",
    padding: Spacing.xl,
  },

  loadingContent: {
    alignItems: "center",
    width: "100%",
    maxWidth: 340,
  },

  ringContainer: {
    width: RING_SIZE,
    height: RING_SIZE,
    justifyContent: "center",
    alignItems: "center",
  },

  ring: {
    position: "absolute",
  },

  loadingIcon: {
    width: 115,
    height: 115,
  },

  progressText: {
    ...Typography.h2,
    marginTop: Spacing.md,
  },

  loadingTitle: {
    ...Typography.h2,
    marginTop: Spacing.md,
    textAlign: "center",
  },

  loadingSubtitle: {
    ...Typography.bodySecondary,
    marginTop: Spacing.sm,
    textAlign: "center",
    lineHeight: 22,
  },
});
