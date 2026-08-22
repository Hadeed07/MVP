import React, { useEffect, useMemo, useRef, useState } from "react";
import {
  Animated,
  Image,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  View,
  useWindowDimensions,
} from "react-native";
import { useLocalSearchParams } from "expo-router";
import Svg, { Polygon } from "react-native-svg";
import { useSafeAreaInsets } from "react-native-safe-area-context";

import { RecommendedBook, ScanResponse } from "@/services/api/types";
import { Colors, Shadows, Typography } from "@/theme";

const OWLY_IMAGE = require("../../assets/images/owly_1.png");

const UI = {
  ink: Colors.text,

  // Owly primary palette
  primary: "#18344A",
  primaryLight: "#294B64",

  // Existing theme colors
  cream: Colors.background,
  surface: Colors.surface,
  white: "#FFFFFF",

  muted: "#747A7F",
  border: Colors.border,
};

export default function ResultsScreen() {
  const { scanResult, imageUri } = useLocalSearchParams<{
    scanResult?: string;
    imageUri?: string;
  }>();

  const insets = useSafeAreaInsets();
  const { height: screenHeight } = useWindowDimensions();

  const result = useMemo<ScanResponse | null>(() => {
    if (!scanResult) return null;

    try {
      return JSON.parse(scanResult);
    } catch {
      return null;
    }
  }, [scanResult]);

  const recommendations = result?.recommendations ?? [];

  const [selectedBook, setSelectedBook] = useState<RecommendedBook | null>(
    null,
  );

  const animation = useRef(new Animated.Value(0)).current;

  /*
   * The hero image now occupies 75% of the screen height.
   *
   * The actual image uses resizeMode="cover", exactly like
   * the ScanScreen preview. This means:
   *
   * - aspect ratio is preserved
   * - the image fills the entire container
   * - excess parts are cropped
   */
  const fullImageHeight = screenHeight * 0.75;

  const collapsedImageHeight = 235;

  useEffect(() => {
    Animated.spring(animation, {
      toValue: selectedBook ? 1 : 0,
      useNativeDriver: false,
      damping: 18,
      stiffness: 150,
      mass: 0.8,
    }).start();
  }, [selectedBook, animation]);

  const animatedImageHeight = animation.interpolate({
    inputRange: [0, 1],
    outputRange: [fullImageHeight, collapsedImageHeight],
  });

  const handleSelectBook = (book: RecommendedBook) => {
    setSelectedBook(book);
  };

  const handleCloseBook = () => {
    setSelectedBook(null);
  };

  return (
    <View
      style={[
        styles.container,
        {
          paddingTop: insets.top + 14,
          paddingBottom: Math.max(insets.bottom, 12),
        },
      ]}
    >
      {/* ---------------------------------------------------------- */}
      {/* HEADER */}
      {/* ---------------------------------------------------------- */}

      <View style={styles.header}>
        <View style={styles.owlyBadge}>
          <Image source={OWLY_IMAGE} style={styles.owlyImage} />
        </View>

        <View style={styles.headerText}>
          <Text style={styles.heading}>Your Next Reads</Text>

          <Text style={styles.subHeading}>
            {recommendations.length}{" "}
            {recommendations.length === 1 ? "book" : "books"} recommended
          </Text>
        </View>

        <View style={styles.countBadge}>
          <Text style={styles.countText}>{recommendations.length}</Text>
        </View>
      </View>

      {/* ---------------------------------------------------------- */}
      {/* HERO IMAGE */}
      {/* ---------------------------------------------------------- */}

      {imageUri && result ? (
        <Animated.View
          style={[
            styles.imageContainer,
            {
              height: animatedImageHeight,
            },
          ]}
        >
          <Image
            source={{ uri: imageUri }}
            style={styles.image}
            resizeMode="stretch"
          />

          {/* Slight dark tint for polygon contrast */}
          <View style={styles.imageTint} />

          {/* BOOK POLYGON OVERLAY */}
          <View style={styles.overlay}>
            <Svg
              width="100%"
              height="100%"
              viewBox={`0 0 ${result.image_width} ${result.image_height}`}
              preserveAspectRatio="none"
            >
              {recommendations.map((book) => {
                const points = book.corners
                  .map(([x, y]) => `${x},${y}`)
                  .join(" ");

                const isSelected = selectedBook?.id === book.id;

                return (
                  <Polygon
                    key={book.id}
                    points={points}
                    fill={
                      isSelected
                        ? "rgba(38, 126, 103, 0.62)"
                        : "rgba(38, 126, 103, 0.20)"
                    }
                    stroke={isSelected ? "#FFFFFF" : "rgba(255,255,255,0.78)"}
                    strokeWidth={isSelected ? 6 : 3}
                    strokeLinejoin="round"
                    onPress={() => handleSelectBook(book)}
                  />
                );
              })}
            </Svg>
          </View>

          {/* IMAGE LABEL */}
          {!selectedBook && recommendations.length > 0 && (
            <View style={styles.imageInstruction}>
              <View style={styles.tapDot} />

              <Text style={styles.imageInstructionText}>
                Tap a highlighted book
              </Text>
            </View>
          )}

          {/* SELECTED BOOK LABEL */}
          {selectedBook && (
            <View style={styles.selectedBadge}>
              <Text style={styles.selectedBadgeText}>SELECTED</Text>
            </View>
          )}
        </Animated.View>
      ) : (
        <View style={styles.emptyImage}>
          <Text style={styles.emptyText}>No bookshelf image available.</Text>
        </View>
      )}

      {/* ---------------------------------------------------------- */}
      {/* RECOMMENDATION DETAILS */}
      {/* ---------------------------------------------------------- */}

      {selectedBook && (
        <Animated.View style={styles.detailsPanel}>
          <View style={styles.panelAccent} />

          <View style={styles.panelHeader}>
            <View>
              <Text style={styles.panelLabel}>RECOMMENDED FOR YOU</Text>

              <View style={styles.panelTitleRow}>
                <View style={styles.greenDot} />

                <Text style={styles.panelTitle}>A book worth exploring</Text>
              </View>
            </View>

            <Pressable
              onPress={handleCloseBook}
              hitSlop={12}
              style={styles.closeButton}
            >
              <Text style={styles.closeText}>×</Text>
            </Pressable>
          </View>

          <ScrollView
            showsVerticalScrollIndicator={false}
            contentContainerStyle={styles.detailsContent}
          >
            <View style={styles.bookHeader}>
              {selectedBook.thumbnail && (
                <Image
                  source={{
                    uri: selectedBook.thumbnail,
                  }}
                  style={styles.thumbnail}
                  resizeMode="cover"
                />
              )}

              <View style={styles.bookHeading}>
                <Text style={styles.bookTitle}>{selectedBook.title}</Text>

                <Text style={styles.author}>{selectedBook.author}</Text>
              </View>
            </View>

            {selectedBook.description && (
              <View style={styles.descriptionSection}>
                <Text style={styles.descriptionLabel}>About this book</Text>

                <Text style={styles.description}>
                  {selectedBook.description}
                </Text>
              </View>
            )}
          </ScrollView>
        </Animated.View>
      )}

      {/* ---------------------------------------------------------- */}
      {/* EMPTY STATE */}
      {/* ---------------------------------------------------------- */}

      {!selectedBook && recommendations.length === 0 && (
        <View style={styles.noRecommendations}>
          <View style={styles.emptyIcon}>
            <Text style={styles.emptyIconText}>📚</Text>
          </View>

          <Text style={styles.emptyTitle}>No recommendations yet</Text>

          <Text style={styles.emptyText}>Try scanning another bookshelf.</Text>
        </View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: UI.cream,
    paddingHorizontal: 18,
  },

  /* -------------------------------------------------------------- */
  /* HEADER */
  /* -------------------------------------------------------------- */

  header: {
    flexDirection: "row",
    alignItems: "center",
    marginBottom: 18,
  },

  /*
   * PERFECT CIRCLE
   */
  owlyBadge: {
    width: 58,
    height: 58,
    justifyContent: "center",
    alignItems: "center",
    marginRight: 12,
  },

  owlyImage: {
    width: 48,
    height: 48,
    resizeMode: "contain",
  },

  headerText: {
    flex: 1,
  },

  heading: {
    ...Typography.h2,
    color: UI.primary,
    fontWeight: "800",
    letterSpacing: -0.5,
  },

  subHeading: {
    ...Typography.bodySecondary,
    color: UI.muted,
    marginTop: 3,
  },

  /*
   * PERFECT CIRCLE
   */
  countBadge: {
    width: 44,
    height: 44,
    borderRadius: 22,
    backgroundColor: UI.primary,
    justifyContent: "center",
    alignItems: "center",
  },

  countText: {
    color: "#FFFFFF",
    fontSize: 15,
    fontWeight: "800",
  },

  /* -------------------------------------------------------------- */
  /* IMAGE */
  /* -------------------------------------------------------------- */

  imageContainer: {
    width: "100%",
    borderRadius: 28,
    overflow: "hidden",
    backgroundColor: UI.primary,
    borderWidth: 2,
    borderColor: "rgba(23,79,66,0.14)",

    ...Shadows.md,
  },

  /*
   * IMPORTANT:
   *
   * This matches the ScanScreen behavior.
   * The image fills the entire container while maintaining
   * its original aspect ratio.
   */
  image: {
    width: "100%",
    height: "100%",
  },

  imageTint: {
    position: "absolute",
    top: 0,
    right: 0,
    bottom: 0,
    left: 0,
    backgroundColor: "rgba(10,45,37,0.05)",
  },

  overlay: {
    position: "absolute",
    top: 0,
    right: 0,
    bottom: 0,
    left: 0,
  },

  imageInstruction: {
    position: "absolute",
    left: 16,
    bottom: 16,

    flexDirection: "row",
    alignItems: "center",

    backgroundColor: "rgba(247,245,238,0.94)",
    paddingHorizontal: 14,
    paddingVertical: 10,
    borderRadius: 20,

    shadowColor: "#000",
    shadowOpacity: 0.12,
    shadowRadius: 8,
    shadowOffset: {
      width: 0,
      height: 3,
    },
    elevation: 4,
  },

  tapDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: UI.primary,
    marginRight: 8,
  },

  imageInstructionText: {
    color: UI.primary,
    fontSize: 13,
    fontWeight: "700",
  },

  selectedBadge: {
    position: "absolute",
    top: 14,
    right: 14,
    backgroundColor: UI.primary,
    paddingHorizontal: 12,
    paddingVertical: 7,
    borderRadius: 14,
  },

  selectedBadgeText: {
    color: "#FFFFFF",
    fontSize: 10,
    fontWeight: "800",
    letterSpacing: 1,
  },

  /* -------------------------------------------------------------- */
  /* DETAILS PANEL */
  /* -------------------------------------------------------------- */

  detailsPanel: {
    flex: 1,
    marginTop: 14,

    backgroundColor: UI.white,
    borderRadius: 24,
    overflow: "hidden",

    ...Shadows.md,
  },

  panelAccent: {
    height: 5,
    backgroundColor: UI.primary,
  },

  panelHeader: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",

    paddingHorizontal: 18,
    paddingTop: 16,
    paddingBottom: 10,
  },

  panelLabel: {
    fontSize: 11,
    fontWeight: "800",
    letterSpacing: 1.2,
    color: UI.primary,
  },

  panelTitleRow: {
    flexDirection: "row",
    alignItems: "center",
    marginTop: 5,
  },

  greenDot: {
    width: 7,
    height: 7,
    borderRadius: 4,
    backgroundColor: UI.primary,
    marginRight: 7,
  },

  panelTitle: {
    fontSize: 13,
    fontWeight: "600",
    color: UI.muted,
  },

  closeButton: {
    width: 38,
    height: 38,
    borderRadius: 19,
    backgroundColor: UI.primaryLight,
    justifyContent: "center",
    alignItems: "center",
  },

  closeText: {
    fontSize: 25,
    lineHeight: 28,
    color: "#FFFFFF",
    fontWeight: "400",
  },

  detailsContent: {
    paddingHorizontal: 18,
    paddingBottom: 28,
  },

  bookHeader: {
    flexDirection: "row",
    alignItems: "flex-start",
    marginTop: 6,
  },

  thumbnail: {
    width: 100,
    height: 148,
    borderRadius: 12,
    backgroundColor: UI.primaryLight,
  },

  bookHeading: {
    flex: 1,
    marginLeft: 16,
    paddingTop: 2,
  },

  bookTitle: {
    ...Typography.h2,
    color: UI.primary,
    fontWeight: "800",
  },

  author: {
    ...Typography.bodySecondary,
    color: UI.muted,
    marginTop: 6,
  },

  descriptionSection: {
    marginTop: 22,
    paddingTop: 18,
    borderTopWidth: 1,
    borderTopColor: UI.border,
  },

  descriptionLabel: {
    fontSize: 16,
    fontWeight: "800",
    color: UI.primary,
    marginBottom: 8,
  },

  description: {
    ...Typography.body,
    color: UI.ink,
    lineHeight: 24,
  },

  /* -------------------------------------------------------------- */
  /* EMPTY */
  /* -------------------------------------------------------------- */

  emptyImage: {
    width: "100%",
    height: 360,
    borderRadius: 28,
    backgroundColor: UI.primaryLight,
    justifyContent: "center",
    alignItems: "center",
  },

  noRecommendations: {
    alignItems: "center",
    paddingTop: 45,
  },

  emptyIcon: {
    width: 62,
    height: 62,
    borderRadius: 31,
    backgroundColor: UI.primaryLight,
    justifyContent: "center",
    alignItems: "center",
    marginBottom: 14,
  },

  emptyIconText: {
    fontSize: 27,
  },

  emptyTitle: {
    fontSize: 17,
    fontWeight: "800",
    color: UI.primary,
    marginBottom: 5,
  },

  emptyText: {
    ...Typography.bodySecondary,
    color: UI.muted,
    textAlign: "center",
  },
});
