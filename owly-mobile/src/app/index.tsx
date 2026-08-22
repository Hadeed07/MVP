import React, { useState } from "react";
import {
  Alert,
  Image,
  Modal,
  Pressable,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { useRouter } from "expo-router";

import RecommendationQuery from "@/components/recommendations/RecommendationQuery";
import { useRecommendation } from "@/context/RecommendationContext";

const COLORS = {
  background: "#F8F6F1",
  surface: "#FFFEFB",

  // Taken from the Owly logo
  navy: "#18344A",
  navyLight: "#294B64",

  text: "#18242D",
  muted: "#747A7F",

  border: "#E5E0D7",
  warm: "#E8E0D3",
  white: "#FFFFFF",
};

type ScanSource = "camera" | "gallery";

export default function HomeScreen() {
  const router = useRouter();

  const { recommendationQuery, setRecommendationQuery } = useRecommendation();

  const [queryModalVisible, setQueryModalVisible] = useState(false);

  const openScan = (source: ScanSource) => {
    if (!recommendationQuery.trim()) {
      setQueryModalVisible(true);
      return;
    }

    router.push({
      pathname: "/scan",
      params: { source },
    });
  };

  const handleQuerySubmit = async (query: string) => {
    try {
      await setRecommendationQuery(query);
      setQueryModalVisible(false);
    } catch {
      Alert.alert(
        "Couldn't Save",
        "Your reading preference could not be saved.",
      );
    }
  };

  const closeQueryModal = () => {
    setQueryModalVisible(false);
  };

  return (
    <SafeAreaView style={styles.safeArea}>
      <View style={styles.container}>
        {/* ───────────────── HEADER ───────────────── */}

        <View style={styles.header}>
          <View style={styles.brand}>
            <Image
              source={require("../../assets/images/owl-icon.png")}
              style={styles.brandLogo}
              resizeMode="contain"
            />

            <Text style={styles.brandName}>Owly</Text>
          </View>
        </View>

        {/* ───────────────── HERO ───────────────── */}

        <View style={styles.hero}>
          <Text style={styles.eyebrow}>YOUR READING COMPANION</Text>

          <Text style={styles.title}>
            What are you{"\n"}
            <Text style={styles.titleAccent}>curious about?</Text>
          </Text>

          <Text style={styles.description}>
            Tell Owly what you're looking for,
            {"\n"}
            or explore your bookshelf.
          </Text>
        </View>

        {/* ───────────────── OWL ILLUSTRATION ───────────────── */}

        <View style={styles.illustrationContainer}>
          <Image
            source={require("../../assets/images/owly_4.png")}
            style={styles.illustration}
            resizeMode="contain"
          />
        </View>

        {/* ───────────────── READING MOOD ───────────────── */}

        <TouchableOpacity
          activeOpacity={0.82}
          style={styles.preference}
          onPress={() => setQueryModalVisible(true)}
        >
          <View style={styles.preferenceIcon}>
            <Text style={styles.preferenceIconText}>✦</Text>
          </View>

          <View style={styles.preferenceContent}>
            <Text style={styles.preferenceLabel}>
              {recommendationQuery ? "YOUR READING MOOD" : "READING MOOD"}
            </Text>

            <Text style={styles.preferenceValue} numberOfLines={1}>
              {recommendationQuery || "What are you in the mood for?"}
            </Text>
          </View>

          <View style={styles.preferenceArrow}>
            <Text style={styles.arrow}>›</Text>
          </View>
        </TouchableOpacity>

        {/* ───────────────── ACTIONS ───────────────── */}

        <View style={styles.actions}>
          <Text style={styles.sectionLabel}>YOUR BOOKSHELF</Text>

          <TouchableOpacity
            activeOpacity={0.86}
            style={styles.primaryButton}
            onPress={() => openScan("camera")}
          >
            <View style={styles.buttonText}>
              <Text style={styles.primaryTitle}>Scan Bookshelf</Text>

              <Text style={styles.primarySubtitle}>Let Owly take a look</Text>
            </View>

            <Text style={styles.primaryArrow}>→</Text>
          </TouchableOpacity>

          {/* <TouchableOpacity
            activeOpacity={0.8}
            style={styles.secondaryButton}
            onPress={() => openScan("gallery")}
          >
            <View style={styles.buttonText}>
              <Text style={styles.secondaryTitle}>Upload Photo</Text>

              <Text style={styles.secondarySubtitle}>
                Choose from your gallery
              </Text>
            </View>

            <Text style={styles.secondaryArrow}>→</Text>
          </TouchableOpacity> */}
        </View>

        {/* ───────────────── FOOTER ───────────────── */}

        <View style={styles.footer}>
          <View style={styles.footerDot} />

          <Text style={styles.footerText}>Private · Secure · AI Powered</Text>
        </View>
      </View>

      {/* ───────────────── READING PREFERENCE ───────────────── */}

      <Modal
        visible={queryModalVisible}
        animationType="slide"
        onRequestClose={closeQueryModal}
      >
        <View style={styles.queryScreen}>
          <Pressable style={styles.queryClose} onPress={closeQueryModal}>
            <Text style={styles.closeText}>×</Text>
          </Pressable>

          <RecommendationQuery
            initialQuery={recommendationQuery}
            onSubmit={handleQuerySubmit}
          />
        </View>
      </Modal>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
    backgroundColor: COLORS.background,
  },

  container: {
    flex: 1,
    paddingHorizontal: 28,
    paddingTop: 26,
    paddingBottom: 16,
    justifyContent: "space-between",
  },

  /* ───────── HEADER ───────── */

  header: {
    flexDirection: "row",
    alignItems: "center",
  },

  brand: {
    flexDirection: "row",
    alignItems: "center",
  },

  brandLogo: {
    width: 42,
    height: 42,
  },

  brandName: {
    marginLeft: 8,
    fontSize: 21,
    fontWeight: "700",
    letterSpacing: -0.5,
    color: COLORS.navy,
  },

  /* ───────── HERO ───────── */

  hero: {
    alignItems: "center",
    marginTop: 6,
  },

  eyebrow: {
    fontSize: 10,
    fontWeight: "700",
    letterSpacing: 2,
    color: COLORS.navy,
    opacity: 0.62,
    marginBottom: 10,
  },

  title: {
    fontSize: 36,
    lineHeight: 41,
    fontWeight: "700",
    letterSpacing: -1.1,
    textAlign: "center",
    color: COLORS.text,
  },

  titleAccent: {
    color: "#E87532",
  },

  description: {
    marginTop: 10,
    fontSize: 14,
    lineHeight: 21,
    textAlign: "center",
    color: COLORS.muted,
  },

  /* ───────── ILLUSTRATION ───────── */

  illustrationContainer: {
    width: "100%",
    alignItems: "center",
    justifyContent: "center",
    marginVertical: 2,
  },

  illustration: {
    width: "100%",
    height: 250,
  },

  /* ───────── READING MOOD ───────── */

  preference: {
    width: "100%",
    minHeight: 68,
    paddingHorizontal: 13,
    paddingVertical: 11,
    borderRadius: 20,

    backgroundColor: COLORS.surface,

    borderWidth: 1,
    borderColor: COLORS.border,

    flexDirection: "row",
    alignItems: "center",
  },

  preferenceIcon: {
    width: 43,
    height: 43,
    borderRadius: 22,

    backgroundColor: COLORS.warm,

    alignItems: "center",
    justifyContent: "center",
  },

  preferenceIconText: {
    fontSize: 20,
    color: COLORS.navy,
  },

  preferenceContent: {
    flex: 1,
    marginLeft: 12,
    marginRight: 8,
  },

  preferenceLabel: {
    fontSize: 9,
    fontWeight: "700",
    letterSpacing: 1.6,
    color: COLORS.muted,
    marginBottom: 4,
  },

  preferenceValue: {
    fontSize: 14,
    fontWeight: "600",
    color: COLORS.text,
  },

  preferenceArrow: {
    width: 34,
    height: 34,
    borderRadius: 17,

    backgroundColor: "#F3F1EC",

    alignItems: "center",
    justifyContent: "center",
  },

  arrow: {
    fontSize: 25,
    lineHeight: 26,
    color: COLORS.navy,
    marginTop: -2,
  },

  /* ───────── ACTIONS ───────── */

  actions: {
    width: "100%",
  },

  sectionLabel: {
    fontSize: 10,
    fontWeight: "700",
    letterSpacing: 1.8,
    color: COLORS.muted,
    marginBottom: 8,
  },

  primaryButton: {
    minHeight: 68,
    paddingHorizontal: 18,
    paddingVertical: 12,

    borderRadius: 19,

    backgroundColor: COLORS.navy,

    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
  },

  buttonText: {
    flex: 1,
  },

  primaryTitle: {
    fontSize: 16,
    fontWeight: "700",
    color: COLORS.white,
    marginBottom: 3,
  },

  primarySubtitle: {
    fontSize: 11,
    color: "rgba(255,255,255,0.65)",
  },

  primaryArrow: {
    fontSize: 24,
    color: COLORS.white,
    marginLeft: 12,
  },

  secondaryButton: {
    minHeight: 62,
    marginTop: 9,

    paddingHorizontal: 18,
    paddingVertical: 11,

    borderRadius: 19,

    backgroundColor: COLORS.surface,

    borderWidth: 1.3,
    borderColor: COLORS.navy,

    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
  },

  secondaryTitle: {
    fontSize: 15,
    fontWeight: "700",
    color: COLORS.navy,
    marginBottom: 3,
  },

  secondarySubtitle: {
    fontSize: 11,
    color: COLORS.muted,
  },

  secondaryArrow: {
    fontSize: 23,
    color: COLORS.navy,
    marginLeft: 12,
  },

  /* ───────── FOOTER ───────── */

  footer: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
  },

  footerDot: {
    width: 5,
    height: 5,
    borderRadius: 3,
    backgroundColor: COLORS.navy,
    opacity: 0.45,
    marginRight: 7,
  },

  footerText: {
    fontSize: 10,
    color: COLORS.muted,
  },

  /* ───────── QUERY ───────── */

  queryScreen: {
    flex: 1,
    backgroundColor: COLORS.background,
    paddingTop: 16,
  },

  queryClose: {
    position: "absolute",
    right: 18,
    top: 12,
    zIndex: 20,

    width: 40,
    height: 40,

    alignItems: "center",
    justifyContent: "center",
  },

  closeText: {
    fontSize: 32,
    lineHeight: 32,
    color: COLORS.navy,
    fontWeight: "300",
  },
});
