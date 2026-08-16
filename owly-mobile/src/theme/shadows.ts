import { ViewStyle } from "react-native";

import { Colors } from "./colors";

export const Shadows: Record<string, ViewStyle> = {
  sm: {
    shadowColor: Colors.shadow,
    shadowOpacity: 0.04,
    shadowRadius: 4,
    shadowOffset: {
      width: 0,
      height: 2,
    },
    elevation: 1,
  },

  md: {
    shadowColor: Colors.shadow,
    shadowOpacity: 0.07,
    shadowRadius: 10,
    shadowOffset: {
      width: 0,
      height: 4,
    },
    elevation: 3,
  },

  lg: {
    shadowColor: Colors.shadow,
    shadowOpacity: 0.1,
    shadowRadius: 16,
    shadowOffset: {
      width: 0,
      height: 8,
    },
    elevation: 6,
  },
};