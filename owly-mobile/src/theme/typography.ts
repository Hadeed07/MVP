import { TextStyle } from 'react-native';

import { Colors } from './colors';

export const Typography: Record<string, TextStyle> = {
  h1: {
    fontSize: 32,
    fontWeight: '700',
    color: Colors.text,
  },

  h2: {
    fontSize: 28,
    fontWeight: '700',
    color: Colors.text,
  },

  h3: {
    fontSize: 22,
    fontWeight: '600',
    color: Colors.text,
  },

  title: {
    fontSize: 18,
    fontWeight: '600',
    color: Colors.text,
  },

  body: {
    fontSize: 16,
    color: Colors.text,
  },

  bodySecondary: {
    fontSize: 15,
    color: Colors.textSecondary,
  },

  caption: {
    fontSize: 13,
    color: Colors.textSecondary,
  },

  button: {
    fontSize: 16,
    fontWeight: '600',
    color: Colors.surface,
  },
};