import * as ImagePicker from 'expo-image-picker';

async function ensureCameraPermission(): Promise<void> {
  const { granted } = await ImagePicker.requestCameraPermissionsAsync();

  if (!granted) {
    throw new Error('Camera permission was denied.');
  }
}

async function ensureMediaPermission(): Promise<void> {
  const { granted } = await ImagePicker.requestMediaLibraryPermissionsAsync();

  if (!granted) {
    throw new Error('Media library permission was denied.');
  }
}

export async function pickImage(): Promise<string | null> {
  await ensureMediaPermission();

  const result = await ImagePicker.launchImageLibraryAsync({
    mediaTypes: ['images'],
    allowsEditing: false,
    quality: 1,
    exif: false,
  });

  if (result.canceled || result.assets.length === 0) {
    return null;
  }

  return result.assets[0].uri;
}

export async function takePhoto(): Promise<string | null> {
  await ensureCameraPermission();

  const result = await ImagePicker.launchCameraAsync({
    allowsEditing: false,
    quality: 1,
    exif: false,
  });

  if (result.canceled || result.assets.length === 0) {
    return null;
  }

  return result.assets[0].uri;
}