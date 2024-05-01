use image::ColorType::{La16, La8, Rgb16, Rgb8, Rgba16, Rgba32F, Rgba8, L16, L8};
use image::DynamicImage;

use crate::ImageMode;

/// Checks whether a `DynamicImage` buffer contains an alpha channel by means of it's color enum
pub fn image_uses_alpha(buffer: &DynamicImage) -> bool {
    matches!(buffer.color(), La8 | Rgba8 | La16 | Rgba16 | Rgba32F)
}

/// Check if the image is single channel (mono)
pub fn image_is_mono(buffer: &DynamicImage) -> bool {
    matches!(buffer.color(), L8 | L16)
}

/// Check if image is two channel mono with alpha.
pub fn image_is_mono_alpha(buffer: &DynamicImage) -> bool {
    matches!(buffer.color(), La8 | La16)
}

/// Check if the image is three channel RGB
pub fn image_is_rgb(buffer: &DynamicImage) -> bool {
    matches!(buffer.color(), Rgb8 | Rgb16)
}

/// Check if the image is four channel RGBA
pub fn image_is_rgba(buffer: &DynamicImage) -> bool {
    matches!(buffer.color(), Rgba8 | Rgba16)
}

/// Determines the bit depth (8, 16 bit) from the `DynamicImage` color enum
pub fn image_bitmode(buffer: &DynamicImage) -> ImageMode {
    match buffer.color() {
        L8 | La8 | Rgb8 | Rgba8 => ImageMode::U8BIT,
        L16 | La16 | Rgb16 | Rgba16 => ImageMode::U16BIT,
        _ => panic!("Unsupported 32-bit image format"),
    }
}
