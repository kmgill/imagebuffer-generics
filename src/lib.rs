use std::ffi::OsStr;
use std::path::Path;

use anyhow::{Error, Result};

use buffer::Buffer;
use imagebuffer::ImageBuffer;

pub mod buffer;
pub mod image;
pub mod imagebuffer;
mod util;

/// Represents a valid number to be used as a generic constraint in `Buffer` and `ImageBuffer`.
pub trait Num:
    num_traits::Num + num_traits::ToPrimitive + num_traits::FromPrimitive + Copy + num_traits::Bounded
{
}

/// Represents a minimum/maximum range of values.
pub struct Range<N: Num> {
    pub min: N,
    pub max: N,
}

/// Represents supported image output formats.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Default)]
pub enum OutputFormat {
    PNG,
    JPEG,
    #[default]
    TIFF,
    // DNG,
}

impl OutputFormat {
    /// Convenience function to convert a string image extension to the appropriate image
    /// format enum value
    pub fn from_string(s: &str) -> Result<OutputFormat> {
        match s.to_uppercase().as_str() {
            "PNG" => Ok(OutputFormat::PNG),
            "JPG" | "JPEG" => Ok(OutputFormat::JPEG),
            "TIF" | "TIFF" => Ok(OutputFormat::TIFF),
            // "DNG" => Ok(OutputFormat::DNG),
            _ => Err(Error::msg(format!(
                "Invalid output format specified: {}",
                s
            ))),
        }
    }

    /// Convenience function to determine the format enum value from the extension on a
    /// given filename.
    pub fn from_filename(s: &str) -> Result<OutputFormat> {
        if let Some(extension) = Path::new(s).extension().and_then(OsStr::to_str) {
            OutputFormat::from_string(extension)
        } else {
            Err(Error::msg("Unable to isolate filename extension"))
        }
    }

    /// Replaces the extension of a given filename with that of the provided new extension.
    pub fn replace_extension_with(filename: &str, new_extension: &str) -> Result<String> {
        if let Some(new_filename) = Path::new(filename).with_extension(new_extension).to_str() {
            Ok(new_filename.to_string())
        } else {
            Err(Error::msg("Unable to replace filename"))
        }
    }

    /// Replaces the extension of a provided filename with that of the enum value on which this
    /// method is called.
    ///
    /// **Example:**
    /// ```
    /// use imagebuffer_generics::OutputFormat;
    ///
    /// let foo = OutputFormat::PNG.replace_extension_for("foo.jpg").unwrap();
    ///
    /// assert_eq!(foo, "foo.png");
    /// ```
    pub fn replace_extension_for(self, filename: &str) -> Result<String> {
        match self {
            // OutputFormat::DNG => replace_extension_with(filename, "dng"),
            OutputFormat::PNG => OutputFormat::replace_extension_with(filename, "png"),
            OutputFormat::TIFF => OutputFormat::replace_extension_with(filename, "tif"),
            OutputFormat::JPEG => OutputFormat::replace_extension_with(filename, "jpg"),
        }
    }
}

/// Represents a supported input/output bit depth.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum ImageMode {
    /// Unsigned one byte (8 bits)
    U8BIT,

    /// Unsigned two bute (16 bits)
    U16BIT,
}

impl Num for f32 {}
impl Num for f64 {}
impl Num for u8 {}
impl Num for u16 {}
impl Num for u32 {}
impl Num for u64 {}
impl Num for i8 {}
impl Num for i16 {}
impl Num for i32 {}
impl Num for i64 {}

pub type Buffer1ByteInt = Buffer<u8>;
pub type Buffer2ByteInt = Buffer<u16>;
pub type Buffer4ByteInt = Buffer<u32>;
pub type Buffer4ByteFloat = Buffer<f32>;
pub type Buffer8ByteFloat = Buffer<f64>;

pub type ImageBuffer1ByteInt = ImageBuffer<u8>;
pub type ImageBuffer2ByteInt = ImageBuffer<u16>;
pub type ImageBuffer4ByteInt = ImageBuffer<u32>;
pub type ImageBuffer4ByteFloat = ImageBuffer<f32>;
pub type ImageBuffer8ByteFloat = ImageBuffer<f64>;
