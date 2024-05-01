use std::ops::{Add, Div, Mul, Sub};

use anyhow::Error;
use image::{open, DynamicImage, Luma};
use itertools::iproduct;

use crate::{buffer::Buffer, util, ImageMode, Num, OutputFormat, Range};

/// Represents a two-dimensional single-channel image buffer.
#[derive(Clone)]
pub struct ImageBuffer<N: Num> {
    pub buffer: Buffer<N>,
    pub width: usize,
    pub height: usize,
}

////////////
// ImageBuffer Addition
////////////

impl<N: Num> Add for ImageBuffer<N> {
    type Output = Self;

    /// Performs the + operation for `ImageBuffer<N>`.
    fn add(self, other: Self) -> Self {
        if !self.dims_match(&other) {
            panic!("Image buffer dimensions do not match!");
        }
        ImageBuffer {
            buffer: self.buffer + other.buffer,
            width: self.width,
            height: self.height,
        }
    }
}

impl<N: Num> Add<N> for ImageBuffer<N> {
    type Output = Self;

    /// Performs the + operation for `ImageBuffer<N>` and a single value of type `N`.
    fn add(self, other: N) -> Self {
        ImageBuffer {
            buffer: self.buffer + other,
            width: self.width,
            height: self.height,
        }
    }
}

impl<N: Num> ImageBuffer<N> {
    /// Adds the values of matching-length `ImageBuffer` into this `ImageBuffer`.
    pub fn add_mut(&mut self, other: &ImageBuffer<N>) {
        self.buffer.add_mut(&other.buffer);
    }

    /// Adds value of type `N` to each value in this `ImageBuffer`.
    pub fn add_into(&self, other: N) -> Self {
        ImageBuffer {
            width: self.width,
            height: self.height,
            buffer: self.buffer.add_into(other),
        }
    }

    /// Adds the value of type `N` to each value of a copy of this `ImageBuffer`.
    pub fn add_into_mut(&mut self, other: N) {
        self.buffer.add_into(other);
    }
}

////////////
// ImageBuffer Subtraction
////////////

impl<N: Num> Sub for ImageBuffer<N> {
    type Output = Self;

    /// Performs the - operation for `ImageBuffer<N>`.
    fn sub(self, other: Self) -> Self {
        if !self.dims_match(&other) {
            panic!("Image buffer dimensions do not match!");
        }
        ImageBuffer {
            buffer: self.buffer - other.buffer,
            width: self.width,
            height: self.height,
        }
    }
}

impl<N: Num> Sub<N> for ImageBuffer<N> {
    type Output = Self;

    /// Performs the - operation for `ImageBuffer<N>` and a single value of type `N`.
    fn sub(self, other: N) -> Self {
        ImageBuffer {
            buffer: self.buffer - other,
            width: self.width,
            height: self.height,
        }
    }
}

impl<N: Num> ImageBuffer<N> {
    /// Subtracts the values of matching-length `ImageBuffer` into this `ImageBuffer`.
    pub fn subtract_mut(&mut self, other: &ImageBuffer<N>) {
        self.buffer.subtract_mut(&other.buffer);
    }

    /// Subtracts value of type `N` from each value in this `ImageBuffer`.
    pub fn subtract_into(&self, other: N) -> Self {
        ImageBuffer {
            width: self.width,
            height: self.height,
            buffer: self.buffer.subtract_into(other),
        }
    }

    /// Subtracts the value of type `N` from each value of a copy of this `ImageBuffer`.
    pub fn subtract_into_mut(&mut self, other: N) {
        self.buffer.subtract_into(other);
    }
}

////////////
// ImageBuffer Multiplication
////////////

impl<N: Num> Mul for ImageBuffer<N> {
    type Output = Self;

    /// Performs the * operation for `ImageBuffer<N>`.
    fn mul(self, other: Self) -> Self {
        if !self.dims_match(&other) {
            panic!("Image buffer dimensions do not match!");
        }
        ImageBuffer {
            buffer: self.buffer * other.buffer,
            width: self.width,
            height: self.height,
        }
    }
}

impl<N: Num> Mul<N> for ImageBuffer<N> {
    type Output = Self;

    /// Performs the * operation for `ImageBuffer<N>` and a single value of type `N`.
    fn mul(self, other: N) -> Self {
        ImageBuffer {
            buffer: self.buffer * other,
            width: self.width,
            height: self.height,
        }
    }
}

impl<N: Num> ImageBuffer<N> {
    /// Multiplies the values of matching-length Buffer into this `ImageBuffer`.
    pub fn multiply_mut(&mut self, other: &ImageBuffer<N>) {
        self.buffer.multiply_mut(&other.buffer);
    }

    /// Multiplies value of type `N` with each value in this `ImageBuffer`.
    pub fn multiply_into(&self, other: N) -> Self {
        ImageBuffer {
            width: self.width,
            height: self.height,
            buffer: self.buffer.multiply_into(other),
        }
    }

    /// Multiplies the value of type `N` with each value of a copy of this `ImageBuffer`.
    pub fn multiply_into_mut(&mut self, other: N) {
        self.buffer.multiply_into(other);
    }
}

////////////
// ImageBuffer Division
////////////

impl<N: Num> Div for ImageBuffer<N> {
    type Output = Self;

    /// Performs the / operation for `ImageBuffer<N>`.
    fn div(self, other: Self) -> Self {
        if !self.dims_match(&other) {
            panic!("Image buffer dimensions do not match!");
        }
        ImageBuffer {
            buffer: self.buffer / other.buffer,
            width: self.width,
            height: self.height,
        }
    }
}

impl<N: Num> Div<N> for ImageBuffer<N> {
    type Output = Self;

    /// Performs the / operation for `Buffer<N>` and a single value of type `N`.
    fn div(self, other: N) -> Self {
        ImageBuffer {
            buffer: self.buffer / other,
            width: self.width,
            height: self.height,
        }
    }
}

impl<N: Num> ImageBuffer<N> {
    /// Divides the values of matching-length Buffer into this `ImageBuffer`.
    pub fn divide_mut(&mut self, other: &ImageBuffer<N>) {
        self.buffer.divide_mut(&other.buffer);
    }

    /// Divides value of type `N` from each value in this `ImageBuffer`.
    pub fn divide_into(&self, other: N) -> Self {
        ImageBuffer {
            width: self.width,
            height: self.height,
            buffer: self.buffer.divide_into(other),
        }
    }

    /// Divides the value of type `N` from each value of a copy of this `ImageBuffer`.
    pub fn divide_into_mut(&mut self, other: N) {
        self.buffer.divide_into(other);
    }
}

////////////////////
// ImageBuffer impl
////////////////////

impl<N: Num> ImageBuffer<N> {
    /// Creatges a new `ImageBuffer` of size `width`x`height` and initialized with all values as
    /// `fill_value`.
    pub fn new(width: usize, height: usize, fill_value: N) -> Self {
        ImageBuffer {
            width,
            height,
            buffer: Buffer::<N>::new_of_length(width * height, fill_value),
        }
    }

    /// Compares the dimensions of two `ImageBuffer`s, returning true if they match.
    pub fn dims_match(&self, other: &ImageBuffer<N>) -> bool {
        self.width == other.width && self.height == other.height
    }

    /// Converts a two-dimensional x/y coordinate to a one-dimensional index.
    fn xy_to_idx(&self, x: usize, y: usize) -> usize {
        y * self.width + x
    }

    /// Retrieves the value at coordinate x/y.
    #[inline(always)]
    pub fn get(&self, x: usize, y: usize) -> N {
        self.buffer.buffer[self.xy_to_idx(x, y)]
    }

    /// Retrieves the value at coordinate x/y, returning it as type `f64`.
    ///
    /// **panics** If the value is incompatible with type `f64`.
    pub fn get_f64(&self, x: usize, y: usize) -> f64 {
        self.buffer.buffer[self.xy_to_idx(x, y)].to_f64().unwrap()
    }

    /// Retrieves the value at coordinate x/y, returning it as type `u8`.
    ///
    /// **panics** If the value is incompatible with type `u8`.
    pub fn get_u8(&self, x: usize, y: usize) -> u8 {
        self.buffer.buffer[self.xy_to_idx(x, y)].to_u8().unwrap()
    }

    /// Retrieves the value at coordinate x/y, returning it as type `u16`.
    ///
    /// **panics** If the value is incompatible with type `u16`.
    pub fn get_u16(&self, x: usize, y: usize) -> u16 {
        self.buffer.buffer[self.xy_to_idx(x, y)].to_u16().unwrap()
    }

    /// Retrieves a value that is interpolated based on the fractional distance between
    /// floor(x) and floor(y) and ceil(x) and ceil(y)
    ///
    /// **panics** If any value used is incompatible with type `f64` or indexes
    /// are out of bounds.
    pub fn get_interpolated_f64(&self, x: f64, y: f64) -> f64 {
        if x < self.width as f64 && y < self.height as f64 {
            let xf = x.floor();
            let xc = xf + 1.0;

            let yf = y.floor();
            let yc = yf + 1.0;

            let xd = x - xf;
            let yd = y - yf;

            let v00 = self.get_f64(xf as usize, yf as usize);
            let v01 = self.get_f64(xc as usize, yf as usize);
            let v10 = self.get_f64(xf as usize, yc as usize);
            let v11 = self.get_f64(xc as usize, yc as usize);

            let v0 = v10 * yd + v00 * (1.0 - yd);
            let v1 = v11 * yd + v01 * (1.0 - yd);
            v1 * xd + v0 * (1.0 - xd)
        } else {
            panic!("Invalid pixel coordinates");
        }
    }

    /// Sets a value within the buffer at the x/y coordinate.
    ///
    /// **panics** If x/y coordinate is out of bounds.
    pub fn set(&mut self, x: usize, y: usize, value: N) {
        let idx = self.xy_to_idx(x, y);
        self.buffer.buffer[idx] = value;
    }

    /// Creates a `Vec<u8>` with the content of this buffer. Each value is casted to `u8`.
    ///
    /// Values are not pre-normalized into a valid `u8` (0-255) range.
    ///
    /// **panics** If a value in this buffer is incompatible with `u8`
    pub fn to_vector_u8(&self) -> Vec<u8> {
        self.buffer.to_vector_u8()
    }

    /// Creates a `Vec<u16>` with the content of this buffer. Each value is casted to `u16`.
    ///
    /// Values are not pre-normalized into a valid `u16` (0-65535) range.
    ///
    /// **panics** If a value in this buffer is incompatible with `u16`
    pub fn to_vector_u16(&self) -> Vec<u16> {
        self.buffer.to_vector_u16()
    }

    /// Creates a `Vec<f64>` with the content of this buffer. Each value is casted to `f64`.
    ///
    /// Values are not pre-normalized into a valid `f64` range.
    ///
    /// **panics** If a value in this buffer is incompatible with `f64`
    pub fn to_vector_f64(&self) -> Vec<f64> {
        self.buffer.to_vector_f64()
    }

    /// Retrieves a horizontal slice subframe of the image for pushframe cameras (e.g. JunoCam raws as a stack of slices)
    ///
    /// **panics** If any bounds are exceeded.
    pub fn get_slice(&self, top_y: usize, len: usize) -> ImageBuffer<N> {
        let start_index = top_y * self.width;
        let stop_index = (top_y + len) * self.width;

        ImageBuffer {
            buffer: self.buffer.get_slice(start_index, stop_index),
            width: self.width,
            height: len,
        }
    }

    /// Creates a cropped image given the top left x/y coordinates and required width and height.
    ///
    /// **panics** If the requested subframe is too large or not positioned entirely on this
    /// image buffer.
    pub fn get_subframe(
        &self,
        left_x: usize,
        top_y: usize,
        width: usize,
        height: usize,
    ) -> ImageBuffer<N> {
        ImageBuffer {
            buffer: self
                .buffer
                .crop_2d(self.width, self.height, left_x, top_y, width, height),
            width,
            height,
        }
    }

    /// Isolates (crops) a square copy of the array centered on coordinates `x & y`. Treats the buffer as a two-dimensional matrix.
    /// The size of the window is determined from `window_size`.
    pub fn isolate_window(&self, window_size: usize, x: usize, y: usize) -> Buffer<N> {
        self.buffer
            .isolate_window_2d(self.width, self.height, window_size, x, y)
    }

    /// Applies a look-up table of the same time. This assumes the pre-LUT values in this buffer
    /// are proper indexes into the LUT table. The buffer values will be replaced with the values
    /// contained in the LUT table.
    ///
    /// **panics** if a pre-LUT value in this buffer is an invalid index (out of bounds) or is
    /// incompatible with type `usize`.
    pub fn apply_lut(&mut self, lut: &[N]) {
        self.buffer.apply_lut(lut);
    }

    /// Computes the sum of all values in the buffer.
    ///
    /// **panics** If any value is incompatible with type `f64`.
    pub fn sum(&self) -> f64 {
        self.buffer.sum()
    }

    /// Computes the mean (average) of all values in the buffer.
    ///
    /// **panics** If any value is incompatible with type `f64`.
    pub fn mean(&self) -> f64 {
        self.buffer.mean()
    }

    /// Computes the statistical variance of all values in the buffer.
    ///
    /// **panics** If any value is incompatible with type `f64`.
    pub fn variance(&self) -> f64 {
        self.buffer.variance()
    }

    /// Computes the statistical cross-correlation of this and `other` buffer.
    ///
    /// **panics** If any value is incompatible with type `f64` or buffer lengths do not match.
    pub fn xcorr(&self, other: &ImageBuffer<N>) -> f64 {
        self.buffer.xcorr(&other.buffer)
    }

    /// Computes the standard deviation of the values in the buffer.
    ///
    /// **panics** If any value in the buffer is incompatible with `f64`.
    pub fn stddev(&self) -> f64 {
        self.buffer.stddev()
    }

    /// Computes the z-score (standard score) of the values in the buffer.
    ///
    /// **panics** If any value in the buffer is incompatible with `f64`.
    pub fn z_score(&self, check_value: f64) -> f64 {
        self.buffer.z_score(check_value)
    }

    /// Creates a copy of this buffer with each value raised to the power of `exponent`.
    ///
    /// **panics** If any value is incompatible with type `f64`.
    pub fn power(&self, exponent: N) -> Self {
        ImageBuffer {
            width: self.width,
            height: self.height,
            buffer: self.buffer.power(exponent),
        }
    }

    /// Replaces all values in this buffer with each raised to the power of `exponent`.
    ///
    /// **panics** If any value in the buffer in incompatible with type `f64`.
    pub fn power_mut(&mut self, exponent: N) {
        self.buffer.power_mut(exponent);
    }

    /// Creates a copy of this buffer with each value clipped to between `clip_min` at a minimum
    /// and `clip_max` at a maximum. If any value exceeds the bounds, it will be replaced with that
    /// respective bound value.
    ///
    /// **panics** If any value is incompatible with type `f64`.
    pub fn clip(&self, clip_min: N, clip_max: N) -> Self {
        ImageBuffer {
            width: self.width,
            height: self.height,
            buffer: self.buffer.clip(clip_min, clip_max),
        }
    }

    /// Replaces each value in this buffer with each value clipped to between `clip_min` at a minimum
    /// and `clip_max` at a maximum. If any value exceeds the bounds, it will be replaced with that
    /// respective bound value.
    ///
    /// **panics** If any value is incompatible with type `f64`.
    pub fn clip_mut(&mut self, clip_min: N, clip_max: N) {
        self.buffer.clip(clip_min, clip_max);
    }

    /// Determines the minimum value contained in this buffer.
    ///
    /// **panics** If any value is incompatible with type `f64`.
    pub fn min(&self) -> N {
        self.buffer.min()
    }

    /// Determines the maximum value contained in this buffer.
    ///
    /// **panics** If any value is incompatible with type `f64`.
    pub fn max(&self) -> N {
        self.buffer.max()
    }

    /// Determines the minimum and maximum value in this buffer.
    ///
    /// **panics** If any value is incompatible with type `f64`.
    pub fn range(&self) -> Range<N> {
        self.buffer.range()
    }

    /// Saves the `ImageBuffer` to disk.
    pub fn save_image_to(
        &self,
        output_file_name: &str,
        format: OutputFormat,
        mode: ImageMode,
    ) -> anyhow::Result<()> {
        let corrected_file_name = format.replace_extension_for(output_file_name)?;
        match mode {
            ImageMode::U8BIT => self.save_image_to_8bpp(&corrected_file_name),
            ImageMode::U16BIT => self.save_image_to_16bpp(&corrected_file_name),
        }
    }

    /// Saves the `ImageBuffer` to disk as unsigned 16 bit.
    fn save_image_to_16bpp(&self, output_file_name: &str) -> anyhow::Result<()> {
        let mut out_img =
            DynamicImage::new_luma16(self.width as u32, self.height as u32).into_luma16();
        iproduct!(0..self.height, 0..self.width).for_each(|(y, x)| {
            out_img.put_pixel(x as u32, y as u32, Luma([self.get_u16(x, y)]));
        });
        if out_img.save(output_file_name).is_ok() {
            Ok(())
        } else {
            Err(Error::msg("Failed to save image"))
        }
    }

    /// Saves the `ImageBuffer` to disk as unsigned 8 bit.
    fn save_image_to_8bpp(&self, output_file_name: &str) -> anyhow::Result<()> {
        let mut out_img =
            DynamicImage::new_luma8(self.width as u32, self.height as u32).into_luma8();
        iproduct!(0..self.height, 0..self.width).for_each(|(y, x)| {
            out_img.put_pixel(x as u32, y as u32, Luma([self.get_u8(x, y)]));
        });
        if out_img.save(output_file_name).is_ok() {
            Ok(())
        } else {
            Err(Error::msg("Failed to save image"))
        }
    }

    /// Loads an unsigned 8 bit image from a `DynamicImage` buffer. If the image being loaded is RGB(A), it
    /// will attept to convert it into a monochrome image before loading. The method in which the
    /// conversion is handled is at the whim of the `image` crate.
    fn open_8bit(buffer: DynamicImage) -> anyhow::Result<ImageBuffer<N>> {
        let image_data = buffer.into_luma8();

        let dims = image_data.dimensions();

        let width = dims.0 as usize;
        let height = dims.1 as usize;

        let mut newbuffer = ImageBuffer::<N>::new(width, height, N::from_u8(0).unwrap());
        for y in 0..height {
            for x in 0..width {
                let pixel = image_data.get_pixel(x as u32, y as u32);
                newbuffer.set(x, y, N::from_u8(pixel[0]).unwrap());
            }
        }
        Ok(newbuffer)
    }

    /// Loads an unsigned 16 bit image from a `DynamicImage` buffer. If the image being loaded is RGB(A), it
    /// will attept to convert it into a monochrome image before loading. The method in which the
    /// conversion is handled is at the whim of the `image` crate.
    fn open_16bit(buffer: DynamicImage) -> anyhow::Result<ImageBuffer<N>> {
        let image_data = buffer.into_luma16();

        let dims = image_data.dimensions();

        let width = dims.0 as usize;
        let height = dims.1 as usize;

        let mut newbuffer = ImageBuffer::<N>::new(width, height, N::from_u16(0).unwrap());
        for y in 0..height {
            for x in 0..width {
                let pixel = image_data.get_pixel(x as u32, y as u32);
                newbuffer.set(x, y, N::from_u16(pixel[0]).unwrap());
            }
        }
        Ok(newbuffer)
    }

    /// Opens an image from the filesystem. `ImageBuffer` is a single channel (monochrome) structure
    /// and as such any image loaded will become grayscale. This method attempts to determine whether
    /// the image being loaded is unsigned 8 or 16 bit (32 bit not yet supported).
    pub fn open(file_path: &str) -> anyhow::Result<ImageBuffer<N>> {
        let buffer = open(file_path).unwrap();

        let _has_alpha = util::image_uses_alpha(&buffer);
        let image_mode = util::image_bitmode(&buffer);

        match image_mode {
            ImageMode::U8BIT => ImageBuffer::<N>::open_8bit(buffer),
            ImageMode::U16BIT => ImageBuffer::<N>::open_16bit(buffer),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use crate::{ImageBuffer1ByteInt, ImageBuffer2ByteInt, ImageBuffer4ByteFloat};

    use super::*;

    #[test]
    fn test_imagebuffer() {
        let image0 = ImageBuffer4ByteFloat::new(1024, 1024, 0.0);
        let image1 = ImageBuffer4ByteFloat::new(1024, 1024, 0.0);
        let image3 = image0 + image1;
        let image4 = image3 + 5.0;

        _ = image4.get(0, 0) as u8;

        let image5 = image4.get_subframe(100, 100, 100, 100);
        assert_eq!(image5.width, 100);
        assert_eq!(image5.height, 100);
        assert_eq!(image5.buffer.len(), image5.width * image5.height);
    }

    #[test]
    #[should_panic]
    fn test_size_mismatch() {
        let image0 = ImageBuffer4ByteFloat::new(1024, 1024, 0.0);
        let image1 = ImageBuffer4ByteFloat::new(2048, 2048, 0.0);
        _ = image0 + image1;
    }

    #[test]
    fn test_save() -> anyhow::Result<()> {
        let image0 = ImageBuffer4ByteFloat::new(1024, 1024, 0.0);

        // Test: Saves as an 8 bit PNG

        image0.save_image_to(
            "target/testsave_image0_8bit_png.png",
            OutputFormat::PNG,
            ImageMode::U8BIT,
        )?;

        assert!(Path::new("target/testsave_image0_8bit_png.png").exists());

        // Test: Saves as an 8 bit TIF, replaces .png extension with .tif

        image0.save_image_to(
            "target/testsave_image0_8bit_png.png",
            OutputFormat::TIFF,
            ImageMode::U8BIT,
        )?;

        assert!(Path::new("target/testsave_image0_8bit_png.tif").exists());

        // Test: Saves as a 16 bit png

        image0.save_image_to(
            "target/testsave_image0_16bit_png.png",
            OutputFormat::PNG,
            ImageMode::U16BIT,
        )?;

        assert!(Path::new("target/testsave_image0_16bit_png.png").exists());

        // Test: Saves as a 16 bit tif, replaces .png extension with .tif
        image0.save_image_to(
            "target/testsave_image0_16bit_png.png",
            OutputFormat::TIFF,
            ImageMode::U16BIT,
        )?;

        assert!(Path::new("target/testsave_image0_16bit_png.tif").exists());

        // Test: Saves as an 8 bit jpeg, appends .jpg to missing extension

        image0.save_image_to(
            "target/testsave_image0_8bit_png",
            OutputFormat::JPEG,
            ImageMode::U8BIT,
        )?;

        assert!(Path::new("target/testsave_image0_8bit_png.jpg").exists());

        Ok(())
    }

    #[test]
    fn create_gray_with_math() {
        // Create a 1024x1024 image, all white
        let myimage = ImageBuffer4ByteFloat::new(1024, 1024, 255.0);

        // Create an image for halving the white image
        let otherimage = ImageBuffer4ByteFloat::new(1024, 1024, 0.5);

        // gray = 255 * 0.5
        let grayimage = myimage * otherimage;

        // Check the resulting value
        assert_eq!(grayimage.get(0, 0), 127.5);
    }

    #[test]
    fn create_gray_with_math_1byte() {
        // Create a 1024x1024 image, all white
        let myimage = ImageBuffer1ByteInt::new(1024, 1024, 255);

        // gray = 255 / 2
        let grayimage = myimage / 2;

        // Check the resulting value
        assert_eq!(grayimage.get(0, 0), 127);
    }

    #[test]
    fn test_open_image_8bit() -> anyhow::Result<()> {
        // The image is in color, but will be converted by the image module to grayscale.
        let myimage = ImageBuffer1ByteInt::open("assets/test-image.jpg")?;
        myimage.save_image_to(
            "target/testsave_testimage_8bit_jpg",
            OutputFormat::JPEG,
            ImageMode::U8BIT,
        )?;

        Ok(())
    }

    #[test]
    fn test_open_image_8bit_save_16bit() -> anyhow::Result<()> {
        // The image is in color, but will be converted by the image module to grayscale.
        let mut myimage = ImageBuffer4ByteFloat::open("assets/test-image.jpg")?;

        // Scale values from 0-255 to 0-65535
        myimage = myimage * 257.0;

        myimage.save_image_to(
            "target/testsave_testimage_8bit_into_16bit_png",
            OutputFormat::PNG,
            ImageMode::U16BIT,
        )?;

        Ok(())
    }

    #[test]
    fn test_open_image_16bit() -> anyhow::Result<()> {
        // The image is in color, but will be converted by the image module to grayscale.
        let myimage = ImageBuffer2ByteInt::open("assets/test-image.png")?;
        myimage.save_image_to(
            "target/testsave_testimage_16bit_png",
            OutputFormat::PNG,
            ImageMode::U16BIT,
        )?;

        Ok(())
    }

    #[test]
    #[should_panic]
    fn test_open_image_16bit_save_8bit() {
        // The image is in color, but will be converted by the image module to grayscale.
        let myimage = ImageBuffer2ByteInt::open("assets/test-image.png").unwrap();
        myimage
            .save_image_to(
                "target/testsave_testimage_16bit_into_8bit_png",
                OutputFormat::PNG,
                ImageMode::U8BIT,
            )
            .unwrap();
    }
}
