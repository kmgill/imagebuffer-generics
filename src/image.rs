use crate::imagebuffer::ImageBuffer;
use crate::{util, ImageMode, Num, OutputFormat};
use anyhow::{Error, Result};
use image::{open, DynamicImage, Luma, LumaA, Rgb, Rgba};
use itertools::iproduct;
use std::ops::{Add, Div, Mul, Sub};

/// Represents a two-dimensional multi-channel image.
#[derive(Clone)]
pub struct Image<N: Num> {
    pub bands: Vec<ImageBuffer<N>>,
    pub width: usize,
    pub height: usize,
}

/////////////////////////////////
// Image Add
/////////////////////////////////

impl<N: Num> Add for Image<N> {
    type Output = Self;

    /// Performs the + operation for `Image<N>`.
    fn add(self, other: Self) -> Self {
        if self.num_bands() != other.num_bands() {
            panic!("Band count mismatch");
        }

        // I'm not wild about using clone() here. Gonna cause a lot of memory churn
        // in a program with a lot of image math
        let bands: Vec<ImageBuffer<N>> = (0..self.num_bands())
            .map(|i| self.bands[i].clone() + other.bands[i].clone())
            .collect();

        let w = self.bands[0].width;
        let h = self.bands[0].height;
        Image {
            bands,
            width: w,
            height: h,
        }
    }
}

impl<N: Num> Add<N> for Image<N> {
    type Output = Self;

    /// Performs the + operation for `ImageBuffer<N>` and a single value of type `N`.
    fn add(self, other: N) -> Self {
        let bands: Vec<ImageBuffer<N>> = (0..self.num_bands())
            .map(|i| self.bands[i].clone() + other)
            .collect();

        let w = self.bands[0].width;
        let h = self.bands[0].height;
        Image {
            bands,
            width: w,
            height: h,
        }
    }
}

/////////////////////////////////
// Image Subtract
/////////////////////////////////

impl<N: Num> Sub for Image<N> {
    type Output = Self;

    /// Performs the - operation for `Image<N>`.
    fn sub(self, other: Self) -> Self {
        if self.num_bands() != other.num_bands() {
            panic!("Band count mismatch");
        }

        // I'm not wild about using clone() here. Gonna cause a lot of memory churn
        // in a program with a lot of image math
        let bands: Vec<ImageBuffer<N>> = (0..self.num_bands())
            .map(|i| self.bands[i].clone() - other.bands[i].clone())
            .collect();

        let w = self.bands[0].width;
        let h = self.bands[0].height;
        Image {
            bands,
            width: w,
            height: h,
        }
    }
}

impl<N: Num> Sub<N> for Image<N> {
    type Output = Self;

    /// Performs the - operation for `ImageBuffer<N>` and a single value of type `N`.
    fn sub(self, other: N) -> Self {
        let bands: Vec<ImageBuffer<N>> = (0..self.num_bands())
            .map(|i| self.bands[i].clone() - other)
            .collect();

        let w = self.bands[0].width;
        let h = self.bands[0].height;
        Image {
            bands,
            width: w,
            height: h,
        }
    }
}

/////////////////////////////////
// Image Multiplication
/////////////////////////////////

impl<N: Num> Mul for Image<N> {
    type Output = Self;

    /// Performs the * operation for `Image<N>`.
    fn mul(self, other: Self) -> Self {
        if self.num_bands() != other.num_bands() {
            panic!("Band count mismatch");
        }

        // I'm not wild about using clone() here. Gonna cause a lot of memory churn
        // in a program with a lot of image math
        let bands: Vec<ImageBuffer<N>> = (0..self.num_bands())
            .map(|i| self.bands[i].clone() * other.bands[i].clone())
            .collect();

        let w = self.bands[0].width;
        let h = self.bands[0].height;
        Image {
            bands,
            width: w,
            height: h,
        }
    }
}

impl<N: Num> Mul<N> for Image<N> {
    type Output = Self;

    /// Performs the * operation for `ImageBuffer<N>` and a single value of type `N`.
    fn mul(self, other: N) -> Self {
        let bands: Vec<ImageBuffer<N>> = (0..self.num_bands())
            .map(|i| self.bands[i].clone() * other)
            .collect();

        let w = self.bands[0].width;
        let h = self.bands[0].height;
        Image {
            bands,
            width: w,
            height: h,
        }
    }
}

/////////////////////////////////
// Image Division
/////////////////////////////////

impl<N: Num> Div for Image<N> {
    type Output = Self;

    /// Performs the / operation for `Image<N>`.
    fn div(self, other: Self) -> Self {
        if self.num_bands() != other.num_bands() {
            panic!("Band count mismatch");
        }

        // I'm not wild about using clone() here. Gonna cause a lot of memory churn
        // in a program with a lot of image math
        let bands: Vec<ImageBuffer<N>> = (0..self.num_bands())
            .map(|i| self.bands[i].clone() / other.bands[i].clone())
            .collect();

        let w = self.bands[0].width;
        let h = self.bands[0].height;
        Image {
            bands,
            width: w,
            height: h,
        }
    }
}

impl<N: Num> Div<N> for Image<N> {
    type Output = Self;

    /// Performs the / operation for `ImageBuffer<N>` and a single value of type `N`.
    fn div(self, other: N) -> Self {
        let bands: Vec<ImageBuffer<N>> = (0..self.num_bands())
            .map(|i| self.bands[i].clone() / other)
            .collect();

        let w = self.bands[0].width;
        let h = self.bands[0].height;
        Image {
            bands,
            width: w,
            height: h,
        }
    }
}

impl<N: Num> Image<N> {
    /// Creates a blank image containing the provided number of bands
    pub fn new_with_num_bands(width: usize, height: usize, num_bands: usize) -> Self {
        let bands: Vec<ImageBuffer<N>> = (0..num_bands)
            .map(|_| ImageBuffer::<N>::new(width, height, N::from_u8(0).unwrap()))
            .collect();

        Image {
            bands,
            width,
            height,
        }
    }

    /// Creates a single channel image using the provided `ImageBuffer`
    pub fn new_with_buffer(buffer: ImageBuffer<N>) -> Self {
        let w = buffer.width;
        let h = buffer.height;
        Image {
            bands: vec![buffer],
            width: w,
            height: h,
        }
    }

    /// Creates a three channel RGB image using the provided `ImageBuffer`s.
    pub fn new_with_rgb_buffers(r: ImageBuffer<N>, g: ImageBuffer<N>, b: ImageBuffer<N>) -> Self {
        if !r.dims_match(&g) || !r.dims_match(&b) {
            panic!("Mismatched buffer dimensions");
        }

        let w = r.width;
        let h = r.height;
        Image {
            bands: vec![r, g, b],
            width: w,
            height: h,
        }
    }

    /// Creates a four channel RGBA image using the provided `ImageBuffer`s.
    pub fn new_with_rgba_buffers(
        r: ImageBuffer<N>,
        g: ImageBuffer<N>,
        b: ImageBuffer<N>,
        a: ImageBuffer<N>,
    ) -> Self {
        if !r.dims_match(&g) || !r.dims_match(&b) || !r.dims_match(&a) {
            panic!("Mismatched buffer dimensions");
        }

        let w = r.width;
        let h = r.height;
        Image {
            bands: vec![r, g, b, a],
            width: w,
            height: h,
        }
    }

    /// Returns the number of bands (channels) in this image
    pub fn num_bands(&self) -> usize {
        self.bands.len()
    }

    fn open_8bit_mono(buffer: DynamicImage) -> Result<Image<N>> {
        let image_data = buffer.into_luma8();

        let dims = image_data.dimensions();

        let width = dims.0 as usize;
        let height = dims.1 as usize;

        let mut newbuffer = ImageBuffer::<N>::new(width, height, N::from_u8(0).unwrap());
        iproduct!(0..height, 0..width).for_each(|(y, x)| {
            let pixel = image_data.get_pixel(x as u32, y as u32);
            newbuffer.set(x, y, N::from_u8(pixel[0]).unwrap());
        });

        Ok(Image {
            bands: vec![newbuffer],
            width,
            height,
        })
    }

    fn open_8bit_mono_alpha(buffer: DynamicImage) -> Result<Image<N>> {
        let image_data = buffer.into_luma_alpha8();

        let dims = image_data.dimensions();

        let width = dims.0 as usize;
        let height = dims.1 as usize;

        let mut lbuffer = ImageBuffer::<N>::new(width, height, N::from_u8(0).unwrap());
        let mut abuffer = ImageBuffer::<N>::new(width, height, N::from_u8(0).unwrap());

        iproduct!(0..height, 0..width).for_each(|(y, x)| {
            let pixel = image_data.get_pixel(x as u32, y as u32);
            lbuffer.set(x, y, N::from_u8(pixel[0]).unwrap());
            abuffer.set(x, y, N::from_u8(pixel[1]).unwrap());
        });
        Ok(Image {
            bands: vec![lbuffer, abuffer],
            width,
            height,
        })
    }

    fn open_8bit_rgb(buffer: DynamicImage) -> Result<Image<N>> {
        let image_data = buffer.into_rgb8();

        let dims = image_data.dimensions();

        let width = dims.0 as usize;
        let height = dims.1 as usize;

        let mut rbuffer = ImageBuffer::<N>::new(width, height, N::from_u8(0).unwrap());
        let mut gbuffer = ImageBuffer::<N>::new(width, height, N::from_u8(0).unwrap());
        let mut bbuffer = ImageBuffer::<N>::new(width, height, N::from_u8(0).unwrap());

        iproduct!(0..height, 0..width).for_each(|(y, x)| {
            let pixel = image_data.get_pixel(x as u32, y as u32);
            rbuffer.set(x, y, N::from_u8(pixel[0]).unwrap());
            gbuffer.set(x, y, N::from_u8(pixel[1]).unwrap());
            bbuffer.set(x, y, N::from_u8(pixel[2]).unwrap());
        });
        Ok(Image {
            bands: vec![rbuffer, gbuffer, bbuffer],
            width,
            height,
        })
    }

    fn open_8bit_rgba(buffer: DynamicImage) -> Result<Image<N>> {
        let image_data = buffer.into_rgba8();

        let dims = image_data.dimensions();

        let width = dims.0 as usize;
        let height = dims.1 as usize;

        let mut rbuffer = ImageBuffer::<N>::new(width, height, N::from_u8(0).unwrap());
        let mut gbuffer = ImageBuffer::<N>::new(width, height, N::from_u8(0).unwrap());
        let mut bbuffer = ImageBuffer::<N>::new(width, height, N::from_u8(0).unwrap());
        let mut abuffer = ImageBuffer::<N>::new(width, height, N::from_u8(0).unwrap());

        iproduct!(0..height, 0..width).for_each(|(y, x)| {
            let pixel = image_data.get_pixel(x as u32, y as u32);
            rbuffer.set(x, y, N::from_u8(pixel[0]).unwrap());
            gbuffer.set(x, y, N::from_u8(pixel[1]).unwrap());
            bbuffer.set(x, y, N::from_u8(pixel[2]).unwrap());
            abuffer.set(x, y, N::from_u8(pixel[3]).unwrap());
        });
        Ok(Image {
            bands: vec![rbuffer, gbuffer, bbuffer, abuffer],
            width,
            height,
        })
    }

    fn open_8bit(buffer: DynamicImage) -> Result<Image<N>> {
        if util::image_is_mono(&buffer) {
            Image::open_8bit_mono(buffer)
        } else if util::image_is_mono_alpha(&buffer) {
            Image::open_8bit_mono_alpha(buffer)
        } else if util::image_is_rgb(&buffer) {
            Image::open_8bit_rgb(buffer)
        } else if util::image_is_rgba(&buffer) {
            Image::open_8bit_rgba(buffer)
        } else {
            Err(Error::msg("Unsupported color channel configuration"))
        }
    }

    fn open_16bit_mono(buffer: DynamicImage) -> Result<Image<N>> {
        let image_data = buffer.into_luma16();

        let dims = image_data.dimensions();

        let width = dims.0 as usize;
        let height = dims.1 as usize;

        let mut newbuffer = ImageBuffer::<N>::new(width, height, N::from_u16(0).unwrap());
        iproduct!(0..height, 0..width).for_each(|(y, x)| {
            let pixel = image_data.get_pixel(x as u32, y as u32);
            newbuffer.set(x, y, N::from_u16(pixel[0]).unwrap());
        });

        Ok(Image {
            bands: vec![newbuffer],
            width,
            height,
        })
    }

    fn open_16bit_mono_alpha(buffer: DynamicImage) -> Result<Image<N>> {
        let image_data = buffer.into_luma_alpha16();

        let dims = image_data.dimensions();

        let width = dims.0 as usize;
        let height = dims.1 as usize;

        let mut lbuffer = ImageBuffer::<N>::new(width, height, N::from_u16(0).unwrap());
        let mut abuffer = ImageBuffer::<N>::new(width, height, N::from_u16(0).unwrap());

        iproduct!(0..height, 0..width).for_each(|(y, x)| {
            let pixel = image_data.get_pixel(x as u32, y as u32);
            lbuffer.set(x, y, N::from_u16(pixel[0]).unwrap());
            abuffer.set(x, y, N::from_u16(pixel[1]).unwrap());
        });
        Ok(Image {
            bands: vec![lbuffer, abuffer],
            width,
            height,
        })
    }

    fn open_16bit_rgb(buffer: DynamicImage) -> Result<Image<N>> {
        let image_data = buffer.into_rgb16();

        let dims = image_data.dimensions();

        let width = dims.0 as usize;
        let height = dims.1 as usize;

        let mut rbuffer = ImageBuffer::<N>::new(width, height, N::from_u16(0).unwrap());
        let mut gbuffer = ImageBuffer::<N>::new(width, height, N::from_u16(0).unwrap());
        let mut bbuffer = ImageBuffer::<N>::new(width, height, N::from_u16(0).unwrap());

        iproduct!(0..height, 0..width).for_each(|(y, x)| {
            let pixel = image_data.get_pixel(x as u32, y as u32);
            rbuffer.set(x, y, N::from_u16(pixel[0]).unwrap());
            gbuffer.set(x, y, N::from_u16(pixel[1]).unwrap());
            bbuffer.set(x, y, N::from_u16(pixel[2]).unwrap());
        });
        Ok(Image {
            bands: vec![rbuffer, gbuffer, bbuffer],
            width,
            height,
        })
    }

    fn open_16bit_rgba(buffer: DynamicImage) -> Result<Image<N>> {
        let image_data = buffer.into_rgba16();

        let dims = image_data.dimensions();

        let width = dims.0 as usize;
        let height = dims.1 as usize;

        let mut rbuffer = ImageBuffer::<N>::new(width, height, N::from_u16(0).unwrap());
        let mut gbuffer = ImageBuffer::<N>::new(width, height, N::from_u16(0).unwrap());
        let mut bbuffer = ImageBuffer::<N>::new(width, height, N::from_u16(0).unwrap());
        let mut abuffer = ImageBuffer::<N>::new(width, height, N::from_u16(0).unwrap());

        iproduct!(0..height, 0..width).for_each(|(y, x)| {
            let pixel = image_data.get_pixel(x as u32, y as u32);
            rbuffer.set(x, y, N::from_u16(pixel[0]).unwrap());
            gbuffer.set(x, y, N::from_u16(pixel[1]).unwrap());
            bbuffer.set(x, y, N::from_u16(pixel[2]).unwrap());
            abuffer.set(x, y, N::from_u16(pixel[3]).unwrap());
        });
        Ok(Image {
            bands: vec![rbuffer, gbuffer, bbuffer, abuffer],
            width,
            height,
        })
    }

    fn open_16bit(buffer: DynamicImage) -> Result<Image<N>> {
        if util::image_is_mono(&buffer) {
            Image::open_16bit_mono(buffer)
        } else if util::image_is_mono_alpha(&buffer) {
            Image::open_16bit_mono_alpha(buffer)
        } else if util::image_is_rgb(&buffer) {
            Image::open_16bit_rgb(buffer)
        } else if util::image_is_rgba(&buffer) {
            Image::open_16bit_rgba(buffer)
        } else {
            Err(Error::msg("Unsupported color channel configuration"))
        }
    }

    /// Opens an image from disk
    pub fn open(file_path: &str) -> Result<Image<N>> {
        let buffer = open(file_path).unwrap();
        let image_mode = util::image_bitmode(&buffer);

        match image_mode {
            ImageMode::U8BIT => Image::<N>::open_8bit(buffer),
            ImageMode::U16BIT => Image::<N>::open_16bit(buffer),
        }
    }

    fn save_to_16bit_mono(&self, output_file_name: &str) -> Result<()> {
        let mut out_img =
            DynamicImage::new_luma16(self.width as u32, self.height as u32).into_luma16();
        iproduct!(0..self.height, 0..self.width).for_each(|(y, x)| {
            out_img.put_pixel(x as u32, y as u32, Luma([self.bands[0].get_u16(x, y)]));
        });
        if out_img.save(output_file_name).is_ok() {
            Ok(())
        } else {
            Err(Error::msg("Failed to save image"))
        }
    }

    fn save_to_16bit_mono_alpha(&self, output_file_name: &str) -> Result<()> {
        let mut out_img =
            DynamicImage::new_luma_a16(self.width as u32, self.height as u32).into_luma_alpha16();
        iproduct!(0..self.height, 0..self.width).for_each(|(y, x)| {
            out_img.put_pixel(
                x as u32,
                y as u32,
                LumaA([self.bands[0].get_u16(x, y), self.bands[1].get_u16(x, y)]),
            );
        });
        if out_img.save(output_file_name).is_ok() {
            Ok(())
        } else {
            Err(Error::msg("Failed to save image"))
        }
    }

    fn save_to_16bit_rgb(&self, output_file_name: &str) -> Result<()> {
        let mut out_img =
            DynamicImage::new_rgb16(self.width as u32, self.height as u32).into_rgb16();
        iproduct!(0..self.height, 0..self.width).for_each(|(y, x)| {
            out_img.put_pixel(
                x as u32,
                y as u32,
                Rgb([
                    self.bands[0].get_u16(x, y),
                    self.bands[1].get_u16(x, y),
                    self.bands[2].get_u16(x, y),
                ]),
            );
        });
        if out_img.save(output_file_name).is_ok() {
            Ok(())
        } else {
            Err(Error::msg("Failed to save image"))
        }
    }

    fn save_to_16bit_rgba(&self, output_file_name: &str) -> Result<()> {
        let mut out_img =
            DynamicImage::new_rgba16(self.width as u32, self.height as u32).into_rgba16();
        iproduct!(0..self.height, 0..self.width).for_each(|(y, x)| {
            out_img.put_pixel(
                x as u32,
                y as u32,
                Rgba([
                    self.bands[0].get_u16(x, y),
                    self.bands[1].get_u16(x, y),
                    self.bands[2].get_u16(x, y),
                    self.bands[3].get_u16(x, y),
                ]),
            );
        });
        if out_img.save(output_file_name).is_ok() {
            Ok(())
        } else {
            Err(Error::msg("Failed to save image"))
        }
    }

    fn save_to_16bit(&self, output_file_name: &str) -> Result<()> {
        match self.num_bands() {
            1 => self.save_to_16bit_mono(output_file_name),
            2 => self.save_to_16bit_mono_alpha(output_file_name),
            3 => self.save_to_16bit_rgb(output_file_name),
            4 => self.save_to_16bit_rgba(output_file_name),
            _ => Err(Error::msg("Unsupported number of bands")),
        }
    }

    fn save_to_8bit_mono(&self, output_file_name: &str) -> Result<()> {
        let mut out_img =
            DynamicImage::new_luma8(self.width as u32, self.height as u32).into_luma8();
        iproduct!(0..self.height, 0..self.width).for_each(|(y, x)| {
            out_img.put_pixel(x as u32, y as u32, Luma([self.bands[0].get_u8(x, y)]));
        });
        if out_img.save(output_file_name).is_ok() {
            Ok(())
        } else {
            Err(Error::msg("Failed to save image"))
        }
    }

    fn save_to_8bit_mono_alpha(&self, output_file_name: &str) -> Result<()> {
        let mut out_img =
            DynamicImage::new_luma_a8(self.width as u32, self.height as u32).into_luma_alpha8();
        iproduct!(0..self.height, 0..self.width).for_each(|(y, x)| {
            out_img.put_pixel(
                x as u32,
                y as u32,
                LumaA([self.bands[0].get_u8(x, y), self.bands[1].get_u8(x, y)]),
            );
        });
        if out_img.save(output_file_name).is_ok() {
            Ok(())
        } else {
            Err(Error::msg("Failed to save image"))
        }
    }

    fn save_to_8bit_rgb(&self, output_file_name: &str) -> Result<()> {
        let mut out_img = DynamicImage::new_rgb8(self.width as u32, self.height as u32).into_rgb8();
        iproduct!(0..self.height, 0..self.width).for_each(|(y, x)| {
            out_img.put_pixel(
                x as u32,
                y as u32,
                Rgb([
                    self.bands[0].get_u8(x, y),
                    self.bands[1].get_u8(x, y),
                    self.bands[2].get_u8(x, y),
                ]),
            );
        });
        if out_img.save(output_file_name).is_ok() {
            Ok(())
        } else {
            Err(Error::msg("Failed to save image"))
        }
    }

    fn save_to_8bit_rgba(&self, output_file_name: &str) -> Result<()> {
        let mut out_img =
            DynamicImage::new_rgba8(self.width as u32, self.height as u32).into_rgba8();
        iproduct!(0..self.height, 0..self.width).for_each(|(y, x)| {
            out_img.put_pixel(
                x as u32,
                y as u32,
                Rgba([
                    self.bands[0].get_u8(x, y),
                    self.bands[1].get_u8(x, y),
                    self.bands[2].get_u8(x, y),
                    self.bands[3].get_u8(x, y),
                ]),
            );
        });
        if out_img.save(output_file_name).is_ok() {
            Ok(())
        } else {
            Err(Error::msg("Failed to save image"))
        }
    }

    fn save_to_8bit(&self, output_file_name: &str) -> Result<()> {
        match self.num_bands() {
            1 => self.save_to_8bit_mono(output_file_name),
            2 => self.save_to_8bit_mono_alpha(output_file_name),
            3 => self.save_to_8bit_rgb(output_file_name),
            4 => self.save_to_8bit_rgba(output_file_name),
            _ => Err(Error::msg("Unsupported number of bands")),
        }
    }

    /// Saves the image to disk
    pub fn save_to(
        &self,
        output_file_name: &str,
        format: OutputFormat,
        mode: ImageMode,
    ) -> Result<()> {
        let corrected_file_name = format.replace_extension_for(output_file_name)?;
        match mode {
            ImageMode::U8BIT => self.save_to_8bit(&corrected_file_name),
            ImageMode::U16BIT => self.save_to_16bit(&corrected_file_name),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;

    #[test]
    fn test_scale_8bit_to_16bit() -> Result<()> {
        let mut a = Image::<u16>::open("assets/test-image.jpg")?;
        a = a * 257;

        a.save_to(
            "target/testsave_image_scaled_16bit_png",
            OutputFormat::PNG,
            ImageMode::U16BIT,
        )?;

        assert!(Path::new("target/testsave_image_scaled_16bit_png.png").exists());

        Ok(())
    }

    #[test]
    fn test_open_rgb_image() -> Result<()> {
        // 8 bit RGB
        let a = Image::<u8>::open("assets/test-image.jpg")?;
        a.save_to(
            "target/testsave_image_8bit_jpg_a",
            OutputFormat::JPEG,
            ImageMode::U8BIT,
        )?;
        assert!(Path::new("target/testsave_image_8bit_jpg_a.jpg").exists());

        // 8 bit RGB as f64
        let b = Image::<f64>::open("assets/test-image.jpg")?;
        b.save_to(
            "target/testsave_image_8bit_jpg_b",
            OutputFormat::JPEG,
            ImageMode::U8BIT,
        )?;
        assert!(Path::new("target/testsave_image_8bit_jpg_b.jpg").exists());

        // 16 bit RGB
        let c = Image::<u16>::open("assets/test-image.png")?;
        c.save_to(
            "target/testsave_image_16bit_png_c",
            OutputFormat::PNG,
            ImageMode::U16BIT,
        )?;
        assert!(Path::new("target/testsave_image_16bit_png_c.png").exists());

        Ok(())
    }
}
