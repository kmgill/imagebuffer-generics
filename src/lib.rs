use std::ops::{Add, Div, Index, IndexMut, Mul, Sub};

use itertools::iproduct;

pub trait Num:
    num_traits::Num + num_traits::ToPrimitive + num_traits::FromPrimitive + Copy
{
}

#[derive(Clone)]
struct Buffer<N: Num> {
    buffer: Vec<N>,
}

///////////////////////
// Buffer Addition
///////////////////////

impl<N: Num> Add for Buffer<N> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        if self.buffer.len() != other.buffer.len() {
            panic!("Buffer lengths do not match");
        }
        Buffer {
            buffer: self
                .buffer
                .iter()
                .zip(other.buffer.iter())
                .map(|(a, b)| *a + *b)
                .collect(),
        }
    }
}

impl<N: Num> Add<N> for Buffer<N> {
    type Output = Self;
    fn add(self, other: N) -> Self {
        Buffer {
            buffer: self.buffer.iter().map(|a| *a + other).collect(),
        }
    }
}

impl<N: Num> Buffer<N> {
    pub fn add_mut(&mut self, other: &Buffer<N>) {
        self.buffer
            .iter_mut()
            .zip(other.buffer.iter())
            .for_each(|(a, b)| *a = *a + *b);
    }

    pub fn add_into_mut(&mut self, value: N) {
        self.buffer.iter_mut().for_each(|a| *a = *a + value);
    }

    pub fn add_into(&self, value: N) -> Self {
        Buffer {
            buffer: self.buffer.iter().map(|a| *a + value).collect(),
        }
    }
}

///////////////////////
// Buffer Subtraction
///////////////////////

impl<N: Num> Sub for Buffer<N> {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        if self.buffer.len() != other.buffer.len() {
            panic!("Buffer lengths do not match");
        }
        Buffer {
            buffer: self
                .buffer
                .iter()
                .zip(other.buffer.iter())
                .map(|(a, b)| *a - *b)
                .collect(),
        }
    }
}

impl<N: Num> Sub<N> for Buffer<N> {
    type Output = Self;
    fn sub(self, other: N) -> Self {
        Buffer {
            buffer: self.buffer.iter().map(|a| *a - other).collect(),
        }
    }
}

impl<N: Num> Buffer<N> {
    pub fn subtract_mut(&mut self, other: &Buffer<N>) {
        self.buffer
            .iter_mut()
            .zip(other.buffer.iter())
            .for_each(|(a, b)| *a = *a - *b);
    }

    pub fn subtract_into_mut(&mut self, value: N) {
        self.buffer.iter_mut().for_each(|a| *a = *a - value);
    }

    pub fn subtract_into(&self, value: N) -> Self {
        Buffer {
            buffer: self.buffer.iter().map(|a| *a - value).collect(),
        }
    }
}

///////////////////////
// Buffer Multiplication
///////////////////////

impl<N: Num> Mul for Buffer<N> {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        if self.buffer.len() != other.buffer.len() {
            panic!("Buffer lengths do not match");
        }
        Buffer {
            buffer: self
                .buffer
                .iter()
                .zip(other.buffer.iter())
                .map(|(a, b)| *a * *b)
                .collect(),
        }
    }
}

impl<N: Num> Mul<N> for Buffer<N> {
    type Output = Self;
    fn mul(self, other: N) -> Self {
        Buffer {
            buffer: self.buffer.iter().map(|a| *a * other).collect(),
        }
    }
}

impl<N: Num> Buffer<N> {
    pub fn multiply_mut(&mut self, other: &Buffer<N>) {
        self.buffer
            .iter_mut()
            .zip(other.buffer.iter())
            .for_each(|(a, b)| *a = *a * *b);
    }

    pub fn multiply_into_mut(&mut self, scalar: N) {
        self.buffer.iter_mut().for_each(|a| *a = *a * scalar);
    }

    pub fn multiply_into(&self, scalar: N) -> Self {
        Buffer {
            buffer: self.buffer.iter().map(|a| *a * scalar).collect(),
        }
    }
}

///////////////////////
// Buffer Division
///////////////////////

impl<N: Num> Div for Buffer<N> {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        if self.buffer.len() != other.buffer.len() {
            panic!("Buffer lengths do not match");
        }
        Buffer {
            buffer: self
                .buffer
                .iter()
                .zip(other.buffer.iter())
                .map(|(a, b)| *a / *b)
                .collect(),
        }
    }
}

impl<N: Num> Div<N> for Buffer<N> {
    type Output = Self;
    fn div(self, other: N) -> Self {
        Buffer {
            buffer: self.buffer.iter().map(|a| *a / other).collect(),
        }
    }
}

impl<N: Num> Buffer<N> {
    pub fn divide_mut(&mut self, other: &Buffer<N>) {
        self.buffer
            .iter_mut()
            .zip(other.buffer.iter())
            .for_each(|(a, b)| *a = *a / *b);
    }

    pub fn divide_into_mut(&mut self, divisor: N) {
        self.buffer.iter_mut().for_each(|a| *a = *a / divisor);
    }

    pub fn divide_into(&self, divisor: N) -> Self {
        Buffer {
            buffer: self.buffer.iter().map(|a| *a / divisor).collect(),
        }
    }
}

///////////////////////
// Buffer Index
///////////////////////

impl<N: Num> Index<usize> for Buffer<N> {
    type Output = N;
    fn index<'a>(&'_ self, i: usize) -> &'_ N {
        &self.buffer[i]
    }
}

impl<N: Num> IndexMut<usize> for Buffer<N> {
    fn index_mut<'a>(&'_ mut self, i: usize) -> &'_ mut N {
        &mut self.buffer[i]
    }
}

///////////////////////
// Buffer impl
///////////////////////

impl<N: Num> Buffer<N> {
    pub fn new_of_length(len: usize, fill_value: N) -> Self {
        Buffer {
            buffer: (0..len).map(|_| fill_value).collect(),
        }
    }

    // unwrap() used here will allow panic if value is out of bounds
    pub fn put_f64(&mut self, idx: usize, v: f64) {
        self.buffer[idx] = N::from_f64(v).unwrap();
    }

    // unwrap() used here will allow panic if value is out of bounds
    pub fn put_f32(&mut self, idx: usize, v: f32) {
        self.buffer[idx] = N::from_f32(v).unwrap();
    }

    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    pub fn to_vector_u8(&self) -> Vec<u8> {
        self.buffer
            .iter()
            .map(|v| (*v).to_u8().unwrap_or(0))
            .collect()
    }

    pub fn to_vector_u16(&self) -> Vec<u16> {
        self.buffer
            .iter()
            .map(|v| (*v).to_u16().unwrap_or(0))
            .collect()
    }

    pub fn to_vector_f64(&self) -> Vec<f64> {
        self.buffer
            .iter()
            .map(|v| (*v).to_f64().unwrap_or(0.0))
            .collect()
    }

    pub fn get_slice(&self, start_index: usize, stop_index: usize) -> Buffer<N> {
        Buffer {
            buffer: self.buffer[start_index..stop_index].to_vec(),
        }
    }

    pub fn crop_2d(
        &self,
        from_width: usize,
        from_height: usize,
        left_x: usize,
        top_y: usize,
        to_width: usize,
        to_height: usize,
    ) -> Buffer<N> {
        if top_y + to_height > from_height || left_x + to_width > from_width {
            panic!("Crop bounds exceed source array");
        }
        Buffer {
            buffer: iproduct!(0..to_height, 0..to_width)
                .map(|(y, x)| self.buffer[(top_y + y) * from_width + (left_x + x)])
                .collect(),
        }
    }

    pub fn isolate_window_2d(
        &self,
        width_2d: usize,
        height_2d: usize,
        window_size: usize,
        x: usize,
        y: usize,
    ) -> Buffer<N> {
        let start = -(window_size as i32 / 2);
        let end = window_size as i32 / 2 + 1;

        Buffer {
            buffer: iproduct!(start..end, start..end)
                .map(|(_y, _x)| {
                    let get_x = x as i32 + _x;
                    let get_y = y as i32 + _y;
                    (get_y, get_x)
                })
                .filter(|(get_y, get_x)| {
                    *get_x >= 0
                        && *get_x < width_2d as i32
                        && *get_y >= 0
                        && *get_y < height_2d as i32
                })
                .map(|(get_y, get_x)| {
                    let idx = get_y * width_2d as i32 + get_x;
                    self.buffer[idx as usize]
                })
                .collect(),
        }
    }

    pub fn apply_lut(&mut self, lut: &[N]) {
        (0..self.buffer.len()).for_each(|i| {
            let idx = self.buffer[i].to_usize().unwrap();
            self.buffer[i] = lut[idx];
        });
    }

    pub fn sum(&self) -> f64 {
        self.buffer
            .iter()
            .map(|v| (*v).to_f64().unwrap_or(0.0))
            .sum()
    }

    pub fn mean(&self) -> f64 {
        self.sum() / self.buffer.len() as f64
    }

    pub fn variance(&self) -> f64 {
        let m = self.mean();
        let sqdiff: f64 = self
            .buffer
            .iter()
            .map(|v| (*v).to_f64().unwrap_or(0.0))
            .map(|v| (v - m) * (v - m))
            .sum();
        sqdiff / self.buffer.len() as f64
    }

    pub fn xcorr(&self, other: &Buffer<N>) -> f64 {
        if self.len() != other.len() {
            panic!("Arrays need to be the same length (for now)");
        }
        let m_x = self.mean();
        let m_y = other.mean();
        let v_x = self.variance();
        let v_y = other.variance();

        let s: f64 = (0..self.len())
            .map(|n| {
                (self.buffer[n].to_f64().unwrap_or(0.0) - m_x)
                    * (other.buffer[n].to_f64().unwrap_or(0.0) - m_y)
            })
            .sum();
        1.0 / self.len() as f64 * s / (v_x * v_y).sqrt()
    }

    pub fn stddev(&self) -> f64 {
        self.variance().sqrt()
    }

    pub fn z_score(&self, check_value: f64) -> f64 {
        (check_value - self.mean()) / self.stddev()
    }

    pub fn power(&self, exponent: N) -> Self {
        let mut b = self.clone();
        b.power_mut(exponent);
        b
    }

    pub fn power_mut(&mut self, exponent: N) {
        let xp: f64 = exponent.to_f64().unwrap();
        (0..self.len())
            .for_each(|i| self[i] = N::from_f64(self[i].to_f64().unwrap().powf(xp)).unwrap());
    }

    fn clip(&self, clip_min: N, clip_max: N) -> Self {
        let mut b = self.clone();
        b.clip_mut(clip_min, clip_max);
        b
    }

    fn clip_mut(&mut self, clip_min: N, clip_max: N) {
        let mn: f64 = clip_min.to_f64().unwrap();
        let mx: f64 = clip_max.to_f64().unwrap();
        (0..self.len()).for_each(|i| {
            let v = self[i].to_f64().unwrap();
            self[i] = if v > mx {
                clip_max
            } else if v < mn {
                clip_min
            } else {
                self[i]
            };
        });
    }

    pub fn min(&self) -> N {
        let mut min = f64::MAX;
        (0..self.len()).for_each(|i| {
            let v = self[i].to_f64().unwrap();
            min = v.min(min);
        });
        N::from_f64(min).unwrap()
    }

    pub fn max(&self) -> N {
        let mut max = f64::MIN;
        (0..self.len()).for_each(|i| {
            let v = self[i].to_f64().unwrap();
            max = v.max(max);
        });
        N::from_f64(max).unwrap()
    }

    pub fn range(&self) -> Range<N> {
        Range {
            min: self.min(),
            max: self.max(),
        }
    }
}

pub struct Range<N: Num> {
    pub min: N,
    pub max: N,
}

#[derive(Clone)]
struct ImageBuffer<N: Num> {
    buffer: Buffer<N>,
    width: usize,
    height: usize,
}

////////////
// ImageBuffer Addition
////////////

impl<N: Num> Add for ImageBuffer<N> {
    type Output = Self;

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

    fn add(self, other: N) -> Self {
        ImageBuffer {
            buffer: self.buffer + other,
            width: self.width,
            height: self.height,
        }
    }
}

impl<N: Num> ImageBuffer<N> {
    pub fn add_mut(&mut self, other: &ImageBuffer<N>) {
        self.buffer.add_mut(&other.buffer);
    }

    pub fn add_into(&self, other: N) -> Self {
        ImageBuffer {
            width: self.width,
            height: self.height,
            buffer: self.buffer.add_into(other),
        }
    }

    pub fn add_into_mut(&mut self, other: N) {
        self.buffer.add_into(other);
    }
}

////////////
// ImageBuffer Subtraction
////////////

impl<N: Num> Sub for ImageBuffer<N> {
    type Output = Self;

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

    fn sub(self, other: N) -> Self {
        ImageBuffer {
            buffer: self.buffer - other,
            width: self.width,
            height: self.height,
        }
    }
}

impl<N: Num> ImageBuffer<N> {
    pub fn subtract_mut(&mut self, other: &ImageBuffer<N>) {
        self.buffer.subtract_mut(&other.buffer);
    }

    pub fn subtract_into(&self, other: N) -> Self {
        ImageBuffer {
            width: self.width,
            height: self.height,
            buffer: self.buffer.subtract_into(other),
        }
    }

    pub fn subtract_into_mut(&mut self, other: N) {
        self.buffer.subtract_into(other);
    }
}

////////////
// ImageBuffer Multiplication
////////////

impl<N: Num> Mul for ImageBuffer<N> {
    type Output = Self;

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

    fn mul(self, other: N) -> Self {
        ImageBuffer {
            buffer: self.buffer * other,
            width: self.width,
            height: self.height,
        }
    }
}

impl<N: Num> ImageBuffer<N> {
    pub fn multiply_mut(&mut self, other: &ImageBuffer<N>) {
        self.buffer.multiply_mut(&other.buffer);
    }

    pub fn multiply_into(&self, other: N) -> Self {
        ImageBuffer {
            width: self.width,
            height: self.height,
            buffer: self.buffer.multiply_into(other),
        }
    }

    pub fn multiply_into_mut(&mut self, other: N) {
        self.buffer.multiply_into(other);
    }
}

////////////
// ImageBuffer Division
////////////

impl<N: Num> Div for ImageBuffer<N> {
    type Output = Self;

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

    fn div(self, other: N) -> Self {
        ImageBuffer {
            buffer: self.buffer / other,
            width: self.width,
            height: self.height,
        }
    }
}

impl<N: Num> ImageBuffer<N> {
    pub fn divide_mut(&mut self, other: &ImageBuffer<N>) {
        self.buffer.divide_mut(&other.buffer);
    }

    pub fn divide_into(&self, other: N) -> Self {
        ImageBuffer {
            width: self.width,
            height: self.height,
            buffer: self.buffer.divide_into(other),
        }
    }

    pub fn divide_into_mut(&mut self, other: N) {
        self.buffer.divide_into(other);
    }
}

////////////////////
// ImageBuffer impl
////////////////////

impl<N: Num> ImageBuffer<N> {
    pub fn new(width: usize, height: usize, fill_value: N) -> Self {
        if width < 0 || height < 0 {
            panic!("Cannot have a negative length image dimension");
        }
        ImageBuffer {
            width,
            height,
            buffer: Buffer::<N>::new_of_length(width * height, fill_value),
        }
    }

    fn dims_match(&self, other: &ImageBuffer<N>) -> bool {
        self.width == other.width && self.height == other.height
    }

    fn xy_to_idx(&self, x: usize, y: usize) -> usize {
        y * self.width + x
    }

    #[inline(always)]
    pub fn get(&self, x: usize, y: usize) -> N {
        self.buffer.buffer[self.xy_to_idx(x, y)]
    }

    pub fn get_f64(&self, x: usize, y: usize) -> f64 {
        self.buffer.buffer[self.xy_to_idx(x, y)]
            .to_f64()
            .unwrap_or(0.0)
    }

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
            let v = v1 * xd + v0 * (1.0 - xd);

            v
        } else {
            panic!("Invalid pixel coordinates");
        }
    }

    pub fn set(&mut self, x: usize, y: usize, value: N) {
        let idx = self.xy_to_idx(x, y);
        self.buffer.buffer[idx] = value;
    }

    pub fn to_vector_u8(&self) -> Vec<u8> {
        self.buffer.to_vector_u8()
    }

    pub fn to_vector_u16(&self) -> Vec<u16> {
        self.buffer.to_vector_u16()
    }

    pub fn to_vector_f64(&self) -> Vec<f64> {
        self.buffer.to_vector_f64()
    }

    /// Retrieves a horizontal slice subframe of the image for pushframe cameras (e.g. JunoCam raws as a stack of slices)
    ///
    pub fn get_slice(&self, top_y: usize, len: usize) -> ImageBuffer<N> {
        let start_index = top_y * self.width;
        let stop_index = (top_y + len) * self.width;

        ImageBuffer {
            buffer: self.buffer.get_slice(start_index, stop_index),
            width: self.width,
            height: len,
        }
    }

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

    pub fn isolate_window(&self, window_size: usize, x: usize, y: usize) -> Buffer<N> {
        self.buffer
            .isolate_window_2d(self.width, self.height, window_size, x, y)
    }

    pub fn apply_lut(&mut self, lut: &[N]) {
        self.buffer.apply_lut(lut);
    }

    pub fn sum(&self) -> f64 {
        self.buffer.sum()
    }

    pub fn mean(&self) -> f64 {
        self.buffer.mean()
    }

    pub fn variance(&self) -> f64 {
        self.buffer.variance()
    }

    pub fn xcorr(&self, other: &ImageBuffer<N>) -> f64 {
        self.buffer.xcorr(&other.buffer)
    }

    pub fn stddev(&self) -> f64 {
        self.buffer.stddev()
    }

    pub fn z_score(&self, check_value: f64) -> f64 {
        self.buffer.z_score(check_value)
    }

    pub fn power(&self, exponent: N) -> Self {
        ImageBuffer {
            width: self.width,
            height: self.height,
            buffer: self.buffer.power(exponent),
        }
    }

    pub fn power_mut(&mut self, exponent: N) {
        self.buffer.power_mut(exponent);
    }

    fn clip(&self, clip_min: N, clip_max: N) -> Self {
        ImageBuffer {
            width: self.width,
            height: self.height,
            buffer: self.buffer.clip(clip_min, clip_max),
        }
    }

    fn clip_mut(&mut self, clip_min: N, clip_max: N) {
        self.buffer.clip(clip_min, clip_max);
    }

    pub fn min(&self) -> N {
        self.buffer.min()
    }

    pub fn max(&self) -> N {
        self.buffer.max()
    }

    pub fn range(&self) -> Range<N> {
        self.buffer.range()
    }
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

type Buffer4ByteFloat = Buffer<f32>;
type Buffer8ByteFloat = Buffer<f64>;

type ImageBuffer4ByteFloat = ImageBuffer<f32>;
type ImageBuffer8ByteFloat = ImageBuffer<f64>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buffer() {
        let a = Buffer::new_of_length(10, 1.0_f32);
        let mut b0 = Buffer4ByteFloat::new_of_length(10, 1.0);
        let b1 = Buffer4ByteFloat::new_of_length(10, 2.0);

        b0.add_mut(&b1);
        let b3 = b0 + b1;
        let b4 = b3 + 5.0;

        let v0 = b4.to_vector_u8();
        assert_eq!(v0[1], 10);
    }

    #[test]
    fn test_imagebuffer() {
        let image0 = ImageBuffer4ByteFloat::new(1024, 1024, 0.0);
        let image1 = ImageBuffer4ByteFloat::new(1024, 1024, 0.0);
        let image3 = image0 + image1;
        let image4 = image3 + 5.0;

        let v0 = image4.get(0, 0) as u8;

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
}
