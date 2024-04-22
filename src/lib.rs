use std::ops::{Add, Div, Index, IndexMut, Mul, Sub};

use itertools::iproduct;

/// Represents a valid number to be used as a generic constraint in `Buffer` and `ImageBuffer`.
pub trait Num:
    num_traits::Num + num_traits::ToPrimitive + num_traits::FromPrimitive + Copy
{
}

/// Represents a one-dimensional single-band buffer.
#[derive(Clone)]
pub struct Buffer<N: Num> {
    buffer: Vec<N>,
}

///////////////////////
// Buffer Addition
///////////////////////

impl<N: Num> Add for Buffer<N> {
    type Output = Self;

    /// Performs the + operation for `Buffer<N>`.
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

    /// Performs the + operation for `Buffer<N>` and a single value of type `N`.
    fn add(self, other: N) -> Self {
        Buffer {
            buffer: self.buffer.iter().map(|a| *a + other).collect(),
        }
    }
}

impl<N: Num> Buffer<N> {
    /// Adds the values of matching-length Buffer into this Buffer.
    pub fn add_mut(&mut self, other: &Buffer<N>) {
        self.buffer
            .iter_mut()
            .zip(other.buffer.iter())
            .for_each(|(a, b)| *a = *a + *b);
    }

    /// Adds value of type `N` to each value in this Buffer.
    pub fn add_into_mut(&mut self, value: N) {
        self.buffer.iter_mut().for_each(|a| *a = *a + value);
    }

    /// Adds the value of type `N` to each value of a copy of this Buffer.
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

    /// Performs the - operation for `Buffer<N>`.
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

    /// Performs the - operation for `Buffer<N>` and a single value of type `N`.
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

    /// Performs the * operation for `Buffer<N>`.
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

    /// Performs the * operation for `Buffer<N>` and a single value of type `N`.
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

    /// Performs the / operation for `Buffer<N>`.
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

    /// Performs the / operation for `Buffer<N>` and a single value of type `N`.
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
// Buffer Iterator
///////////////////////

pub struct BufferIter<'a, N: Num> {
    vec: &'a Buffer<N>,
    curr: usize,
    next: usize,
}

impl<N: Num> BufferIter<'_, N> {
    pub fn new(vec: &Buffer<N>) -> BufferIter<'_, N> {
        BufferIter {
            vec,
            curr: 0,
            next: 1,
        }
    }
}

impl<N: Num> Iterator for BufferIter<'_, N> {
    type Item = N;

    fn next(&mut self) -> Option<Self::Item> {
        let v = if self.curr >= self.vec.len() {
            None
        } else {
            Some(self.vec[self.curr])
        };

        let new_next = self.next + 1;
        self.curr = self.next;
        self.next = new_next;

        v
    }
}

///////////////////////
// Buffer impl
///////////////////////

impl<N: Num> Buffer<N> {
    /// Creates a new buffer object. It is preinitialized of lenth `len` and filled with `fill_value`.
    pub fn new_of_length(len: usize, fill_value: N) -> Self {
        Buffer {
            buffer: (0..len).map(|_| fill_value).collect(),
        }
    }

    /// Inserts a value of type `f64` into index `idx`.
    ///
    /// **panics** If value is incompatible with type `N` or `idx` is out of bounds.
    pub fn put_f64(&mut self, idx: usize, v: f64) {
        self.buffer[idx] = N::from_f64(v).unwrap();
    }

    /// Inserts a value of type `f32` into index `idx`.
    ///
    /// **panics** If value is incompatible with type `N` or `idx` is out of bounds.
    pub fn put_f32(&mut self, idx: usize, v: f32) {
        self.buffer[idx] = N::from_f32(v).unwrap();
    }

    /// Returns the length of the buffer.
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Creates a `Vec<u8>` with the content of this buffer. Each value is casted to `u8`.
    ///
    /// Values are not pre-normalized into a valid `u8` (0-255) range.
    ///
    /// **panics** If a value in this buffer is incompatible with `u8`
    pub fn to_vector_u8(&self) -> Vec<u8> {
        self.buffer.iter().map(|v| (*v).to_u8().unwrap()).collect()
    }

    /// Creates a `Vec<u16>` with the content of this buffer. Each value is casted to `u16`.
    ///
    /// Values are not pre-normalized into a valid `u8` (0-65535) range.
    ///
    /// **panics** If a value in this buffer is incompatible with `u16`
    pub fn to_vector_u16(&self) -> Vec<u16> {
        self.buffer.iter().map(|v| (*v).to_u16().unwrap()).collect()
    }

    /// Creates a `Vec<u64>` with the content of this buffer. Each value is casted to `u64`.
    ///
    /// Values are not pre-normalized into a valid `u64` range.
    ///
    /// **panics** If a value in this buffer is incompatible with `u64`
    pub fn to_vector_f64(&self) -> Vec<f64> {
        self.buffer
            .iter()
            .map(|v| (*v).to_f64().unwrap_or(0.0))
            .collect()
    }

    /// Retrieves a sub-section of the buffer of length `stop_index - start_index`.
    ///
    /// **panics** If indexes are out of bounds.
    pub fn get_slice(&self, start_index: usize, stop_index: usize) -> Buffer<N> {
        Buffer {
            buffer: self.buffer[start_index..stop_index].to_vec(),
        }
    }

    /// Creates a cropped copy of the buffer, treating it as a two-dimensional matrix.
    ///
    /// **panics** If the parameters lead to out-of-bounds indexing errors.
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

    /// Isolates (crops) a square copy of the array centered on coordinates `x & y`. Treats the buffer as a two-dimensional matrix.
    /// The size of the window is determined from `window_size`.
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

    /// Applies a look-up table of the same time. This assumes the pre-LUT values in this buffer
    /// are proper indexes into the LUT table. The buffer values will be replaced with the values
    /// contained in the LUT table.
    ///
    /// **panics** if a pre-LUT value in this buffer is an invalid index (out of bounds) or is
    /// incompatible with type `usize`.
    pub fn apply_lut(&mut self, lut: &[N]) {
        (0..self.buffer.len()).for_each(|i| {
            let idx = self.buffer[i].to_usize().unwrap();
            self.buffer[i] = lut[idx];
        });
    }

    /// Computes the sum of all values in the buffer.
    ///
    /// **panics** If any value is incompatible with type `f64`.
    pub fn sum(&self) -> f64 {
        self.buffer.iter().map(|v| (*v).to_f64().unwrap()).sum()
    }

    /// Computes the mean (average) of all values in the buffer.
    ///
    /// **panics** If any value is incompatible with type `f64`.
    pub fn mean(&self) -> f64 {
        self.sum() / self.buffer.len() as f64
    }

    /// Computes the statistical variance of all values in the buffer.
    ///
    /// **panics** If any value is incompatible with type `f64`.
    pub fn variance(&self) -> f64 {
        let m = self.mean();
        let sqdiff: f64 = self
            .buffer
            .iter()
            .map(|v| (*v).to_f64().unwrap())
            .map(|v| (v - m) * (v - m))
            .sum();
        sqdiff / self.buffer.len() as f64
    }

    /// Computes the statistical cross-correlation of this and `other` buffer.
    ///
    /// **panics** If any value is incompatible with type `f64` or buffer lengths do not match.
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
                (self.buffer[n].to_f64().unwrap() - m_x) * (other.buffer[n].to_f64().unwrap() - m_y)
            })
            .sum();
        1.0 / self.len() as f64 * s / (v_x * v_y).sqrt()
    }

    /// Computes the standard deviation of the values in the buffer.
    ///
    /// **panics** If any value in the buffer is incompatible with `f64`.
    pub fn stddev(&self) -> f64 {
        self.variance().sqrt()
    }

    /// Computes the z-score (standard score) of the values in the buffer.
    ///
    /// **panics** If any value in the buffer is incompatible with `f64`.
    pub fn z_score(&self, check_value: f64) -> f64 {
        (check_value - self.mean()) / self.stddev()
    }

    /// Creates a copy of this buffer with each value raised to the power of `exponent`.
    ///
    /// **panics** If any value is incompatible with type `f64`.
    pub fn power(&self, exponent: N) -> Self {
        let mut b = self.clone();
        b.power_mut(exponent);
        b
    }

    /// Replaces all values in this buffer with each raised to the power of `exponent`.
    ///
    /// **panics** If any value in the buffer in incompatible with type `f64`.
    pub fn power_mut(&mut self, exponent: N) {
        let xp: f64 = exponent.to_f64().unwrap();
        (0..self.len())
            .for_each(|i| self[i] = N::from_f64(self[i].to_f64().unwrap().powf(xp)).unwrap());
    }

    /// Creates a copy of this buffer with each value clipped to between `clip_min` at a minimum
    /// and `clip_max` at a maximum. If any value exceeds the bounds, it will be replaced with that
    /// respective bound value.
    ///
    /// **panics** If any value is incompatible with type `f64`.
    fn clip(&self, clip_min: N, clip_max: N) -> Self {
        let mut b = self.clone();
        b.clip_mut(clip_min, clip_max);
        b
    }

    /// Replaces each value in this buffer with each value clipped to between `clip_min` at a minimum
    /// and `clip_max` at a maximum. If any value exceeds the bounds, it will be replaced with that
    /// respective bound value.
    ///
    /// **panics** If any value is incompatible with type `f64`.
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

    /// Determines the minimum value contained in this buffer.
    ///
    /// **panics** If any value is incompatible with type `f64`.
    pub fn min(&self) -> N {
        let mut min = f64::MAX;
        (0..self.len()).for_each(|i| {
            let v = self[i].to_f64().unwrap();
            min = v.min(min);
        });
        N::from_f64(min).unwrap()
    }

    /// Determines the maximum value contained in this buffer.
    ///
    /// **panics** If any value is incompatible with type `f64`.
    pub fn max(&self) -> N {
        let mut max = f64::MIN;
        (0..self.len()).for_each(|i| {
            let v = self[i].to_f64().unwrap();
            max = v.max(max);
        });
        N::from_f64(max).unwrap()
    }

    /// Determines the minimum and maximum value in this buffer.
    ///
    /// **panics** If any value is incompatible with type `f64`.
    pub fn range(&self) -> Range<N> {
        Range {
            min: self.min(),
            max: self.max(),
        }
    }

    /// Returns an iterator to be used to cycle through the buffer values.
    pub fn iter(&self) -> BufferIter<'_, N> {
        BufferIter::new(self)
    }
}

/// Represents a minimum/maximum range of values.
pub struct Range<N: Num> {
    pub min: N,
    pub max: N,
}

/// Represents a two-dimensional single-channel image buffer.
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
    fn dims_match(&self, other: &ImageBuffer<N>) -> bool {
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
            let v = v1 * xd + v0 * (1.0 - xd);

            v
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
    fn clip(&self, clip_min: N, clip_max: N) -> Self {
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
    fn clip_mut(&mut self, clip_min: N, clip_max: N) {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buffer() {
        _ = Buffer::new_of_length(10, 1.0_f32);
        let mut b0 = Buffer4ByteFloat::new_of_length(10, 1.0);
        let b1 = Buffer4ByteFloat::new_of_length(10, 2.0);

        b0.add_mut(&b1);
        let b3 = b0 + b1;
        let b4 = b3 + 5.0;

        let v0 = b4.to_vector_u8();
        assert_eq!(v0[1], 10);

        let b5 = Buffer4ByteFloat::new_of_length(100, 2.0);
        b5.iter().for_each(|v| assert_eq!(v, 2.0));
    }

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
}
