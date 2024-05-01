use std::ops::{Add, Div, Index, IndexMut, Mul, Sub};

use itertools::iproduct;

use crate::Num;
use crate::Range;

/// Represents a one-dimensional single-band buffer.
#[derive(Clone)]
pub struct Buffer<N: Num> {
    pub buffer: Vec<N>,
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
    /// Adds the values of matching-length Buffer into this `Buffer`.
    pub fn add_mut(&mut self, other: &Buffer<N>) {
        self.buffer
            .iter_mut()
            .zip(other.buffer.iter())
            .for_each(|(a, b)| *a = *a + *b);
    }

    /// Adds value of type `N` to each value in this `Buffer`.
    pub fn add_into_mut(&mut self, value: N) {
        self.buffer.iter_mut().for_each(|a| *a = *a + value);
    }

    /// Adds the value of type `N` to each value of a copy of this `Buffer`.
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
    /// Subtracts the values of matching-length Buffer into this `Buffer`.
    pub fn subtract_mut(&mut self, other: &Buffer<N>) {
        self.buffer
            .iter_mut()
            .zip(other.buffer.iter())
            .for_each(|(a, b)| *a = *a - *b);
    }

    /// Subtracts value of type `N` from each value in this `Buffer`.
    pub fn subtract_into_mut(&mut self, value: N) {
        self.buffer.iter_mut().for_each(|a| *a = *a - value);
    }

    /// Subtracts the value of type `N` from each value of a copy of this `Buffer`.
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
    /// Multiplies the values of matching-length Buffer into this `Buffer`.
    pub fn multiply_mut(&mut self, other: &Buffer<N>) {
        self.buffer
            .iter_mut()
            .zip(other.buffer.iter())
            .for_each(|(a, b)| *a = *a * *b);
    }

    /// Multiplies value of type `N` with each value in this `Buffer`.
    pub fn multiply_into_mut(&mut self, scalar: N) {
        self.buffer.iter_mut().for_each(|a| *a = *a * scalar);
    }

    /// Multiplies the value of type `N` with each value of a copy of this `Buffer`.
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
    /// Divides the values of matching-length Buffer into this `Buffer`.
    pub fn divide_mut(&mut self, other: &Buffer<N>) {
        self.buffer
            .iter_mut()
            .zip(other.buffer.iter())
            .for_each(|(a, b)| *a = *a / *b);
    }

    /// Divides value of type `N` from each value in this `Buffer`.
    pub fn divide_into_mut(&mut self, divisor: N) {
        self.buffer.iter_mut().for_each(|a| *a = *a / divisor);
    }

    /// Divides the value of type `N` from each value of a copy of this `Buffer`.
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

    /// Performs the indexing (`container[index]`) operation.
    ///
    /// **Panics** If the index is out of bounds.
    fn index<'a>(&'_ self, i: usize) -> &'_ N {
        &self.buffer[i]
    }
}

impl<N: Num> IndexMut<usize> for Buffer<N> {
    /// Performs the mutable indexing (`container[index]`) operation.
    ///
    /// **Panics** If the index is out of bounds.
    fn index_mut<'a>(&'_ mut self, i: usize) -> &'_ mut N {
        &mut self.buffer[i]
    }
}

///////////////////////
// Buffer Iterator
///////////////////////

/// Represents an iterator across values contained in an instance of `Buffer<N>`.
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

    /// Returns true is the buffer is zero-length
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Creates a new `Buffer` from an array of values of type `N`.
    pub fn from_vector(values: &[N]) -> Self {
        Buffer {
            buffer: values.to_vec(),
        }
    }

    /// Creates a new `Buffer` from an array of values of type `u8`.
    ///
    /// **panics** If `u8` is incompatible with type `N`.
    pub fn from_vector_u8(values: &[u8]) -> Self {
        Buffer {
            buffer: values.iter().map(|v| N::from_u8(*v).unwrap()).collect(),
        }
    }

    /// Creates a new `Buffer` from an array of values of type `u16`.
    ///
    /// **panics** If `u16` is incompatible with type `N`.
    pub fn from_vector_u16(values: &[u16]) -> Self {
        Buffer {
            buffer: values.iter().map(|v| N::from_u16(*v).unwrap()).collect(),
        }
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
    pub fn clip(&self, clip_min: N, clip_max: N) -> Self {
        let mut b = self.clone();
        b.clip_mut(clip_min, clip_max);
        b
    }

    /// Replaces each value in this buffer with each value clipped to between `clip_min` at a minimum
    /// and `clip_max` at a maximum. If any value exceeds the bounds, it will be replaced with that
    /// respective bound value.
    ///
    /// **panics** If any value is incompatible with type `f64`.
    pub fn clip_mut(&mut self, clip_min: N, clip_max: N) {
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

#[cfg(test)]
mod tests {
    use crate::Buffer4ByteFloat;

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
}
