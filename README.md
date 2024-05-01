# imagebuffer-generics

A prototype redesign of the `ImageBuffer` struct in the `sciimg` repo that uses generics for dynamic typing.
Types are restricted to primitive numericals compatible with `f64`.

## Examples:

```rust
fn create_black_image() -> Result<()> {

    // Create an image of width 1024 by height 1024. Initialize all
    // pixel values to zero.
    let myimage = ImageBuffer4ByteFloat::new(1024, 1024, 0.0);

    // Save the image to the target directory as an 8 bit PNG.
    image0.save_image_to(
        "target/testsave_image0_8bit",
        OutputFormat::PNG,
        ImageMode::U8BIT,
    )?;

    // Check that the output file indeed exists.
    assert!(Path::new("target/testsave_image0_8bit.png").exists());

    Ok(())
}

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

fn create_gray_with_math_1byte() {
    // Create a 1024x1024 image, all white
    let myimage = ImageBuffer1ByteInt::new(1024, 1024, 255);

    // gray = 255 / 2
    let grayimage = myimage / 2;

    // Check the resulting value
    assert_eq!(grayimage.get(0, 0), 127);
}

fn open_and_save_rgb_image() -> Result<()> {
    // Opens an image using unsigned byte format
    let a = Image::<u8>::open("assets/test-image.jpg")?;

    // Saves as an unsigned byte JPEG
    a.save_to(
        "target/testsave_image_8bit_jpg_a",
        OutputFormat::JPEG,
        ImageMode::U8BIT,
    )?;

    // Check that the image exists
    assert!(Path::new("target/testsave_image_8bit_jpg_a.jpg").exists());

    Ok(())
}
```


