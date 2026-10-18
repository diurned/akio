mod ffi;
use crate::ffi::*;

fn main() {
    unsafe { ggml_backend_load_all() };
}
