use std::ffi::CStr;
use std::sync::atomic::{AtomicU32, Ordering};

use crate::ffi::*;

/// Global minimum log level.  Messages with a level below this threshold are
/// silently dropped.  The special `CONT` level (5) inherits the level of the
/// preceding message, so we track that too.
static MIN_LEVEL: AtomicU32 = AtomicU32::new(ggml_log_level_GGML_LOG_LEVEL_ERROR);
static LAST_LEVEL: AtomicU32 = AtomicU32::new(ggml_log_level_GGML_LOG_LEVEL_NONE);

/// Set the minimum log level that will be printed to stderr.
pub fn set_min_level(level: ggml_log_level) {
    MIN_LEVEL.store(level, Ordering::Relaxed);
}

/// Install this as the llama.cpp / ggml log callback via `llama_log_set`.
pub unsafe extern "C" fn log_callback(
    level: ggml_log_level,
    text: *const ::std::os::raw::c_char,
    _user_data: *mut ::std::os::raw::c_void,
) {
    if text.is_null() {
        return;
    }

    // CONT continues the previous message at whatever level it had.
    let effective = if level == ggml_log_level_GGML_LOG_LEVEL_CONT {
        LAST_LEVEL.load(Ordering::Relaxed)
    } else {
        LAST_LEVEL.store(level, Ordering::Relaxed);
        level
    };

    if effective >= MIN_LEVEL.load(Ordering::Relaxed) {
        let s = CStr::from_ptr(text).to_string_lossy();
        eprint!("{s}");
    }
}
