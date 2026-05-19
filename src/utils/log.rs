use crate::ffi::*;

pub fn parse_log_level(s: &str) -> ggml_log_level {
    match s.to_ascii_lowercase().as_str() {
        "none" => ggml_log_level_GGML_LOG_LEVEL_NONE,
        "debug" => ggml_log_level_GGML_LOG_LEVEL_DEBUG,
        "info" => ggml_log_level_GGML_LOG_LEVEL_INFO,
        "warn" => ggml_log_level_GGML_LOG_LEVEL_WARN,
        "error" => ggml_log_level_GGML_LOG_LEVEL_ERROR,
        _ => ggml_log_level_GGML_LOG_LEVEL_ERROR,
    }
}
