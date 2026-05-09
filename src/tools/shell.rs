use anyhow::Result;
use serde_json::{json, Value};

use super::Tool;

pub struct ShellTool;

impl Tool for ShellTool {
    fn name(&self) -> &str {
        "shell"
    }

    fn description(&self) -> &str {
        "Execute a shell command and return its combined stdout and stderr output."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to run (executed via /bin/sh -c)."
                }
            },
            "required": ["command"]
        })
    }

    fn execute(&self, args: Value) -> Result<String> {
        let command = args["command"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("missing required argument: command"))?;

        let output = std::process::Command::new("/bin/sh")
            .arg("-c")
            .arg(command)
            .output()?;

        let mut result = String::new();
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);

        if !stdout.is_empty() {
            result.push_str(&stdout);
        }
        if !stderr.is_empty() {
            if !result.is_empty() {
                result.push('\n');
            }
            result.push_str("[stderr]\n");
            result.push_str(&stderr);
        }
        if !output.status.success() {
            result.push_str(&format!("\n[exit code: {}]", output.status));
        }
        if result.is_empty() {
            result.push_str("(no output)");
        }
        Ok(result)
    }
}
