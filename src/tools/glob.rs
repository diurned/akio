use anyhow::Result;
use serde_json::{json, Value};

use super::Tool;

pub struct GlobTool;

impl Tool for GlobTool {
    fn name(&self) -> &str {
        "glob"
    }

    fn description(&self) -> &str {
        "Return a list of file paths that match a glob pattern (e.g. \"src/**/*.rs\")."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "A glob pattern such as \"**/*.txt\" or \"src/**/*.rs\"."
                }
            },
            "required": ["pattern"]
        })
    }

    fn execute(&self, args: Value) -> Result<String> {
        let pattern = args["pattern"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("missing required argument: pattern"))?;

        let mut paths: Vec<String> = glob::glob(pattern)?
            .filter_map(|entry| entry.ok())
            .map(|p| p.display().to_string())
            .collect();
        paths.sort();

        if paths.is_empty() {
            return Ok("(no matches)".into());
        }
        Ok(paths.join("\n"))
    }
}
