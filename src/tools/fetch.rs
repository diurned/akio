use anyhow::{anyhow, Result};
use reqwest::Url;
use serde_json::{json, Value};

use super::Tool;

pub struct FetchTool;

impl FetchTool {
    fn fetch(&self, url: &str) -> Result<String> {
        let url = Url::parse(url).map_err(|e| anyhow!("invalid URL: {e}"))?;
        let url_string = url.to_string();
        let (status, body) = std::thread::spawn(move || -> Result<(u16, String)> {
            let response = reqwest::blocking::Client::builder()
                .user_agent("Mozilla/5.0 (compatible; akio/0.1)")
                .build()?
                .get(&url_string)
                .send()?;
            let status = response.status().as_u16();
            let body = response.text()?;
            Ok((status, body))
        })
        .join()
        .map_err(|_| anyhow!("fetch thread panicked"))??;

        const MAX_LEN: usize = 8000;
        let body = if body.len() > MAX_LEN {
            format!(
                "{}\n\n... [truncated, {} of {} bytes shown]",
                &body[..MAX_LEN],
                MAX_LEN,
                body.len()
            )
        } else {
            body
        };

        if body.trim().is_empty() {
            return Ok(format!("(status {status}) empty response body"));
        }

        Ok(format!("(status {status})\n{body}"))
    }
}

impl Tool for FetchTool {
    fn name(&self) -> &str {
        "fetch"
    }

    fn description(&self) -> &str {
        "Fetch the contents of a URL via an HTTP GET request."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to fetch."
                }
            },
            "required": ["url"]
        })
    }

    fn execute(&self, args: Value) -> Result<String> {
        let url = args["url"]
            .as_str()
            .ok_or_else(|| anyhow!("missing required argument: url"))?;

        self.fetch(url)
    }
}
