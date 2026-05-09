use anyhow::{anyhow, Result};
use regex::Regex;
use reqwest::Url;
use serde_json::{json, Value};

use super::Tool;

pub struct WebSearchTool;

struct SearchResult {
    title: String,
    url: String,
    snippet: String,
}

fn extract_url(href: &str) -> String {
    // DuckDuckGo wraps URLs like //duckduckgo.com/l/?uddg=ENCODED_URL&...
    let full = if href.starts_with("//") {
        format!("https:{href}")
    } else {
        href.to_string()
    };
    if let Ok(parsed) = Url::parse(&full) {
        if let Some((_, uddg)) = parsed.query_pairs().find(|(k, _)| k == "uddg") {
            return uddg.to_string();
        }
    }
    href.to_string()
}

impl WebSearchTool {
    fn search(&self, query: &str, num_results: usize) -> Result<Vec<SearchResult>> {
        let mut url = Url::parse("https://html.duckduckgo.com/html")?;
        url.query_pairs_mut().append_pair("q", query);
        let url_string = url.to_string();

        // reqwest::blocking spawns its own Tokio runtime, which panics when called
        // from inside an existing async runtime. Spawn a plain OS thread to escape it.
        let response = std::thread::spawn(move || -> Result<String> {
            Ok(reqwest::blocking::Client::builder()
                .user_agent("Mozilla/5.0 (compatible; akio/0.1)")
                .build()?
                .get(&url_string)
                .send()?
                .text()?)
        })
        .join()
        .map_err(|_| anyhow!("search thread panicked"))??;

        let title_re = Regex::new(r#"<a[^>]*class="result__a"[^>]*>(.*?)</a>"#)?;
        let url_re = Regex::new(r#"<a[^>]*class="result__a"[^>]*href="([^"]*)""#)?;
        let snippet_re = Regex::new(r#"(?s)<a[^>]*class="result__snippet"[^>]*>(.*?)</a>"#)?;
        let bold_re = Regex::new(r"</?b>")?;

        let mut results = Vec::new();

        for chunk in response.split("<div class=\"result results_links").skip(1) {
            if results.len() >= num_results {
                break;
            }

            let title = title_re
                .captures(chunk)
                .and_then(|c| c.get(1))
                .map(|m| m.as_str().trim().to_string());

            let raw_url = url_re
                .captures(chunk)
                .and_then(|c| c.get(1))
                .map(|m| m.as_str().to_string());

            let snippet = snippet_re
                .captures(chunk)
                .and_then(|c| c.get(1))
                .map(|m| bold_re.replace_all(m.as_str(), "").trim().to_string());

            if let (Some(title), Some(raw_url), Some(snippet)) = (title, raw_url, snippet) {
                let url = extract_url(&raw_url);
                if !title.is_empty() && !url.is_empty() && !snippet.is_empty() {
                    results.push(SearchResult { title, url, snippet });
                }
            }
        }

        Ok(results)
    }
}

impl Tool for WebSearchTool {
    fn name(&self) -> &str {
        "websearch"
    }

    fn description(&self) -> &str {
        "Search the web using DuckDuckGo and return results."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to perform."
                },
                "num_results": {
                    "type": "integer",
                    "description": "Number of results to return (default: 2)."
                }
            },
            "required": ["query"]
        })
    }

    fn execute(&self, args: Value) -> Result<String> {
        let query = args["query"]
            .as_str()
            .ok_or_else(|| anyhow!("missing required argument: query"))?;

        let num_results = args["num_results"]
            .as_u64()
            .unwrap_or(2) as usize;

        let results = self.search(query, num_results)?;

        if results.is_empty() {
            return Ok("No results found.".to_string());
        }

        let mut output = String::new();
        for (i, result) in results.iter().enumerate() {
            output.push_str(&format!(
                "{}. {}\n   URL: {}\n   {}\n\n",
                i + 1,
                result.title.trim(),
                result.url,
                result.snippet.trim()
            ));
        }

        Ok(output.trim_end().to_string())
    }
}
