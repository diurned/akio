use anyhow::{anyhow, Result};
use reqwest::Url;
use scraper::{Html, Selector};
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
        let query = query.to_string();

        // Run the blocking HTTP request on a dedicated thread to avoid
        // panicking when called from within a tokio async runtime.
        let response = std::thread::spawn(move || -> Result<String> {
            let client = reqwest::blocking::Client::builder()
                .user_agent("Mozilla/5.0 (compatible; AkioBot/0.1)")
                .build()?;

            let mut url = Url::parse("https://html.duckduckgo.com/html/").unwrap();
            url.query_pairs_mut().append_pair("q", &query);

            Ok(client.get(url).send()?.text()?)
        })
        .join()
        .map_err(|_| anyhow!("search thread panicked"))??;

        let document = Html::parse_document(&response);
        let result_sel = Selector::parse(".result").unwrap();
        let title_sel = Selector::parse(".result__a").unwrap();
        let snippet_sel = Selector::parse(".result__snippet").unwrap();

        let mut results = Vec::new();
        for element in document.select(&result_sel) {
            if results.len() >= num_results {
                break;
            }

            let title = element
                .select(&title_sel)
                .next()
                .map(|e| e.text().collect::<String>())
                .unwrap_or_default();

            let url = element
                .select(&title_sel)
                .next()
                .and_then(|e| e.value().attr("href"))
                .map(extract_url)
                .unwrap_or_default();

            let snippet = element
                .select(&snippet_sel)
                .next()
                .map(|e| e.text().collect::<String>())
                .unwrap_or_default();

            if !title.is_empty() {
                results.push(SearchResult { title, url, snippet });
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
