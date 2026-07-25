pub mod glob;
pub mod read;
pub mod shell;
pub mod write;
pub mod websearch;
pub mod fetch;

use anyhow::Result;
use serde_json::Value;


pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn parameters_schema(&self) -> Value;
    fn execute(&self, args: Value) -> Result<String>;
}

pub struct ToolRegistry {
    tools: Vec<Box<dyn Tool>>,
}

impl ToolRegistry {
    pub fn new(tools: Vec<Box<dyn Tool>>) -> Self {
        Self { tools }
    }

    pub fn all(&self) -> &[Box<dyn Tool>] {
        &self.tools
    }

    pub fn find(&self, name: &str) -> Option<&dyn Tool> {
        self.tools.iter().find(|t| t.name() == name).map(|t| t.as_ref())
    }
}

pub fn default_registry() -> ToolRegistry {
    ToolRegistry::new(vec![
        Box::new(shell::ShellTool),
        Box::new(write::WriteTool),
        Box::new(read::ReadTool),
        Box::new(glob::GlobTool),
        Box::new(websearch::WebSearchTool),
        Box::new(fetch::FetchTool),
    ])
}

pub fn build_system_prompt(registry: &ToolRegistry) -> String {
    let mut tools_json: Vec<Value> = Vec::new();
    for tool in registry.all() {
        tools_json.push(serde_json::json!({
            "type": "function",
            "function": {
                "name": tool.name(),
                "description": tool.description(),
                "parameters": tool.parameters_schema()
            }
        }));
    }
    let tools_str = serde_json::to_string_pretty(&tools_json)
        .unwrap_or_else(|_| "[]".into());

    format!(
        "You are a helpful assistant with access to tools.\n\n\
        # Available tools\n\n\
        {tools_str}\n\n\
        When you need to call a tool, respond with ONLY a tool call block in this exact format, \
        with no other text before or after it:\n\n\
        <tool_call>\n\
        {{\"name\": \"<tool_name>\", \"arguments\": {{...}}}}\n\
        </tool_call>\n\n\
        After the tool result is returned to you, you may call another tool or provide a final answer.\n\
        Never fabricate tool results."
    )
}

#[derive(Debug, Clone)]
pub struct ToolCall {
    pub name: String,
    pub arguments: Value,
}

pub fn parse_tool_calls(response: &str) -> Vec<ToolCall> {
    let mut calls = Vec::new();
    let mut remaining = response;

    while let Some(start) = remaining.find("<tool_call>") {
        let after_open = &remaining[start + "<tool_call>".len()..];
        if let Some(end) = after_open.find("</tool_call>") {
            let body = after_open[..end].trim();
            if let Ok(v) = serde_json::from_str::<Value>(body) {
                if let Some(name) = v["name"].as_str() {
                    let arguments = v["arguments"].clone();
                    calls.push(ToolCall {
                        name: name.to_owned(),
                        arguments: if arguments.is_null() {
                            serde_json::json!({})
                        } else {
                            arguments
                        },
                    });
                }
            }
            remaining = &after_open[end + "</tool_call>".len()..];
        } else {
            break;
        }
    }

    calls
}
