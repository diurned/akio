#![allow(dead_code)]

use std::path::PathBuf;

use anyhow::Result;
use rmcp::model::{CallToolRequestParams, Tool};
use rmcp::service::RunningService;
use rmcp::transport::{ConfigureCommandExt, TokioChildProcess};
use rmcp::{RoleClient, ServiceExt};
use serde::{Deserialize, Serialize};
use tokio::process::Command;

/// Persistent config for registered MCP servers, stored in `~/.akio/mcp.json`.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct McpConfig {
    pub servers: Vec<McpServerEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpServerEntry {
    pub name: String,
    pub command: String,
    pub args: Vec<String>,
}

impl McpConfig {
    fn config_path() -> PathBuf {
        let home = std::env::var("HOME")
            .or_else(|_| std::env::var("USERPROFILE"))
            .expect("cannot determine home directory");
        PathBuf::from(home).join(".akio").join("mcp.json")
    }

    pub fn load() -> Result<Self> {
        let path = Self::config_path();
        if !path.exists() {
            return Ok(Self::default());
        }
        let data = std::fs::read_to_string(&path)?;
        Ok(serde_json::from_str(&data)?)
    }

    pub fn save(&self) -> Result<()> {
        let path = Self::config_path();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let data = serde_json::to_string_pretty(self)?;
        std::fs::write(&path, data)?;
        Ok(())
    }

    pub fn add(&mut self, entry: McpServerEntry) {
        // Replace if same name exists
        self.servers.retain(|s| s.name != entry.name);
        self.servers.push(entry);
    }

    pub fn remove(&mut self, name: &str) -> bool {
        let before = self.servers.len();
        self.servers.retain(|s| s.name != name);
        self.servers.len() < before
    }
}

/// An MCP client that connects to a server via stdio and exposes its tools.
pub struct MCPClient {
    session: Option<RunningService<RoleClient, ()>>,
}

impl MCPClient {
    pub fn new() -> Self {
        Self { session: None }
    }

    /// Connect to an MCP server by spawning the given command with args.
    pub async fn connect_to_server(&mut self, command: &str, args: &[String]) -> Result<Vec<Tool>> {
        let args = args.to_vec();
        let transport = TokioChildProcess::new(Command::new(command).configure(move |cmd| {
            for arg in &args {
                cmd.arg(arg);
            }
        }))?;

        let client = ().serve(transport).await?;

        let tools = client.list_all_tools().await?;
        println!(
            "\nConnected to server with tools: {:?}",
            tools.iter().map(|t| &t.name).collect::<Vec<_>>()
        );

        self.session = Some(client);
        Ok(tools)
    }

    /// Call a tool on the connected MCP server.
    pub async fn call_tool(&self, tool_name: &str, tool_args: serde_json::Value) -> Result<String> {
        let session = self
            .session
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Not connected to any server"))?;

        let mut params = CallToolRequestParams::new(tool_name.to_owned());
        if let serde_json::Value::Object(map) = tool_args {
            let json_obj: serde_json::Map<String, serde_json::Value> = map;
            params = params.with_arguments(json_obj);
        }

        let result = session.call_tool(params).await?;

        let text = result
            .content
            .iter()
            .filter_map(|c| c.as_text().map(|t| t.text.as_str()))
            .collect::<Vec<_>>()
            .join("\n");

        Ok(text)
    }

    /// List tools available on the connected server.
    pub async fn list_tools(&self) -> Result<Vec<Tool>> {
        let session = self
            .session
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Not connected to any server"))?;
        Ok(session.list_all_tools().await?)
    }

    /// Shut down the connection to the MCP server.
    pub async fn cleanup(mut self) -> Result<()> {
        if let Some(session) = self.session.take() {
            session.cancel().await?;
        }
        Ok(())
    }
}
