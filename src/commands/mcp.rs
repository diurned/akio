use anyhow::{bail, Result};

use crate::mcp::client::{MCPClient, McpConfig, McpServerEntry};

/// `akio mcp list` — show all registered servers and, if reachable, their tools.
pub async fn list() -> Result<()> {
    let config = McpConfig::load()?;
    if config.servers.is_empty() {
        println!("No MCP servers registered. Use `akio mcp add` to add one.");
        return Ok(());
    }

    for entry in &config.servers {
        println!(
            "{} ({} {})",
            entry.name,
            entry.command,
            entry.args.join(" ")
        );

        let mut client = MCPClient::new();
        match client.connect_to_server(&entry.command, &entry.args).await {
            Ok(tools) => {
                for tool in &tools {
                    println!(
                        "  - {}: {}",
                        tool.name,
                        tool.description.as_deref().unwrap_or("")
                    );
                }
                let _ = client.cleanup().await;
            }
            Err(e) => {
                println!("  (could not connect: {e})");
            }
        }
    }

    Ok(())
}

/// `akio mcp add --name <name> --command <cmd> --args <args...>`
pub fn add(name: &str, command: &str, args: &[String]) -> Result<()> {
    let mut config = McpConfig::load()?;
    config.add(McpServerEntry {
        name: name.to_owned(),
        command: command.to_owned(),
        args: args.to_vec(),
    });
    config.save()?;

    println!(
        "Added MCP server '{}' → {} {}",
        name,
        command,
        args.join(" ")
    );
    Ok(())
}

/// `akio mcp remove <name>`
pub fn remove(name: &str) -> Result<()> {
    let mut config = McpConfig::load()?;
    if config.remove(name) {
        config.save()?;
        println!("Removed MCP server '{name}'");
    } else {
        bail!("no MCP server named '{name}' found");
    }
    Ok(())
}
