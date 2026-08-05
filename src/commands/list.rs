use anyhow::Result;


// TODO: Add tags in table
pub fn list(all: bool) -> Result<()> {
    if all {
        // Show every model in the registry, marking which are downloaded.
        let dir = crate::models::models_dir();
        println!("{:<30} {:<40} {}", "FILE", "REPO", "STATUS");
        println!("{}", "-".repeat(78));
        for entry in crate::models::WHITELISTED_MODELS {
            let downloaded = dir.join(entry.filename).exists();
            let status = if downloaded { "\x1b[32m✔\x1b[0m" } else { "\x1b[31m✘\x1b[0m" };
            println!("{:<30} {:<40} {}", entry.filename, entry.repo, status);
        }
    } else {
        // Show only downloaded models.
        let dir = crate::models::models_dir();
        if !dir.exists() {
            println!("No models downloaded yet. Run `akio pull <repo>` to download one.");
            return Ok(());
        }
        let mut found = false;
        for entry in crate::models::WHITELISTED_MODELS {
            // FIXME: Image generation models lack filenames, which causes them
            // to appear in the `akio list` even when they are not downloaded.
            let path = dir.join(entry.filename);
            if path.exists() {
                if !found {
                    println!("{:<30} {:<40}", "FILE", "REPO");
                    println!("{}", "-".repeat(72));
                    found = true;
                }
                println!("{:<30} {:<40}", entry.filename, entry.repo);
            }
        }
        if !found {
            println!("No models downloaded yet. Run `akio pull <repo>` to download one.");
        }
    }
    Ok(())
}
