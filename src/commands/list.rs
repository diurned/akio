use anyhow::Result;


// TODO: Add tags in table
pub fn list(all: bool) -> Result<()> {
    if all {
        // Show every model in the registry, marking which are downloaded.
        let dir = crate::models::models_dir();
        println!("{:<40} {:<30} {}", "REPO", "FILE", "STATUS");
        println!("{}", "-".repeat(80));
        for entry in crate::models::WHITELISTED_MODELS {
            let downloaded = dir.join(entry.filename).exists();
            let status = if downloaded { "downloaded" } else { "not downloaded" };
            println!("{:<40} {:<30} {}", entry.repo, entry.filename, status);
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
                    println!("{:<40} {}", "REPO", "FILE");
                    println!("{}", "-".repeat(72));
                    found = true;
                }
                println!("{:<40} {}", entry.repo, entry.filename);
            }
        }
        if !found {
            println!("No models downloaded yet. Run `akio pull <repo>` to download one.");
        }
    }
    Ok(())
}
