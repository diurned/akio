use crossterm::event::{KeyCode, KeyModifiers};
use nu_ansi_term::{Color, Style};
use reedline::{
    default_vi_insert_keybindings, default_vi_normal_keybindings, DefaultPrompt,
    DefaultPromptSegment, EditCommand, FileBackedHistory, Highlighter, Reedline, ReedlineEvent,
    Signal, StyledText, Vi,
};
use std::{env, path};

struct CommandHighlighter {
    commands: Vec<String>,
}

impl Highlighter for CommandHighlighter {
    fn highlight(&self, line: &str, _cursor: usize) -> StyledText {
        let mut styled = StyledText::new();
        if line.starts_with('/') {
            let is_known = self
                .commands
                .iter()
                .any(|c| c == line || c.starts_with(line));
            let color = if is_known { Color::Green } else { Color::Red };
            styled.push((Style::new().fg(color), line.to_string()));
        } else {
            styled.push((Style::new(), line.to_string()));
        }
        styled
    }
}

pub fn read_input() -> Option<String> {
    let mut history_path =
        path::PathBuf::from(env::home_dir().unwrap_or_else(|| path::PathBuf::from(".")));
    history_path.push(".akio");
    history_path.push("history");
    history_path.set_extension("txt");

    let history = Box::new(
        FileBackedHistory::with_file(5, history_path.into())
            .expect("Error configuring history with file"),
    );

    let mut insert_keybindings = default_vi_insert_keybindings();

    insert_keybindings.add_binding(
        KeyModifiers::SHIFT,
        KeyCode::Enter,
        ReedlineEvent::Edit(vec![EditCommand::InsertNewline]),
    );

    // NOTE: Some terminals can't distinguish shift+enter from enter so use alt.
    insert_keybindings.add_binding(
        KeyModifiers::ALT,
        KeyCode::Enter,
        ReedlineEvent::Edit(vec![EditCommand::InsertNewline]),
    );

    let commands = vec!["/quit".into(), "/exit".into()];

    let mut line_editor = Reedline::create()
        .with_edit_mode(Box::new(Vi::new(
            insert_keybindings,
            default_vi_normal_keybindings(),
        )))
        .with_history(history)
        .with_highlighter(Box::new(CommandHighlighter { commands }));

    let prompt = DefaultPrompt::new(
        DefaultPromptSegment::Basic(("Ask something... ").to_string()),
        DefaultPromptSegment::Empty,
    );

    loop {
        let sig = line_editor.read_line(&prompt);
        match sig {
            Ok(Signal::Success(buffer)) => {
                return Some(buffer);
            }
            Ok(Signal::CtrlD) | Ok(Signal::CtrlC) => {
                println!("\nbye");
                break None;
            }
            x => {
                println!("Event: {:?}", x);
            }
        }
    }
}
