use crossterm::event::{KeyCode, KeyModifiers};
use reedline::{
    default_vi_insert_keybindings, default_vi_normal_keybindings, DefaultPrompt,
    DefaultPromptSegment, EditCommand, FileBackedHistory, Reedline, ReedlineEvent, Signal, Vi,
};
use std::{env, path};

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

    let mut line_editor = Reedline::create()
        .with_edit_mode(Box::new(Vi::new(
            insert_keybindings,
            default_vi_normal_keybindings(),
        )))
        .with_history(history);

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
