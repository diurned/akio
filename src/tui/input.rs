use std::io;
use std::io::Write;
use crossterm::{
    event::{self, Event, KeyCode, KeyModifiers},
    terminal::{disable_raw_mode, enable_raw_mode},
};

pub struct Input {
    pub value: String,
    pub character_index: usize,
}

impl Input {
    pub fn new() -> Self {
        Self { value: String::new(), character_index: 0 }
    }

    fn move_cursor_left(&mut self) {
        self.character_index = self.character_index.saturating_sub(1);
    }

    fn move_cursor_right(&mut self) {
        self.character_index = self.character_index.saturating_add(1).min(self.value.chars().count());
    }

    fn move_cursor_home(&mut self) { self.character_index = 0; }
    fn move_cursor_end(&mut self)  { self.character_index = self.value.chars().count(); }

    fn enter_char(&mut self, c: char) {
        let byte_idx = self.byte_index();
        self.value.insert(byte_idx, c);
        self.move_cursor_right();
    }

    fn delete_char(&mut self) {
        if self.character_index == 0 { return; }
        let before = self.value.chars().take(self.character_index - 1);
        let after  = self.value.chars().skip(self.character_index);
        self.value = before.chain(after).collect();
        self.move_cursor_left();
    }

    fn delete_char_forward(&mut self) {
        if self.character_index >= self.value.chars().count() { return; }
        let before = self.value.chars().take(self.character_index);
        let after  = self.value.chars().skip(self.character_index + 1);
        self.value = before.chain(after).collect();
    }

    fn submit(&mut self) -> String {
        let submitted = self.value.clone();
        self.value.clear();
        self.character_index = 0;
        submitted
    }

    fn byte_index(&self) -> usize {
        self.value
            .char_indices()
            .nth(self.character_index)
            .map(|(i, _)| i)
            .unwrap_or(self.value.len())
    }

    fn redraw(&self) {
        let display = self.value.replace('\n', "↵ ");
        print!("\r\x1b[2K\x1b[34m> \x1b[0m{}", display);
        let chars_after = self.value.chars().count() - self.character_index;
        if chars_after > 0 {
            print!("\x1b[{}D", chars_after);
        }
        io::stdout().flush().ok();
    }

    pub fn read_input() -> Option<String> {
        let mut input = Input::new();

        print!("\n\n\x1b[34m> \x1b[0m");
        io::stdout().flush().ok();
        enable_raw_mode().ok();

        loop {
            if let Ok(Event::Key(key)) = event::read() {
                match (key.code, key.modifiers) {
                    (KeyCode::Char('c'), KeyModifiers::CONTROL)
                    | (KeyCode::Char('d'), KeyModifiers::CONTROL) => {
                        disable_raw_mode().ok();
                        println!();
                        return None;
                    }

                    (KeyCode::Enter, KeyModifiers::NONE) => {
                        let msg = input.submit();
                        if !msg.is_empty() {
                            disable_raw_mode().ok();
                            println!();
                            return Some(msg);
                        }
                        // empty — re-prompt
                        print!("\n\n\x1b[34m> \x1b[0m");
                        io::stdout().flush().ok();
                        continue;
                    }

                    (KeyCode::Enter,     KeyModifiers::SHIFT) => input.enter_char('\n'),
                    (KeyCode::Left,      _) => input.move_cursor_left(),
                    (KeyCode::Right,     _) => input.move_cursor_right(),
                    (KeyCode::Home,      _) => input.move_cursor_home(),
                    (KeyCode::End,       _) => input.move_cursor_end(),
                    (KeyCode::Backspace, _) => input.delete_char(),
                    (KeyCode::Delete,    _) => input.delete_char_forward(),

                    (KeyCode::Char(c), KeyModifiers::NONE | KeyModifiers::SHIFT) => {
                        input.enter_char(c);
                    }

                    _ => {}
                }

                input.redraw();
            }
        }
    }
}
