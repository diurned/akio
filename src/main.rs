mod ffi;
mod inference;

fn main() {
    let model = "Ornith-1.5-9B-Q4_K_M.gguf";
    inference::llm::serve(model.to_string());
}
