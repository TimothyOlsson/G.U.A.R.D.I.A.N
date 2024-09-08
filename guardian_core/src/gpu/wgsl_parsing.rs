use regex::Regex;

fn find_and_replace<R>(mut wgsl: String, to_find: &str, replace_fn: R) -> String
where
    R: Fn(String) -> String
{
    let re = Regex::new(&to_find).unwrap();
    let template = re
        .captures(&wgsl)
        .unwrap()
        .get(1)
        .unwrap()
        .as_str()
        .to_string();
    let to_replace = replace_fn(template);
    wgsl = re.replace(&wgsl, to_replace).to_string();
    wgsl
}


pub fn add_buffers(mut wgsl: String, group_bindings: Vec<[u32; 2]>) -> String {
    let n_buffers = group_bindings.len();
    let [group, binding] = group_bindings.last().unwrap();
    wgsl = find_and_replace(wgsl, "//!buffers last (.*)", |template| {
        template
            .replace("BUFFER_INDEX", &format!("{}", n_buffers - 1))
            .replace("GROUP", &format!("{}", group))
            .replace("BINDING", &format!("{}", binding))
    });

    // The last buffer is the only buffer if only 1
    if n_buffers <= 1 {
        return wgsl;
    }

    for (i, [group, binding]) in group_bindings.iter().take(n_buffers - 1).enumerate() {  // Skip last
        wgsl = find_and_replace(wgsl, "//!buffers !last (.*)", |template| {
            template
                .replace("BUFFER_INDEX", &format!("{}", i))
                .replace("GROUP", &format!("{}", group))
                .replace("BINDING", &format!("{}", binding))
        });
    }
    wgsl
}


pub fn add_cases(mut wgsl: String, n_buffers: usize) -> String {
    let cases_fn = |template: String| {
        let mut cases = String::new();
        for case in 0..n_buffers {
            cases += &(template.replace("BUFFER_INDEX", &format!("{}", case)) + "\n");
        }
        cases
    };
    wgsl = find_and_replace(wgsl, "//!cases read (.*)", cases_fn);
    wgsl = find_and_replace(wgsl, "//!cases write (.*)", cases_fn);
    wgsl
}


pub fn prepare_terminal_shader(mut wgsl: String) -> String {
    // Temporary, take this as an input with network settings
    let n_values = 1000;
    let n_values_per_array = 500;
    let n_buffers = n_values / n_values_per_array;
    let reminder = n_values % n_values_per_array;
    let group_bindings = (0..n_buffers).into_iter().map(|i| [1, i]).collect();

    // If divisable, means every buffer is filled, even the last
    let n_values_in_last_array = if reminder == 0 { n_values_per_array } else { reminder };

    // Start with sizes and constants

    wgsl = wgsl
        .replace("$TERMINALS_ARRAY_SIZE", &format!("{}", n_values_per_array))
        .replace("$TERMINALS_LAST_ARRAY_SIZE", &format!("{}", n_values_in_last_array))
        .replace("$TERMINALS_PER_Neuron", &format!("{}", 128))
        .replace("$TERMINAL_SIZE", &format!("{}", 128));
    wgsl = add_buffers(wgsl, group_bindings);
    wgsl = add_cases(wgsl, 1);
    wgsl
}


pub fn prepare_stage(mut wgsl: String) -> String {
    let ranges = include_str!("shaders/utils/ranges.wgsl").to_string();
    let params = include_str!("shaders/inputs/params.wgsl").to_string();
    let utils = include_str!("shaders/utils/utils.wgsl").to_string();
    wgsl = wgsl
        .replace("$RANGES", &ranges)
        .replace("$UTILS", &utils)
        .replace("$PARAMS", &params);
    wgsl
}
