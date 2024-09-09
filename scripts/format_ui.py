import unicodedata

import gradio as gr
import regex as re
from modules import script_callbacks, scripts, shared

"""
Formatting settings
"""
SPACE_COMMAS = True
BRACKET2WEIGHT = True
CONV_SPUND = "None"

"""
Regex stuff
"""
brackets_opening = set("([{")
brackets_closing = set(")]}")
bracket_pairs = dict(zip("([{", ")]}"))
bracket_pairs_reverse = dict(zip(")]}", "([{"))
re_angle_bracket = re.compile(r"<[^>]+>")
re_networks = re.compile(r"<.+?>")
re_brackets = re.compile(r'([([{<])|([)\]}>])')
re_brackets_open = re.compile(r"(?<!\\)(\(+|\[+)")

"""
References
"""
ui_prompts = set()

"""
Functions
"""

def get_bracket_closing(c: str):
    return bracket_pairs.get(c, '')

def get_bracket_opening(c: str):
    return bracket_pairs_reverse.get(c, '')

def normalize_characters(data: str):
    return unicodedata.normalize("NFKC", data)

def tokenize(data: str) -> list:
    return re.split(r',', data)

def remove_whitespace_excessive(prompt: str):
    prompt = re.sub(r'\n\s*\n+', "\n", prompt)
    lines = prompt.split("\n")
    cleaned_lines = [" ".join(re.split(r'[^\S\r\n]+', line)).strip() for line in lines]
    return "\n".join(cleaned_lines).strip()

def align_brackets(prompt: str):
    return re_brackets.sub(lambda m: m.group(1) or m.group(2), prompt)

def space_and(prompt: str):
    def helper(match: re.Match):
        return " ".join(match.groups())

    return re.sub(r"(.*?)\s*(AND)\s*(.*?)", helper, prompt)

def align_colons(prompt: str):
    def process(match):
        content = match.group(1)
        if content.startswith('<') or content.startswith('('):
            return re.sub(r'\s*:\s*', ':', content)
        return re.sub(r'(\S+)\s*:\s*', r'\1: ', content)
    
    return re.sub(r'(<[^>]*>|\([^)]*\)|[^<>(]+)', process, prompt)

def align_commas(prompt: str):
    if not SPACE_COMMAS:
        return prompt

    split = [s.strip() for s in prompt.split(',') if s.strip()]
    return ", ".join(split)

def extract_networks(tokens: list):
    return [token for token in tokens if re_networks.match(token)]

def remove_networks(tokens: list):
    return [token for token in tokens if not re_networks.match(token)]

def remove_mismatched_brackets(prompt: str):
    stack = []
    pos = []
    ret = ""

    for i, c in enumerate(prompt):
        if c in brackets_opening:
            stack.append(c)
            pos.append(i)
            ret += c
        elif c in brackets_closing:
            if not stack:
                continue
            if stack[-1] == get_bracket_opening(c):
                stack.pop()
                pos.pop()
                ret += c
        else:
            ret += c

    while stack:
        bracket = stack.pop()
        p = pos.pop()
        ret = ret[:p] + ret[p + 1 :]

    return ret

def space_brackets(prompt: str):
    def helper(match: re.Match):
        return " ".join(match.groups())

    parts = re.split(r'(<[^>]+>)', prompt)
    for i in range(len(parts)):
        if not parts[i].startswith('<'):
            parts[i] = re.sub(r"([)\]}>])([([{<])", helper, parts[i])

    return ''.join(parts)

def align_alternating(prompt: str):
    return re.sub(r"\s*(\|)\s*", lambda match: match.group(1), prompt)

def bracket_to_weights(prompt: str):
    if not BRACKET2WEIGHT:
        return prompt
    
    # Identify regions enclosed within angle brackets
    excluded_regions = []
    for match in re_angle_bracket.finditer(prompt):
        excluded_regions.append((match.start(), match.end()))

    # Split the prompt into sections that are not within angle brackets
    segments = []
    previous_position = 0
    for start, end in excluded_regions:
        segments.append(prompt[previous_position:start])
        previous_position = end
    segments.append(prompt[previous_position:])

    # Process each segment separately
    updated_segments = []
    for segment in segments:
        depths, gradients, brackets = get_mappings(segment)
        pos = 0
        ret = segment

        while pos < len(ret):
            if ret[pos] in brackets_opening:
                open_bracketing = re_brackets_open.match(ret, pos)
                if open_bracketing:
                    consecutive = len(open_bracketing.group(0))
                    gradient_search = "".join(
                        map(
                            str,
                            reversed(
                                range(
                                    int(depths[pos]) - 1,
                                    int(depths[pos]) + consecutive
                                ),
                            )
                        )
                    )
                    is_square_brackets = "[" in open_bracketing.group(0)

                    insert_at, weight, valid_consecutive = get_weight(
                        ret,
                        gradients,
                        depths,
                        brackets,
                        open_bracketing.end(),
                        consecutive,
                        gradient_search,
                        is_square_brackets,
                    )

                    if weight:
                        # If weight already exists, ignore
                        current_weight = re.search(
                            r"(?<=:)(\d+.?\d*|\d*.?\d+)(?=[)\]]$)",
                            ret[:insert_at + 1]
                        )
                        if current_weight:
                            ret = (
                                ret[:open_bracketing.start()]
                                + "("
                                + ret[
                                    open_bracketing.start() + valid_consecutive : insert_at
                                ]
                                + ")"
                                + ret[insert_at + consecutive :]
                            )
                        else:
                            ret = (
                                ret[:open_bracketing.start()]
                                + "("
                                + ret[
                                    open_bracketing.start() + valid_consecutive : insert_at
                                ]
                                + f":{weight:.2f}".rstrip("0").rstrip(".")
                                + ")"
                                + ret[insert_at + consecutive :]
                            )

                    depths, gradients, brackets = get_mappings(ret)
                    pos += 1

            match = re.search(r"(?<!\\)[([]", ret[pos:])

            if not match:  # no more potential weight brackets to parse
                break

            pos += match.start()
        updated_segments.append(ret)

    # Reassemble the final prompt with the excluded regions
    final_prompt = ""
    for i, segment in enumerate(updated_segments):
        final_prompt += segment
        if i < len(excluded_regions):
            final_prompt += prompt[excluded_regions[i][0]:excluded_regions[i][1]]

    # Remove round brackets with weight 1
    final_prompt = re.sub(r'(?<!\\)\(([^:]+):1(?:\.0*)?\)', r'\1', final_prompt)

    return final_prompt

def depth_and_gradient(s: str):
    depth = 0
    depth_map = []
    gradient = []
    for c in s:
        if c in brackets_opening:
            depth += 1
            gradient.append('^')
        elif c in brackets_closing:
            depth -= 1
            gradient.append('v')
        else:
            gradient.append('-')
        depth_map.append(str(depth))
    return ''.join(depth_map), ''.join(gradient)

def get_mappings(s: str):
    depth_map, gradient = depth_and_gradient(s)
    brackets = ''.join(c if c in "[]()<>" else " " for c in s)
    return depth_map, gradient, brackets

def calculate_weight(d: str, is_square_brackets: bool):
    return 1 / 1.1 ** int(d) if is_square_brackets else 1 * 1.1 ** int(d)

def get_weight(
    prompt: str,
    map_gradient: list,
    map_depth: list,
    map_brackets: list,
    pos: int,
    ctv: int,
    gradient_search: str,
    is_square_brackets: bool = False,
):
    """Returns 0 if bracket was recognized as prompt editing, alternation, or composable."""
    # CURRENTLY DOES NOT TAKE INTO ACCOUNT COMPOSABLE?? DO WE EVEN NEED TO?
    # E.G. [a AND B :1.2] == (a AND B:1.1) != (a AND B:1.1) ????
    while pos + ctv <= len(prompt):
        if ctv == 0:
            return prompt, 0, 1
        a, b = pos, pos + ctv
        if prompt[a] in ":|" and is_square_brackets:
            if map_depth[-2] == map_depth[a]:
                return prompt, 0, 1
            if map_depth[a] in gradient_search:
                gradient_search = gradient_search.replace(map_depth[a], "")
                ctv -= 1
        elif map_gradient[a:b] == "v" * ctv and map_depth[a - 1 : b] == gradient_search:
            return a, calculate_weight(ctv, is_square_brackets), ctv
        elif "v" == map_gradient[a] and map_depth[a - 1 : b - 1] in gradient_search:
            narrowing = map_gradient[a:b].count("v")
            gradient_search = gradient_search[narrowing:]
            ctv -= 1
        pos += 1

    msg = f"Somehow weight index searching has gone outside of prompt length with prompt: {prompt}"
    raise Exception(msg)

def space_to_underscore(prompt: str):
    if CONV_SPUND == "None":
        return prompt
    elif CONV_SPUND == "Spaces to underscores":
        match = re.compile(r"(?<!BREAK) +(?!BREAK|[^<]*>)")
        replace = "_"
    else:
        match = re.compile(r"(?<!BREAK|_)_(?!_|BREAK|[^<]*>)")
        replace = " "

    tokens = [t.strip() for t in prompt.split(",")]
    tokens = [re.sub(match, replace, t) for t in tokens]

    return ",".join(tokens)

def escape_bracket_index(token, symbols, start_index=0):
    token_length = len(token)
    open_symbols = set(symbols)
    close_symbols = set(bracket_pairs[s] for s in symbols if s in bracket_pairs)
    
    depth = 0
    for i in range(start_index, token_length):
        if token[i] in open_symbols:
            depth += 1
        elif token[i] in close_symbols:
            depth -= 1
            if depth == 0:
                return i
    
    return token_length - 1

def dedup_tokens(prompt: str):
    # Find segments inside angle brackets
    angle_bracket_matches = list(re_angle_bracket.finditer(prompt))

    # Separate the prompt into chunks outside and inside angle brackets
    segments = []
    previous_position = 0

    for match in angle_bracket_matches:
        start, end = match.span()
        # Everything before the bracketed segment
        segments.append(('text', prompt[previous_position:start]))
        # The bracketed segment
        segments.append(('bracket', prompt[start:end]))
        previous_position = end

    # Add the last unbracketed segment, if any
    if previous_position < len(prompt):
        segments.append(('text', prompt[previous_position:]))

    # Deduplicate tokens across all text segments
    all_text = ' '.join([segment[1] for segment in segments if segment[0] == 'text'])
    tokens = [token.strip() for token in all_text.split(',')]
    unique_tokens = list(dict.fromkeys(tokens))  # Deduplicate

    # Reconstruct the prompt
    result = []
    for segment_type, segment_text in segments:
        if segment_type == 'bracket':
            result.append(segment_text)
        else:
            if unique_tokens:
                result.append(unique_tokens.pop(0))
                while unique_tokens and not re_angle_bracket.match(unique_tokens[0]):
                    result.append(', ' + unique_tokens.pop(0))

    prompt = ' '.join(result)
    prompt = re.sub(r'(<)', r' \1', prompt)
    prompt = re.sub(r'(>)(?!,)', r'\1 ', prompt)

    return prompt.strip()

def comma_after_bracket(prompt: str):
    return re.sub(r'(>)(\s*)([a-zA-Z])', r'\1,\2\3', prompt)

def comma_before_bracket(prompt: str):
    return re.sub(r',\s*(<)', r' \1', prompt)

def format_prompt(*prompts: tuple[dict]):
    sync_settings()

    ret = []

    for component, prompt in prompts[0].items():
        if not prompt or prompt.strip() == "":
            ret.append("")
            continue

        # Clean up the string
        prompt = normalize_characters(prompt)
        prompt = remove_mismatched_brackets(prompt)
        prompt = comma_after_bracket(prompt)

        # Remove dups
        prompt = dedup_tokens(prompt)

        # Clean up whitespace for cool beans
        prompt = remove_whitespace_excessive(prompt)
        prompt = space_to_underscore(prompt)
        prompt = align_brackets(prompt)
        prompt = space_and(prompt)  # for proper compositing alignment on colons
        prompt = space_brackets(prompt)
        prompt = align_colons(prompt)
        prompt = align_commas(prompt)
        prompt = align_alternating(prompt)
        prompt = bracket_to_weights(prompt)
        prompt = comma_before_bracket(prompt)

        ret.append(prompt)

    return ret

def convert_tags(*prompts: tuple[dict]):
    converted_prompts = []

    for component, prompt in prompts[0].items():
        if not prompt or prompt.strip() == "":
            converted_prompts.append("")
            continue

        # Replace underscores with spaces, escape round brackets, and join with commas
        normal_tags = ", ".join(
            tag.replace("_", " ").replace("(", r"\(").replace(")", r"\)")
            for tag in prompt.strip().split()
        )

        converted_prompts.append(normal_tags)

    return converted_prompts

def on_before_component(component: gr.component, **kwargs: dict):
    elem_id = kwargs.get("elem_id", None)

    if elem_id:
        if elem_id in ["txt2img_prompt", "txt2img_neg_prompt", "img2img_prompt", "img2img_neg_prompt", "hires_prompt", "hires_neg_prompt"]:
            ui_prompts.add(component)
        elif elem_id == "paste":
            with gr.Blocks(analytics_enabled=False) as ui_component:
                format_button = gr.Button(value="💫", elem_classes="tool", elem_id="format", tooltip="Format and clean up the prompt")
                format_button.click(fn=format_prompt, inputs=ui_prompts, outputs=ui_prompts)

                convert_button = gr.Button(value="✒️", elem_classes="tool", elem_id="convert_tags", tooltip="Convert Danbooru tags to comma-separated format")
                convert_button.click(fn=convert_tags, inputs=ui_prompts, outputs=ui_prompts)

                return ui_component
    return None

def on_ui_settings():
    section = ("pformat", "Prompt Formatter")
    shared.opts.add_option(
        "pformat_space_commas",
        shared.OptionInfo(
            True,
            "Add a spaces after comma",
            gr.Checkbox,
            {"interactive": True},
            section=section,
        ),
    )
    shared.opts.add_option(
        "pfromat_bracket2weight",
        shared.OptionInfo(
            True,
            "Convert excessive brackets to weights",
            gr.Checkbox,
            {"interactive": True},
            section=section,
        ),
    )
    shared.opts.add_option(
        "pfromat_convert_space_underscore",
        shared.OptionInfo(
            "None",
            "Space/underscore convert handling",
            gr.Radio,
            {"choices": ["None", "Spaces to underscores", "Underscores to spaces"]},
            section=section,
        ),
    )

    sync_settings()

def sync_settings():
    global SPACE_COMMAS, BRACKET2WEIGHT, CONV_SPUND
    SPACE_COMMAS = shared.opts.pformat_space_commas
    BRACKET2WEIGHT = shared.opts.pfromat_bracket2weight
    CONV_SPUND = shared.opts.pfromat_convert_space_underscore

script_callbacks.on_before_component(on_before_component)
script_callbacks.on_ui_settings(on_ui_settings)
