"""
This module converts CNN/Dailymail dataset from the original .story form 
into jsonl file {text: [str], summary:[str]}
i.e. performs preprocessing, sentence splitting and saving
"""
# Some chunks are from https://github.com/abisee/cnn-dailymail/blob/master/make_datafiles.py
import regex
import argparse
from functools import partial as p
from functools import reduce
import regex as re
from pathlib import Path
import json
from itertools import takewhile, dropwhile
from typing import List, Tuple, Dict
def compose(*functions):
    return reduce(lambda f, g: lambda x: f(g(x)), functions, lambda x: x)

# #### Special character normalization
decode_caret_return = p(re.sub, "_x000D_", "\r")

remove_caret_return = p(re.sub, "\r", "")
remove_linefeed = p(re.sub, "\n", "")
remove_space_inbetween_punctuation = p(re.sub, "(?<=[.,!?;])\s+(?=[.,?!;])", "")

deduplicate_commas = p(re.sub, ",{2,}", ",")
deduplicate_multiple_linefeeds = p(re.sub, "\n{1,}", "\n", flags=re.M)
deduplicate_spaces = p(re.sub, " {2,}", " ")

ensure_space_after_semicolon = p(re.sub, r";(?=\w)", "; ")
strip = lambda x: x.strip()

cleaning_function = compose(
    *reversed([
    # return caret normalization
    decode_caret_return,
    remove_caret_return,
    remove_space_inbetween_punctuation,
    # misc
    deduplicate_commas,
    # cosmetics
    deduplicate_multiple_linefeeds,
    ensure_space_after_semicolon,
    deduplicate_spaces,
    strip,
    ])
)

def fix_missing_period(line):
    """Adds a period to a line that is missing a period"""
    if "@highlight" in line: return line
    if line=="": return line
    if line[-1] in END_TOKENS: return line
    # print line[-1]
    return line + " ."

num_expected_cnn_stories = 92579
num_expected_dm_stories = 219506


HIGHLIGHTS = "@highlight"

dm_single_close_quote = u'\u2019' # unicode
dm_double_close_quote = u'\u201d'
END_TOKENS = ['.', '!', '?', '...', "'", "`", '"', dm_single_close_quote, dm_double_close_quote, ")"] # acceptable ways to end a sentence

# These are the number of .story files we expect there to be in cnn_stories_dir and dm_stories_dir


# f = "ffff522cebe5ad9dcfb6dfc476b8f423f3f8dd34.story"
def process_file(file_path: Path) -> Tuple[List[str], List[str]]:
    with open(file_path, "r") as inp:
        lines = inp.readlines()
        lines = map(cleaning_function, lines)
        lines = map(fix_missing_period, lines)
        lines = list(filter(lambda x: len(x)>2, lines))
        text_lines = list(takewhile(lambda x: HIGHLIGHTS not in x, lines))
        highlight_lines = dropwhile(lambda x: HIGHLIGHTS not in x, lines)
        highlight_lines = list(filter(lambda x: HIGHLIGHTS not in x, highlight_lines))

        return (text_lines, highlight_lines)

def append_line(file_path: Path, line: str):
    with open(file_path, "a") as fout:
        fout.write(line + "\n")

def convert_dataset(arguments):
    input_path = Path(arguments.data_path).resolve()
    output_path = Path(arguments.write_path).resolve()
    output_path.mkdir(exist_ok=True)
    output_file = output_path / "dataset.jsonl"
    cnn_path = input_path.joinpath("cnn", "stories")
    dm_path = input_path.joinpath("cnn", "stories")

    # Check contents
    assert cnn_path.exists() and cnn_path.is_dir(), "CNN folder not in provided path"
    assert dm_path.exists() and dm_path.is_dir(), "DM folder not in provided path"
    # if not len(list(cnn_path.glob(r"*.story"))) == num_expected_cnn_stories:
    #     print("Unexpected num of cnn stories")
    # if not len(list(dm_path.glob(r"*.story"))) == num_expected_dm_stories:
    #     print("Unexpected num of dm stories")

    for folder in [cnn_path, dm_path]:
        print("Processing ", folder.parent.name)
        for file in folder.iterdir():
            text, summary = process_file(file)
            if (not text) or (not summary):
                print(f"{file.name} is missing data")
                continue
            line = json.dumps(dict(text=text, summary=summary), ensure_ascii=False)
            append_line(output_file, line)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Create sentence ranking for futher truncation'
    )
    parser.add_argument('--data_path', type=str, required=True,
        help='Path to the folder with the original CNN/DM dataset in a form of [cnn/dailymail]/*.story')
    parser.add_argument('--write_path', type=str, required=True,
        help='Path to store the rankings {sent_id: [int]}')

    args = parser.parse_args()
    assert Path(args.data_path).exists()
    convert_dataset(args)