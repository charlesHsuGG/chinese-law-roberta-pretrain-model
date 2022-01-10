from __future__ import absolute_import, division, print_function

import argparse

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_file", default=None, type=str, required=True,
                        help="The input data file (a text file).")
    parser.add_argument("--max_lines", type=int, default=50000000,
                        help="Preprocess Max limit Num of trian lines")
    args = parser.parse_args()

    file_id = 1
    text_line = ""
    with open(args.data_file, encoding="utf-8") as f:
        for index, line in enumerate(tqdm(f, total=num_lines)):
            if index > args.max_lines:
                text_line += line + "\n"
            
            if index % args.max_lines == 0:
                with open(args.data_file + "_" + str(file_id), 'wb') as handle:
                    handle.write(text_line)
                    text_line = ""
                    file_id += 1

if __name__ == "__main__":
    main()

