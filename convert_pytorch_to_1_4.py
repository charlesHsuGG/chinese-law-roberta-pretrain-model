import torch
import argparse
import transformers
from transformers import BertModel, BertTokenizer

if __name__ == "__main__":
    print(transformers.__version__)
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_folder", default=None, type=str, required=True,
                        help="The input data file (a text file).")
    args = parser.parse_args()
    file = args.data_folder + "pytorch_model.bin"
    state_dict = torch.load(file, map_location="cpu")
    torch.save(state_dict, file, _use_new_zipfile_serialization=False)
    tokenizer = BertTokenizer.from_pretrained(args.data_folder, do_lower_case=True)
    model = BertModel.from_pretrained(args.data_folder)
    print(model)