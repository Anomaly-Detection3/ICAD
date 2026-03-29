import pandas as pd
import numpy as np
import os
import torch
from einops import rearrange


def read_text_file(directory_path):
    text_file_context = []

    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory_path, filename)

            with open(file_path, "r", encoding='utf-8') as file:
                context = file.read()
                text_file_context.append(context)

    return text_file_context


