import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from difflib import SequenceMatcher


class AI(nn.Module):
    def __init__(self):
        super(AI, self).__init__()
        self.SiLU = None
        self.hidden2_to_output = None
        self.hidden_to_output = None
        self.input_to_hidden = None
        self.chars = {
            "а": 0, "б": 1, "в": 2, "г": 3, "д": 4, "е": 5, "ё": 6, "ж": 7, "з": 8, "и": 9, "й": 10, "к": 11, "л": 12,
            "м": 13, "н": 14, "о": 15, "п": 16, "р": 17, "с": 18, "т": 19, "у": 20, "ф": 21, "х": 22, "ч": 23, "ш": 24,
            "щ": 25, "ь": 26, "ы": 27, "ъ": 28, "э": 29, "ю": 30, "я": 31, "ц": 32, "0": 33, "1": 34, "2": 35, "3": 36,
            "4": 37, "5": 38, "6": 39, "7": 40, "8": 41, "9": 42, "w": 43, "e": 44, "b": 45, "i": 46, "t": 47,
        }
        self.inputwords = []

    def create_model(self, n):
        self.input_to_hidden = nn.Linear(48, 48 + n * 2)
        self.hidden_to_output = nn.Linear(48 + n * 2, 48 + n * 2)
        self.hidden2_to_output = nn.Linear(48 + n * 2, n)
        self.SiLU = nn.SiLU()

    def forward(self, x):
        hidden = self.SiLU(self.input_to_hidden(x))
        hidden2 = self.SiLU(self.hidden_to_output(hidden))
        output = self.SiLU(self.hidden2_to_output(hidden2))
        return output

    def save(self):
        torch.save(self.state_dict(), 'model_weights.pth')
        np.save(f'input_data', self.inputwords)

    def load(self):
        self.inputwords = np.load(f'input_data.npy').tolist()
        self.create_model(len(self.inputwords))
        self.load_state_dict(torch.load('model_weights.pth'))

    def training_model(self):
        self.load()
        criterion = nn.MSELoss()
        self.create_model(len(self.inputwords))
        optimizer = optim.SGD(self.parameters(), lr=0.01)

        epochs = 5
        for epoch in range(epochs):
            e_loss = 0
            e_correct = 0

            for i in range(len(self.inputwords)):
                los = 0.9
                layer = torch.zeros(len(self.inputwords))
                temp = torch.zeros(48)

                for ii in range(len(self.inputwords[i])):
                    temp[self.chars[self.inputwords[i][ii]]] = los + temp[self.chars[self.inputwords[i][ii]]] - (ii/10)

                layer[i] = 1
                inp12 = temp.unsqueeze(0)
                label = layer.unsqueeze(0)

                optimizer.step()

                output = self.forward(inp12)
                loss = criterion(output, label)
                e_loss += loss.item()
                e_correct += int(output.argmax() == label.argmax())

                loss.backward()
                optimizer.step()
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {e_loss:.6f}, Correct: {e_correct}/{len(self.inputwords)}")

        self.save()

    def return_word(self, word):
        self.load()
        los = 0.9
        if word not in self.inputwords:
            temp = torch.zeros(48)
            for i in range(len(word)):
                temp[self.chars[word[i]]] = los + temp[self.chars[word[i]]] - (i/10)
            inp2 = temp.unsqueeze(0)

            output = self.forward(inp2)
            tag = output.argmax().item()
        else:
            tag = self.inputwords.index(word)
        print(self.inputwords[tag])
        return tag

    def add_words(self, text):
        temp = text.replace(':', '')
        temp_words = temp.split()
        self.load()
        for word in temp_words:
            is_unique = True
            if word not in self.inputwords:
                for other_word in self.inputwords:
                    if word != other_word:
                        similarity = SequenceMatcher(None, word, other_word).ratio()
                        if similarity >= 0.8:
                            is_unique = False
                            break
                if is_unique:
                    self.inputwords.append(word)
        self.create_model(len(self.inputwords))
        self.save()


def add_words(input: list[list[str]]):
    text = ""
    tags_ai = AI()
    remove_chars = '.,?"—-'
    translation_table = str.maketrans("", "", remove_chars)
    for i in range(len(input)):
        for inp in input[i]:
            text += f" {inp.translate(translation_table).lower()}"
    tags_ai.add_words(text=text)
    tags_ai.training_model()


tags_ai = AI()
tags_ai.return_word("общежитие")