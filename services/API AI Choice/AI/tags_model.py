import asyncio
from fastapi import HTTPException

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


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
            "щ": 25, "ь": 26, "ы": 27, "ъ": 28, "э": 29, "ю": 30, "я": 31, "ц": 32
        }
        self.inputwords = []

    def create_model(self, n):
        self.input_to_hidden = nn.Linear(33, 33 + n * 2)
        self.hidden_to_output = nn.Linear(33 + n * 2, 33 + n * 2)
        self.hidden2_to_output = nn.Linear(33 + n * 2, n)
        self.SiLU = nn.SiLU()

    def forward(self, x):
        hidden = self.SiLU(self.input_to_hidden(x))
        hidden2 = self.SiLU(self.hidden_to_output(hidden))
        output = self.SiLU(self.hidden2_to_output(hidden2))
        return output

    def save(self):
        torch.save(self.state_dict(), 'ai_cashe/model_weights.pth')
        np.save(f'ai_cashe/input_data', self.inputwords)

    def load(self):
        self.inputwords = np.load(f'ai_cashe/input_data.npy').tolist()
        self.create_model(len(self.inputwords))
        self.load_state_dict(torch.load('ai_cashe/model_weights.pth'))

    def training_model(self):
        self.load()
        criterion = nn.MSELoss()
        self.create_model(len(self.inputwords))
        optimizer = optim.SGD(self.parameters(), lr=0.01)

        epochs = 36
        for epoch in range(epochs):
            e_loss = 0
            e_correct = 0

            for i in range(len(self.inputwords)):
                los = 0.9
                layer = torch.zeros(len(self.inputwords))
                temp = torch.zeros(33)

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
            temp = torch.zeros(33)
            for i in range(len(word)):
                temp[self.chars[word[i]]] = los + temp[self.chars[word[i]]] - (i/10)
            inp2 = temp.unsqueeze(0)

            output = self.forward(inp2)
            tag = output.argmax().item()
        else:
            tag = self.inputwords.index(word)
        return tag

    def add_words(self, text):
        temp = text.replace(',', '').replace('.', '').replace('!', '').replace('"', '').replace('?', '').replace('–', '').replace('-', '').replace(':', '')
        temp_words = temp.split()
        self.load()
        for word in temp_words:
            if word not in self.inputwords:
                self.inputwords.append(word)
        self.create_model(len(self.inputwords))
        self.save()


tags_ai = AI()
tags_ai.add_words(text="предусмотрены ли у вас льготы по оплате за обучение если да то какие документы нужно представить могу ли я подать документы на несколько специальностей я хочу поступить в ваш колледж нужно ли мне сдавать огэ или егэ можно ли поступить в ваш колледж на базе классов можно ли подать документы в приемную комиссию без документа об образовании а позже предоставить его в приёмную комиссию проводятся ли вступительные испытания в техникум как считается средний балл аттестата каким образом происходит зачисление получу ли я отсрочку от службы в армии существуют ли льготы при поступлении как происходит поступление на коммерческую платную основу по какому адресу принимают документы на поступление предоставляется ли студентам  техникума  общежитие какие документы понадобятся для поступления на программы спо как происходит зачисление на программы спо")
tags_ai.training_model()
print(tags_ai.inputwords)

async def convert_prompt_to_tag(prompt: str) -> list[str]:
    prompt = "".join([i for i in prompt.lower() if i in tags_ai.chars or i == " "])
    if len(prompt) == 0:
        raise HTTPException(status_code=400, detail=f"Не понял вас. Запрос некорректный")
    tags = [tags_ai.inputwords[tags_ai.return_word(i)] for i in prompt.split(" ")]
    return tags
