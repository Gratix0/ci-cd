import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import tests2 as tags
import tests3 as data_inp_out


class AI(nn.Module):
    def __init__(self):
        super(AI, self).__init__()
        self.SiLU = None
        self.hidden2_to_output = None
        self.hidden_to_output = None
        self.hidden_to_output2 = None
        self.input_to_hidden = None
        self.inputwords = np.load(f'input_data.npy').tolist()
        self.output = []
        self.input = []

    def create_model(self, n, n2):
        self.input_to_hidden = nn.Linear(n, n * 3)
        self.hidden_to_output = nn.Linear(n * 3, n * 3)
        self.hidden_to_output2 = nn.Linear(n * 3, n * 3)
        self.hidden2_to_output = nn.Linear(n * 3, n2)
        self.SiLU = nn.SiLU()

    def forward(self, x):
        hidden = self.SiLU(self.input_to_hidden(x))
        hidden2 = self.SiLU(self.hidden_to_output(hidden))
        hidden3 = self.SiLU(self.hidden_to_output2(hidden2))
        output = self.SiLU(self.hidden2_to_output(hidden3))
        return output

    def save(self):
        torch.save(self.state_dict(), 'model_weights2.pth')

    def load(self):
        self.create_model(len(self.inputwords), len(self.output))
        state_dict = torch.load('model_weights2.pth', weights_only=True)
        self.load_state_dict(state_dict)

    def training_model(self, _input: list[list[str]], _output: list[str]):
        #self.load()
        tag_ai = tags.AI()
        remove_chars = '.,?"—-:'
        translation_table = str.maketrans("", "", remove_chars)
        self.input = _input
        self.output = _output
        criterion = nn.MSELoss()
        self.create_model(len(self.inputwords), len(self.output))
        optimizer = optim.SGD(self.parameters(), lr=0.01)

        epochs = 15
        for epoch in range(epochs):
            e_loss = 0
            e_correct = 0

            for i in range(len(self.input)):
                layer = torch.zeros(len(self.output))
                temp = torch.zeros(len(self.inputwords))
                for inp in self.input[i]:
                    los = 0.9
                    ii = 0
                    for word in inp.split(" "):
                        if word in self.inputwords:
                            temp[self.inputwords.index(word.translate(translation_table).lower())] = los + (ii / 6)
                            ii += 0.01
                        else:
                            temp[tag_ai.return_word(word.translate(translation_table).lower())] = los + (ii / 6)
                            ii += 0.01
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
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {e_loss:.6f}, Correct: {e_correct}/{len(self.output)}")

        self.save()

    def return_output(self, prompt, _output):
        self.output = _output
        self.load()
        count = 0
        tag_ai = tags.AI()
        remove_chars = '.,?"—-:'
        translation_table = str.maketrans("", "", remove_chars)
        los = 0.9
        temp = torch.zeros(len(self.inputwords))
        ii = 0
        for word in prompt.split(" "):
            if word in self.inputwords:
                temp[self.inputwords.index(word.translate(translation_table).lower())] = los + (ii / 6)
                ii += 0.5
                count += 1
            else:
                temp[tag_ai.return_word(word.translate(translation_table).lower())] = los + (ii / 6)
                ii += 0.5
        inp2 = temp.unsqueeze(0)

        output = self.forward(inp2)
        tag = output.argmax().item()
        if count >= 3:
           return self.output[tag]
        else:
            return "Я не понял ваш запрос"



tags_ai = AI()
_input, _output = data_inp_out.process_xlsx() # получаем дату
tags.add_words(_input) # создаем словарь и обучаем нейросеть для тегов

tags_ai.training_model(_input=_input, _output=_output)
def ret():
    e = ""
    while e != "exit":
        e = input("Ваш запрос: ")
        print(f"Ответ: {tags_ai.return_output(e, _output=_output)}")
ret()