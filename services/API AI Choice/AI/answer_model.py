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
        self.hidden_to_output2 = None
        self.input_to_hidden = None
        self.inputwords = np.load(f'ai_cashe/input_data.npy').tolist()
        self.output = ["Зачисление абитуриентов происходит по показателю среднего балла аттестата.",
                       "При поступлении по программы СПО понадобятся следующие документы: документ, удостоверяющий личность, документ об образовании, СНИЛС.). Заявление будет составлено сотрудниками Приемной комиссии в присутствии абитуриента при личной подаче документов, либо рассмотрено в онлайн модуле при дистанционной подаче.",
                       "Нет, техникум  общежитие не предоставляет.",
                       "Приемная коммссия  москвского приборостроительног техникума работает по двум адресам: ул. Нежинская д. 7 и Нахимовский проспект д.21",
                       "Для поступления на коммерческую основу обучения необходимо подать полный пакет документов произвести оплату первого периода обучения. Факт оплаты является подтверждением согласия поступающего с условиями договора на оказание платных образовательных услуг. При поступлении на платное обучение несовершеннолетних граждан договор заключает один из родителей или законный представитель.",
                       'Все категории граждан поступают на равных условиях согласно Приказу Минпросвещения России от 02.09.2020 N 457 (ред. от 30.04.2021) «Об утверждении Порядка приема на обучение по образовательным программам среднего профессионального образования" поступление в колледж является общедоступным. Значение имеет средний балл аттестата поступающего."',
                       'В соответствии с изменениями в статью 24 Федерального закона «О воинской обязанности и военной службе» от 28.03.1998г. №53-ФЗ обучающимся предоставляется отсрочка от призыва на военную службу в течение всего периода обучения в колледже, но не свыше сроков получения среднего профессионального образования, установленных федеральными государственными образовательными стандартами.',
                       "По каждой специальности формируется рейтинговый список из подавших документы абитуриентов, от высшего к низшему баллу аттестата. После окончания приёма (17 августа 2022 года) к зачислению представляются абитуриенты, подавшие оригинал документа об образовании, и вошедшие в число, соответствующее контрольным цифрам приема по специальности.",
                       "Средний балл аттестата представляет собой сумму оценок по всем дисциплинам из приложения аттестата, разделенную на количество дисциплин.",
                       "Вступительные испытания не проводятся. При приеме на обучение в колледж учитывается только средний балл аттестата (конкурс аттестатов).",
                       "Без документа об образовании или его заверенной ксерокопии, документы приемной комиссией не принимаются. К зачислению представляются абитуриенты, предоставившие оригинал аттестата.",
                       "Прием в наше учебное заведение осуществляется только на базе 9 и 11 классов.",
                       "Для поступления в Колледж  нужен всего лишь аттестат. При поступлении в Колледж действует конкурс аттестатов.",
                       "Документы можно подавать на несколько специпльностей в порядке приоритета.",
                       "К сожалению, при поступлении на программы СПО льгот не предусмотрено.", ]
        self.input = ["как происходит зачисление на программы спо",
                      "какие документы понадобятся для поступления на программы спо",
                      "предоставляется ли студентам  техникума  общежитие",
                      "по какому адресу принимают документы на поступление",
                      "как происходит поступление на коммерческую платную основу",
                      "существуют ли льготы при поступлении",
                      "получу ли я отсрочку от службы в армии",
                      "каким образом происходит зачисление",
                      "как считается средний балл аттестата",
                      "проводятся ли вступительные испытания в техникум",
                      "можно ли подать документы в приемную комиссию без документа об образовании а позже "
                      "предоставить его в приёмную комиссию",
                      "можно ли поступить в ваш колледж на базе классов",
                      "я хочу поступить в ваш колледж нужно ли мне сдавать огэ или егэ",
                      "могу ли я подать документы на несколько специальностей",
                      "предусмотрены ли у вас льготы по оплате за обучение если да то какие документы нужно представить"]

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
        torch.save(self.state_dict(), 'ai_cashe/model_weights2.pth')

    def load(self):
        self.create_model(len(self.inputwords), len(self.output))
        self.load_state_dict(torch.load('ai_cashe/model_weights2.pth'))

    def training_model(self):
        self.load()
        criterion = nn.MSELoss()
        self.create_model(len(self.inputwords), len(self.output))
        optimizer = optim.SGD(self.parameters(), lr=0.01)

        epochs = 20
        for epoch in range(epochs):
            e_loss = 0
            e_correct = 0

            for i in range(len(self.output)):
                los = 0.9
                layer = torch.zeros(len(self.output))
                temp = torch.zeros(len(self.inputwords))
                ii = 0
                for word in self.input[i].split(" "):
                    if word in self.inputwords:
                        temp[self.inputwords.index(word)] = los + (ii / 6)
                        ii += 0.5

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

    def return_output(self, prompt):
        self.load()
        los = 0.9
        temp = torch.zeros(len(self.inputwords))
        ii = 0
        for word in prompt.split(" "):
            if word in self.inputwords:
                temp[self.inputwords.index(word)] = los + (ii / 6)
                ii += 0.5
        inp2 = temp.unsqueeze(0)

        output = self.forward(inp2)
        tag = output.argmax().item()
        return self.output[tag]


tags_ai = AI()
tags_ai.training_model()


async def convert_tag_to_answer(tags: list[str]) -> str:
    return tags_ai.return_output(tags)
