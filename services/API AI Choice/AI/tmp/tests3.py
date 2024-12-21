import openpyxl


def process_xlsx():
    workbook = openpyxl.load_workbook("questions.xlsx")
    sheet = workbook.active

    input_data = []
    output_data = []
    _input_data = []

    _answer = ""
    _numbernone = 0
    _lastNumber = 0
    block = [32, 38, 40, 41, 44]

    for row in sheet.iter_rows(values_only=True):
        Number = row[0]
        question = row[1]
        answer = row[2]
        if Number:
            _lastNumber = Number
        if 0 < _lastNumber < 51 and _lastNumber not in block:
            if Number:
                if answer and question and answer != '' and question != '':
                    if _lastNumber > 1:
                        input_data.append(_input_data)
                    _input_data = []
                    _numbernone = 0
                    _answer = answer
                    _input_data.append(question)
                    output_data.append(answer)
                else:
                    if _lastNumber > 1:
                        input_data.append(_input_data)
                    _input_data = []
                    _numbernone = Number
                    _answer = ''
            else:
                if question and _numbernone == 0 and question != '' and _answer and _answer != '':
                    _input_data.append(question)

    return input_data, output_data


