import openpyxl


def parse_xlsx(path):
    xlsx_file = path

    wb = openpyxl.load_workbook(xlsx_file)
    sheet = wb.active

    unnormalized_records = []

    for row in sheet.iter_rows(min_row=9, min_col=1, max_col=1, values_only=True):
        if row[0] is not None and row[0] != 'Итого':
            unnormalized_records.append(row[0].strip())

    return unnormalized_records
