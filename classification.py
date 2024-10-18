def get_classes_v1(records):
    classification = {}

    for record in records:
        first_word = record.split()[0]

        if first_word in classification:
            classification[first_word].append(record)
        else:
            classification[first_word] = [record]

    return classification
