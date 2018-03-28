import random as rd


def parse_csv(path):
    dataset = []
    header = "No header"
    is_header = True
    with open(path, "r") as file:
        for line in file:
            if not(is_header):
                instance = line.split(",")
                dataset.append(instance)
            else:
                header = line
                is_header = False
        file.close()
    return dataset, header


def reduce_data(dataset, nr_shuffle, nr_findings, attr_to_rm):
    new_dataset = []
    no_findings = []
    for instance in dataset:
        label = instance[1].split("|")[0]
        if label == "No Finding":
            no_findings.append(instance)
        elif not(label in attr_to_rm):
            new_dataset.append(instance)
    for i in range(nr_shuffle):
        rd.shuffle(no_findings)
    for instance in no_findings[0:nr_findings]:
        new_dataset.append(instance)
    for i in range(nr_shuffle):
        rd.shuffle(new_dataset)
    return new_dataset


def save_csv(path, dataset, header):
    with open(path, "w") as file:
        file.write(header)
        for instance in dataset:
            file.write(",".join(instance))
        file.close()


def main():
    input_path = "data/sample_labels.csv"
    output_path = "data/reduced_sample_labels.csv"
    nr_findings = 459
    nr_shuffle = 1000
    attr_to_rm = ["Hernia", "Pneumonia", "Fibrosis", "Edema", "Emphysema", "Cardiomegaly", "Pleural_Thickening"]
    dataset, header = parse_csv(input_path)
    new_dataset = reduce_data(dataset, nr_shuffle, nr_findings, attr_to_rm)
    save_csv(output_path, new_dataset, header)
    print("Done!")


if __name__ == "__main__":
    main()
