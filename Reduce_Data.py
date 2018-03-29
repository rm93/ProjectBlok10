import os
import random as rd
from shutil import copyfile
from tqdm import tqdm


def parse_csv(path):
    print("Parsing dataset...")
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
    print("Parsed dataset!")
    return dataset, header


def reduce_data(dataset, nr_shuffle, nr_findings, attr_to_rm):
    print("Creating new dataset...")
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
    print("Created new dataset!")
    return new_dataset


def create_new_imageset(image_input_path, image_output_path, dataset):
    files_used = []
    for instance in dataset:
        files_used.append(instance[0])
    try:
        os.stat(image_output_path)
    except:
        os.mkdir(image_output_path)
    files = next(os.walk(image_input_path))[2]
    print("Creating a new imageset...")
    for filename in tqdm(files):
        if filename in files_used:
            src = image_input_path + "/" + filename
            dst = image_output_path + "/" + filename
            copyfile(src, dst)
    print("Created new imageset!")


def save_csv(path, dataset, header):
    print("Saving csv...")
    with open(path, "w") as file:
        file.write(header)
        for instance in dataset:
            file.write(",".join(instance))
        file.close()
    print("Saved csv!")


def main():
    input_path = "data/sample_labels.csv"
    output_path = "data/reduced_sample_labels.csv"
    image_input_path = "data/images"
    image_output_path = "data/reduced_images"
    nr_findings = 459
    nr_shuffle = 1000
    attr_to_rm = ["Hernia", "Pneumonia", "Fibrosis", "Edema", "Emphysema", "Cardiomegaly", "Pleural_Thickening"]
    dataset, header = parse_csv(input_path)
    new_dataset = reduce_data(dataset, nr_shuffle, nr_findings, attr_to_rm)
    create_new_imageset(image_input_path, image_output_path, new_dataset)
    save_csv(output_path, new_dataset, header)
    print("Done!")


if __name__ == "__main__":
    main()
