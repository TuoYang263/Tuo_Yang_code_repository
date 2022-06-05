import os
import math
import codecs


def info_extraction():
    # the directory of storing images
    dir = './datasets/'

    # could use sort method to do ordering for list
    files = os.listdir(dir)

    files.sort()

    # ordering is done by ascending order
    # print(files)

    # read training,testing and validation dataset label files
    with codecs.open('./labels/train.txt', 'w', 'utf-8') as train_labels:
        train_labels.seek(0)
    with codecs.open('./labels/validation.txt', 'w', 'utf-8') as validation_labels:
        validation_labels.seek(0)
    with codecs.open('./labels/test.txt', 'w', 'utf-8') as test_labels:
        test_labels.seek(0)

    # print(train_labels)

    # we totally got 41 class data samples
    species_range = 41

    # the sum of this list should be 2942, and the marked result is correct
    # print(sum(species_range))

    # take 35% for training,15 % for validation, and 50% for testing
    for i in range(1, species_range+1):
        current_label = i - 1
        current_class_samples = []
        for file_name in files:
            if i < 10:
                compare_string = '0'+str(i)
            else:
                compare_string = str(i)
            if file_name.startswith(compare_string):
                current_class_samples.append(file_name)

        training_num = math.ceil(len(current_class_samples) * 0.5)
        validation_num = math.ceil(len(current_class_samples) * 0.15)
        testing_num = len(current_class_samples) - training_num

        for j in range(training_num):
            species_train = './datasets/'+current_class_samples[j]+' '+str(current_label)
            with codecs.open('./labels/train.txt', 'a', 'utf-8') as train_labels:
                train_labels.write(species_train)
                train_labels.write('\r\n')

        for k in range(validation_num):
            species_validation = './datasets/' + current_class_samples[k] + ' ' + str(current_label)
            with codecs.open('./labels/validation.txt', 'a', 'utf-8') as validation_labels:
                validation_labels.write(species_validation)
                validation_labels.write('\r\n')

        for m in range(len(current_class_samples)-testing_num+1, len(current_class_samples)):
            species_test = './datasets/' + current_class_samples[m] + ' ' + str(current_label)
            with codecs.open('./labels/test.txt', 'a', 'utf-8') as test_labels:
                test_labels.write(species_test)
                test_labels.write('\r\n')









