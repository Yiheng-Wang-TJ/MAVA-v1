import json
from model.app import predict
import re

def convert_to_number(letter):
    mapping = {
        '1': 0,
        '2': 1,
        '3': 2,
        '4': 3,
        '5': 4
    }
    return mapping.get(letter, 9)

def main():

    total = 0
    correct = 0

    with open('subset_answers.json', 'r', encoding='utf-8') as file:
        file = json.load(file)
        for key, value in file.items():

            with open('questions.json', 'r', encoding='utf-8') as q_file:
                egoschema_data = json.load(q_file)

                for i in range(0, len(egoschema_data)):
                    q_uid = egoschema_data[i]['q_uid']
                    if q_uid != key:
                        continue
                    print('round ' + str(i))

                    video_path = q_uid + ".mp4"
                    question = egoschema_data[i]['question']

                    task_id = 100  # set task_id for log check
                    key = "AIzaSyDq6yR8-iCvktapOKmmLL51RhADFIchY_U"

                    prompt = ["You will be given a question about a video and five possible answer options, where C "
                              "refers to the person wearing the camera. You will be provided frames from the video, "
                              "sampled across the video."]
                    prompt.append("Question:" + question + " Possible answer choices:"
                               "(1)" + egoschema_data[i]['option 0'] + "(2)" + egoschema_data[i]['option 1'] +
                               "(3)" + egoschema_data[i]['option 2'] + "(4)" + egoschema_data[i]['option 3'] +
                               "(5)" + egoschema_data[i]['option 4'] + 'Output the final answer in the format "(X)" '
                               'where X is the correct digit choice. DO NOT OUTPUT with the full answer text or any other words.')

                    interval = 20
                    num_frames = 8
                    compress_ratio = 3
                    pred = predict(task_id, key, video_path, question, prompt, interval, num_frames, compress_ratio)
                    match = re.search(r'\((\d)\)', pred).group(1)
                    print(convert_to_number(match))
                    total += 1
                    if convert_to_number(match) == value:
                        correct += 1

    acc = (correct / total) * 100
    formatted_accuracy = f"{acc:.2f}%"

    print(formatted_accuracy)

if __name__ == '__main__':
    main()
