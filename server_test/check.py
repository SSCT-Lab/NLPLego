import json
import copy

if __name__ == "__main__":

    # 读处理后的源文件
    file_name = "all_booleans_dev.json"
    path = "./" + file_name
    boolean_file = open(path, mode="r", encoding='utf-8')
    boolean_json = json.load(boolean_file)


    file_name = "predictions_dev.json"
    path = "./" + file_name
    predict_file = open(path, mode="r", encoding='utf-8')
    predict_json = json.load(predict_file)

    l = 0
    right = 0
    wrong_list = []
    for k, v in predict_json.items():
        l += 1
        if not boolean_json[k]:
            wrong_list.append([k, v])
            right += 1
    print(l, right)

    # 读处理后的源文件
    file_name = "dev-v2.0.json"
    path = "./" + file_name
    predict_file = open(path, mode="r", encoding='utf-8')
    prediction_json = json.load(predict_file)
    prediction_data = prediction_json["data"]
    item_list = []
    index = 0
    result_dic = {}
    for i in prediction_data:
        prediction_item = i["paragraphs"]
        for j in prediction_item:
            temp = j["qas"]
            for k in temp:
                if len(k["answers"]) == 0:
                    result_dic[k["id"]] = [k["question"], j["context"], [a["text"] for a in k["plausible_answers"]], False]
                else:
                    result_dic[k["id"]] = [k["question"], j["context"], [a["text"] for a in k["answers"]], True]

    f = open("wrong_answer.txt", 'w', encoding="utf-8")
    f1 = open("wrong_no_answer.txt", 'w', encoding="utf-8")
    a = 0
    b = 0
    for i in wrong_list:
        if i[1] != "":
            a+=1
            content = result_dic[i[0]]
            f.write("context: " + content[1] + "\n")
            f.write("ques: " + content[0] + "\n")
            if content[3]:
                f.write("answers: "+str(content[2]) + "\n")
            else:
                f.write("plausible_answers: " + str(content[2]) + "\n")
            f.write("predict_answers: " + i[1] + "\n")
            f.write("\n")
        else:
            b+=1
            content = result_dic[i[0]]
            f1.write("context: " + content[1] + "\n")
            f1.write("ques: " + content[0] + "\n")
            if content[3]:
                f1.write("answers: "+str(content[2]) + "\n")
            else:
                f1.write("plausible_answers: " + str(content[2]) + "\n")
            f1.write("predict_answers: " + i[1] + "\n")
            f1.write("\n")
    f.close()
    f1.close()
    print(a,b)


