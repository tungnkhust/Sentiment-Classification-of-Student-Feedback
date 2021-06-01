import requests


def test_predict(inputs):
    res = requests.post('http://localhost:5000/predict', json=inputs)
    print(res.json())


if __name__ == '__main__':
    inputs = {
        "text": "Cô giáo nhiệt tình thân thiện, giảng bài rất hay",
        "model_type": "bilstm"
    }
    test_predict(inputs)
