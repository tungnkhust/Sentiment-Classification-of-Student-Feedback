import axios from 'axios'

const headers = {
    'Content-Type': 'application/json',
}

const api = {
    getPredict: async (sentence, modelType) => {
        return axios.post(
            'http://localhost:5000/predict',
            {
                text: sentence,
                model_type: modelType
            },
            {
                headers: headers
            }
        )
    }
}

export default api