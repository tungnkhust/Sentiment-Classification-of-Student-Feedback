import React from 'react'
import { Form, Col, Row, Button } from 'react-bootstrap'
import api from './api/api'
import 'bootstrap/dist/css/bootstrap.min.css';
import colormap from 'colormap'
class App extends React.Component {
  constructor(props) {
    super(props)
    this.state = {
      text: '',
      model: 'bilstm',
      print: false
    }
  }
  changHandler = (e) => {

    let name = e.target.name;
    let value = e.target.value;
    this.setState({ [name]: value ,print:false})
  }
  submit = async (e) => {
    e.preventDefault()
    try {
      let res = await api.getPredict(this.state.text, this.state.model)
      let data = res.data;

      this.setState({ data:data, print:true });
      console.log(data)
    } catch (err) {
      console.log(err)
    }
  }

  colors = colormap({
    colormap: 'bluered',
    nshades: 255,
    format: 'hex',
    alpha: 1
  })

  getBackGroundColor = (v) => {
    let x = parseInt(254*v)
    return this.colors[x]
  }

  getColorMap = () => {
    let row = []
    for(let i = 0; i<254; i++){
      row.push(
        <div style= {{backgroundColor: this.colors[i], width:"3px", height:"50px"}} title={i/254.0}></div>
      )
    }
    return row
  }

  getSentimentDistributed = (text,w) => {
    let texts = text.split(" ");
    return texts.map((text,index) => {
      return <span title={w[index].toFixed(3)} className ="ml-1" style={{color: this.getBackGroundColor(w[index]) , cursor:"default"}}>{text}</span>
    })
  }

  getPredictLabel = () => {
    let attention 
    let colorMap
    if (this.state.model === 'attention' || this.state.model === 'character-attention') {
      attention = <tr>
        <td className='min'><b>Attention distribution</b></td>
        <td className='max'>
          <div className="d-flex justify-content-center flex-wrap">
            {this.getSentimentDistributed(this.state.data.text,this.state.data.sentiment_att_weight)}
          </div>
        </td>
        <td className='max'>
        <div className="d-flex justify-content-center flex-wrap">
            {this.getSentimentDistributed(this.state.data.text,this.state.data.topic_att_weight)}
          </div>
        </td>
      </tr>
      colorMap = this.getColorMap()
    }
    return (
      <div>
        <br></br>
        <div className="row">
          <table className='table table-bordered table-striped' style={{ width: '100%', fontSize: '22px' }}>
            <thead>
              <tr>
                <th></th>
                <th className='text-center'><h2>Sentiment </h2></th>
                <th className='text-center'><h2>Topic</h2></th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td className='min'><b>Phân loại</b></td>
                <td className='max'>{this.state.data.sentiment}</td>
                <td className='max'>{this.state.data.topic}</td>
              </tr>
              <tr>
                <td className='min'><b>Độ chắc chắn</b></td>
                <td className='max'>{(this.state.data.sentiment_confidence * 100).toFixed(2)}%</td>
                <td className='max'>{(this.state.data.topic_confidence * 100).toFixed(2)}%</td>
              </tr>
              {attention}
            </tbody>
          </table>
        </div>
        <div className="d-flex justify-content-center">
            {colorMap}
        </div>
      </div>
    )
  }

  render() {
    return (
      <div className="container">
        <Form onSubmit={this.submit}>
          <Form.Group controlId="exampleForm.ControlSelect1">
            <br />
            <Row>
              <div className="h1">Demo nhóm 19 - Phân tích quan điểm</div>
            </Row>
            <br />
            <br />
            <br />
            <Row>
              <Col xs={8}>
                <Form.Label><h6>Câu thử nghiệm</h6></Form.Label>
                <Form.Control as="textarea" value={this.state.text} required name="text" rows={4} onChange={this.changHandler} placeholder="Nhập câu cần dự đoán">
                </Form.Control>
              </Col>
              <Col xs={4}>
                <Form.Label><h6>Model</h6></Form.Label>
                <Form.Control as="select" value={this.state.model} name="model" onChange={this.changHandler}>
                  <option value="bilstm">Bi-LSTM</option>
                  <option value="character">Character</option>
                  <option value="attention">Attention</option>
                  <option value="character-attention">Character attention</option>
                </Form.Control>
              </Col>
            </Row>
          </Form.Group>
          <br />
          <br />
          <Form.Group>
            <Form.Row className="justify-content-center">
              <Col xs={4}>
                <Button variant="primary" type="submit" className="w-100">Dự đoán</Button>
              </Col>
            </Form.Row>
          </Form.Group>
        </Form>
        <div>
          {this.state.print && this.getPredictLabel()}
        </div>
      </div>
    )
  }
}

export default App;
