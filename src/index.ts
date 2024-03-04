import "./styles/main.css"
import Renderer from "./lib/renderer";

export interface InfoInterface {
    renderTime: HTMLParagraphElement
}

const infoRenderTime = document.getElementById('renderTime') as HTMLParagraphElement
const info: InfoInterface = {renderTime: infoRenderTime}

let numCircles = Number(prompt('Select the number of particles', '10000'))
if (!(numCircles >= 1 && numCircles <= 10000000)) {
    numCircles = 10000
}

const canvas = document.getElementById('root') as HTMLCanvasElement
// canvas.width = window.innerWidth
// canvas.height = window.innerHeight
const renderer = new Renderer(canvas, info, numCircles)
if (await renderer.init()) {
    renderer.update()
} else {
    document.body.innerHTML = '<div class="not-supported"><h1>WebGPU not supported!</h1></div>'
}
